//! The core PDF parsing module. This includes the `PdfParser` struct, which is somewhat tuned for
//! academic PDFs. In particular, it skips images and tables by default. This behavior might change
//! later. The parser also handles common math symbols and converts them to their corresponding
//! LaTeX equivalents.

use itertools::Itertools;
use log;
use ordered_float::OrderedFloat;
use std::char::decode_utf16;
use std::f32;
use std::str::Utf8Error;
use std::{error::Error, str};

use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::LazyLock;

use crate::edits::{Edit, EditType, apply_edits};
use crate::fonts::{FONT_TRANSFORMS, FontEncoding, FontSizeMarker, font_transform};
use crate::tokenizer::{Token, tokenize};
use lopdf::{Document, Object};

const ASCII_PLUS: u8 = b'+';

/// The default kerning tolerance between adjacent elements in a `TJ` to mark them as part of the same
/// word.
pub const DEFAULT_SAME_WORD_THRESHOLD: f32 = 60.0;
/// The default distance threshold to check alignment efforts in a table.
pub const DEFAULT_TABLE_EUCLIDEAN_THRESHOLD: f32 = 40.0;

/// A wrapper for all PDF parsing errors
#[derive(Debug, thiserror::Error)]
enum PdfError {
    #[error("Failed to get page content")]
    ContentError,
    #[error("Font key \"{0}\" not found in dictionary")]
    FontNotFound(String),
    #[error("Internal error: {0}")]
    InternalError(String),
    #[error("BaseFont value isn't a valid name")]
    InvalidFontName,
    #[error("Font name isn't valid UTF-8")]
    InvalidUtf8,
    #[error("Font object missing BaseFont field")]
    MissingBaseFont,
    #[error("Encoding error: {0}")]
    EncodingError(String),
    #[error("Font object missing Subtype field")]
    MissingSubtype,
    #[error("Failed to get page fonts")]
    PageFontError,
}

impl From<Utf8Error> for PdfError {
    fn from(_value: Utf8Error) -> Self {
        Self::InvalidUtf8
    }
}

/// Configuration for PDF parsing
#[derive(Debug)]
struct PdfParserConfig {
    /// Threshold for determining when to join words
    same_word_threshold: f32,
    /// Euclidean distance threshold between `Td` alignment to declare a table
    table_alignment_threshold: f32,
}

impl Default for PdfParserConfig {
    fn default() -> Self {
        Self {
            same_word_threshold: DEFAULT_SAME_WORD_THRESHOLD,
            table_alignment_threshold: DEFAULT_TABLE_EUCLIDEAN_THRESHOLD,
        }
    }
}

/// A detected section boundary.
#[derive(Debug, Clone)]
pub struct SectionBoundary {
    /// 0-indexed page number
    pub page_number: usize,
    /// Byte index into the extracted text
    pub byte_index: usize,
    /// Header level: 0 = title, 1 = section, 2 = subsection, etc.
    pub level: usize,
    /// Index of parent section for focal context traversal
    pub parent_idx: Option<usize>,
    /// The font size of the header
    pub font_size: f32,
}

impl PartialEq for SectionBoundary {
    fn eq(&self, other: &Self) -> bool {
        self.page_number == other.page_number && self.byte_index == other.byte_index
    }
}
impl Eq for SectionBoundary {}
impl std::hash::Hash for SectionBoundary {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.page_number.hash(state);
        self.byte_index.hash(state);
    }
}

/// The return type of `parse_content`. This includes the extracted text and the detected section
/// boundaries.
#[derive(Debug, Clone)]
pub struct ExtractedContent {
    /// The extracted text
    pub text_content: String,
    /// The list of detected section boundaries
    pub sections: Vec<SectionBoundary>,
    /// Page count
    pub page_count: usize,
}

impl ExtractedContent {
    /// Get the length of the content
    #[must_use]
    pub const fn len(&self) -> usize {
        self.text_content.len()
    }

    /// Check if there is any content present.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// A lazy-loaded hashmap of octal character replacements post-parsing.
/// Some of these come across because of ligature support in fonts. This
/// is not exhaustive, however.
static OCTAL_REPLACEMENTS: LazyLock<HashMap<&str, &str>> = LazyLock::new(|| {
    let mut m = HashMap::new();
    m.insert("\\050", "(");
    m.insert("\\051", ")");
    m.insert("\\002", "fi");
    m.insert("\\017", "*");
    m.insert("\\227", "--");
    m.insert("\\247", "Section ");
    m.insert("\\223", "\"");
    m.insert("\\224", "\"");
    m.insert("\\000", "-");

    m
});

struct PdfParser {
    /// The config
    config: PdfParserConfig,
    /// The current font we're using. This is not the font string in the dictionary (e.g. "F28"),
    /// but rather the font's name itself (e.g. "CMMI10").
    cur_font: String,
    /// The current font ID we're using. This is not the font string in the dictionary (e.g. "F28"),
    /// but rather the font's ID itself (e.g. "F28").
    cur_font_id: String,
    /// Current font size
    cur_font_size: f32,
    /// The \baselineskip set by the user.
    /// TODO: Actually compute this; for now, this is set to the pdflatex default of 1.2
    cur_baselineskip: f32,
    /// The current page's map of whether a font is a CID-keyed font or not.
    font_type: HashMap<(PageID, String), FontEncoding>,
}

/// `lopdf` references individual pages by a tuple of unsigned integers. Usually, the specific
/// values are irrelevant, and it is more useful to think about the page ID itself as one "thing".
type PageID = (u32, u16);

/// Having collected all the positions where the y position was changed, collect the edits
/// necessary to add sub/superscript markers. The core idea here is that when we "come back"
/// from a script, the y position will return to one that we've seen. This creates a span of
/// indices to look through. Within this span, we can parse nested scripts (but note that
/// these might not all follow the same "direction", particularly because for sums, the
/// upper limit comes before the summation symbol for some reason in LaTeX-created content
/// streams. We use the sign in the difference between y positions to determine what kind of
/// a script it is.
///
/// # Arguments
///
/// * `y_history`: A slice of (position, font size) tuples where each `position` refers to an
///   index in `parsed`.
/// * `parsed`: A mutable reference to a string that should be updated with sub/superscript
///   markers.
///
/// # Returns
///
/// A list of `Edit`s to apply.
#[must_use]
fn get_script_marker_edits(y_history: &[(usize, f32)], parsed: &mut String) -> Vec<Edit> {
    let mut additions: Vec<Edit> = Vec::new();
    let mut i = y_history.len().saturating_sub(1);
    while i > 0 {
        // Find the last index where the y position was equal to the y position recorded by
        // `y_history[i]`.
        #[allow(clippy::float_cmp)]
        let j = (0..i).rev().find(|k| y_history[*k].1 == y_history[i].1);

        if j.is_none() {
            i -= 1;
            continue;
        }

        // Start at the next position...
        let j_orig = j.unwrap();
        let mut j = j_orig + 1;

        // ...and go in pairs. We can only collect the additions at this point, since they may
        // be overlapping.
        while j < i {
            const BACKSLASH_ASCII: u8 = 92;

            // The offset measures how much we need to shift the opening curly braces by. This
            // is because while symbols are single characters in math fonts (such as CMEX),
            // they expand to a longer string, so we account for the difference in lengths.
            let offset = if parsed.as_bytes().get(y_history[j].0) == Some(&BACKSLASH_ASCII) {
                parsed[y_history[j].0..].find(' ').unwrap()
            } else {
                0
            };

            additions.push(Edit {
                start: y_history[j + 1].0.saturating_sub(1),
                end: y_history[j + 1].0.saturating_sub(1) + 1,
                r#type: EditType::Insert("}".into()),
            });

            // TODO: Refine the below rule.
            additions.push(Edit {
                start: y_history[j].0 + offset,
                end: y_history[j].0 + offset + 2, // both cases below have length 2
                r#type: EditType::Insert(if y_history[j].1 > y_history[j_orig].1 {
                    "^{".into()
                } else {
                    "_{".into()
                }),
            });

            j += 2;
        }

        i = j_orig.saturating_sub(1);
    }

    additions
}

impl Default for PdfParser {
    fn default() -> Self {
        Self::new(PdfParserConfig::default())
    }
}

/// An intermediate representation that contains the result of parsing one page.
struct ParseResult {
    /// The contents of a page
    content: String,
    /// A record of all font size changes, including the page number, byte index (in that page),
    /// new font size, and new font name
    font_size_markers: Vec<FontSizeMarker>,
    /// Body font size
    body_font_size: Option<f32>,
}

impl PdfParser {
    /// Creates a new parser with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use for parsing
    fn new(config: PdfParserConfig) -> Self {
        Self {
            config,
            cur_font: String::new(),
            cur_font_id: String::new(),
            cur_font_size: 12.0,   // Doesn't really matter
            cur_baselineskip: 1.2, // The pdflatex default
            font_type: HashMap::new(),
        }
    }

    fn with_default_config() -> Self {
        Self::new(PdfParserConfig::default())
    }

    /// Checks if a given font in a specific page of a document is a CID-keyed font.
    ///
    /// If it is, returns the `ToUnicode` CMap for that font; otherwise, returns
    /// FontEncoding::Simple. This function is *not* pure: it updates `self.font_type`, which
    /// acts as a cache for these results. Users of `parse_content` are responsible for emptying
    /// this between pages. However, `extract_text`, which is meant for external users, does this
    /// automatically.
    ///
    /// # Returns
    ///
    /// If the specified font is a CID-keyed font, returns `FontEncoding::CIDKeyed` with a
    /// reference to the `ToUnicode` CMap. Otherwise, returns `FontEncoding::Simple`.
    ///
    /// # Errors
    ///
    /// * `PdfError::PageFontError` if getting page fonts failed.
    /// * `PdfError::FontNotFound` if the font key does not exist.
    /// * `PdfError::EncodingError` if the font dictionary:
    ///      * Has an /Encoding that could not be read
    ///      * Indicates a CID-keyed font but does not have a ToUnicode CMap.
    ///      * Has a ToUnicode CMap that could not be read or deflated.
    ///      * Has a ToUnicode CMap without a `beginbfchar` or `endbfchar` marker.
    /// * `PdfError::InternalError` if the font dictionary:
    ///      * Does not have a /Subtype that could be read.
    ///      * Has a /ToUnicode reference that is invalid.
    /// * `PdfError::InvalidUtf8` if the deflated ToUnicode CMap is not valid UTF-8.
    ///
    /// # Panics
    ///
    /// * If any of the keys in the font dictionary are not valid UTF-8.
    #[allow(clippy::too_many_lines)]
    fn is_cid_keyed_font(
        &mut self,
        doc: &Document,
        page_id: PageID,
        font_key: &str,
    ) -> Result<&FontEncoding, PdfError> {
        // Use the entry API to avoid borrow checker issues
        match self.font_type.entry((page_id, font_key.to_string())) {
            Entry::Occupied(entry) => Ok(entry.into_mut()),
            Entry::Vacant(entry) => {
                let font_obj = get_font(doc, page_id, font_key)?;

                // Determine the font encoding type
                let font_encoding_type = {
                    // Test 1: if the font has:
                    //   /Subtype /Type1
                    //   /Subtype /TrueType
                    // then it is likely a "simple" font.
                    let font_subtype = str::from_utf8(
                        font_obj
                            .get("Subtype")
                            .ok_or(PdfError::MissingSubtype)?
                            .as_name()
                            .map_err(|_| {
                                PdfError::InternalError(format!(
                                    "Expected font {font_key}'s Subtype key to be a `name`"
                                ))
                            })?,
                    )
                    .unwrap();

                    if ["Type1", "TrueType"].contains(&font_subtype) {
                        FontEncoding::Simple
                    } else {
                        // Test 2: if the font has:
                        //   /Encoding /Identity-H
                        //   /Encoding /Identity-V
                        // then it is likely a CID-keyed font. However, if it has:
                        //   /Encoding /WinAnsiEncoding
                        //   /Encoding /MacRomanEncoding
                        // it is likely a "simple" font.
                        let encoding = font_obj.get("Encoding");
                        if let Some(font_encoding) = encoding {
                            let font_encoding =
                                str::from_utf8(font_encoding.as_name().map_err(|_| {
                                    PdfError::EncodingError(format!(
                                        "Expected font {font_key}'s Encoding key to be a `name`"
                                    ))
                                })?)
                                .unwrap();

                            if ["WinAnsiEncoding", "MacRomanEncoding"].contains(&font_encoding) {
                                FontEncoding::Simple
                            } else if ["Identity-H", "Identity-V"].contains(&font_encoding)
                                || font_encoding == "Type0"
                            {
                                // Test 3: if the font has:
                                //   /Subtype /Type0
                                // then it is likely a CID-keyed font.
                                let cmap_ref = font_obj
                            .get("ToUnicode")
                            .ok_or(PdfError::EncodingError(format!(
                                "CID-keyed font {font_key} does not have a ToUnicode CMap.",
                            )))?
                            .as_reference()
                            .map_err(|_| {
                                PdfError::EncodingError(format!(
                                    "CID-keyed font {font_key}'s ToUnicode CMap could not be read."
                                ))
                            })?;

                                let f = doc.get_object(cmap_ref);
                                let decompressed = f
                            .map_err(|_| {
                                PdfError::InternalError(format!(
                                    "Font {font_key}'s ToUnicode CMap points to invalid reference.",
                                ))
                            })?
                            .as_stream()
                            .map_err(|_| {
                                PdfError::EncodingError(format!(
                                    "CID-keyed font {font_key}'s ToUnicode CMap could not be read."
                                ))
                            })?
                            .decompressed_content()
                            .map_err(|_| {
                                PdfError::EncodingError(format!(
                                    "CID-keyed font {font_key}'s ToUnicode CMap could not be deflated."
                                ))
                            })?;

                                let cmap = String::from_utf8(decompressed)
                                    .map_err(|_| PdfError::InvalidUtf8)?;
                                let csrange_begin = cmap.find("beginbfchar").ok_or(
                            PdfError::EncodingError(format!(
                                "Deflated ToUnicode CMap for font {font_key} has no `beginbfchar`."
                            )),
                        )? + "beginbfchar".len();
                                let csrange_end = cmap.find("endbfchar").ok_or(PdfError::EncodingError(
                            format!(
                                "Deflated ToUnicode CMap for font {font_key} has no `endbfchar`."
                            ),
                        ))?;

                                let csrange = cmap[csrange_begin..csrange_end].trim();
                                let lines = csrange.split('\n');

                                let mut mappings = HashMap::new();

                                // Within the CMap, each line has the following form:
                                //   <001B> <0041>
                                // In this case, the 2-byte CID 001B maps to U+0041.
                                for line in lines {
                                    let parts: Vec<&str> = line.split_whitespace().collect();
                                    if parts.len() == 2 {
                                        let cid = parts[0].trim_matches(|c| c == '<' || c == '>');

                                        // In some cases, `parts[1]` can have multiple Unicode code points. This is
                                        // sometimes done to handle ligatures, among others. For example, the ligature
                                        // `ff` is represented as two U+066 code points.
                                        let code_units: Vec<u16> = parts[1]
                                            .trim_matches(|c| c == '<' || c == '>')
                                            .as_bytes()
                                            .chunks_exact(4)
                                            .map(|chunk| {
                                                u16::from_str_radix(
                                                    std::str::from_utf8(chunk).unwrap(),
                                                    16,
                                                )
                                            })
                                            .collect::<Result<_, _>>()
                                            .map_err(|_| PdfError::InvalidUtf8)?;

                                        let unicode: String = decode_utf16(code_units.into_iter())
                                            .map(|r| {
                                                r.map_err(|e| {
                                                    PdfError::EncodingError(format!(
                                                        "Invalid UTF-16: {e}"
                                                    ))
                                                })
                                            })
                                            .collect::<Result<_, _>>()?;

                                        mappings.insert(cid.to_string().to_lowercase(), unicode);
                                    }
                                }

                                FontEncoding::CIDKeyed(mappings)
                            } else {
                                // If we got here, then we don't have a good idea; emit a warning and assume Simple.
                                log::warn!(
                                    "No heuristic matched for font {font_key}; assuming simple. This is likely wrong."
                                );
                                FontEncoding::Simple
                            }
                        } else {
                            log::warn!(
                                "Could not determine font type for {font_key}, assuming simple. This may be wrong."
                            );
                            FontEncoding::Simple
                        }
                    }
                };

                // Insert into cache via vacant entry and return reference
                Ok(entry.insert(font_encoding_type))
            }
        }
    }

    /// Get the `N` whitespace-separated words in `content` before position `pos`. This is used to
    /// get the operators for a PDF command.
    ///
    /// # Errors
    /// * `PdfError::InternalError` if `N` > number of words in `content[..pos]`
    #[allow(clippy::unused_self)]
    #[allow(dead_code)]
    fn get_params<'a, const N: usize>(
        &self,
        content: &'a str,
        pos: usize,
    ) -> Result<[&'a str; N], PdfError> {
        let mut params = [""; N];
        let mut start_pos = pos - 1;

        for i in 0..N {
            let idx = content[..start_pos].rfind([' ', '/', '\n', '\t']).ok_or(
                PdfError::InternalError("No valid separator in content at {start_pos}".into()),
            )? + 1;
            params[N - 1 - i] = &content[idx..start_pos];
            start_pos = idx - 1;
        }

        Ok(params)
    }

    /// Extract N number tokens that appear immediately before an operator token.
    ///
    /// This scans backwards through the token slice to find the N numbers that precede
    /// the given operator. Useful for extracting Tf parameters (font_id, font_size) or
    /// Td parameters (x, y).
    ///
    /// # Arguments
    ///
    /// * `tokens` - The token slice to search
    /// * `op_idx` - The index of the operator token
    ///
    /// # Returns
    ///
    /// An array of N byte slices representing the number tokens, in order (not reversed)
    ///
    /// # Errors
    ///
    /// * `PdfError::InternalError` if there aren't enough Number tokens before the operator
    fn get_params_from_tokens<'a, const N: usize>(
        tokens: &'a [Token<'_>],
        op_idx: usize,
    ) -> Result<[&'a [u8]; N], PdfError> {
        let mut params_vec: Vec<&'a [u8]> = Vec::new();
        let mut idx = op_idx;

        // Scan backwards to find N number/name tokens
        while idx > 0 && params_vec.len() < N {
            idx -= 1;
            match &tokens[idx] {
                Token::Number(num) => params_vec.push(num),
                Token::Name(name) => params_vec.push(name),
                _ => {}
            }
        }

        if params_vec.len() < N {
            return Err(PdfError::InternalError(format!(
                "Expected {N} parameters before operator at index {op_idx}, found {}",
                params_vec.len()
            )));
        }

        // Reverse since we scanned backwards
        params_vec.reverse();

        // Convert Vec to array
        params_vec
            .try_into()
            .map_err(|_| PdfError::InternalError("Failed to convert params vec to array".into()))
    }

    /// Given a string `content` and a position `pos`, look *from* `pos` and search for an image.
    /// If an image is detected, attempt to return the position of the end of the caption. Note
    /// that this means that the returned position will be inside a `BT`...`ET`. This function
    /// assumes the position given to it is the position of an `ET`.
    ///
    /// # Returns:
    /// * `Some(idx)` where `idx` is the index of the space after the TJ where the image caption is
    ///   shown.
    /// * `None` if no image is detected.
    #[allow(clippy::similar_names)]
    #[allow(dead_code)]
    fn get_image_bounds(&self, content: &str, pos: usize) -> Option<usize> {
        let bt_idx = content[pos..].find("BT")? + pos;
        content[pos..bt_idx].find("/Im")?;

        let mut tj_idx = content[bt_idx..].find("TJ")? + bt_idx;

        /* If a caption is long enough, it will line-break and go to the next line. In pdflatex,
         * the line spacing between these two depends on the \baselineskip value. We currently
         * assume that \baselineskip is always 1.2 (which is the default), and parse lines accordingly. */
        loop {
            /* Find the next Td that is:
             *  1. After the current TJ
             *  2. Before the next TJ that is also before the next ET. */
            let next_et_idx = content[tj_idx..].find("ET")? + tj_idx;
            let Some(next_tj_idx) = content[tj_idx + 2..next_et_idx].find("TJ") else {
                // If there is no TJ before the next ET, the caption has ended.
                return Some(tj_idx + 2); // +2 for len("TJ")
            };

            let Some(next_td_idx) = content[tj_idx..tj_idx + 2 + next_tj_idx].find("Td") else {
                // There's no vertical spacing after, so this condition does not apply.
                return Some(tj_idx + 2);
            };

            let y = self
                .get_params::<2>(content, tj_idx + next_td_idx)
                .ok()
                .and_then(|[_, y]| y.parse::<f32>().ok())?;

            if y.abs() <= self.cur_font_size * self.cur_baselineskip {
                // This is a line break, keep skipping.
                tj_idx = next_tj_idx + tj_idx + 2;
            } else {
                return Some(tj_idx + 2);
            }
        }
    }

    /// Checks for a font change command (`/F`) within a specified range of the content string
    /// and updates the parser's current font state accordingly.
    ///
    /// This function searches for `/F` font resource references (e.g., `/F28`) within the
    /// given range `[start_idx, end_idx)` of the content string. If a font change is detected,
    /// it extracts the font ID, resolves the font name using the document's font dictionary,
    /// and updates both `self.cur_font` and `self.cur_font_id`.
    ///
    /// # Arguments
    ///
    /// * `doc` - The `lopdf::Document` object containing font information
    /// * `page_id` - The page ID where the font is referenced
    /// * `content` - The PDF content stream as a string
    /// * `start_idx` - The starting index of the range to search
    /// * `end_idx` - The ending index of the range to search
    fn check_and_update_font(
        &mut self,
        doc: &Document,
        page_id: PageID,
        content: &str,
        start_idx: usize,
        end_idx: usize,
    ) {
        let slice = &content[start_idx..end_idx];
        let Some(font_begin_idx) = slice.find("/F") else {
            return;
        };

        // Find the space after the font ID
        let Some(space_offset) = slice[font_begin_idx..].find(' ') else {
            // No space found in slice - this /F might be inside text content, skip it
            return;
        };
        let font_end_idx = font_begin_idx + space_offset;

        let font_id = &slice[font_begin_idx + 1..font_end_idx];

        // Skip empty font IDs or IDs that don't start with an alphanumeric character
        // Valid font IDs look like "F28", "F1", etc.
        if font_id.is_empty() || !font_id.starts_with(|c: char| c.is_ascii_alphanumeric()) {
            return;
        }

        // Try to get the font name - if it fails, this might not be a valid font reference
        if let Ok(font_name) = get_font_name(doc, page_id, font_id) {
            self.cur_font = font_name.to_string();
            self.cur_font_id = font_id.to_string();
        }
    }

    /// Process a TJ block given its tokens, extracting text with proper spacing.
    ///
    /// Takes a slice of tokens that represent the contents of a TJ array (literals, hex strings,
    /// and spacing numbers) and processes them according to the current font encoding.
    ///
    /// # Arguments
    ///
    /// * `tokens` - A slice of tokens from inside a TJ array (between [ and ])
    /// * `doc` - The PDF document
    /// * `page_id` - The current page ID
    ///
    /// # Returns
    ///
    /// The extracted text string with proper spacing applied
    ///
    /// # Errors
    ///
    /// Returns an error if font information cannot be retrieved
    fn process_tj_tokens(
        &mut self,
        tokens: &[Token<'_>],
        doc: &Document,
        page_id: PageID,
    ) -> Result<String, PdfError> {
        let mut result = String::new();
        let font_id = self.cur_font_id.clone();

        // Skip if we don't have a valid font ID yet
        if font_id.is_empty() {
            return Ok(result);
        }

        let cur_font = self.cur_font.clone();
        let same_word_threshold = self.config.same_word_threshold;

        let font_encoding = self.is_cid_keyed_font(doc, page_id, &font_id)?;
        let mut i = 0;

        while i < tokens.len() {
            match &tokens[i] {
                Token::Literal(text) => {
                    // Simple font encoding - handle math fonts
                    match font_encoding {
                        FontEncoding::Simple => {
                            let text_str = std::str::from_utf8(text).unwrap_or("");
                            if let Some(transform) = FONT_TRANSFORMS.get(cur_font.as_str()) {
                                result += &font_transform(text_str, *transform);
                            } else {
                                result += text_str;
                            }
                        }
                        FontEncoding::CIDKeyed(_) => {
                            // This shouldn't happen - CID-keyed fonts use Hex tokens
                            log::warn!("Unexpected Literal token in CID-keyed font");
                        }
                    }

                    // Check for spacing after this literal
                    if i + 1 < tokens.len() {
                        if let Token::Number(spacing_bytes) = tokens[i + 1] {
                            let spacing_str = std::str::from_utf8(spacing_bytes).unwrap_or("0");
                            let spacing = spacing_str.parse::<f32>().unwrap_or(0.0).abs();

                            if !(0.0..=same_word_threshold).contains(&spacing) {
                                result += " ";
                            }
                            i += 1; // Skip the number token
                        } else {
                            result += " ";
                        }
                    } else {
                        result += " ";
                    }
                }
                Token::Hex(hex_str) => {
                    // CID-keyed font - process hex string
                    match font_encoding {
                        FontEncoding::CIDKeyed(cmap) => {
                            let hex_text = std::str::from_utf8(hex_str).unwrap_or("");
                            let mut j = 0;
                            while j + 4 <= hex_text.len() {
                                let cid = hex_text[j..j + 4].to_lowercase();
                                if let Some(unicode) = cmap.get(&cid) {
                                    result += unicode;
                                } else {
                                    log::warn!("CID {cid} not found in ToUnicode CMap");
                                }
                                j += 4;
                            }
                        }
                        FontEncoding::Simple => {
                            log::warn!("Unexpected Hex token in simple font");
                        }
                    }

                    // Check for spacing after this hex string
                    if i + 1 < tokens.len() {
                        if let Token::Number(spacing_bytes) = tokens[i + 1] {
                            let spacing_str = std::str::from_utf8(spacing_bytes).unwrap_or("0");
                            let spacing = spacing_str.parse::<f32>().unwrap_or(0.0).abs();

                            if !(0.0..=same_word_threshold).contains(&spacing) {
                                result += " ";
                            }
                            i += 1; // Skip the number token
                        } else {
                            result += " ";
                        }
                    } else {
                        result += " ";
                    }
                }
                Token::Number(_) | Token::Op(_) | Token::Name(_) => {
                    // Standalone numbers (not after a literal/hex) are just spacing.
                    // They're handled as part of literal/hex processing above.
                    // We shouldn't encounter operators inside TJ arrays, but skip them.
                }
            }

            i += 1;
        }

        Ok(result)
    }

    /// Given a string `content` and a position `pos`, look *around* `pos` and search for likely
    /// boundaries for a table. This function uses the heuristic that tables are likely to be
    /// near `ET` blocks, because tables typically have some graphics (lines for borders, etc.).
    /// Under this assumption, this function looks at `Td` commands starting from the *previous*
    /// `BT` from `pos`, and compares the position described there to positions in subsequent `Td`
    /// commands starting from `pos`, using the Euclidean distance. If this distance is below a
    /// threshold, it is assumed that these are alignment efforts.
    ///
    /// We need to use *all* the `Td`s from the previous `BT` since LaTeX can use one `BT`..`ET`
    /// block for both non-table and table content.
    ///
    /// We do not need to keep a running track of where we are by adding the `Td` movements across
    /// `BT`..`ET` blocks: from the PDF Reference Manual, Section 7.2.3:
    ///
    /// >  Each time a text object begins, the current point is set to the origin of the page's
    /// >  coordinate system.
    ///
    /// This function also updates the font accordingly, since it is possible for the font to
    /// change within the bounds returned. Therefore, the core parsing method itself can safely
    /// assume that `self.cur_font` is accurate if it skipped over the returned bounds.
    ///
    /// # Returns
    ///
    /// * `Some((start_idx, end_idx))`, where `start_idx` is the index of the `BT` command where it is suspected that
    ///   the table begins, and `end_idx` is the index of the `ET` command where the table is suspected to end.
    /// * If no table is detected, returns `None`.
    #[allow(dead_code)]
    fn get_table_bounds(
        &mut self,
        content: &str,
        pos: usize,
        doc: &Document,
        page_id: PageID,
    ) -> Option<(usize, usize)> {
        let bt_idx = content[..pos].rfind("BT")?; // First `BT` we see
        let first_td_idx = content[..pos].rfind("Td")?;

        // Collect all the `Td`s after the previous `BT` up to where we are. LaTeX is free to use a
        // single BT..ET block for non-table and table content, so the table might start from one
        // of the intermediate `Td`s.
        let mut td_positions = content[bt_idx..pos]
            .split("Td")
            .filter_map(|s| {
                self.get_params::<2>(s, s.len() - 1)
                    .ok()
                    .and_then(|[x, y]| x.parse::<f32>().ok().zip(y.parse::<f32>().ok()))
            })
            .collect::<Vec<_>>();

        if td_positions.is_empty() {
            return None;
        }

        // Accumulate `Td`s, since these are movements
        for i in 1..td_positions.len() {
            td_positions[i] = (
                td_positions[i].0 + td_positions[i - 1].0,
                td_positions[i].1 + td_positions[i - 1].1,
            );
        }

        // The "first" (x, y) is the accumulated movements so far; see the function's documentation
        // pointing to Section 7.2.3 of the PDF reference manual for reasoning.
        let (first_x, first_y) = td_positions[td_positions.len() - 1];

        // Running position in `content`
        let mut cur_pos = pos;

        // How many `BT`s have we skipped?
        let mut bt_count = 0;

        loop {
            // Try to find the next BT
            let Some(next_bt) = content[cur_pos..].find("BT") else {
                if bt_count > 0 {
                    // If we've processed at least one BT and reached the end, return what we have
                    log::debug!("Could not find a BT, is the table at the end of the page?");
                    return Some((bt_idx, cur_pos));
                }

                // Not a table
                return None;
            };
            let next_bt_pos = cur_pos + next_bt;

            // Try to find a Td command after this BT
            let Some(td_offset) = content[next_bt_pos..].find("Td") else {
                log::debug!("Could not find a Td after a BT, ignoring possible table.");
                return None;
            };

            // Calculate current alignment position
            let cur_td_idx = next_bt_pos + td_offset;

            let params_result = self.get_params::<2>(content, cur_td_idx).ok();
            let Some((cur_x, cur_y)) = params_result
                .and_then(|[x, y]| Some((x.parse::<f32>().ok()?, y.parse::<f32>().ok()?)))
            else {
                // Could not parse parameters
                log::info!("Could not parse Td parameters, ignoring possible table.");
                return None;
            };

            let distance = ((cur_x - first_x).powi(2) + (cur_y - first_y).powi(2)).sqrt();

            if distance < self.config.table_alignment_threshold {
                bt_count += 1;

                // Before we move, check if the font has changed
                self.check_and_update_font(doc, page_id, content, 0, content.len() - 1);

                cur_pos = cur_td_idx;
            } else if bt_count > 0 {
                // We've found the end of the table
                return Some((first_td_idx + 3, next_bt_pos)); // +3 for "Td " length
            } else {
                // Not a table
                return None;
            }
        }
    }

    /// The actual PDF parser itself. Parses UTF-8 encoded code points in a best-effort manner,
    /// making reasonable assumptions along the way. Such assumptions are documented.
    #[allow(clippy::too_many_lines)]
    fn parse_content(
        &mut self,
        doc: &Document,
        page_id: PageID,
        page_number: usize,
        compute_body_font_size: bool,
    ) -> Result<ParseResult, PdfError> {
        let content = doc
            .get_page_content(page_id)
            .map_err(|_| PdfError::ContentError)?;
        let mut parsed = String::new();

        let tokens = tokenize(&content);

        // Keep track of the font sizes markers (from Tf) and associated positions
        let mut tf_history: Vec<FontSizeMarker> = Vec::new();
        // Keep track of vertical movements (second arg of Td) and associated positions
        let mut y_history: Vec<(usize, f32)> = Vec::new();

        // Index in tokens where TJ started (this is the first `Token::Literal`).
        let mut tj_start_idx: Option<usize> = None;
        for (token_idx, token) in tokens.iter().enumerate() {
            match token {
                Token::Literal(_) | Token::Hex(_) if tj_start_idx.is_none() => {
                    tj_start_idx = Some(token_idx);
                }
                Token::Op(b"TJ") => {
                    parsed += &self.process_tj_tokens(
                        &tokens[tj_start_idx.unwrap()..token_idx],
                        doc,
                        page_id,
                    )?;
                    tj_start_idx = None;
                }
                Token::Op(b"Tf") => {
                    if let Ok([font_id_bytes, font_size_bytes]) =
                        // Skip if there aren't enough tokens before this operator
                        PdfParser::get_params_from_tokens(&tokens, token_idx)
                        && let Ok(font_id) = std::str::from_utf8(font_id_bytes)
                        && let Ok(font_size_str) = std::str::from_utf8(font_size_bytes)
                        && let Ok(font_name) = get_font_name(doc, page_id, font_id)
                        && let Ok(font_size) = font_size_str.parse::<f32>()
                    {
                        self.cur_font = font_name.into();
                        self.cur_font_id = font_id.into();
                        self.cur_font_size = font_size;

                        tf_history.push(FontSizeMarker {
                            page_number,
                            byte_index: parsed.len(),
                            font_size: OrderedFloat(font_size),
                            font_name: self.cur_font.clone(),
                        });
                    }
                }
                Token::Op(b"Td") => {
                    // Skip if there aren't enough tokens before this operator
                    if let Ok([_x_bytes, vert_bytes]) =
                        PdfParser::get_params_from_tokens(&tokens, token_idx)
                        && let Ok(vert_str) = std::str::from_utf8(vert_bytes)
                        && let Ok(vert) = vert_str.parse::<f32>()
                    {
                        // `Td` args are movements, not absolute
                        let (_, cur_y) = y_history.last().unwrap_or(&(0, 0.0));
                        let new_y = cur_y + vert;

                        if vert != 0.0 {
                            y_history.push((parsed.len(), new_y));
                        }
                    }
                }
                _ => {}
            }
        }

        let mut edits = Vec::new();

        // Collect edits for replacing ligatures with constitutent characters
        for (from, to) in OCTAL_REPLACEMENTS.iter() {
            edits.extend_from_slice(
                &parsed
                    .match_indices(from)
                    .map(|(idx, _)| Edit {
                        start: idx,
                        end: idx + from.len(),
                        r#type: EditType::Replace((**to).into()),
                    })
                    .collect::<Vec<_>>(),
            );
        }

        // Add edits for sub/super-script markers
        let script_edits = get_script_marker_edits(&y_history, &mut parsed);
        edits.extend_from_slice(&script_edits);

        apply_edits(&edits, &mut parsed, &mut tf_history);

        let body_font_size = if compute_body_font_size && tf_history.len() > 1 {
            Some(Into::<f32>::into(
                tf_history
                    .iter()
                    .zip(tf_history.iter().skip(1))
                    .map(|(a, b)| (a.font_size, b.byte_index.saturating_sub(a.byte_index)))
                    .max_by_key(|f| f.1)
                    .unwrap()
                    .0,
            ))
        } else {
            None
        };

        Ok(ParseResult {
            content: parsed,
            font_size_markers: tf_history,
            body_font_size,
        })
    }
}

/// Given a PDF `Document` reference, a page ID, and a font key (e.g., "F19"), return the font
/// object.
///
/// # Arguments
///
/// * `doc` - The `lopdf::Document` object.
/// * `page_id` - The `lopdf` page ID. Different pages can have different font dictionaries,
///   otherwise operations such as joining PDFs would be more complicated than they already are.
/// * `font_key` - The font key as used in the PDF content stream.
///
/// # Returns
///
/// A `HashMap` with string keys mapping to `lopdf::Object` references.
///
/// # Errors
///
/// * `PdfError::PageFontError` if getting page fonts failed.
/// * `PdfError::FontNotFound` if the font key does not exist.
///
/// # Panics
///
/// * If any of the keys in the font dictionary are not valid UTF-8.
fn get_font<'a>(
    doc: &'a Document,
    page_id: PageID,
    font_key: &str,
) -> Result<HashMap<&'a str, &'a Object>, PdfError> {
    // Get the font dictionary for the page
    let fonts = doc
        .get_page_fonts(page_id)
        .map_err(|_| PdfError::PageFontError)?;

    let font_obj = fonts
        .get(font_key.as_bytes())
        .ok_or(PdfError::FontNotFound(font_key.into()))?;
    let font_hash = font_obj.as_hashmap();

    Ok(font_hash
        .iter()
        .map(|(k, v)| (str::from_utf8(k).expect("Invalid UTF in font object"), v))
        .collect())
}

/// Given a PDF `Document` reference, a page ID, and a font key (e.g., "F19"), return the font
/// name.
///
/// # Arguments
///
/// * `doc` - The `lopdf::Document` object.
/// * `page_id` - The `lopdf` page ID. Different pages can have different font dictionaries,
///   otherwise operations such as joining PDFs would be more complicated than they already are.
/// * `font_key` - The font key as used in the PDF content stream.
///
/// # Returns
///
/// The font name
///
/// # Errors
///
/// * `PdfError::PageFontError` if getting page fonts failed.
/// * `PdfError::FontNotFound` if the font key does not exist.
/// * `PdfError::MissingBaseFont` if the `BaseFont` key is missing in the font dictionary.
///
/// # Panics
///
/// * If any of the keys in the font dictionary are not valid UTF-8.
fn get_font_name<'a>(
    doc: &'a Document,
    page_id: PageID,
    font_key: &'a str,
) -> Result<&'a str, PdfError> {
    let font_obj = get_font(doc, page_id, font_key)?;
    let base_font = font_obj.get("BaseFont").ok_or(PdfError::MissingBaseFont)?;
    match base_font.as_name() {
        Ok(name) => {
            let idx = match name.iter().position(|&byte| byte == ASCII_PLUS) {
                Some(i) => i + 1,
                None => 0,
            };
            let (_, font_name) = name.split_at(idx);
            str::from_utf8(font_name).map_err(|_| PdfError::InvalidUtf8)
        }
        Err(_) => Err(PdfError::InvalidFontName),
    }
}

fn is_bold_font(font_name: &str) -> bool {
    font_name.contains("BX") || font_name.ends_with('B')
}

/// Fill in the `parent_idx` field for a set of sections that are ordered in the same manner as
/// they appear in the document.
///
/// # Arguments
///
/// * sections - A list of sections in document order.
fn compute_parent_indices(sections: &mut [SectionBoundary]) {
    for i in 1..sections.len() {
        let my_level = sections[i].level;
        sections[i].parent_idx = (0..i).rev().find(|&j| sections[j].level < my_level);
    }
}

/// Extracts text content from a PDF file at the given path.
///
/// # Errors
/// Returns an error if the file cannot be loaded or if text extraction fails.
pub fn extract_text(file_path: &str) -> Result<ExtractedContent, Box<dyn Error>> {
    let doc = Document::load(file_path)?;

    let mut full_text = String::new();
    let mut sections = Vec::new();
    let mut body_font_size: Option<f32> = None;

    let page_count = doc.get_pages().len();

    for (page_num, page_id) in doc.page_iter().enumerate() {
        log::debug!("\tParsing page {} of {page_count}", page_num + 1);

        let byte_offset = full_text.len();
        let mut parser = PdfParser::with_default_config();
        let result = parser.parse_content(&doc, page_id, page_num, body_font_size.is_none())?;

        if let Some(size) = result.body_font_size {
            body_font_size = Some(size);
        }

        for mut marker in result.font_size_markers {
            if body_font_size.is_some_and(|s| {
                marker.font_size > OrderedFloat(s)
                    || ((marker.font_size - s).abs() < f32::EPSILON
                        && is_bold_font(&marker.font_name))
            }) {
                sections.push(SectionBoundary {
                    page_number: marker.page_number,
                    byte_index: marker.byte_index,
                    font_size: Into::<f32>::into(marker.font_size),
                    level: 0,
                    parent_idx: None,
                });
            }

            marker.byte_index += byte_offset;
            marker.page_number = page_num;
        }

        full_text.push_str(&result.content);
    }

    let mut font_sizes = sections
        .iter()
        .map(|s| OrderedFloat(s.font_size))
        .collect::<Vec<_>>();

    font_sizes = font_sizes.into_iter().unique().collect();
    font_sizes.sort();
    font_sizes.reverse();

    let levels = font_sizes
        .iter()
        .enumerate()
        .map(|(i, f)| (f, i))
        .collect::<HashMap<_, _>>();

    for section in &mut sections {
        section.level = *levels
            .get(&OrderedFloat(section.font_size))
            .unwrap_or(&(levels.len() - 1));
    }

    sections.retain(|f| f.level < 3);

    compute_parent_indices(&mut sections);

    Ok(ExtractedContent {
        text_content: full_text,
        sections,
        page_count,
    })
}

/// Tests for the core PDF parser. While the tests themselves are nothing special, there are a few
/// useful tools here for maintainers. These are specific tests whose purpose is to help debugging:
///
/// * `test_pdf_content` shows the raw PDF content stream for the first page of a specified file.
///   Feel free to change this filename, or the (0-indexed) page number across PRs, this does not
///   need to be kept constant.
/// * `test_font_properties` is usually the second thing you'll use after the above. This test
///   prints out information about a font as obtained from the page's font dictionary. If
///   available, it will also print out the font's CMap, but you can disable this by commenting out
///   those lines.
/// * `test_get_content_around_object` shows a context window around some "anchor" text on a
///   specific page. This is useful for showing you nearby PDF content stream operators when
///   working on a feature and you need to test with a real PDF with potentially large content
///   streams.
#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::{env, fs};

    use super::*;

    /// Get the raw content stream for page `page_num` for the PDF.
    fn get_raw_content_stream(doc: &Document, page_num: usize) -> Result<String, PdfError> {
        let page_id: PageID = doc
            .page_iter()
            .nth(page_num)
            .ok_or(PdfError::ContentError)?;

        let page_content = doc
            .get_page_content(page_id)
            .map_err(|_| PdfError::ContentError)?;
        let content = String::from_utf8_lossy(&page_content);

        Ok(content.to_string())
    }

    fn check_pdf_contains(file_name: &str, queries: &[&str]) {
        let path = PathBuf::from("assets").join(file_name);
        let content = extract_text(path.to_str().unwrap()).unwrap().text_content;

        for query in queries {
            assert!(
                content.contains(*query),
                "content of {file_name} did not contain '{query}'\n\nContent was:\n{content}"
            );
        }
    }

    #[test]
    fn test_parsing_works() {
        // Test 1: "test1.pdf"
        check_pdf_contains("test1.pdf", &["Oversampling", "GHOST", "Deep Learning"]);

        // Test 2: "ntk.pdf"
        check_pdf_contains("ntk.pdf", &["\\theta", "\\otimes", "\\sum", "\\mathbb{E}"]);

        // Test 3: "manifold.pdf", contains CID-keyed fonts
        check_pdf_contains("manifold.pdf", &["Manifold", "Dimension"]);
    }

    #[test]
    #[ignore = "Manual test for debugging PDF content"]
    fn test_pdf_content() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        // NOTE: Maintainers: use this as a way to quickly get the UTF-8 content of raw PDF commands.
        let path = PathBuf::from("assets").join("subtables.pdf");

        let doc = Document::load(path).unwrap();
        let content = get_raw_content_stream(&doc, 0).unwrap();

        dbg!(content);
    }

    #[test]
    #[ignore = "Manual test for debugging font properties"]
    fn test_font_properties() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        // NOTE: Maintainers: use this as a way to quickly inspect fonts.

        /* In PDFs, a simplified view of fonts is as triply-nested dictionaries.
         * First, pages have a resources dictionary, which includes a font dictionary; that dictionary maps
         * font resource names (e.g., "F28") to font objects themselves--the second level of redirection.
         * Each font object has various properties of the font. This might include, for example, CMaps
         * (explained below), the font's name (e.g., "CMR10"), and other properties. It's also worth noting:
         * the 10 in CMR10 only gives the *design size* of the font in points--the size for which it was
         * designed and optimized. You still need to look at Tf for the font sizes.
         *
         * The only font that will have a ToUnicode map is `F9` in `manifold.pdf`.
         */
        let font_key = "F172";
        let path = PathBuf::from("assets")
            .join("test_papers")
            .join("mono2micro.pdf");
        let expect_cid_keyed_font = true;

        let doc = Document::load(path).unwrap();
        let page_id = doc.page_iter().next().unwrap();

        // Get the font dictionary for the page
        let readable_font_obj = get_font(&doc, page_id, font_key).unwrap();
        dbg!(&readable_font_obj);

        let font_subtype = readable_font_obj.get("Subtype").unwrap().as_name().unwrap();
        dbg!(&str::from_utf8(font_subtype));

        /* Quick primer: in PDFs, a CMap (character map) is an object that maps character codes to
         * Unicode values or to intermediate glyph identifiers. There are two main kinds of CMaps:
         *
         * - ToUnicode CMaps: map font character codes to actual Unicode values.
         * - CID-based CMaps: map character codes to CIDs (character IDs), which then map to GIDs
         *   or glyphs inside the font file--used especially in CJK and composite fonts.
         *
         * Here, we read the first kind (ToUnicdoe). Note: this will be a long object, so comment
         * it out if you don't need it!
         */
        if expect_cid_keyed_font {
            let cmap_ref = readable_font_obj
                .get("ToUnicode")
                .unwrap()
                .as_reference()
                .unwrap();

            let f = doc.get_object(cmap_ref);
            let decompressed = f
                .unwrap()
                .as_stream()
                .unwrap()
                .decompressed_content()
                .unwrap();
            print!("{}", String::from_utf8(decompressed).unwrap());
        }
    }

    #[test]
    #[ignore = "Manual test for debugging PDF content stream"]
    fn test_get_content_around_object() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }
        let page_number = 0; // 0-indexed page number to inspect
        let anchor = "Int"; // What should be found (case-sensitive)
        let context = 60; // Characters around the anchor

        let path = PathBuf::from("assets")
            .join("test_papers")
            .join("mono2micro.pdf");
        let doc = Document::load(path).unwrap();
        let raw_content = get_raw_content_stream(&doc, page_number).unwrap();
        dbg!(&raw_content);

        let idx = raw_content.find(anchor).unwrap();
        let start = idx.saturating_sub(context);
        let end = idx.saturating_add(anchor.len()).saturating_add(context);

        dbg!(raw_content[start..end].to_string());
    }

    #[test]
    fn test_sections_detected_correctly() {
        let path = PathBuf::from("assets").join("sections.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();
        let text = content.text_content;

        dbg!(&content.sections);
        assert_eq!(content.sections.len(), 4);

        assert_eq!(
            text[content.sections.first().unwrap().byte_index..][..5].to_string(),
            "title"
        );
        // The author is, for now, incorrectly detected as a section.
        assert_eq!(
            text[content.sections.get(1).unwrap().byte_index..][..6].to_string(),
            "author"
        );
        assert_eq!(
            text[content.sections.get(2).unwrap().byte_index..][..9].to_string(),
            "1 section"
        );
        assert_eq!(
            text[content.sections.get(3).unwrap().byte_index..][..10].to_string(),
            "1.1 subsec"
        );
    }

    /// A harder version of the above, which uses a real paper. This is by no means the hardest
    /// PDF, but it's meant to be more "real-world".
    #[test]
    fn test_sections_detected_correctly_hard() {
        let path = PathBuf::from("assets")
            .join("test_papers")
            .join("mono2micro.pdf");
        let res = extract_text(path.to_str().unwrap()).unwrap();

        const TESTS: [&str; 5] = [
            "Mono2Micro",
            "ABSTRACT",
            "CCS CONCEPTS",
            "KEYWORDS",
            "1 INTRODUCTION",
        ];
        let mut satisfied = [false; TESTS.len()];

        for s in &res.sections {
            let nearby_content = &res.text_content[s.byte_index..][..20];

            for (i, test) in TESTS.iter().enumerate() {
                if nearby_content.contains(test) {
                    satisfied[i] = true;
                }
            }
        }

        assert!(satisfied.iter().all(|s| *s));
    }

    #[test]
    fn test_same_size_sections_detected_correctly() {
        // This paper has section headings that are the same font size as the text, with
        // sections being bold and subsections being italicized.
        let path = PathBuf::from("assets")
            .join("test_papers")
            .join("mono2micro.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();
        assert!(!content.sections.is_empty());

        for section in &content.sections {
            let section_text = content.text_content[section.byte_index..][..30].to_string();
            println!("Page: {}", section.page_number);
            println!("Text: {section_text}");
            println!("Font size: {}", section.font_size);
            println!();

            if section_text.contains("Ref") {
                break;
            }
        }
    }

    #[test]
    fn test_real_papers_parse_without_errors() {
        let path = PathBuf::from("assets").join("test_papers");

        for file in fs::read_dir(path).unwrap() {
            let file = file.unwrap().path();
            let file = file.to_str().unwrap();
            let content = extract_text(file);

            if let Err(e) = content {
                println!("Error in {file}: {e}");
                panic!();
            }
        }
    }

    #[test]
    fn test_fonts_identified_correctly() {
        let path = PathBuf::from("assets").join("symbols.pdf");

        let doc = Document::load(path).unwrap();
        let content = get_raw_content_stream(&doc, 0).unwrap();

        const TEST_QUERIES: [&str; 3] = ["F21", "F27", "F30"];
        for test in TEST_QUERIES {
            assert!(content.contains(test));
        }

        let page_id = doc.page_iter().next().unwrap();
        let font_name = get_font_name(&doc, page_id, "F30").unwrap();

        assert_eq!(font_name, "CMMI7");
    }

    #[test]
    fn test_math_parsing_works() {
        let path = PathBuf::from("assets").join("symbols.pdf");

        let content = extract_text(path.to_str().unwrap());
        assert!(content.is_ok());

        let content = content.unwrap().text_content;
        dbg!(&content);
        for op in [r"\int", r"\sum", r"\infty"] {
            assert!(content.contains(op));
        }
    }

    #[test]
    fn test_get_table_bounds_works() {
        let path = PathBuf::from("assets").join("table.pdf");

        let doc = Document::load(&path).unwrap();
        let page_id = doc.page_iter().next().unwrap();
        let content = get_raw_content_stream(&doc, 0).unwrap();

        let mut parser = PdfParser::with_default_config();

        // The indices will not exactly line up, because "\n"s seem to be two separate characters. This is okay.
        let first_et = content.find("ET").unwrap();
        assert_eq!(first_et, 342);
        assert_eq!(
            parser.get_table_bounds(&content, first_et, &doc, page_id),
            Some((305, 707))
        );
    }

    #[test]
    fn test_tables_are_ignored() {
        let path = PathBuf::from("assets").join("table.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap().text_content;
        let tests = ["r1c1", "r1c2", "r2c1", "r2c2"];
        for text in tests {
            assert!(!content.contains(text));
        }
    }

    #[test]
    fn test_subtables_are_ignored() {
        let path = PathBuf::from("assets").join("subtables.pdf");
        let content = extract_text(path.to_str().unwrap())
            .expect("Failed to extract content from subtables.pdf")
            .text_content;

        // NOTE: This should also ignore "quux2" and "Caption", but it currently doesn't. This is
        // left to a future story, because the current implementation is already much better than
        // the older version, where it failed to capture the entirety of the first subtable.
        let tests = ["foo", "bar", "baz", "quux1"];
        for text in tests {
            assert!(!content.contains(text));
        }
    }

    #[test]
    fn test_images_are_ignored() {
        let path = PathBuf::from("assets").join("images.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap().text_content;

        // NOTE: This should also ignore "caption" and "HERE", which is a regression from the older
        // non-tokenizer-based parser. I'll get back to this later.
        let tests = ["Figure", "Caption", "is", "good"];
        for text in tests {
            assert!(!content.contains(text));
        }

        let tests = ["begin1", "end1", "begin2", "end2"];
        for text in tests {
            assert!(content.contains(text));
        }
    }

    #[test]
    fn test_hyperlinks_are_ignored() {
        let path = PathBuf::from("assets").join("hyperlinks.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap().text_content;

        let tests = ["google.com", "sec:2", "cite.yedida"];
        for text in tests {
            assert!(!content.contains(text));
        }
    }

    #[test]
    fn test_process_tj_tokens() {
        // Test processing a simple TJ block with literals and spacing
        let tokens = vec![
            Token::Literal(b"Hello"),
            Token::Number(b"-250"),
            Token::Literal(b"World"),
        ];

        let path = PathBuf::from("assets").join("symbols.pdf");
        let doc = Document::load(path).unwrap();
        let page_id = doc.page_iter().next().unwrap();

        let mut parser = PdfParser::with_default_config();

        // F30 is CMMI7
        parser.cur_font_id = "F30".to_string();
        parser.cur_font = "CMMI7".to_string();

        let result = parser.process_tj_tokens(&tokens, &doc, page_id);
        assert!(result.is_ok(), "process_tj_tokens failed: {result:?}");

        let text = result.unwrap();
        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    #[test]
    fn test_get_params_from_tokens() {
        // Test: /F28 12.0 Tf
        let tokens = vec![
            Token::Name(b"F28"),
            Token::Number(b"12.0"),
            Token::Op(b"Tf"),
        ];

        let result = PdfParser::get_params_from_tokens::<2>(&tokens, 2);

        assert!(result.is_ok());
        let [font_id, font_size] = result.unwrap();
        assert_eq!(font_id, b"F28");
        assert_eq!(font_size, b"12.0");
    }

    #[test]
    fn test_get_params_from_tokens_td() {
        // Test: 100.5 -20.3 Td
        let tokens = vec![
            Token::Number(b"100.5"),
            Token::Number(b"-20.3"),
            Token::Op(b"Td"),
        ];

        let result = PdfParser::get_params_from_tokens::<2>(&tokens, 2);

        assert!(result.is_ok());
        let [x, y] = result.unwrap();
        assert_eq!(x, b"100.5");
        assert_eq!(y, b"-20.3");
    }
}
