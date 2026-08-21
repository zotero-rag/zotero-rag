//! The core PDF parsing module. This includes the `PdfParser` struct, which is somewhat tuned for
//! academic PDFs. In particular, it skips images and tables by default. This behavior might change
//! later. The parser also handles common math symbols and converts them to their corresponding
//! LaTeX equivalents.

use std::collections::HashMap;
use std::error::Error;
use std::rc::Rc;
use std::str::Utf8Error;
use std::sync::LazyLock;
use std::{f32, str};

use itertools::Itertools;
use log;
use lopdf::Document;
use ordered_float::OrderedFloat;

use crate::edits::{Edit, EditType, apply_edits};
use crate::fonts::{
    CMap, DEFAULT_SPACE_WIDTH, FONT_TRANSFORMS, FontEncoding, FontSizeMarker, SPACE_WIDTH_FRACTION,
    compute_font_encoding, font_transform, get_font, get_space_width,
};
use crate::tokenizer::{Token, tokenize};

const ASCII_PLUS: u8 = b'+';

/// The default distance threshold to check alignment efforts in a table.
pub const DEFAULT_TABLE_EUCLIDEAN_THRESHOLD: f32 = 40.0;
/// The default number of `Td`s within a `BT` after which the `BT..ET` block is declared to be a
/// paragraph as opposed to a table.
pub const DEFAULT_TBL_TD_THRESHOLD: usize = 5;

/// A wrapper for all PDF parsing errors
#[derive(Debug, thiserror::Error)]
pub(crate) enum PdfError {
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
struct PdfParserThresholds {
    /// Euclidean distance threshold between `Td` alignment to declare a table
    table_alignment: f32,
    /// Threshold number of `Td`s within a text block to declare a paragraph as opposed to a table
    tbl_td: usize,
}

impl Default for PdfParserThresholds {
    fn default() -> Self {
        Self {
            table_alignment: DEFAULT_TABLE_EUCLIDEAN_THRESHOLD,
            tbl_td: DEFAULT_TBL_TD_THRESHOLD,
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
    thresholds: PdfParserThresholds,
    /// The current font we're using. This is not the font string in the dictionary (e.g. "F28"),
    /// but rather the font's name itself (e.g. "CMMI10").
    cur_font: String,
    /// The current font ID we're using. This is not the font string in the dictionary (e.g. "F28"),
    /// but rather the font's ID itself (e.g. "F28").
    cur_font_id: Rc<str>,
    /// Current font size
    cur_font_size: f32,
    /// The \baselineskip set by the user.
    /// TODO: Actually compute this; for now, this is set to the pdflatex default of 1.2
    cur_baselineskip: f32,
    /// The current page's map of whether a font is a CID-keyed font or not.
    font_type: HashMap<(PageID, String), Rc<FontEncoding>>,
    /// Cache of each font's space width, since it is constant per font but used for every `TJ`
    /// number token.
    space_width: HashMap<(PageID, Rc<str>), f32>,
}

/// `lopdf` references individual pages by a tuple of unsigned integers. Usually, the specific
/// values are irrelevant, and it is more useful to think about the page ID itself as one "thing".
pub(crate) type PageID = (u32, u16);

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
            let offset = if parsed.as_bytes().get(y_history[j].0) == Some(&BACKSLASH_ASCII)
                && let Some(space_pos) = parsed[y_history[j].0..].find(' ')
            {
                space_pos
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
        Self::new(PdfParserThresholds::default())
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
    /// The number of characters that were skipped on this page because they used CID-keyed fonts
    /// without a usable `ToUnicode` CMap.
    skipped_chars: usize,
}

/// Look up a one-byte character code in a simple font's `ToUnicode` CMap without allocating a key.
fn simple_cmap_mapping(mappings: &CMap, code: u8) -> Option<&str> {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let key = [HEX[usize::from(code >> 4)], HEX[usize::from(code & 0x0f)]];
    let key = std::str::from_utf8(&key).expect("hexadecimal digits are always valid UTF-8");
    mappings.get(key).map(String::as_str)
}

/// Append unmapped simple-font bytes as UTF-8, applying a configured math-font transformation.
///
/// A simple font may have no `ToUnicode` CMap or one that does not cover every character code.
/// This fallback preserves direct text extraction and math-to-LaTeX normalization for those
/// unmapped codes.
fn append_simple_fallback(result: &mut String, text: &[u8], font_name: &str) {
    let text = std::str::from_utf8(text).unwrap_or("");
    if let Some(transform) = FONT_TRANSFORMS.get(font_name) {
        result.push_str(&font_transform(text, *transform));
    } else {
        result.push_str(text);
    }
}

/// Append one simple-font character code using the available decoding methods in priority order.
///
/// Explicit math-to-LaTeX mappings take priority over `ToUnicode` so mathematical output retains
/// its normalized representation. Codes without either mapping use [`append_simple_fallback`].
///
/// # Arguments
///
/// * `result` - The extracted text buffer to append the decoded character to.
/// * `code` - The one-byte character code to decode.
/// * `mappings` - The font's parsed `ToUnicode` mappings, if it has a usable CMap.
/// * `font_name` - The font's base name, used to select any configured math-font transformation.
fn append_simple_code(result: &mut String, code: u8, mappings: Option<&CMap>, font_name: &str) {
    if let Some(transform) = FONT_TRANSFORMS.get(font_name)
        && let Some(transformed) = transform(code)
    {
        // known math symbol
        result.push_str(&transformed);
    } else if let Some(unicode) = mappings.and_then(|map| simple_cmap_mapping(map, code)) {
        // `ToUnicode` mapping available
        result.push_str(unicode);
    } else {
        // push the raw byte as-is
        append_simple_fallback(result, std::slice::from_ref(&code), font_name);
    }
}

/// Decode a hexadecimal string for a simple font, giving `ToUnicode` mappings priority.
///
/// Whitespace between hexadecimal digits is ignored. Character codes present in `mappings` are
/// converted to Unicode, while unmapped codes use the simple-font fallback decoder. As required by
/// ISO 32000-1:2008, §7.3.4.3, "Hexadecimal strings", an odd final digit is padded with a zero.
///
/// # Arguments
///
/// * `result` - The extracted text buffer to append decoded text to.
/// * `hex` - The hexadecimal digits from the PDF string, with optional ASCII whitespace.
/// * `mappings` - The font's parsed `ToUnicode` mappings, if it has a usable CMap.
/// * `font_name` - The font's base name, used to select any configured math-font transformation.
fn append_simple_hex(result: &mut String, hex: &[u8], mappings: Option<&CMap>, font_name: &str) {
    let mut high_nibble = None;
    for &digit in hex.iter().filter(|digit| !digit.is_ascii_whitespace()) {
        let Some(nibble) = char::from(digit)
            .to_digit(16)
            .and_then(|value| u8::try_from(value).ok())
        else {
            log::warn!("Invalid hexadecimal digit in simple-font text string");
            return;
        };

        if let Some(high) = high_nibble.take() {
            let code = high << 4 | nibble;
            append_simple_code(result, code, mappings, font_name);
        } else {
            high_nibble = Some(nibble);
        }
    }

    // The standard pads an odd final hexadecimal digit with a low-order zero.
    if let Some(high) = high_nibble {
        let code = high << 4;
        append_simple_code(result, code, mappings, font_name);
    }
}

impl PdfParser {
    /// Creates a new parser with the given configuration
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use for parsing
    fn new(config: PdfParserThresholds) -> Self {
        Self {
            thresholds: config,
            cur_font: String::new(),
            cur_font_id: Rc::from(""),
            cur_font_size: 12.0,   // Doesn't really matter
            cur_baselineskip: 1.2, // The pdflatex default
            font_type: HashMap::new(),
            space_width: HashMap::new(),
        }
    }

    /// Gets the encoding for a font on a specific page.
    ///
    /// This function is *not* pure: it updates `self.font_type`, which acts as a cache for these
    /// results. Entries are keyed by `(page_id, font_key)`, so the cache does not need to be cleared
    /// between pages.
    ///
    /// # Returns
    ///
    /// The font encoding and any available `ToUnicode` mappings.
    ///
    /// # Errors
    ///
    /// * `PdfError::PageFontError` if getting page fonts failed.
    /// * `PdfError::FontNotFound` if the font key does not exist.
    /// * `PdfError::EncodingError` if the font dictionary:
    ///      * Has an /Encoding that could not be read
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
    fn font_encoding(
        &mut self,
        doc: &Document,
        page_id: PageID,
        font_key: &str,
    ) -> Result<Rc<FontEncoding>, PdfError> {
        let key = (page_id, font_key.to_string());
        if let Some(encoding) = self.font_type.get(&key) {
            return Ok(encoding.clone());
        }

        let font_obj = get_font(doc, page_id, font_key)?;
        let encoding = Rc::new(
            compute_font_encoding(doc, &font_obj, font_key).unwrap_or(FontEncoding::Unmappable),
        );
        self.font_type.insert(key, encoding.clone());
        Ok(encoding)
    }

    /// Get the word-break gap threshold for the current font: its space width scaled by
    /// [`SPACE_WIDTH_FRACTION`]. The space width is constant per font but consulted for every
    /// `TJ` number token, so computed widths are cached. As with [`Self::font_type`], entries are
    /// keyed by `(page_id, font_key)`, so the cache does not need to be cleared between pages.
    ///
    /// # Arguments
    ///
    /// * `doc` - The PDF document.
    /// * `page_id` - The page the font appears on.
    /// * `font_encoding` - The font's encoding, e.g. from [`Self::font_encoding`].
    ///
    /// # Returns
    ///
    /// The word-break threshold in text-space units: a negative `TJ` adjustment whose magnitude
    /// exceeds this value is emitted as a space.
    ///
    /// # Errors
    ///
    /// * `PdfError::PageFontError` if getting page fonts failed.
    /// * `PdfError::FontNotFound` if the font key does not exist.
    fn space_threshold(
        &mut self,
        doc: &Document,
        page_id: PageID,
        font_encoding: &FontEncoding,
    ) -> Result<f32, PdfError> {
        let font_key = &self.cur_font_id;
        if let Some(&width) = self.space_width.get(&(page_id, font_key.clone())) {
            return Ok(width * SPACE_WIDTH_FRACTION);
        }

        let font_obj = get_font(doc, page_id, font_key)?;
        let width = get_space_width(doc, &font_obj, font_key, font_encoding);
        self.space_width.insert((page_id, font_key.clone()), width);

        Ok(width * SPACE_WIDTH_FRACTION)
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

    /// Given a sequence of `tokens` and an index `pos` of an `/Im` into that sequence, look *from*
    /// `pos` and attempt to return the position of the first token after the image's caption.
    /// Specifically, the token pointed to will be right after a `TJ` token.
    ///
    /// # Returns
    ///
    /// `Some(idx)` where `idx` is the position of the first token after the caption.
    fn get_image_bounds(&self, tokens: &[Token<'_>], pos: usize) -> Option<usize> {
        if pos >= tokens.len() {
            return None;
        }
        if let Token::Name(name) = &tokens[pos] {
            if !name.starts_with(b"Im") {
                return None;
            }
        } else {
            return None;
        }

        let mut i = pos + 1;
        while i < tokens.len() {
            // Find a `TJ`
            while i < tokens.len() && tokens[i] != Token::Op(b"TJ") {
                i += 1;
            }
            i += 1; // Go past the `TJ`
            let tj_idx = i;

            /* Go until one of the following conditions is met:
             *  1. We find an `ET` - return `None`.
             *  2. We find a `TJ`, but haven't yet found a `Td` - return `i + 1`.
             *  3. We find a `Td` before a `TJ` - break
             */
            while i < tokens.len() {
                match tokens[i] {
                    Token::Op(b"ET") => {
                        return Some(tj_idx);
                    }
                    Token::Op(b"TJ") => {
                        return Some(i + 1);
                    }
                    Token::Op(b"Td") => {
                        break;
                    }
                    _ => i += 1,
                }
            }

            if i >= tokens.len() {
                return Some(tj_idx);
            }

            // Get the params for this `Td`.
            let Some(y) = Self::get_params_from_tokens(tokens, i)
                .ok()
                .and_then(|[y]| std::str::from_utf8(y).ok())
                .and_then(|y| y.parse::<f32>().ok())
            else {
                return Some(tj_idx);
            };

            if y.abs() > self.cur_font_size * self.cur_baselineskip {
                return Some(tj_idx);
            }

            i += 1; // Move past the `Td`
        }

        None
    }

    /// Process a TJ block given its tokens, extracting text with proper spacing.
    ///
    /// Takes a slice of tokens that represent the contents of a TJ array (literals, hex strings,
    /// and spacing numbers) and processes them according to the current font encoding.
    ///
    /// # Arguments
    ///
    /// * `text_op` - The token that caused this function to be called. There are subtle differences
    ///   in how different text operators (such as `TJ` and `Tj`) need to be processed.
    /// * `tokens` - A slice of tokens from inside a TJ array (between [ and ])
    /// * `doc` - The PDF document
    /// * `page_id` - The current page ID
    ///
    /// # Returns
    ///
    /// A tuple of the extracted text string with proper spacing applied, and the number of
    /// characters that were skipped because the current font is CID-keyed but has no usable
    /// `ToUnicode` CMap (an estimate based on the number of CIDs skipped).
    #[allow(clippy::too_many_lines)]
    fn process_tj_tokens(
        &mut self,
        text_op: Token<'_>,
        tokens: &[Token<'_>],
        doc: &Document,
        page_id: PageID,
    ) -> (String, usize) {
        let mut result = String::new();
        let mut skipped_chars = 0;
        let font_id = self.cur_font_id.clone();

        // Skip if we don't have a valid font ID yet
        if font_id.is_empty() {
            return (result, skipped_chars);
        }

        let cur_font = self.cur_font.clone();

        let mut i = 0;
        let font_encoding = self
            .font_encoding(doc, page_id, &font_id)
            .unwrap_or_else(|_| Rc::new(FontEncoding::Unmappable));

        while i < tokens.len() {
            match &tokens[i] {
                Token::Literal(text) => {
                    // Simple font encoding - handle math fonts
                    match &*font_encoding {
                        FontEncoding::Simple { mappings, .. } => {
                            if let Some(mappings) = mappings {
                                for &code in *text {
                                    append_simple_code(
                                        &mut result,
                                        code,
                                        Some(mappings),
                                        cur_font.as_str(),
                                    );
                                }
                            } else {
                                // Decode the complete string so contiguous UTF-8 remains intact.
                                append_simple_fallback(&mut result, text, cur_font.as_str());
                            }
                        }
                        FontEncoding::CIDKeyed { .. } => {
                            // This shouldn't happen - CID-keyed fonts use Hex tokens
                            log::warn!("Unexpected Literal token in CID-keyed font");
                        }
                        FontEncoding::Unmappable => {
                            // In a CID-keyed font, literal strings contain two-byte character
                            // codes, so halve the byte count to estimate skipped characters.
                            let num_cids = text.len().div_ceil(2);
                            log::debug!(
                                "Skipping {num_cids} CID(s) of text in unmappable font {font_id}"
                            );
                            skipped_chars += num_cids;
                        }
                    }

                    // Check for spacing after this literal
                    if i + 1 < tokens.len() {
                        if let Token::Number(spacing_bytes) = tokens[i + 1] {
                            let spacing_str = std::str::from_utf8(spacing_bytes).unwrap_or("0");
                            let spacing = spacing_str.parse::<f32>().unwrap_or(0.0);
                            let space_threshold = self
                                .space_threshold(doc, page_id, &font_encoding)
                                .unwrap_or(DEFAULT_SPACE_WIDTH * SPACE_WIDTH_FRACTION);

                            // `spacing` < 0 opens a gap of |spacing|/1000 em; `spacing` > 0 is a kern.
                            if spacing < -space_threshold
                                && result.as_bytes().last().is_some_and(|c| *c != b' ')
                            {
                                result += " ";
                            }
                            i += 1; // Skip the number token
                        }
                    } else if text_op == Token::Op(b"TJ") {
                        // After the last literal, emit a space
                        result += " ";
                    }
                }
                Token::Hex(hex_str) => {
                    // CID-keyed font - process hex string
                    match &*font_encoding {
                        FontEncoding::CIDKeyed { mappings, .. } => {
                            let hex_text = std::str::from_utf8(hex_str).unwrap_or("");
                            let mut j = 0;
                            while j + 4 <= hex_text.len() {
                                let cid = hex_text[j..j + 4].to_lowercase();
                                if let Some(unicode) = mappings.get(&cid) {
                                    result += unicode;
                                } else {
                                    log::warn!("CID {cid} not found in ToUnicode CMap");
                                }
                                j += 4;
                            }
                        }
                        FontEncoding::Simple { mappings, .. } => {
                            append_simple_hex(
                                &mut result,
                                hex_str,
                                mappings.as_ref(),
                                cur_font.as_str(),
                            );
                        }
                        FontEncoding::Unmappable => {
                            // CIDs are two bytes each, encoded as four hex digits; use the
                            // number of skipped CIDs as an estimate of skipped characters.
                            // Whitespace is permitted inside PDF hex strings, so count only
                            // hex digits to avoid inflating the estimate.
                            let num_cids = hex_str
                                .iter()
                                .filter(|b| b.is_ascii_hexdigit())
                                .count()
                                .div_ceil(4);
                            log::debug!(
                                "Skipping {num_cids} CID(s) of text in unmappable font {font_id}"
                            );
                            skipped_chars += num_cids;
                        }
                    }

                    // Check for spacing after this hex string
                    if i + 1 < tokens.len() {
                        if let Token::Number(spacing_bytes) = tokens[i + 1] {
                            let spacing_str = std::str::from_utf8(spacing_bytes).unwrap_or("0");
                            let spacing = spacing_str.parse::<f32>().unwrap_or(0.0);
                            let space_threshold = self
                                .space_threshold(doc, page_id, &font_encoding)
                                .unwrap_or(DEFAULT_SPACE_WIDTH * SPACE_WIDTH_FRACTION);

                            // `spacing` < 0 opens a gap of |spacing|/1000 em; `spacing` > 0 is a kern.
                            if spacing < -space_threshold
                                && result.as_bytes().last().is_some_and(|c| *c != b' ')
                            {
                                result += " ";
                            }
                            i += 1; // Skip the number token
                        }
                    } else if text_op == Token::Op(b"TJ") {
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

        (result, skipped_chars)
    }

    /// Given a token slice and an index `pos` of an `ET` token, look *around* `pos` and search for
    /// likely boundaries for a table. This function uses the heuristic that tables are likely to be
    /// near `ET` blocks, because tables typically have some graphics (lines for borders, etc.).
    /// Under this assumption, this function looks at `Td` commands starting from the `BT` at `bt_pos`
    /// up to `pos`, accumulating their movements to form a reference position `(first_x, first_y)`.
    /// It then scans forward from `pos`, comparing the first `Td` in each subsequent `BT` block against
    /// that reference using Euclidean distance. If the distance is below `self.thresholds.table_alignment`,
    /// those BT blocks are considered part of the same table.
    ///
    /// Early-exit heuristics prevent false positives:
    /// - If the current BT block has no `Td` operators, we cannot form a reference position.
    /// - If the current BT block has more than `self.thresholds.tbl_td` `Td` operators, it is
    ///   likely a paragraph rather than a table cell, and we return `None`.
    ///
    /// We do not need to keep a running track of where we are by adding the `Td` movements across
    /// `BT`..`ET` blocks: from the PDF Reference Manual, Section 7.2.3:
    ///
    /// >  Each time a text object begins, the current point is set to the origin of the page's
    /// >  coordinate system.
    ///
    /// # Arguments
    ///
    /// * `tokens` - The full token slice for the page.
    /// * `pos` - The index of the `ET` token at the end of the current BT block.
    /// * `bt_pos` - The index of the `BT` token that opened the current block.
    ///
    /// # Returns
    ///
    /// * `Some((start_idx, end_idx))`, where `start_idx` is the token index of the last `Td`
    ///   inside the current BT block (i.e., where the table content begins), and `end_idx` is the
    ///   token index of the `BT` that starts the first non-matching block after the table (i.e.,
    ///   where the caller should resume processing).
    /// * `None` if no table is detected.
    fn get_table_bounds(
        &self,
        tokens: &[Token<'_>],
        pos: usize,
        bt_pos: usize,
    ) -> Option<(usize, usize)> {
        if bt_pos >= tokens.len() || pos >= tokens.len() || pos <= bt_pos {
            return None;
        }

        let (mut first_x, mut first_y) = (0.0, 0.0);
        // Index of the *last* `Td` in the text block (holds accumulated position)
        let mut first_acc_td_idx: Option<usize> = None;
        let mut td_count = 0;
        let mut i = bt_pos;

        while i < pos {
            if let Token::Op(b"Td") = tokens[i]
                && i > 1
                && let Token::Number(_) = tokens[i - 2]
                && let Token::Number(_) = tokens[i - 1]
            {
                first_x += tokens[i - 2].parse::<f32>()?;
                first_y += tokens[i - 1].parse::<f32>()?;
                first_acc_td_idx = Some(i);
                td_count += 1;
            }

            i += 1;
        }

        // If there were no Td operators in the current BT block, we can't determine a reference
        // position, so we can't detect a table.
        let first_acc_td_idx = first_acc_td_idx?;

        // If there are too many Td operators, this is likely a paragraph, not a table boundary.
        // Tables typically have just a few Td operators per BT block (for positioning).
        if td_count > self.thresholds.tbl_td {
            return None;
        }

        // How many `BT`s have we skipped?
        let mut bt_count = 0;

        let mut cur_bt_pos: Option<usize> = None;

        while i < tokens.len() {
            match tokens[i] {
                Token::Op(b"BT") => {
                    cur_bt_pos = Some(i);
                }
                Token::Op(b"Td") => {
                    if cur_bt_pos.is_some()
                        && i > 1
                        && let Token::Number(_) = tokens[i - 2]
                        && let Token::Number(_) = tokens[i - 1]
                    {
                        let cur_x: f32 = tokens[i - 2].parse()?;
                        let cur_y: f32 = tokens[i - 1].parse()?;

                        let distance =
                            ((cur_x - first_x).powi(2) + (cur_y - first_y).powi(2)).sqrt();
                        if distance < self.thresholds.table_alignment {
                            bt_count += 1;
                            cur_bt_pos = None; // consume this BT; wait for next one
                        } else if bt_count > 0 {
                            // We've found the end of the table
                            return Some((first_acc_td_idx + 1, cur_bt_pos.unwrap()));
                        } else {
                            return None;
                        }
                    }
                }
                _ => (),
            }

            i += 1;
        }

        if bt_count > 0 {
            // If we've processed at least one BT and reached the end, return what we have
            log::debug!("Could not find a BT, is the table at the end of the page?");
            return Some((first_acc_td_idx + 1, tokens.len()));
        }

        // Not a table
        None
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
        let mut skipped_chars = 0;

        // Keep track of the font sizes markers (from Tf) and associated positions
        let mut tf_history: Vec<FontSizeMarker> = Vec::new();
        // Keep track of vertical movements (second arg of Td) and associated positions
        let mut y_history: Vec<(usize, f32)> = Vec::new();

        let mut token_idx = 0;
        let mut bt_pos = 0;
        // Length of `parsed` at the time the current BT was seen; used to rewind
        // if we later detect that the BT block was a table.
        let mut parsed_len_at_bt = 0;

        // Index in tokens where TJ started (this is the first `Token::Literal`).
        let mut tj_start_idx: Option<usize> = None;
        while token_idx < tokens.len() {
            let token = &tokens[token_idx];
            match token {
                Token::Literal(_) | Token::Hex(_) if tj_start_idx.is_none() => {
                    tj_start_idx = Some(token_idx);
                }
                Token::Name(name) if name.starts_with(b"Im") => {
                    if let Some(idx) = self.get_image_bounds(&tokens, token_idx) {
                        token_idx = idx;
                        continue;
                    }
                }
                Token::Op(b"BT") => {
                    bt_pos = token_idx;
                    parsed_len_at_bt = parsed.len();
                }
                Token::Op(b"ET") => {
                    if let Some((_, tbl_end_idx)) =
                        self.get_table_bounds(&tokens, token_idx, bt_pos)
                    {
                        // Rewind any content that was already parsed from this BT block,
                        // and drop any history entries that reference rewound positions.
                        parsed.truncate(parsed_len_at_bt);
                        tf_history.retain(|m| m.byte_index <= parsed_len_at_bt);
                        y_history.retain(|(idx, _)| *idx <= parsed_len_at_bt);
                        // tbl_end_idx points to a BT token; update bt tracking state
                        // since we're jumping over the BT token itself.
                        bt_pos = tbl_end_idx;
                        parsed_len_at_bt = parsed.len();
                        token_idx = tbl_end_idx;
                        tj_start_idx = None;
                    }
                }
                Token::Op(b"TJ" | b"Tj" | b"'" | b"\"") => {
                    if let Some(start_idx) = tj_start_idx {
                        let (text, skipped) = self.process_tj_tokens(
                            *token,
                            &tokens[start_idx..token_idx],
                            doc,
                            page_id,
                        );

                        if matches!(*token, Token::Op(b"'" | b"\"")) {
                            // These operators include an implicit `T*`, so emit a newline before
                            // emitting text.
                            parsed += "\n";
                        }

                        parsed += &text;
                        skipped_chars += skipped;
                        tj_start_idx = None;
                    }
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

            token_idx += 1;
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
            skipped_chars,
        })
    }
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
    let mut skipped_chars_total = 0;

    let page_count = doc.get_pages().len();

    for (page_num, page_id) in doc.page_iter().enumerate() {
        log::debug!("\tParsing page {} of {page_count}", page_num + 1);

        let byte_offset = full_text.len();
        let mut parser = PdfParser::default();
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
                    byte_index: marker.byte_index + byte_offset,
                    font_size: Into::<f32>::into(marker.font_size),
                    level: 0,
                    parent_idx: None,
                });
            }

            marker.byte_index += byte_offset;
            marker.page_number = page_num;
        }

        full_text.push_str(&result.content);
        skipped_chars_total += result.skipped_chars;
    }

    if skipped_chars_total > 0 {
        let total = skipped_chars_total + full_text.chars().count();
        let pct = 100.0 * (skipped_chars_total as f64) / (total as f64);

        // Decide on a log level based on `pct`; we don't want to alarm users needlessly.
        // TODO: Consider making this part of [`PdfParserThresholds`]
        if pct >= 5.0 {
            log::warn!(
                "Skipped {skipped_chars_total} character(s) ({pct:.1}% of the document): text uses CID-keyed fonts with no usable ToUnicode CMap and could not be extracted."
            );
        } else {
            log::debug!(
                "Skipped {skipped_chars_total} character(s) ({pct:.1}% of the document): text uses CID-keyed fonts with no usable ToUnicode CMap and could not be extracted."
            );
        }
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

    sections.retain(|f| f.level < 4);

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

    use zqa_macros::{test_contains_all, test_eq, test_ok};

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

        test_contains_all!(content, *queries);
    }

    #[test]
    fn test_parsing_works() {
        // Test 1: "test1.pdf"
        check_pdf_contains("test1.pdf", &["Oversampling", "GHOST", "Deep Learning"]);

        // Test 2: "ntk.pdf"
        check_pdf_contains(
            "ntk.pdf",
            &[
                "\\theta",
                "\\otimes",
                "\\sum",
                "\\mathbb{E}",
                "\\in",
                "\\partial",
                "\\nabla",
            ],
        );

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

        test_ok!(content);

        let content = content.unwrap();
        let text = content.text_content;

        test_eq!(content.sections.len(), 5);

        test_eq!(
            text[content.sections.first().unwrap().byte_index..][..5].to_string(),
            "title"
        );
        // The author is, for now, incorrectly detected as a section.
        test_eq!(
            text[content.sections.get(1).unwrap().byte_index..][..6].to_string(),
            "author"
        );
        test_eq!(
            text[content.sections.get(2).unwrap().byte_index..][..9].to_string(),
            "1 section"
        );
        test_eq!(
            text[content.sections.get(3).unwrap().byte_index..][..10].to_string(),
            "1.1 subsec"
        );
        test_eq!(
            text[content.sections.get(4).unwrap().byte_index..][..15].to_string(),
            "1.1.1 subsubsec"
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

        test_ok!(content);

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

        test_eq!(font_name, "CMMI7");
    }

    #[test]
    fn test_math_parsing_works() {
        let path = PathBuf::from("assets").join("symbols.pdf");

        let content = extract_text(path.to_str().unwrap());
        test_ok!(content);

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
        let content = get_raw_content_stream(&doc, 0).unwrap();
        let tokens = tokenize(content.as_bytes());

        let config = PdfParserThresholds {
            tbl_td: 10,
            ..Default::default()
        };
        let parser = PdfParser::new(config);

        test_eq!(parser.get_table_bounds(&tokens, 69, 0), Some((61, 167)));
    }

    #[test]
    fn test_tables_are_ignored() {
        let path = PathBuf::from("assets").join("table.pdf");
        let content = extract_text(path.to_str().unwrap());

        test_ok!(content);

        let content = content.unwrap().text_content;
        let tests = ["r1c1", "r1c2", "r2c1", "r2c2"];
        for text in tests {
            dbg!(&text);
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
            dbg!(&text);
            assert!(!content.contains(text));
        }
    }

    #[test]
    fn test_images_are_ignored() {
        let path = PathBuf::from("assets").join("images.pdf");
        let content = extract_text(path.to_str().unwrap());

        test_ok!(content);

        let content = content.unwrap().text_content;

        let tests = ["Figure", "Caption", "is", "good", "caption", "HERE"];
        for text in tests {
            dbg!(&text);
            assert!(!content.contains(text));
        }

        let tests = ["begin1", "end1", "begin2", "end2"];
        for text in tests {
            dbg!(&text);
            assert!(content.contains(text));
        }
    }

    #[test]
    fn test_hyperlinks_are_ignored() {
        let path = PathBuf::from("assets").join("hyperlinks.pdf");
        let content = extract_text(path.to_str().unwrap());

        test_ok!(content);

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

        let mut parser = PdfParser {
            cur_font_id: Rc::from("F30"),
            cur_font: "CMMI7".to_string(),
            ..PdfParser::default()
        };

        // F30 is CMMI7
        parser.cur_font_id = Rc::from("F30");
        parser.cur_font = "CMMI7".to_string();

        let (text, _) = parser.process_tj_tokens(Token::Op(b"TJ"), &tokens, &doc, page_id);

        assert!(text.contains("Hello"));
        assert!(text.contains("World"));
    }

    /// Build a minimal in-memory PDF with a single page whose content stream is `content` and
    /// whose resource dictionary has a single font `F1` described by `font_dict_entries`.
    fn doc_with_font(
        content: &[u8],
        mut font_dict_entries: Vec<(&'static str, lopdf::Object)>,
        to_unicode: Option<&[u8]>,
    ) -> (Document, PageID) {
        let mut doc = Document::with_version("1.5");

        if let Some(to_unicode) = to_unicode {
            let cmap_id = doc.add_object(lopdf::Stream::new(
                lopdf::Dictionary::new(),
                to_unicode.to_vec(),
            ));
            font_dict_entries.push(("ToUnicode", lopdf::Object::Reference(cmap_id)));
        }
        let font_id = doc.add_object(lopdf::Dictionary::from_iter(font_dict_entries));

        let content_id = doc.add_object(lopdf::Stream::new(
            lopdf::Dictionary::new(),
            content.to_vec(),
        ));

        let fonts = lopdf::Dictionary::from_iter(vec![(b"F1", lopdf::Object::Reference(font_id))]);
        let resources_id = doc.add_object(lopdf::Dictionary::from_iter(vec![(
            "Font",
            lopdf::Object::Dictionary(fonts),
        )]));

        let page_obj_id = doc.add_object(lopdf::Dictionary::from_iter(vec![
            ("Type", lopdf::Object::Name(b"Page".to_vec())),
            ("Resources", lopdf::Object::Reference(resources_id)),
            ("Contents", lopdf::Object::Reference(content_id)),
        ]));

        let pages_id = doc.add_object(lopdf::Dictionary::from_iter(vec![
            ("Type", lopdf::Object::Name(b"Pages".to_vec())),
            (
                "Kids",
                lopdf::Object::Array(vec![lopdf::Object::Reference(page_obj_id)]),
            ),
            ("Count", lopdf::Object::Integer(1)),
        ]));

        if let Ok(lopdf::Object::Dictionary(page)) = doc.get_object_mut(page_obj_id) {
            page.set("Parent", pages_id);
        }

        let catalog_id = doc.add_object(lopdf::Dictionary::from_iter(vec![
            ("Type", lopdf::Object::Name(b"Catalog".to_vec())),
            ("Pages", lopdf::Object::Reference(pages_id)),
        ]));
        doc.trailer.set("Root", catalog_id);

        (doc, page_obj_id)
    }

    /// Build a minimal in-memory PDF with a single page whose content stream is `content`,
    /// using a Type0 font (`F1`) with an Identity-H encoding and no `ToUnicode` CMap.
    fn doc_with_unmappable_type0_font(content: &[u8]) -> (Document, PageID) {
        doc_with_font(
            content,
            vec![
                ("Type", lopdf::Object::Name(b"Font".to_vec())),
                ("Subtype", lopdf::Object::Name(b"Type0".to_vec())),
                ("BaseFont", lopdf::Object::Name(b"FakeFont".to_vec())),
                ("Encoding", lopdf::Object::Name(b"Identity-H".to_vec())),
            ],
            None,
        )
    }

    /// Build a minimal in-memory PDF with a single page whose content stream is `content`,
    /// using a simple Type1 font (`F1`) that `parse_content` treats as directly mappable.
    fn doc_with_simple_type1_font(content: &[u8]) -> (Document, PageID) {
        doc_with_font(
            content,
            vec![
                ("Type", lopdf::Object::Name(b"Font".to_vec())),
                ("Subtype", lopdf::Object::Name(b"Type1".to_vec())),
                ("BaseFont", lopdf::Object::Name(b"Helvetica".to_vec())),
            ],
            None,
        )
    }

    /// Build a minimal in-memory PDF using a simple Type1 font with a `ToUnicode` CMap.
    fn doc_with_simple_type1_tounicode_font(content: &[u8]) -> (Document, PageID) {
        let cmap = b"\
1 beginbfchar
<41> <03a9>
<42> <03b2>
endbfchar";
        doc_with_font(
            content,
            vec![
                ("Type", lopdf::Object::Name(b"Font".to_vec())),
                ("Subtype", lopdf::Object::Name(b"Type1".to_vec())),
                ("BaseFont", lopdf::Object::Name(b"Helvetica".to_vec())),
                ("Encoding", lopdf::Object::Name(b"WinAnsiEncoding".to_vec())),
            ],
            Some(cmap),
        )
    }

    #[test]
    fn test_wellformed_tj_extracted() {
        // The common case: a simple string drawn with Tj.
        let (doc, page_id) = doc_with_simple_type1_font(b"BT /F1 12 Tf (Hello World) Tj ET");

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing a well-formed Tj should not be an error");

        assert!(
            result.content.contains("Hello World"),
            "Expected extracted text to contain 'Hello World', got {:?}",
            result.content
        );
    }

    #[test]
    fn test_simple_font_tounicode_overrides_encoding() {
        // A and B would decode literally under WinAnsiEncoding, but ToUnicode has priority for both
        // literal and hexadecimal PDF string syntax.
        let content = b"BT /F1 12 Tf (A) Tj <42> Tj ET";
        let (doc, page_id) = doc_with_simple_type1_tounicode_font(content);

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("A simple font's ToUnicode CMap should be used");

        assert!(
            result.content.contains("Ωβ"),
            "Expected ToUnicode mappings to override WinAnsiEncoding, got {:?}",
            result.content
        );
        assert!(
            !result.content.contains('A') && !result.content.contains('B'),
            "Raw character codes leaked into extracted text: {:?}",
            result.content
        );
    }

    #[test]
    fn test_tj_quote_operator_extracted() {
        // `'` is equivalent to `T* Tj`; `"` additionally sets Tw and Tc. Text shown with
        // these operators must be extracted like text shown with Tj.
        let (doc, page_id) = doc_with_simple_type1_font(b"BT /F1 12 Tf (line1) ' (line2) ' ET");

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing should not be an error");

        assert!(
            result.content.contains("line1\nline2"),
            "got {:?}",
            result.content
        );
    }

    #[test]
    fn test_actual_text_not_leaked_into_output() {
        // Marked content with an /ActualText property list: the (mapped text) literal is an
        // accessibility mapping, not drawn text, and must not appear in the output.
        let content = b"BT /F1 12 Tf /Span <</ActualText (mapped text)>> BDC (real) Tj EMC ET";
        let (doc, page_id) = doc_with_simple_type1_font(content);

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing should not be an error");

        assert!(
            !result.content.contains("mapped text"),
            "/ActualText literal leaked into extracted text: {:?}",
            result.content
        );
        assert!(
            result.content.contains("real"),
            "Expected extracted text to contain 'real', got {:?}",
            result.content
        );
    }

    #[test]
    fn test_inline_image_data_not_leaked_into_output() {
        // Inline image data between ID and EI is raw bytes that can tokenize into strings;
        // it must not be interpreted as text by a following Tj.
        let content =
            b"BT /F1 12 Tf BI /W 1 /H 1 /CS /DeviceGray /BPC 8 ID raw (junk) bytes EI (real) Tj ET";
        let (doc, page_id) = doc_with_simple_type1_font(content);

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing should not be an error");

        assert!(
            !result.content.contains("junk"),
            "Inline image data leaked into extracted text: {:?}",
            result.content
        );
        assert!(
            result.content.contains("real"),
            "Expected extracted text to contain 'real', got {:?}",
            result.content
        );
    }

    #[test]
    fn test_adjacent_tj_ops_not_split_into_words() {
        // Multi-font words drawn as consecutive Tj ops (common with e.g. bold infixes):
        // these are one word on the same line and should not have spaces inserted between
        // them.
        let content = b"BT /F1 12 Tf (Hel) Tj /F1 12 Tf (lo) Tj ET";
        let (doc, page_id) = doc_with_simple_type1_font(content);

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing should not be an error");

        assert!(
            result.content.contains("Hello"),
            "Expected adjacent Tj ops to form one word 'Hello', got {:?}",
            result.content
        );
    }

    #[test]
    fn test_type0_font_without_tounicode_is_unmappable() {
        use crate::fonts::compute_font_encoding;

        let (doc, page_id) = doc_with_unmappable_type0_font(b"");

        let font_obj = get_font(&doc, page_id, "F1").unwrap();
        let encoding = compute_font_encoding(&doc, &font_obj, "F1")
            .expect("A Type0 font without a ToUnicode CMap should not be an error");

        assert!(
            matches!(encoding, FontEncoding::Unmappable),
            "Expected Unmappable, got {encoding:?}"
        );
    }

    #[test]
    fn test_unmappable_font_text_is_skipped_and_counted() {
        // One TJ block with a hex string (8 hex chars = 2 CIDs), and one with a literal string
        // (4 bytes = 2 two-byte CIDs).
        let (doc, page_id) =
            doc_with_unmappable_type0_font(b"BT /F1 12 Tf [<ABCD1234>]TJ [(ABCD)]TJ ET");

        let mut parser = PdfParser::default();
        let result = parser
            .parse_content(&doc, page_id, 0, false)
            .expect("Parsing text in an unmappable font should not be an error");

        test_eq!(result.skipped_chars, 4);
        assert!(
            result.content.trim().is_empty(),
            "Expected no extracted content, got {:?}",
            result.content
        );
    }

    #[test]
    fn test_process_tj_tokens_counts_skipped_cids() {
        let (doc, page_id) = doc_with_unmappable_type0_font(b"");

        let mut parser = PdfParser {
            cur_font_id: Rc::from("F1"),
            cur_font: "FakeFont".to_string(),
            ..PdfParser::default()
        };

        // 8 hex digits = 2 CIDs; 4 literal bytes = 2 CIDs.
        let tokens = vec![Token::Hex(b"ABCD1234"), Token::Literal(b"ABCD")];
        let (text, skipped) = parser.process_tj_tokens(Token::Op(b"TJ"), &tokens, &doc, page_id);

        test_eq!(skipped, 4);
        assert!(text.trim().is_empty(), "Expected no text, got {text:?}");

        // Whitespace inside a hex string is permitted by the PDF spec and must not inflate the
        // skipped count: this is still 8 hex digits = 2 CIDs.
        let tokens = vec![Token::Hex(b"AB CD 12\n34")];
        let (_, skipped) = parser.process_tj_tokens(Token::Op(b"TJ"), &tokens, &doc, page_id);

        test_eq!(skipped, 2);
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

        test_ok!(result);
        let [font_id, font_size] = result.unwrap();
        test_eq!(font_id, b"F28");
        test_eq!(font_size, b"12.0");
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

        test_ok!(result);
        let [x, y] = result.unwrap();
        test_eq!(x, b"100.5");
        test_eq!(y, b"-20.3");
    }
}
