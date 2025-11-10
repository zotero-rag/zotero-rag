//! The core PDF parsing module. This includes the `PdfParser` struct, which is somewhat tuned for
//! academic PDFs. In particular, it skips images and tables by default. This behavior might change
//! later. The parser also handles common math symbols and converts them to their corresponding
//! LaTeX equivalents.

use core::str;
use log;
use std::error::Error;

use std::collections::HashMap;
use std::sync::LazyLock;

use lopdf::Document;

use crate::math::{from_cmex, from_cmmi, from_cmsy, from_msbm};

const ASCII_PLUS: u8 = b'+';
const DEFAULT_SAME_WORD_THRESHOLD: f32 = 60.0;
const DEFAULT_TABLE_EUCLIDEAN_THRESHOLD: f32 = 20.0;

/// A wrapper for all PDF parsing errors
#[derive(Debug, thiserror::Error)]
enum PdfError {
    ContentError,
    FontNotFound,
    InternalError(String),
    InvalidFontName,
    InvalidUtf8,
    MissingBaseFont,
    PageFontError,
}

impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdfError::ContentError => write!(f, "Failed to get page content"),
            PdfError::FontNotFound => write!(f, "Font key not found in dictionary"),
            PdfError::InternalError(e) => write!(f, "{e}"),
            PdfError::InvalidFontName => write!(f, "BaseFont value isn't a valid name"),
            PdfError::InvalidUtf8 => write!(f, "Font name isn't valid UTF-8"),
            PdfError::MissingBaseFont => write!(f, "Font object missing BaseFont field"),
            PdfError::PageFontError => write!(f, "Failed to get page fonts"),
        }
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

/// A type to convert from bytes in math fonts to LaTeX code
type ByteTransformFn = fn(u8) -> String;

/// A zero-allocation iterator for octal escape sequences and raw bytes. This is useful for parsing
/// octal escape codes that are used in math fonts when non-printable characters are used to
/// represent symbols.
pub struct IterCodepoints<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Iterator for IterCodepoints<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b == b'\\' {
            // Possible octal escape
            let rem = self.bytes.len() - self.pos;
            if rem >= 4  // Length of octal escape sequence
                && self.bytes[self.pos + 1] == b'0'
                && self.bytes[self.pos + 2].is_ascii_digit()
                && self.bytes[self.pos + 2] < b'8'
                && self.bytes[self.pos + 3].is_ascii_digit()
                && self.bytes[self.pos + 3] < b'8'
            {
                let oct = &self.bytes[self.pos + 1..=self.pos + 3];
                let code = (oct[1] - b'0') * 8 + (oct[2] - b'0');
                self.pos += 4;
                Some(code)
            } else {
                // Just a backslash or malformed escape
                self.pos += 1;
                Some(b'\\')
            }
        } else {
            self.pos += 1;
            Some(b)
        }
    }
}

fn font_transform(input: &str, transform: ByteTransformFn) -> String {
    IterCodepoints {
        bytes: input.as_bytes(),
        pos: 0,
    }
    .map(transform)
    .collect::<String>()
}

/// A lazy-loaded hashmap storing conversions from math fonts to LaTeX code
/// Handles most common math fonts, but does not yet support specialized math fonts.
static FONT_TRANSFORMS: LazyLock<HashMap<&'static str, ByteTransformFn>> = LazyLock::new(|| {
    let mut m: HashMap<&'static str, ByteTransformFn> = HashMap::new();

    m.insert("CMMI5", from_cmmi);
    m.insert("CMMI6", from_cmmi);
    m.insert("CMMI7", from_cmmi);
    m.insert("CMMI8", from_cmmi);
    m.insert("CMMI9", from_cmmi);
    m.insert("CMMI10", from_cmmi);
    m.insert("CMMI12", from_cmmi);

    m.insert("CMSY5", from_cmsy);
    m.insert("CMSY6", from_cmsy);
    m.insert("CMSY7", from_cmsy);
    m.insert("CMSY8", from_cmsy);
    m.insert("CMSY9", from_cmsy);
    m.insert("CMSY10", from_cmsy);

    m.insert("CMEX10", from_cmex);

    m.insert("MSBM5", from_msbm);
    m.insert("MSBM6", from_msbm);
    m.insert("MSBM7", from_msbm);
    m.insert("MSBM8", from_msbm);
    m.insert("MSBM9", from_msbm);
    m.insert("MSBM10", from_msbm);

    m
});

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
    /// Current font size
    cur_font_size: f32,
    /// The \baselineskip set by the user.
    /// TODO: Actually compute this; for now, this is set to the pdflatex default of 1.2
    cur_baselineskip: f32,
}

impl PdfParser {
    /// Creates a new parser with the given configuration
    ///
    /// # Arguments
    /// * `config` - The configuration to use for parsing
    fn new(config: PdfParserConfig) -> Self {
        Self {
            config,
            cur_font: String::new(),
            cur_font_size: 12.0,   // Doesn't really matter
            cur_baselineskip: 1.2, // The pdflatex default
        }
    }

    fn with_default_config() -> Self {
        Self::new(PdfParserConfig::default())
    }

    fn contains_unescaped(&self, content: &str, ch: char) -> bool {
        self.find_next_unescaped(content, ch).is_some()
    }

    #[allow(clippy::unused_self)]
    fn find_next_unescaped(&self, content: &str, ch: char) -> Option<usize> {
        let mut start_idx: usize = 0;

        loop {
            let rel = content.get(start_idx..)?.find(ch)?;
            let idx = start_idx + rel;

            // If the match is preceded by a backslash, it may be escaped. Count
            // the number of consecutive backslashes immediately preceding it.
            let mut escape_count = 0;
            for c in content[..idx].chars().rev() {
                if c == '\\' {
                    escape_count += 1;
                } else {
                    break;
                }
            }

            if escape_count % 2 == 0 {
                return Some(idx);
            }

            // Continue searching after this escaped occurrence
            start_idx = idx + 1;
        }
    }

    /// Get the `N` whitespace-separated words in `content` before position `pos`. This is used to
    /// get the operators for a PDF command.
    ///
    /// # Errors
    /// * `PdfError::InternalError` if `N` > number of words in `content[..pos]`
    #[allow(clippy::unused_self)]
    fn get_params<'a, const N: usize>(
        &self,
        content: &'a str,
        pos: usize,
    ) -> Result<[&'a str; N], PdfError> {
        let parts: Vec<&str> = content[..pos].split_whitespace().collect();
        let n_parts = parts.len();

        if n_parts < N {
            let operator = content[pos..]
                .split_whitespace()
                .next()
                .unwrap_or("unknown");

            return Err(PdfError::InternalError(format!(
                "get_params expected {N} params for {operator}, but got {n_parts} instead"
            )));
        }

        Ok(std::array::from_fn(|i| parts[n_parts - N + i]))
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

    /// Given a string `content` and a position `pos`, look *around* `pos` and search for likely
    /// boundaries for a table. This function uses the heuristic that tables are likely to be
    /// near `ET` blocks, because tables typically have some graphics (lines for borders, etc.).
    /// Under this assumption, this function looks for the first `Td` command after the *previous*
    /// `BT` from `pos`, and compares the position described there to positions in the first `Td`
    /// command after every `BT` starting from `pos`, using the Euclidean distance. If this
    /// distance is below a threshold, it is assumed that these are alignment efforts.
    ///
    /// # Returns:
    /// * `Some((start_idx, end_idx))`, where `start_idx` is the index of the `BT` command where it is suspected that
    ///   the table begins, and `end_idx` is the index of the `ET` command where the table is suspected to end.
    /// * If no table is detected, returns `None`.
    fn get_table_bounds(&self, content: &str, pos: usize) -> Option<(usize, usize)> {
        let bt_idx = content[..pos].rfind("BT")?; // First `BT` we see
        let prev_td = content[bt_idx..].find("Td")? + bt_idx; // The immediately following `Td`

        let params_result = self.get_params::<2>(content, prev_td).ok();
        let (first_x, first_y) = params_result
            .and_then(|[x, y]| Some((x.parse::<f32>().ok()?, y.parse::<f32>().ok()?)))?;

        // Running position
        let mut cur_pos = pos;

        // How many `BT`s have we skipped?
        let mut bt_count = 0;

        loop {
            // Try to find the next BT
            let Some(next_bt) = content[cur_pos..].find("BT") else {
                if bt_count > 0 {
                    // If we've processed at least one BT and reached the end, return what we have
                    log::debug!("Could not find a BT, is the table at the end of the document?");
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

            // Parse the x,y parameters using and_then for cleaner error handling
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
                cur_pos = cur_td_idx;
            } else if bt_count > 0 {
                // We've found the end of the table
                return Some((bt_idx, next_bt_pos));
            } else {
                // Not a table
                return None;
            }
        }
    }

    /// The actual PDF parser itself. Parses UTF-8 encoded code points in a best-effort manner,
    /// making reasonable assumptions along the way. Such assumptions are documented.
    #[allow(clippy::too_many_lines)]
    fn parse_content(&mut self, doc: &Document, page_id: (u32, u16)) -> Result<String, PdfError> {
        let content = doc
            .get_page_content(page_id)
            .map_err(|_| PdfError::ContentError)?;
        let content = String::from_utf8_lossy(&content).to_string();
        let mut cur_parse_idx = 0;
        let mut parsed = String::new();

        // Keep track of the font sizes (Tf), and vertical movements (second arg of Td) and associated positions
        let mut tf_history: Vec<(f32, usize)> = Vec::new();
        let mut y_history: Vec<(f32, usize)> = Vec::new();

        /* A loop over TJ blocks. The broad idea is:
         * 1. Look for tables/images--and skip them.
         * 2. Handle super/sub-scripts and footnotes.
         * 3. Handle math fonts to parse equations.
         * 4. Iterate over parenthesized blocks in the TJ to parse text. */
        loop {
            /* We need to look for an ET so that we can exclude tables/images. However, it is possible to
             * find an ET that is a table but is too far away to worry about for now; so we actually need to
             * look for both an ET and a TJ. *However*, it is also possible for the immediate TJ to be part of
             * the table that we are trying to avoid. So the following code isn't particularly efficient, but it
             * should cover all the cases. */
            let Some(tj_idx) = content[cur_parse_idx..].find("TJ") else {
                // No more TJs, so nothing left to parse.
                break;
            };

            // Generally, we *should* find an ET here, since we already found a TJ (which means
            // there's a text block to end).
            if let Some(et_idx) = content[cur_parse_idx..].find("ET") {
                // Handle tables
                if let Some((tbl_begin_idx, tbl_end_idx)) =
                    self.get_table_bounds(&content, cur_parse_idx + et_idx)
                    && tbl_begin_idx < cur_parse_idx + tj_idx
                {
                    // Skip over the table
                    cur_parse_idx = tbl_end_idx;

                    // We've invalidated some indexes in the conditions above, so we
                    // actually can't just proceed.
                    continue;
                }

                // Handle images. The TJ has to be after the next ET--otherwise, it's
                // unlikely to be a caption. We assume here that figure captions occur after
                // the figure itself.
                if tj_idx > et_idx
                    && let Some(im_end_idx) =
                        self.get_image_bounds(&content, cur_parse_idx + et_idx)
                {
                    cur_parse_idx = im_end_idx;
                    continue;
                }
            }

            let et_idx = content[cur_parse_idx..].find("ET").unwrap_or(usize::MAX);

            // Get the current font size, if it's set.
            if let Some(tf_idx) = content[cur_parse_idx..].find("Tf")
                && tf_idx < et_idx
            {
                let [font_size_str] = self.get_params::<1>(&content, cur_parse_idx + tf_idx)?;

                if let Ok(font_size) = font_size_str.parse::<f32>() {
                    tf_history.push((font_size, parsed.len()));

                    self.cur_font_size = font_size;
                }
            }

            // `y_history` maintains a stack of every time we changed vertical position, and the
            // current position in `parsed` each time that happened. If the y-delta is zero, then
            // we don't care about this.
            if let Some(td_idx) = content[cur_parse_idx..].find("Td")
                // We need this Td to be before an ET, otherwise we could be looking too far ahead
                && td_idx < et_idx
            {
                let [_, vert_str] = self.get_params::<2>(&content, cur_parse_idx + td_idx)?;

                if let Ok(vert) = vert_str.parse::<f32>() {
                    // `Td` args are movements, not absolute
                    let (cur_y, _) = y_history.last().unwrap_or(&(0.0, 0));
                    let new_y = cur_y + vert;

                    if vert != 0.0 {
                        y_history.push((new_y, parsed.len()));
                    }
                }
            }
            /* TODO: The above logic also captures footnotes, so we might want to parse those while
             * we're here. */

            let end_idx = content[cur_parse_idx..]
                .find("TJ")
                .ok_or(PdfError::ContentError)?;

            // Check the font, if it has been set.
            if content[cur_parse_idx..cur_parse_idx + end_idx].contains("/F") {
                let font_begin_idx = content[cur_parse_idx..cur_parse_idx + end_idx]
                    .find("/F")
                    .ok_or(PdfError::ContentError)?;
                let font_end_idx =
                    content[cur_parse_idx + font_begin_idx..].find(' ').unwrap() + font_begin_idx;

                let font_id =
                    content[cur_parse_idx..][font_begin_idx + 1..font_end_idx].to_string();
                self.cur_font = get_font(doc, page_id, &font_id).unwrap_or("").to_string();
            }

            // We need to match the ] immediately preceding TJ with its [, but papers have references
            // that are written inside [], so a naive method doesn't work. Yes--right now, this doesn't
            // need a stack, but if it turns out we need to do this for other characters, we might want
            // it later.
            let mut begin_idx = end_idx;
            let mut stack = Vec::with_capacity(50);
            while let Some(val) =
                content[cur_parse_idx..cur_parse_idx + begin_idx].rfind(|c| ['[', ']'].contains(&c))
            {
                let char_at_val = content.as_bytes()[cur_parse_idx + val] as char;
                if char_at_val == ']' {
                    stack.push(']');
                } else if char_at_val == '[' {
                    if stack.is_empty() {
                        parsed += "[";
                        break;
                    }
                    if let Some(']') = stack.last() {
                        stack.pop();
                    }
                }

                begin_idx = val;
            }

            let mut cur_content = &content[cur_parse_idx..][begin_idx..end_idx];

            /* Here's our strategy. We'll look for pairs of (), consuming words inside.
             * Then, we'll consume an integer. If that integer is less than SAME_WORD_THRESHOLD, the next
             * chunk will be appended to the current word. Otherwise, we add a space. */
            // TODO: Handle paragraphs
            while self.contains_unescaped(cur_content, '(') {
                let idx1 = self
                    .find_next_unescaped(cur_content, '(')
                    .ok_or(PdfError::ContentError)?;
                let idx2 = self
                    .find_next_unescaped(cur_content, ')')
                    .ok_or(PdfError::ContentError)?;

                if idx1 >= idx2 {
                    break;
                }

                if let Some(transform) = FONT_TRANSFORMS.get(self.cur_font.as_str()) {
                    parsed += &font_transform(&cur_content[idx1 + 1..idx2], *transform);
                } else {
                    parsed += &cur_content[idx1 + 1..idx2];
                }

                if !self.contains_unescaped(&cur_content[idx2..], '(') {
                    parsed += " ";
                    break;
                }

                let idx3 = self.find_next_unescaped(&cur_content[idx2..], '(').unwrap() + idx2;
                let spacing = cur_content[idx2 + 1..idx3]
                    .parse::<f32>()
                    .unwrap_or(self.config.same_word_threshold + 1.0)
                    .abs();

                if !(0.0..=self.config.same_word_threshold).contains(&spacing) {
                    parsed += " ";
                }

                cur_content = &cur_content[idx2 + 1..];
            }

            cur_parse_idx += end_idx + 2;
        }

        // Replace ligatures with constitutent characters
        for (from, to) in OCTAL_REPLACEMENTS.iter() {
            parsed = parsed.replace(from, to);
        }

        // Having collected all the positions where the y position was changed, we can now work
        // backwards to add sub/superscript markers. The core idea here is that when we "come back"
        // from a script, the y position will return to one that we've seen. This creates a span of
        // indices to look through. Within this span, we can parse nested scripts (but note that
        // these might not all follow the same "direction", particularly because for sums, the
        // upper limit comes before the summation symbol for some reason in LaTeX-created content
        // streams. We use the sign in the difference between y positions to determine what kind of
        // a script it is.
        let mut additions: Vec<(usize, &str)> = Vec::new();
        let mut i = y_history.len().saturating_sub(1);
        while i > 0 {
            // Find the last index where the y position was equal to the y position recorded by
            // `y_history[i]`.
            #[allow(clippy::float_cmp)]
            let j = (0..i).rev().find(|k| y_history[*k].0 == y_history[i].0);

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
                let offset = if parsed.as_bytes().get(y_history[j].1) == Some(&BACKSLASH_ASCII) {
                    parsed[y_history[j].1..].find(' ').unwrap()
                } else {
                    0
                };

                additions.push((y_history[j + 1].1 - 1, "}"));
                // TODO: Refine the below rule.
                additions.push((
                    y_history[j].1 + offset,
                    if y_history[j].0 > y_history[j_orig].0 {
                        "^{"
                    } else {
                        "_{"
                    },
                ));

                j += 2;
            }

            i = j_orig.saturating_sub(1);
        }

        // Sort in descending order and then perform the insertions
        additions.sort_by(|a, b| b.0.cmp(&a.0));
        for (pos, s) in additions {
            if pos < parsed.len() {
                parsed.insert_str(pos, s);
            }
        }

        Ok(parsed)
    }
}

fn get_font<'a>(doc: &'a Document, page_id: (u32, u16), font_key: &str) -> Result<&'a str, PdfError> {
    // Get the font dictionary for the page
    let fonts = doc
        .get_page_fonts(page_id)
        .map_err(|_| PdfError::PageFontError)?;

    let font_obj = fonts
        .get(font_key.as_bytes())
        .ok_or(PdfError::FontNotFound)?;

    let base_font = font_obj
        .as_hashmap()
        .get("BaseFont".as_bytes())
        .ok_or(PdfError::MissingBaseFont)?;

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

/// Extracts text content from a PDF file at the given path.
///
/// # Errors
/// Returns an error if the file cannot be loaded or if text extraction fails.
pub fn extract_text(file_path: &str) -> Result<String, Box<dyn Error>> {
    let doc = Document::load(file_path)?;
    let mut parser = PdfParser::with_default_config();

    let page_count = doc.get_pages().len();

    let content = doc
        .page_iter()
        .enumerate()
        .map(|(page_num, page_id)| {
            log::debug!("\tParsing page {} of {page_count}", page_num + 1);

            parser
                .parse_content(&doc, page_id)
                .unwrap_or_else(|_| String::new())
        })
        .collect::<String>();

    Ok(content)
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::path::PathBuf;

    use super::*;

    /// Get the raw content stream for page `page_num` for the PDF.
    fn get_raw_content_stream(doc: &Document, page_num: usize) -> Result<String, PdfError> {
        let page_id: (u32, u16) = doc
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
        let content = extract_text(path.to_str().unwrap()).unwrap();
        for query in queries {
            assert!(
                content.contains(*query),
                "content of {file_name} did not contain '{query}'"
            );
        }
    }

    #[test]
    fn test_parsing_works() {
        // Test 1: "test1.pdf"
        check_pdf_contains("test1.pdf", &["Oversampling", "GHOST", "Deep Learning"]);

        // Test 2: "ntk.pdf"
        check_pdf_contains("ntk.pdf", &["\\theta", "\\otimes", "\\sum", "\\mathbb{E}"]);
    }

    #[test]
    #[ignore = "Manual test for debugging PDF content"]
    fn test_pdf_content() {
        if env::var("CI").is_ok() {
            // Skip this test in CI environments
            return;
        }

        // NOTE: Maintainers: use this as a way to quickly get the UTF-8 content of raw PDF commands.
        let path = PathBuf::from("assets").join("symbols.pdf");

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
        use lopdf::Object;

        /* In PDFs, a simplified view of fonts is as triply-nested dictionaries.
         * First, pages have a resources dictionary, which includes a font dictionary; that dictionary maps
         * font resource names (e.g., "F28") to font objects themselves--the second level of redirection.
         * Each font object has various properties of the font. This might include, for example, CMaps
         * (explained below), the font's name (e.g., "CMR10"), and other properties. It's also worth noting:
         * the 10 in CMR10 only gives the *design size* of the font in points--the size for which it was
         * designed and optimized. You still need to look at Tf for the font sizes.
         */
        let font_key = "F11";
        let path = PathBuf::from("assets").join("ntk.pdf");

        let doc = Document::load(path).unwrap();
        let page_id = doc.page_iter().next().unwrap();

        // Get the font dictionary for the page
        let fonts = doc.get_page_fonts(page_id).unwrap();

        let font_obj = fonts.get(font_key.as_bytes()).unwrap();
        let font_hash = font_obj.as_hashmap();

        let readable_font_obj: HashMap<String, &Object> = font_hash
            .iter()
            .map(|(k, v)| (String::from_utf8(k.clone()).unwrap(), v))
            .collect();
        dbg!(&readable_font_obj);

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
        let font_name = get_font(&doc, page_id, "F30");
        assert!(font_name.is_ok());
        assert_eq!(font_name.unwrap(), "CMMI7");
    }

    #[test]
    fn test_math_parsing_works() {
        let path = PathBuf::from("assets").join("symbols.pdf");

        let content = extract_text(path.to_str().unwrap());
        assert!(content.is_ok());

        let content = content.unwrap();
        for op in [r"\int", r"\sum", r"\infty"] {
            assert!(content.contains(op));
        }
    }

    #[test]
    fn test_get_table_bounds_works() {
        let path = PathBuf::from("assets").join("table.pdf");

        let doc = Document::load(&path).unwrap();
        let content = get_raw_content_stream(&doc, 0).unwrap();

        let parser = PdfParser::with_default_config();

        // The indices will not exactly line up, because "\n"s seem to be two separate characters. This is okay.
        // Test 1: The first ET in this should be ignored.
        let first_et = content.find("ET").unwrap();
        assert_eq!(first_et, 342);
        assert!(parser.get_table_bounds(&content, first_et).is_none());

        // Test 2: The second ET should capture the table (excluding the caption).
        let second_et = content[first_et + 1..].find("ET").unwrap() + first_et + 1;
        assert_eq!(second_et, 471);
        assert_eq!(
            parser.get_table_bounds(&content, second_et),
            Some((412, 707))
        );
    }

    #[test]
    fn test_tables_are_ignored() {
        let path = PathBuf::from("assets").join("table.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();
        let tests = ["r1c1", "r1c2", "r2c1", "r2c2"];
        for text in tests {
            assert!(!content.contains(text));
        }
    }

    #[test]
    fn test_images_are_ignored() {
        let path = PathBuf::from("assets").join("images.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();

        let tests = ["Figure", "Caption", "is", "caption", "good", "HERE"];
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

        let content = content.unwrap();

        let tests = ["google.com", "sec:2", "cite.yedida"];
        for text in tests {
            assert!(!content.contains(text));
        }
    }
}
