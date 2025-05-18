use core::str;
use log;
use std::error::Error;

use once_cell::sync::Lazy;
use std::collections::HashMap;

use lopdf::Document;

use crate::math::{from_cmex, from_cmmi, from_cmsy, from_msbm};

const ASCII_PLUS: u8 = b'+';
const DEFAULT_SAME_WORD_THRESHOLD: f32 = 60.0;
const DEFAULT_SUBSCRIPT_THRESHOLD: f32 = 9.0;
const DEFAULT_TABLE_EUCLIDEAN_THRESHOLD: f32 = 20.0;

/// A wrapper for all PDF parsing errors
#[derive(Debug)]
enum PdfError {
    ContentError,
    PageFontError,
    FontNotFound,
    MissingBaseFont,
    InvalidFontName,
    InvalidUtf8,
    InternalError(String),
}

impl Error for PdfError {}
impl std::fmt::Display for PdfError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            PdfError::ContentError => write!(f, "Failed to get page content"),
            PdfError::PageFontError => write!(f, "Failed to get page fonts"),
            PdfError::FontNotFound => write!(f, "Font key not found in dictionary"),
            PdfError::MissingBaseFont => write!(f, "Font object missing BaseFont field"),
            PdfError::InvalidFontName => write!(f, "BaseFont value isn't a valid name"),
            PdfError::InvalidUtf8 => write!(f, "Font name isn't valid UTF-8"),
            PdfError::InternalError(e) => write!(f, "{}", e),
        }
    }
}

/// Configuration for PDF parsing
#[derive(Debug)]
struct PdfParserConfig {
    /// Threshold for determining when to join words
    same_word_threshold: f32,
    /// Vertical movement threshold to declare sub/superscript
    subscript_threshold: f32,
    /// Euclidean distance threshold between `Td` alignment to declare a table
    table_alignment_threshold: f32,
}

impl Default for PdfParserConfig {
    fn default() -> Self {
        Self {
            same_word_threshold: DEFAULT_SAME_WORD_THRESHOLD,
            subscript_threshold: DEFAULT_SUBSCRIPT_THRESHOLD,
            table_alignment_threshold: DEFAULT_TABLE_EUCLIDEAN_THRESHOLD,
        }
    }
}

/// A type to convert from bytes in math fonts to LaTeX code
type ByteTransformFn = fn(&u8) -> String;

fn font_transform(input: String, transform: ByteTransformFn) -> String {
    input.as_bytes().iter().map(transform).collect::<String>()
}

/// A lazy-loaded hashmap storing conversions from math fonts to LaTeX code
/// Handles most common math fonts, but does not yet support specialized math fonts.
static FONT_TRANSFORMS: Lazy<HashMap<&'static str, ByteTransformFn>> = Lazy::new(|| {
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
static OCTAL_REPLACEMENTS: Lazy<HashMap<&str, &str>> = Lazy::new(|| {
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
    config: PdfParserConfig,
    current_font: String,
    script_status: i32,
}

impl PdfParser {
    /// Creates a new parser with the given configuration
    ///
    /// # Arguments
    /// * `config` - The configuration to use for parsing
    fn new(config: PdfParserConfig) -> Self {
        Self {
            config,
            current_font: String::new(),
            script_status: 0,
        }
    }

    fn with_default_config() -> Self {
        Self::new(PdfParserConfig::default())
    }

    fn contains_unescaped(&self, content: &str, ch: char) -> bool {
        self.find_next_unescaped(content, ch)
            .is_some_and(|pos| pos < content.len())
    }

    fn find_next_unescaped(&self, content: &str, ch: char) -> Option<usize> {
        let mut start_idx: usize = 0;

        loop {
            let idx = content[start_idx..].find(ch)?;

            // If ) is preceded by a \, then it may be escaped
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
            } else {
                start_idx = idx;
            }
        }
    }

    /// Get the `N` whitespace-separated words in `content` before position `pos`. This is used to
    /// get the operators for a PDF command.
    ///
    /// # Errors
    /// * `PdfError::InternalError` if `N` > number of words in `content[..pos]`
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
                "get_params expected {} params for {}, but got {} instead",
                N, operator, n_parts
            )));
        }

        Ok(std::array::from_fn(|i| parts[n_parts - N + i]))
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
            if let Some(next_bt) = content[cur_pos..].find("BT") {
                let next_bt_pos = cur_pos + next_bt;

                // Try to find a Td command after this BT
                if let Some(td_offset) = content[next_bt_pos..].find("Td") {
                    // Calculate current alignment position
                    let cur_td_idx = next_bt_pos + td_offset;

                    // Parse the x,y parameters using and_then for cleaner error handling
                    let params_result = self.get_params::<2>(content, cur_td_idx).ok();
                    if let Some((cur_x, cur_y)) = params_result
                        .and_then(|[x, y]| Some((x.parse::<f32>().ok()?, y.parse::<f32>().ok()?)))
                    {
                        let distance =
                            ((cur_x - first_x).powi(2) + (cur_y - first_y).powi(2)).sqrt();

                        if distance < self.config.table_alignment_threshold {
                            bt_count += 1;
                            cur_pos = cur_td_idx;
                            continue;
                        } else if bt_count > 0 {
                            // We've found the end of the table
                            return Some((bt_idx, next_bt_pos));
                        } else {
                            // Not a table
                            return None;
                        }
                    } else {
                        // Could not parse parameters
                        log::warn!("Could not parse Td parameters, ignoring possible table.");
                        return None;
                    }
                } else {
                    log::warn!("Could not find a Td after a BT, ignoring possible table.");
                    return None;
                }
            } else if bt_count > 0 {
                // If we've processed at least one BT and reached the end, return what we have
                log::warn!("Could not find a BT, is the table at the end of the document?");
                return Some((bt_idx, cur_pos));
            } else {
                // Not a table
                return None;
            }
        }
    }

    /// The actual PDF parser itself. Parses UTF-8 encoded code points in a best-effort manner,
    /// making reasonable assumptions along the way. Such assumptions are documented.
    fn parse_content(&mut self, doc: &Document, page_id: (u32, u16)) -> Result<String, PdfError> {
        let content = doc
            .get_page_content(page_id)
            .map_err(|_| PdfError::ContentError)?;
        let content = String::from_utf8_lossy(&content).to_string();
        let mut rem_content = content.clone();
        let mut parsed = String::new();

        loop {
            if !rem_content.contains("TJ") {
                break;
            }

            /* Heuristic: look for <number> <number> Td. If the second number (vertical) is
             * positive and less than 9 (which is a reasonable line height), we treat it as a superscript
             * until we find the same number, but negative. We do the same with subscripts. */
            if let Some(td_idx) = rem_content.find("Td") {
                let [vert_str] = self.get_params::<1>(&rem_content, td_idx)?;
                let vert = vert_str.parse::<f32>().unwrap_or_else(|err| {
                    panic!(
                        "Failed to parse what should've been a number: '{}': {}",
                        vert_str, err
                    );
                });

                // We shouldn't include 0 in these ranges
                if (0.1..=self.config.subscript_threshold).contains(&vert) {
                    if self.script_status < 0 {
                        parsed += "}"; // end the subscript level
                    } else {
                        parsed += "^{"; // begin a superscript level
                    }
                    self.script_status += 1;
                } else if (-self.config.subscript_threshold..0.0).contains(&vert) {
                    if self.script_status <= 0 {
                        parsed += "_{";
                    } else {
                        parsed += "}";
                    }
                    self.script_status -= 1;
                }
            }
            /* TODO: The above logic also captures footnotes, so we might want to parse those while
             * we're here. */

            let end_idx = rem_content.find("TJ").ok_or(PdfError::ContentError)?;

            // Check the font, if it has been set.
            if rem_content[..end_idx].contains("/F") {
                let font_begin_idx = rem_content[..end_idx]
                    .find("/F")
                    .ok_or(PdfError::ContentError)?;
                let font_end_idx =
                    rem_content[font_begin_idx..].find(" ").unwrap() + font_begin_idx;

                let font_id = rem_content[font_begin_idx + 1..font_end_idx].to_string();
                self.current_font = get_font(doc, page_id, font_id).unwrap_or("").to_string();
            }

            // We need to match the ] immediately preceding TJ with its [, but papers have references
            // that are written inside [], so a naive method doesn't work. Yes--right now, this doesn't
            // need a stack, but if it turns out we need to do this for other characters, we might want
            // it later.
            let mut begin_idx = end_idx;
            let mut stack = Vec::with_capacity(50);
            while let Some(val) = rem_content[..begin_idx].rfind(|c| ['[', ']'].contains(&c)) {
                let char_at_val = rem_content.as_bytes()[val] as char;
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

            let mut cur_content = &rem_content[begin_idx..end_idx];

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

                if let Some(transform) = FONT_TRANSFORMS.get(self.current_font.as_str()) {
                    parsed += &font_transform(cur_content[idx1 + 1..idx2].to_string(), *transform);
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

            rem_content = rem_content[end_idx + 2..].to_string();
        }

        // Parse the weird octal representations
        for (from, to) in OCTAL_REPLACEMENTS.iter() {
            parsed = parsed.replace(from, to);
        }

        Ok(parsed)
    }
}

fn get_font(doc: &Document, page_id: (u32, u16), font_key: String) -> Result<&str, PdfError> {
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
            let idx = name
                .iter()
                .position(|&byte| byte == ASCII_PLUS)
                .unwrap_or(0)
                + 1;
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
            log::debug!("\tParsing page {} of {}", page_num, page_count);

            parser
                .parse_content(&doc, page_id)
                .unwrap_or("".to_string())
        })
        .collect::<Vec<_>>()
        .join("");

    Ok(content)
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::*;

    #[test]
    fn test_parsing_works() {
        let path = PathBuf::from("assets").join("test1.pdf");
        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();

        const TEST_QUERIES: [&str; 3] = ["Oversampling", "GHOST", "Deep Learning"];
        for test in TEST_QUERIES {
            assert!(content.contains(test));
        }
    }

    #[test]
    fn test_fonts_identified_correctly() {
        let path = PathBuf::from("assets").join("symbols.pdf");

        let doc = Document::load(path).unwrap();
        let page_id = doc.page_iter().next().unwrap();
        let page_content = doc.get_page_content(page_id).unwrap();
        let content = String::from_utf8_lossy(&page_content);

        const TEST_QUERIES: [&str; 3] = ["F21", "F27", "F30"];
        for test in TEST_QUERIES {
            assert!(content.contains(test));
        }

        let font_name = get_font(&doc, page_id, String::from_str("F30").unwrap());
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
        let page_id = doc.page_iter().next().unwrap();
        let pre_content = doc.get_page_content(page_id).unwrap();
        let content = String::from_utf8_lossy(&pre_content);

        let parser = PdfParser::with_default_config();

        // NOTE: Maintainers: The indices will not exactly line up, because "\n"s seem
        // to be two separate characters. This is okay.
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
}
