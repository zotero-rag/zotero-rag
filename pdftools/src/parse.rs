use core::str;
use std::error::Error;

use once_cell::sync::Lazy;
use std::collections::HashMap;

use lopdf::Document;

use crate::math::{from_cmex, from_cmmi, from_cmsy, from_msbm};

const ASCII_PLUS: u8 = b'+';
const DEFAULT_SAME_WORD_THRESHOLD: f32 = 60.0;
const DEFAULT_SUBSCRIPT_THRESHOLD: f32 = 9.0;

/// A wrapper for all PDF parsing errors
#[derive(Debug)]
enum PdfError {
    ContentError,
    PageFontError,
    FontNotFound,
    MissingBaseFont,
    InvalidFontName,
    InvalidUtf8,
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
}

impl Default for PdfParserConfig {
    fn default() -> Self {
        Self {
            same_word_threshold: DEFAULT_SAME_WORD_THRESHOLD,
            subscript_threshold: DEFAULT_SUBSCRIPT_THRESHOLD,
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

    /// The actual PDF parser itself. Parses UTF-8 encoded code points in a best-effort manner,
    /// making reasonable assumptions along the way. Such assumptions are documented.
    fn parse_content(&mut self, doc: &Document, page_id: (u32, u16)) -> Result<String, PdfError> {
        let content = doc
            .get_page_content(page_id)
            .map_err(|_| PdfError::ContentError)?;
        let content = String::from_utf8_lossy(&content).to_string();
        let mut rem_content = content;
        let mut parsed = String::new();

        loop {
            if !rem_content.contains("TJ") {
                break;
            }

            /* Heuristic: look for <number> <number> Td. If the second number (vertical) is
             * positive and less than 9 (which is a reasonable line height), we treat it as a superscript
             * until we find the same number, but negative. We do the same with subscripts. */
            if let Some(td_idx) = rem_content.find("Td") {
                let space_idx = rem_content[..td_idx - 1]
                    .rfind(" ")
                    .ok_or(PdfError::ContentError)?;
                let vert = rem_content[space_idx + 1..td_idx - 1]
                    .parse::<f32>()
                    .unwrap_or_else(|err| {
                        panic!(
                            "Failed to parse what should've been a number: '{}': {}",
                            &rem_content[space_idx + 1..td_idx - 1],
                            err
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
            while cur_content.contains('(') {
                let idx1 = cur_content.find('(').ok_or(PdfError::ContentError)?;
                let idx2 = cur_content.find(')').ok_or(PdfError::ContentError)?;

                // TODO: If ) is preceded by a \, then it may be escaped, and we may have to have
                // logic to account for this.

                if idx1 >= idx2 {
                    break;
                }

                if let Some(transform) = FONT_TRANSFORMS.get(self.current_font.as_str()) {
                    parsed += &font_transform(cur_content[idx1 + 1..idx2].to_string(), *transform);
                } else {
                    parsed += &cur_content[idx1 + 1..idx2];
                }

                if !cur_content[idx2..].contains('(') {
                    parsed += " ";
                    break;
                }

                let idx3 = cur_content[idx2..].find('(').unwrap() + idx2;
                let spacing = cur_content[idx2 + 1..idx3].parse::<f32>().unwrap().abs();

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

    let content = doc
        .page_iter()
        .map(|page_id| {
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

        let doc = Document::load(&path).unwrap();
        let page_id = doc.page_iter().next().unwrap();
        let pre_content = doc.get_page_content(page_id).unwrap();
        dbg!(String::from_utf8_lossy(&pre_content));
        println!();

        let content = extract_text(path.to_str().unwrap());

        assert!(content.is_ok());

        let content = content.unwrap();
        dbg!(content);
    }
}
