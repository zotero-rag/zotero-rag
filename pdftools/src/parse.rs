use core::str;
use std::error::Error;

use lopdf::Document;

#[derive(Debug)]
enum PdfError {
    PageFontError,
    FontNotFound,    // Font key not found in dictionary (shouldn't happen)
    MissingBaseFont, // Font object missing BaseFont field
    InvalidFontName, // BaseFont value isn't a valid name
    InvalidUtf8,     // Font name isn't valid UTF-8
}

fn get_font(doc: &Document, page_id: (u32, u16), font_key: String) -> Result<&str, PdfError> {
    const ASCII_PLUS: u8 = b'+';

    // Get the fonts dictionary for the page
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

fn is_math_font(font_name: &str) -> bool {
    font_name.starts_with("CMMI")  // Computer Modern Math Italic
        || font_name.starts_with("CMSY")  // Computer Modern Symbol
        || font_name.starts_with("CMEX")  // Computer Modern Extension
        || font_name.starts_with("CMR") // Computer Modern Roman
}

fn parse_content(doc: &Document, page_id: (u32, u16)) -> String {
    /* Parser v1. Only extracts text, with some heuristics about what should and
     * should not be a single word.
     *
     * Known issues:
     * - Footer text on the left column is added as part of the left column.
     * - The way we do spacing is not perfect, there are some cases where we add a space where it
     * shouldn't be there.
     * - We do not parse tables well (or at all, really)
     * - We don't handle more complex equations yet. */
    let content = doc.get_page_content(page_id).unwrap();
    let content = String::from_utf8_lossy(&content).to_string();

    const SAME_WORD_THRESHOLD: i32 = 60;
    const SUBSCRIPT_THRESHOLD: f32 = 9.0;
    let mut rem_content = content.clone();
    let mut parsed = String::new();
    let mut cur_font: &str = "";

    /* Are we in a subscript/superscript?
     * 0 = no
     * positive: in a subscript, value determines the order of subscript (we can have e^x^2, e.g.)
     * negative: in a superscript, same as above */
    let mut script_status: i32 = 0;

    loop {
        if !rem_content.contains("TJ") {
            break;
        }

        /* Heuristic: look for <number> <number> Td. If the second number (vertical) is
         * positive and less than 9 (which is a reasonable line height), we treat it as a superscript
         * until we find the same number, but negative. We do the same with subscripts. */
        if let Some(td_idx) = rem_content.find("Td") {
            let space_idx = rem_content[..td_idx - 1].rfind(" ").unwrap_or_else(|| {
                panic!("Found a Td command, but no words before it.");
            });

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
            if (0.1..=SUBSCRIPT_THRESHOLD).contains(&vert) {
                if script_status < 0 {
                    parsed += "}"; // end the subscript level
                } else {
                    parsed += "^{"; // begin a superscript level
                }
                script_status += 1;
            } else if (-SUBSCRIPT_THRESHOLD..0.0).contains(&vert) {
                if script_status <= 0 {
                    parsed += "_{";
                } else {
                    parsed += "}";
                }
                script_status -= 1;
            }
        }
        /* TODO: The above logic also captures footnotes, so we might want to parse those while
         * we're here. */

        let end_idx = rem_content.find("TJ").unwrap();

        // Check the font, if it has been set.
        if rem_content[..end_idx].contains("/F") {
            let font_begin_idx = rem_content[..end_idx].find("/F").unwrap();
            let font_end_idx = rem_content[font_begin_idx..].find(" ").unwrap() + font_begin_idx;

            let font_id = rem_content[font_begin_idx + 1..font_end_idx].to_string();
            cur_font = get_font(doc, page_id, font_id).unwrap_or("");
        }

        // We need to match the ] immediately preceding TJ with its [, but papers have references
        // that are written inside [], so a naive method doesn't work. Yes--right now, this doesn't
        // need a stack, but if it turns out we need to do this for other characters, we might want
        // it later.
        let mut begin_idx = end_idx;
        let mut stack = Vec::new();
        while let Some(val) = rem_content[..begin_idx].rfind(|c| ['[', ']'].contains(&c)) {
            match rem_content.as_bytes()[val] as char {
                ']' => stack.push(']'),
                '[' => {
                    if stack.is_empty() {
                        parsed += "[";
                        break;
                    }

                    if *stack.last().unwrap() == ']' {
                        stack.pop();
                    }
                }
                _ => {
                    unreachable!("Invalid pathway reached");
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
            let idx1 = cur_content.find('(').unwrap();
            let idx2 = cur_content.find(')').unwrap();

            if idx1 >= idx2 {
                break;
            }

            if is_math_font(cur_font) {
                // TODO
                parsed += &cur_content[idx1 + 1..idx2];
            } else {
                parsed += &cur_content[idx1 + 1..idx2];
            }

            if !cur_content[idx2..].contains('(') {
                parsed += " ";
                break;
            }

            let idx3 = cur_content[idx2..].find('(').unwrap() + idx2;
            let spacing = cur_content[idx2 + 1..idx3].parse::<i32>().unwrap().abs();

            if !(0..=SAME_WORD_THRESHOLD).contains(&spacing) {
                parsed += " ";
            }

            cur_content = &cur_content[idx2 + 1..];
        }

        rem_content = rem_content[end_idx + 2..].to_string();
    }

    // Parse the weird octal representations
    parsed = parsed
        .replace("\\050", "(")
        .replace("\\051", ")")
        .replace("\\002", "fi")
        .replace("\\017", "*")
        .replace("\\227", "--")
        .replace("\\247", "Section ")
        .replace("\\223", "\"")
        .replace("\\224", "\"")
        .replace("\\000", "-");

    parsed
}

pub fn extract_text(file_path: &str) -> Result<String, Box<dyn Error>> {
    let doc = Document::load(file_path)?;
    let mut content: String = String::new();

    // An easy way to look at specific pages in the paper.
    for page_id in doc.page_iter() {
        content += &parse_content(&doc, page_id);
    }

    dbg!("\nParsed: {}", &content);

    Ok(content)
}

#[cfg(test)]
mod tests {
    use std::str::FromStr;

    use super::*;

    #[test]
    fn test_parsing_works() {
        let mut path = std::env::current_dir().expect("Failed to get cwd");
        path.push("assets/test1.pdf");
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
        let mut path = std::env::current_dir().expect("Failed to get cwd");
        path.push("assets/symbols.pdf");

        let doc = Document::load(path).unwrap();
        let page_id = doc.page_iter().next().unwrap();
        let page_content = doc.get_page_content(page_id).unwrap();
        let content = String::from_utf8_lossy(&page_content);

        dbg!(&content);

        const TEST_QUERIES: [&str; 3] = ["F21", "F27", "F30"];
        for test in TEST_QUERIES {
            assert!(content.contains(test));
        }

        let font_name = get_font(&doc, page_id, String::from_str("F30").unwrap());
        assert!(font_name.is_ok());
        assert_eq!(font_name.unwrap(), "CMMI7");
    }
}
