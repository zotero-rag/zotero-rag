use std::error::Error;

use lopdf::Document;

fn parse_content(content: String) -> String {
    /* Parser v1. Only extracts text, with some heuristics about what should and
     * should not be a single word.
     *
     * Known issues:
     * - Footer text on the left column is added as part of the left column.
     * - The way we do spacing is not perfect, there are some cases where we add a space where it
     * shouldn't be there.
     * - We do not parse tables well (or at all, really) */
    const SAME_WORD_THRESHOLD: i32 = 60;
    const SUBSCRIPT_THRESHOLD: f32 = 9.0;
    let mut rem_content = content.clone();
    let mut parsed = String::new();

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
         * Then, we'll consume an integer. If that integer is less than 50, the next
         * chunk will be appended to the current word. Otherwise, we add a space. */
        // TODO: Handle paragraphs
        while cur_content.contains('(') {
            let idx1 = cur_content.find('(').unwrap();
            let idx2 = cur_content.find(')').unwrap();

            if idx1 >= idx2 {
                break;
            }

            parsed += &cur_content[idx1 + 1..idx2];

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
    // TODO: Remove this later
    let mut i = 0;
    for page_id in doc.page_iter() {
        i += 1;
        if i != 3 {
            continue;
        }
        let contents = doc.get_page_content(page_id)?;
        let text_content = String::from_utf8_lossy(&contents);

        content += text_content.as_ref();
    }

    dbg!("{}", &content[..7000]);
    let parsed_text = parse_content(content);
    dbg!("\nParsed: {}", &parsed_text);

    Ok(parsed_text)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_name() {
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
}
