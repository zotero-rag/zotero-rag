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
     */
    const THRESHOLD: i32 = 60;
    let mut rem_content = content.clone();
    let mut parsed = String::new();

    loop {
        if !rem_content.contains("TJ") {
            break;
        }

        // Text array
        let end_idx = rem_content.find("TJ").unwrap();

        // We need to match the ] immediately preceding TJ with its [, but papers have references
        // that are written inside [], so a naive method doesn't work. Yes--right now, this doesn't
        // need a stack, but if it turns out we need to do this for other characters, we might want
        // it later.
        let mut begin_idx = end_idx;
        let mut stack = Vec::new();
        while let Some(val) = rem_content[..begin_idx].rfind(|c| c == '[' || c == ']') {
            match rem_content.as_bytes()[val] as char {
                ']' => stack.push(']'),
                '[' => {
                    if stack.is_empty() {
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

            parsed += &cur_content[idx1 + 1..idx2];

            if !cur_content[idx2..].contains('(') {
                parsed += " ";
                break;
            }

            let idx3 = cur_content[idx2..].find('(').unwrap() + idx2;
            let spacing = cur_content[idx2 + 1..idx3].parse::<i32>().unwrap().abs();

            if !(0..=THRESHOLD).contains(&spacing) {
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
        .replace("\\247", "Section ");

    parsed
}

pub fn extract_text(file_path: &str) -> Result<String, Box<dyn Error>> {
    let doc = Document::load(file_path)?;
    let mut content: String = String::new();

    for page_id in doc.page_iter() {
        let contents = doc.get_page_content(page_id)?;
        let text_content = String::from_utf8_lossy(&contents);

        content += text_content.as_ref();
        // TODO: Remove this after we finish testing
        break;
    }

    let parsed_text = parse_content(content);

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
