use std::error::Error;

use lopdf::Document;

fn parse_content(content: String) -> String {
    /* Parser v1. Only extracts text, with some heuristics about what should and
     * should not be a single word. */
    const THRESHOLD: i32 = 50;
    let mut rem_content = content.clone();
    let mut parsed = String::new();

    loop {
        if rem_content.contains("TJ") {
            // Text array
            let end_idx = rem_content.find("TJ").unwrap();
            let begin_idx = rem_content[..end_idx].rfind('[').unwrap_or(usize::MAX);

            if begin_idx == usize::MAX {
                break;
            }

            let mut cur_content = &rem_content[begin_idx..end_idx];
            println!("rem_content = {}", &rem_content[..100]);

            /* Here's our strategy. We'll look for pairs of (), consuming words inside.
             * Then, we'll consume an integer. If that integer is less than 50, the next
             * chunk will be appended to the current word. Otherwise, we add a space. */
            // TODO: Handle paragraphs
            while cur_content.contains('(') {
                let idx1 = cur_content.find('(').unwrap();
                let idx2 = cur_content.find(')').unwrap();
                println!(
                    "idx1 = {}, idx2= {}, cur_content = {}",
                    idx1, idx2, cur_content
                );

                parsed += &cur_content[idx1 + 1..idx2];

                if !cur_content[idx2..].contains('(') {
                    parsed += " ";
                    break;
                }

                let idx3 = cur_content[idx2..].find('(').unwrap() + idx2;
                let spacing = cur_content[idx2 + 1..idx3].parse::<i32>().unwrap();

                if !(0..=THRESHOLD).contains(&spacing) {
                    parsed += " ";
                }

                cur_content = &cur_content[idx2 + 1..];
            }

            rem_content = rem_content[end_idx + 2..].to_string();
        } else {
            break;
        }
    }

    parsed
}

pub fn extract_text(file_path: &str) -> Result<String, Box<dyn Error>> {
    let doc = Document::load(file_path)?;
    let mut content: String = String::new();

    for page_id in doc.page_iter() {
        let contents = doc.get_page_content(page_id)?;
        let text_content = String::from_utf8_lossy(&contents);

        content += text_content.as_ref();
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
        assert_eq!(content.unwrap(), "Test");
    }
}
