#[derive(Debug, PartialEq, Eq)]
pub(crate) enum Token<'a> {
    Op(&'a [u8]), // PDF operators, e.g., "TJ", "Td", "Tf"
    Number(&'a [u8]),
    Literal(&'a [u8]), // Inside (..) in TJ blocks
    Hex(&'a [u8]),     // Inside <..> in TJ blocks
    Name(&'a [u8]),    // Name tokens, like "/F28"
}

enum State {
    Hex {
        start: usize,
    },
    Literal {
        depth: usize, // PDF allows nested parens
        start: usize, // For the lowest depth, track start pos
    },
    Normal,
    Number {
        start: usize,
        dot_seen: bool, // Track decimal point
    },
    Op {
        start: usize,
    },
    Name {
        start: usize,
    },
}

#[allow(clippy::too_many_lines)]
pub(crate) fn tokenize(content: &[u8]) -> Vec<Token<'_>> {
    let mut tokens = Vec::with_capacity(content.len() / 5); // Heuristic

    let mut i = 0;
    let mut state = State::Normal;

    let len = content.len();
    while i < len {
        match state {
            State::Normal => {
                match content[i] {
                    // Literal
                    b'(' => {
                        state = State::Literal {
                            depth: 1,
                            start: i + 1,
                        };
                        i += 1;
                    }
                    // Hex
                    b'<' if i + 1 < len && content[i + 1] != b'<' => {
                        state = State::Hex { start: i + 1 };
                        i += 1;
                    }
                    // Number
                    b'0'..=b'9' | b'-' | b'.' => {
                        state = State::Number {
                            start: i,
                            dot_seen: content[i] == b'.',
                        };
                        i += 1;
                    }
                    // Operator
                    b'A'..=b'Z' | b'a'..=b'z' => {
                        state = State::Op { start: i };
                        i += 1;
                    }
                    // Name
                    b'/' => {
                        state = State::Name { start: i + 1 };
                        i += 1;
                    }
                    _ => i += 1,
                }
            }
            State::Literal {
                ref mut depth,
                start,
            } => match content[i] {
                b'\\' => {
                    i += if i + 1 < len { 2 } else { 1 };
                }
                b'(' => {
                    *depth += 1;
                    i += 1;
                }
                b')' => {
                    *depth -= 1;
                    if *depth == 0 {
                        state = State::Normal;
                        tokens.push(Token::Literal(&content[start..i]));
                    }
                    i += 1;
                }
                _ => i += 1,
            },
            State::Hex { start } => {
                if content[i] == b'>' {
                    tokens.push(Token::Hex(&content[start..i]));
                    state = State::Normal;
                }
                i += 1;
            }
            State::Number {
                start,
                ref mut dot_seen,
            } => {
                if i < len && matches!(content[i], b'0'..=b'9' | b'.') {
                    // Check for duplicate decimal points
                    if content[i] == b'.' {
                        if *dot_seen {
                            // Multiple decimal points - this is invalid, emit token and reset
                            tokens.push(Token::Number(&content[start..i]));
                            state = State::Normal;
                            continue;
                        }
                        *dot_seen = true;
                    }
                    i += 1;
                } else {
                    tokens.push(Token::Number(&content[start..i]));
                    state = State::Normal;
                }
            }
            State::Op { start } => {
                if content[i].is_ascii_alphabetic() {
                    i += 1;
                } else {
                    tokens.push(Token::Op(&content[start..i]));
                    state = State::Normal;
                }
            }
            State::Name { start } => {
                if content[i].is_ascii_alphanumeric() {
                    i += 1;
                } else {
                    tokens.push(Token::Name(&content[start..i]));
                    state = State::Normal;
                }
            }
        }
    }

    match state {
        State::Number { start, dot_seen: _ } => tokens.push(Token::Number(&content[start..len])),
        State::Op { start } => tokens.push(Token::Op(&content[start..len])),
        State::Name { start } => tokens.push(Token::Name(&content[start..len])),
        _ => {}
    }

    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tokenize_tj_block() {
        let content = b"[(Hello) -250 (World)] TJ";
        let tokens = tokenize(content);
        let expected = vec![
            Token::Literal(b"Hello"),
            Token::Number(b"-250"),
            Token::Literal(b"World"),
            Token::Op(b"TJ"),
        ];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_hex_strings() {
        let content = b"<48656C6C6F> <576F726C64> TJ";
        let tokens = tokenize(content);
        let expected = vec![
            Token::Hex(b"48656C6C6F"),
            Token::Hex(b"576F726C64"),
            Token::Op(b"TJ"),
        ];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_nested_parens() {
        let content = b"(outer (nested) text) TJ";
        let tokens = tokenize(content);
        let expected = vec![Token::Literal(b"outer (nested) text"), Token::Op(b"TJ")];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_escaped_chars() {
        let content = b"(escaped \\( paren) TJ";
        let tokens = tokenize(content);
        let expected = vec![Token::Literal(b"escaped \\( paren"), Token::Op(b"TJ")];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_font_command() {
        let content = b"/F28 12.0 Tf";
        let tokens = tokenize(content);
        let expected = vec![
            Token::Name(b"F28"),
            Token::Number(b"12.0"),
            Token::Op(b"Tf"),
        ];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_td_command() {
        let content = b"100.5 -20.3 Td";
        let tokens = tokenize(content);
        let expected = vec![
            Token::Number(b"100.5"),
            Token::Number(b"-20.3"),
            Token::Op(b"Td"),
        ];

        assert_eq!(tokens, expected);
    }

    #[test]
    fn test_tokenize_complex_stream() {
        // Simulates a realistic PDF content stream with multiple operations
        let content = b"BT /F1 12.0 Tf 100 700 Td [(Hello) -250 (World)] TJ ET";
        let tokens = tokenize(content);

        // Should contain all the key tokens
        assert!(tokens.contains(&Token::Op(b"BT")));
        assert!(tokens.contains(&Token::Name(b"F1")));
        assert!(tokens.contains(&Token::Number(b"12.0")));
        assert!(tokens.contains(&Token::Op(b"Tf")));
        assert!(tokens.contains(&Token::Number(b"100")));
        assert!(tokens.contains(&Token::Number(b"700")));
        assert!(tokens.contains(&Token::Op(b"Td")));
        assert!(tokens.contains(&Token::Literal(b"Hello")));
        assert!(tokens.contains(&Token::Number(b"-250")));
        assert!(tokens.contains(&Token::Literal(b"World")));
        assert!(tokens.contains(&Token::Op(b"TJ")));
        assert!(tokens.contains(&Token::Op(b"ET")));
    }
}
