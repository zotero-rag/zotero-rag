pub fn from_msbm(ch: &u8) -> String {
    match ch {
        65..=90 | 97..=122 => format!("\\mathbb{{{}}}", ch),
        _ => ch.to_string(),
    }
}
