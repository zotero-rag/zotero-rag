pub fn from_msbm(ch: &u8) -> String {
    match ch {
        65..=90 | 97..=122 => format!("\\mathbb{{{}}}", char::from(*ch)),
        _ => char::from(*ch).to_string(),
    }
}
