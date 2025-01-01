pub fn from_cmsy(ch: &u8) -> String {
    match ch {
        49 => "\\infty".to_string(), // 1
        _ => char::from(*ch).to_string(),
    }
}
