pub fn from_cmex(ch: &u8) -> String {
    match ch {
        88 => "\\sum".to_string(), // X
        90 => "\\int".to_string(), // Z
        _ => char::from(*ch).to_string(),
    }
}
