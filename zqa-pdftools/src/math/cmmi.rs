pub(crate) fn from_cmmi(ch: u8) -> String {
    match ch {
        64 => "\\partial".to_string(), // @
        18 => "\\theta".to_string(),   // \022
        _ => char::from(ch).to_string(),
    }
}
