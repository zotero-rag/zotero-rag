pub fn from_cmmi(ch: &u8) -> String {
    match ch {
        65..=90 | 97..=122 => ch.to_string(),
        _ => ch.to_string(),
    }
}
