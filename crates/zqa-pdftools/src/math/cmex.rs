use std::borrow::Cow;

/// Map common Computer Modern Math Extension character codes to LaTeX.
///
/// Only complete operators are mapped. Extensible delimiter fragments are deliberately omitted
/// because they do not represent standalone characters.
///
/// # Arguments
///
/// * `ch` - A one-byte character code in the CMEX font encoding.
///
/// # Returns
///
/// The corresponding LaTeX command, or `None` for delimiters and assembly fragments.
pub(crate) fn from_cmex(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        70 | 71 => Some(Cow::Borrowed("\\bigsqcup")),
        72 | 73 => Some(Cow::Borrowed("\\oint")),
        74 | 75 => Some(Cow::Borrowed("\\bigodot")),
        76 | 77 => Some(Cow::Borrowed("\\bigoplus")),
        78 | 79 => Some(Cow::Borrowed("\\bigotimes")),
        80 | 88 => Some(Cow::Borrowed("\\sum")),
        81 | 89 => Some(Cow::Borrowed("\\prod")),
        82 | 90 => Some(Cow::Borrowed("\\int")),
        83 | 91 => Some(Cow::Borrowed("\\bigcup")),
        84 | 92 => Some(Cow::Borrowed("\\bigcap")),
        85 | 93 => Some(Cow::Borrowed("\\biguplus")),
        86 | 94 => Some(Cow::Borrowed("\\bigwedge")),
        87 | 95 => Some(Cow::Borrowed("\\bigvee")),
        96 | 97 => Some(Cow::Borrowed("\\coprod")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_cmex_operators() {
        assert_eq!(from_cmex(72).as_deref(), Some("\\oint"));
        assert_eq!(from_cmex(80).as_deref(), Some("\\sum"));
        assert_eq!(from_cmex(89).as_deref(), Some("\\prod"));
        assert_eq!(from_cmex(95).as_deref(), Some("\\bigvee"));
        assert_eq!(from_cmex(48), None);
    }
}
