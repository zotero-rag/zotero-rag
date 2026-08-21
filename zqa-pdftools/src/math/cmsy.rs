use std::borrow::Cow;

pub(crate) fn from_cmsy(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        10 => Some(Cow::Borrowed("\\otimes")), // \\012 (newline/LF)
        b'1' => Some(Cow::Borrowed("\\infty")),
        _ => None,
    }
}
