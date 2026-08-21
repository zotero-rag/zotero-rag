use std::borrow::Cow;

pub(crate) fn from_cmmi(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        b'@' => Some(Cow::Borrowed("\\partial")),
        18 => Some(Cow::Borrowed("\\theta")), // \022 (non-printable)
        _ => None,
    }
}
