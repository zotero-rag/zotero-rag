use std::borrow::Cow;

pub(crate) fn from_cmex(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        b'X' => Some(Cow::Borrowed("\\sum")),
        b'Z' => Some(Cow::Borrowed("\\int")),
        _ => None,
    }
}
