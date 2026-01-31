use std::borrow::Cow;

pub(crate) fn from_cmex(ch: u8) -> Cow<'static, str> {
    match ch {
        b'X' => Cow::Borrowed("\\sum"),
        b'Z' => Cow::Borrowed("\\int"),
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
