use std::borrow::Cow;

pub(crate) fn from_cmex(ch: u8) -> Cow<'static, str> {
    match ch {
        88 => Cow::Borrowed("\\sum"), // X
        90 => Cow::Borrowed("\\int"), // Z
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
