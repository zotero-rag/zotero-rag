use std::borrow::Cow;

pub(crate) fn from_cmmi(ch: u8) -> Cow<'static, str> {
    match ch {
        64 => Cow::Borrowed("\\partial"), // @
        18 => Cow::Borrowed("\\theta"),   // \022
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
