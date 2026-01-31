use std::borrow::Cow;

pub(crate) fn from_cmsy(ch: u8) -> Cow<'static, str> {
    match ch {
        10 => Cow::Borrowed("\\otimes"), // \\012
        49 => Cow::Borrowed("\\infty"),  // 1
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
