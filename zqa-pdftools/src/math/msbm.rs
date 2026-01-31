use std::borrow::Cow;

pub(crate) fn from_msbm(ch: u8) -> Cow<'static, str> {
    match ch {
        b'A'..=b'Z' | b'a'..=b'z' => Cow::Owned(format!("\\mathbb{{{}}}", char::from(ch))),
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
