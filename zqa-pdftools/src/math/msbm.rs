use std::borrow::Cow;

pub(crate) fn from_msbm(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        b'A'..=b'Z' | b'a'..=b'z' => Some(Cow::Owned(format!("\\mathbb{{{}}}", char::from(ch)))),
        _ => None,
    }
}
