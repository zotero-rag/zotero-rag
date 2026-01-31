use std::borrow::Cow;

pub(crate) fn from_msbm(ch: u8) -> Cow<'static, str> {
    match ch {
        65..=90 | 97..=122 => Cow::Owned(format!("\\mathbb{{{}}}", char::from(ch))),
        _ => Cow::Owned(char::from(ch).to_string()),
    }
}
