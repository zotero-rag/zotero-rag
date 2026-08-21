use std::borrow::Cow;

/// Map common AMS Blackboard Bold character codes to LaTeX.
///
/// # Arguments
///
/// * `ch` - A one-byte character code in the MSBM font encoding.
///
/// # Returns
///
/// The corresponding LaTeX command, or `None` for unsupported character codes.
pub(crate) fn from_msbm(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        0 => Some(Cow::Borrowed("\\lneq")),
        1 => Some(Cow::Borrowed("\\gneq")),
        2 => Some(Cow::Borrowed("\\nleq")),
        3 => Some(Cow::Borrowed("\\ngeq")),
        4 => Some(Cow::Borrowed("\\nless")),
        5 => Some(Cow::Borrowed("\\ngtr")),
        6 => Some(Cow::Borrowed("\\nprec")),
        7 => Some(Cow::Borrowed("\\nsucc")),
        12 => Some(Cow::Borrowed("\\lneqq")),
        13 => Some(Cow::Borrowed("\\gneqq")),
        14 => Some(Cow::Borrowed("\\npreceq")),
        15 => Some(Cow::Borrowed("\\nsucceq")),
        28 => Some(Cow::Borrowed("\\nsim")),
        29 => Some(Cow::Borrowed("\\ncong")),
        32 => Some(Cow::Borrowed("\\nsubseteq")),
        33 => Some(Cow::Borrowed("\\nsupseteq")),
        44 => Some(Cow::Borrowed("\\nparallel")),
        63 => Some(Cow::Borrowed("\\varnothing")),
        64 => Some(Cow::Borrowed("\\nexists")),
        b'A'..=b'Z' => Some(Cow::Owned(format!("\\mathbb{{{}}}", char::from(ch)))),
        96 => Some(Cow::Borrowed("\\Finv")),
        97 => Some(Cow::Borrowed("\\Game")),
        102 => Some(Cow::Borrowed("\\mho")),
        103 => Some(Cow::Borrowed("\\eth")),
        105 => Some(Cow::Borrowed("\\beth")),
        106 => Some(Cow::Borrowed("\\gimel")),
        107 => Some(Cow::Borrowed("\\daleth")),
        108 => Some(Cow::Borrowed("\\lessdot")),
        109 => Some(Cow::Borrowed("\\gtrdot")),
        112 => Some(Cow::Borrowed("\\shortmid")),
        113 => Some(Cow::Borrowed("\\shortparallel")),
        122 => Some(Cow::Borrowed("\\digamma")),
        123 => Some(Cow::Borrowed("\\varkappa")),
        125 | 126 => Some(Cow::Borrowed("\\hbar")),
        127 => Some(Cow::Borrowed("\\backepsilon")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_msbm_symbols() {
        assert_eq!(from_msbm(2).as_deref(), Some("\\nleq"));
        assert_eq!(from_msbm(44).as_deref(), Some("\\nparallel"));
        assert_eq!(from_msbm(b'R').as_deref(), Some("\\mathbb{R}"));
        assert_eq!(from_msbm(105).as_deref(), Some("\\beth"));
        assert_eq!(from_msbm(b'b'), None);
    }
}
