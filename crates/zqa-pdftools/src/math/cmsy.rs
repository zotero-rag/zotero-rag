use std::borrow::Cow;

/// Map common Computer Modern Math Symbols character codes to LaTeX.
///
/// # Arguments
///
/// * `ch` - A one-byte character code in the CMSY font encoding.
///
/// # Returns
///
/// The corresponding LaTeX command, or `None` for unsupported character codes.
pub(crate) fn from_cmsy(ch: u8) -> Option<Cow<'static, str>> {
    if ch <= 64 {
        from_cmsy_low(ch)
    } else {
        from_cmsy_high(ch)
    }
}

/// Map the lower half of the CMSY encoding.
fn from_cmsy_low(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        0 => Some(Cow::Borrowed("-")),
        1 => Some(Cow::Borrowed("\\cdot")),
        2 => Some(Cow::Borrowed("\\times")),
        3 => Some(Cow::Borrowed("\\ast")),
        4 => Some(Cow::Borrowed("\\div")),
        5 => Some(Cow::Borrowed("\\diamond")),
        6 => Some(Cow::Borrowed("\\pm")),
        7 => Some(Cow::Borrowed("\\mp")),
        8 => Some(Cow::Borrowed("\\oplus")),
        9 => Some(Cow::Borrowed("\\ominus")),
        10 => Some(Cow::Borrowed("\\otimes")),
        11 => Some(Cow::Borrowed("\\oslash")),
        12 => Some(Cow::Borrowed("\\odot")),
        14 => Some(Cow::Borrowed("\\circ")),
        15 => Some(Cow::Borrowed("\\bullet")),
        16 => Some(Cow::Borrowed("\\asymp")),
        17 => Some(Cow::Borrowed("\\equiv")),
        18 => Some(Cow::Borrowed("\\subseteq")),
        19 => Some(Cow::Borrowed("\\supseteq")),
        20 => Some(Cow::Borrowed("\\leq")),
        21 => Some(Cow::Borrowed("\\geq")),
        22 => Some(Cow::Borrowed("\\preceq")),
        23 => Some(Cow::Borrowed("\\succeq")),
        24 => Some(Cow::Borrowed("\\sim")),
        25 => Some(Cow::Borrowed("\\approx")),
        26 => Some(Cow::Borrowed("\\subset")),
        27 => Some(Cow::Borrowed("\\supset")),
        28 => Some(Cow::Borrowed("\\ll")),
        29 => Some(Cow::Borrowed("\\gg")),
        30 => Some(Cow::Borrowed("\\prec")),
        31 => Some(Cow::Borrowed("\\succ")),
        32 => Some(Cow::Borrowed("\\leftarrow")),
        33 => Some(Cow::Borrowed("\\rightarrow")),
        34 => Some(Cow::Borrowed("\\uparrow")),
        35 => Some(Cow::Borrowed("\\downarrow")),
        36 => Some(Cow::Borrowed("\\leftrightarrow")),
        37 => Some(Cow::Borrowed("\\nearrow")),
        38 => Some(Cow::Borrowed("\\searrow")),
        39 => Some(Cow::Borrowed("\\simeq")),
        40 => Some(Cow::Borrowed("\\Leftarrow")),
        41 => Some(Cow::Borrowed("\\Rightarrow")),
        42 => Some(Cow::Borrowed("\\Uparrow")),
        43 => Some(Cow::Borrowed("\\Downarrow")),
        44 => Some(Cow::Borrowed("\\Leftrightarrow")),
        45 => Some(Cow::Borrowed("\\nwarrow")),
        46 => Some(Cow::Borrowed("\\swarrow")),
        47 => Some(Cow::Borrowed("\\propto")),
        48 => Some(Cow::Borrowed("\\prime")),
        49 => Some(Cow::Borrowed("\\infty")),
        50 => Some(Cow::Borrowed("\\in")),
        51 => Some(Cow::Borrowed("\\ni")),
        52 => Some(Cow::Borrowed("\\triangle")),
        53 => Some(Cow::Borrowed("\\triangledown")),
        55 => Some(Cow::Borrowed("\\mapsto")),
        56 => Some(Cow::Borrowed("\\forall")),
        57 => Some(Cow::Borrowed("\\exists")),
        58 => Some(Cow::Borrowed("\\neg")),
        59 => Some(Cow::Borrowed("\\emptyset")),
        60 => Some(Cow::Borrowed("\\Re")),
        61 => Some(Cow::Borrowed("\\Im")),
        62 => Some(Cow::Borrowed("\\top")),
        63 => Some(Cow::Borrowed("\\perp")),
        64 => Some(Cow::Borrowed("\\aleph")),
        _ => None,
    }
}

/// Map the upper half of the CMSY encoding.
fn from_cmsy_high(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        b'A'..=b'Z' => Some(Cow::Owned(format!("\\mathcal{{{}}}", char::from(ch)))),
        91 => Some(Cow::Borrowed("\\cup")),
        92 => Some(Cow::Borrowed("\\cap")),
        93 => Some(Cow::Borrowed("\\uplus")),
        94 => Some(Cow::Borrowed("\\wedge")),
        95 => Some(Cow::Borrowed("\\vee")),
        98 => Some(Cow::Borrowed("\\lfloor")),
        99 => Some(Cow::Borrowed("\\rfloor")),
        100 => Some(Cow::Borrowed("\\lceil")),
        101 => Some(Cow::Borrowed("\\rceil")),
        102 => Some(Cow::Borrowed("\\{")),
        103 => Some(Cow::Borrowed("\\}")),
        104 => Some(Cow::Borrowed("\\langle")),
        105 => Some(Cow::Borrowed("\\rangle")),
        106 => Some(Cow::Borrowed("|")),
        107 => Some(Cow::Borrowed("\\|")),
        108 => Some(Cow::Borrowed("\\updownarrow")),
        109 => Some(Cow::Borrowed("\\Updownarrow")),
        110 => Some(Cow::Borrowed("\\backslash")),
        111 => Some(Cow::Borrowed("\\wr")),
        112 => Some(Cow::Borrowed("\\sqrt")),
        113 => Some(Cow::Borrowed("\\coprod")),
        114 => Some(Cow::Borrowed("\\nabla")),
        115 => Some(Cow::Borrowed("\\int")),
        116 => Some(Cow::Borrowed("\\sqcup")),
        117 => Some(Cow::Borrowed("\\sqcap")),
        118 => Some(Cow::Borrowed("\\sqsubseteq")),
        119 => Some(Cow::Borrowed("\\sqsupseteq")),
        120 => Some(Cow::Borrowed("\\S")),
        121 => Some(Cow::Borrowed("\\dagger")),
        122 => Some(Cow::Borrowed("\\ddagger")),
        123 => Some(Cow::Borrowed("\\P")),
        124 => Some(Cow::Borrowed("\\clubsuit")),
        125 => Some(Cow::Borrowed("\\diamondsuit")),
        126 => Some(Cow::Borrowed("\\heartsuit")),
        127 => Some(Cow::Borrowed("\\spadesuit")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_cmsy_symbols() {
        assert_eq!(from_cmsy(10).as_deref(), Some("\\otimes"));
        assert_eq!(from_cmsy(32).as_deref(), Some("\\leftarrow"));
        assert_eq!(from_cmsy(50).as_deref(), Some("\\in"));
        assert_eq!(from_cmsy(b'F').as_deref(), Some("\\mathcal{F}"));
        assert_eq!(from_cmsy(114).as_deref(), Some("\\nabla"));
        assert_eq!(from_cmsy(13), None);
    }
}
