use std::borrow::Cow;

/// Map common Computer Modern Math Italic character codes to LaTeX.
///
/// # Arguments
///
/// * `ch` - A one-byte character code in the CMMI font encoding.
///
/// # Returns
///
/// The corresponding LaTeX command, or `None` when the character can be emitted as-is.
pub(crate) fn from_cmmi(ch: u8) -> Option<Cow<'static, str>> {
    match ch {
        0 => Some(Cow::Borrowed("\\Gamma")),
        1 => Some(Cow::Borrowed("\\Delta")),
        2 => Some(Cow::Borrowed("\\Theta")),
        3 => Some(Cow::Borrowed("\\Lambda")),
        4 => Some(Cow::Borrowed("\\Xi")),
        5 => Some(Cow::Borrowed("\\Pi")),
        6 => Some(Cow::Borrowed("\\Sigma")),
        7 => Some(Cow::Borrowed("\\Upsilon")),
        8 => Some(Cow::Borrowed("\\Phi")),
        9 => Some(Cow::Borrowed("\\Psi")),
        10 => Some(Cow::Borrowed("\\Omega")),
        11 => Some(Cow::Borrowed("\\alpha")),
        12 => Some(Cow::Borrowed("\\beta")),
        13 => Some(Cow::Borrowed("\\gamma")),
        14 => Some(Cow::Borrowed("\\delta")),
        15 => Some(Cow::Borrowed("\\varepsilon")),
        16 => Some(Cow::Borrowed("\\zeta")),
        17 => Some(Cow::Borrowed("\\eta")),
        18 => Some(Cow::Borrowed("\\theta")),
        19 => Some(Cow::Borrowed("\\iota")),
        20 => Some(Cow::Borrowed("\\kappa")),
        21 => Some(Cow::Borrowed("\\lambda")),
        22 => Some(Cow::Borrowed("\\mu")),
        23 => Some(Cow::Borrowed("\\nu")),
        24 => Some(Cow::Borrowed("\\xi")),
        25 => Some(Cow::Borrowed("\\pi")),
        26 => Some(Cow::Borrowed("\\rho")),
        27 => Some(Cow::Borrowed("\\sigma")),
        28 => Some(Cow::Borrowed("\\tau")),
        29 => Some(Cow::Borrowed("\\upsilon")),
        30 => Some(Cow::Borrowed("\\phi")),
        31 => Some(Cow::Borrowed("\\chi")),
        32 => Some(Cow::Borrowed("\\psi")),
        33 => Some(Cow::Borrowed("\\omega")),
        34 => Some(Cow::Borrowed("\\epsilon")),
        35 => Some(Cow::Borrowed("\\vartheta")),
        36 => Some(Cow::Borrowed("\\varpi")),
        37 => Some(Cow::Borrowed("\\varrho")),
        38 => Some(Cow::Borrowed("\\varsigma")),
        39 => Some(Cow::Borrowed("\\varphi")),
        46 => Some(Cow::Borrowed("\\triangleright")),
        47 => Some(Cow::Borrowed("\\triangleleft")),
        b'@' => Some(Cow::Borrowed("\\partial")),
        b'[' => Some(Cow::Borrowed("\\flat")),
        b'\\' => Some(Cow::Borrowed("\\natural")),
        b']' => Some(Cow::Borrowed("\\sharp")),
        b'`' => Some(Cow::Borrowed("\\ell")),
        b'{' => Some(Cow::Borrowed("\\imath")),
        b'|' => Some(Cow::Borrowed("\\jmath")),
        b'}' => Some(Cow::Borrowed("\\wp")),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn maps_common_cmmi_symbols() {
        assert_eq!(from_cmmi(11).as_deref(), Some("\\alpha"));
        assert_eq!(from_cmmi(35).as_deref(), Some("\\vartheta"));
        assert_eq!(from_cmmi(b'@').as_deref(), Some("\\partial"));
        assert_eq!(from_cmmi(b'}').as_deref(), Some("\\wp"));
        assert_eq!(from_cmmi(b'A'), None);
    }
}
