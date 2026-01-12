//! Functions and structs to handle fonts in PDFs.

use ordered_float::OrderedFloat;

use crate::math::{from_cmex, from_cmmi, from_cmsy, from_msbm};
use std::collections::HashMap;
use std::sync::LazyLock;

/// A struct to keep track of font size changes. This includes all the metadata you might need
/// about font changes. The primary purpose of this is to track sections, subsections, etc., but
/// the additional metadata here can also be used to chunk text by section.
#[derive(Debug, Clone)]
pub(crate) struct FontSizeMarker {
    /// 0-indexed page number. Initialized after the full PDF is parsed.
    pub page_number: usize,
    /// Byte index into the extracted text
    pub byte_index: usize,
    /// The font size in points
    pub font_size: f32,
    /// Font name (e.g., "CMR10", "CMBX12")
    pub font_name: String,
}

/// Use the history of font size markers to determine the font size of the body text (via the
/// mode of font sizes) and the font sizes of section levels.
///
/// # Arguments
///
/// * content_length - The total document length.
/// * tf_history - A list of font size markers in the order they appear in the document.
/// * min_size_count - The minimum number of times a font size has to appear to be considered. This
///   is useful to reduce noise, remove the effect of a large-font title, etc.
/// * max_depth - The maximum number of section levels to consider. For example, setting this to 3
///   considers sections, subsections, and subsubsections, but not paragraphs with a bold preface.
///
/// # Returns
///
/// A tuple containing:
///
/// * The body font size if a mode can be detected, otherwise a default value of 10.0.
/// * Sorted font sizes (in descending order) of sections
pub(crate) fn get_document_font_sizes(
    content_length: usize,
    tf_history: &Vec<FontSizeMarker>,
    min_size_count: usize,
    max_depth: usize,
) -> (f32, Vec<OrderedFloat<f32>>) {
    let mut counts = HashMap::<OrderedFloat<f32>, usize>::new();
    for (cur, next) in tf_history.iter().zip(tf_history.iter().skip(1)) {
        *counts.entry(OrderedFloat(cur.font_size)).or_default() +=
            next.byte_index.saturating_sub(cur.byte_index);
    }
    if let Some(last) = tf_history.last() {
        *counts.entry(OrderedFloat(last.font_size)).or_default() +=
            content_length.saturating_sub(last.byte_index);
    }

    let body_font_size = counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map_or(10.0, |f| (*f.0).into());

    let mut font_sizes: Vec<OrderedFloat<f32>> = counts
        .iter()
        .filter(|(size, _)| **size > OrderedFloat(body_font_size))
        .filter(|(_, c)| **c > min_size_count)
        .map(|f| *f.0)
        .collect();

    font_sizes.sort();
    font_sizes.reverse();

    (
        body_font_size,
        font_sizes.into_iter().take(max_depth).collect(),
    )
}

/// A type to convert from bytes in math fonts to LaTeX code
type ByteTransformFn = fn(u8) -> String;

/// A zero-allocation iterator for octal escape sequences and raw bytes. This is useful for parsing
/// octal escape codes that are used in math fonts when non-printable characters are used to
/// represent symbols.
pub(crate) struct IterCodepoints<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl Iterator for IterCodepoints<'_> {
    type Item = u8;

    fn next(&mut self) -> Option<u8> {
        if self.pos >= self.bytes.len() {
            return None;
        }
        let b = self.bytes[self.pos];
        if b == b'\\' {
            // Possible octal escape
            let rem = self.bytes.len() - self.pos;
            if rem >= 4  // Length of octal escape sequence
                && self.bytes[self.pos + 1] == b'0'
                && self.bytes[self.pos + 2].is_ascii_digit()
                && self.bytes[self.pos + 2] < b'8'
                && self.bytes[self.pos + 3].is_ascii_digit()
                && self.bytes[self.pos + 3] < b'8'
            {
                let oct = &self.bytes[self.pos + 1..=self.pos + 3];
                let code = (oct[1] - b'0') * 8 + (oct[2] - b'0');
                self.pos += 4;
                Some(code)
            } else {
                // Just a backslash or malformed escape
                self.pos += 1;
                Some(b'\\')
            }
        } else {
            self.pos += 1;
            Some(b)
        }
    }
}

pub(crate) fn font_transform(input: &str, transform: ByteTransformFn) -> String {
    IterCodepoints {
        bytes: input.as_bytes(),
        pos: 0,
    }
    .map(transform)
    .collect::<String>()
}

/// A lazy-loaded hashmap storing conversions from math fonts to LaTeX code
/// Handles most common math fonts, but does not yet support specialized math fonts.
pub(crate) static FONT_TRANSFORMS: LazyLock<HashMap<&'static str, ByteTransformFn>> =
    LazyLock::new(|| {
        let mut m: HashMap<&'static str, ByteTransformFn> = HashMap::new();

        m.insert("CMMI5", from_cmmi);
        m.insert("CMMI6", from_cmmi);
        m.insert("CMMI7", from_cmmi);
        m.insert("CMMI8", from_cmmi);
        m.insert("CMMI9", from_cmmi);
        m.insert("CMMI10", from_cmmi);
        m.insert("CMMI12", from_cmmi);

        m.insert("CMSY5", from_cmsy);
        m.insert("CMSY6", from_cmsy);
        m.insert("CMSY7", from_cmsy);
        m.insert("CMSY8", from_cmsy);
        m.insert("CMSY9", from_cmsy);
        m.insert("CMSY10", from_cmsy);

        m.insert("CMEX10", from_cmex);

        m.insert("MSBM5", from_msbm);
        m.insert("MSBM6", from_msbm);
        m.insert("MSBM7", from_msbm);
        m.insert("MSBM8", from_msbm);
        m.insert("MSBM9", from_msbm);
        m.insert("MSBM10", from_msbm);

        m
    });

/// The mappings of a ToUnicode CMap. For now, we do not support other kinds of CMaps.
pub(crate) type CMap = HashMap<String, String>;

/// The type of encoding used by a font. Either `SIMPLE` (human-readable) or `CID_KEYED`
/// (Unicode/glyph ID-encoded)
#[derive(Debug)]
pub(crate) enum FontEncoding {
    /// Human-readable, "simple" encoding. Under this encoding, each TJ block has contents that can
    /// be parsed as plain text.
    Simple,
    /// CID-keyed, or glyph ID-encoded font. For this encoding, the font usually is a subsetted
    /// embedded font (i.e., a CID-keyed subset of a font that's embedded), and we need to check
    /// the `ToUnicode` CMap for that font. It is possible to also use non-ToUnicode CMaps; we do
    /// not yet handle this case.
    ///
    /// Modern PDFs, such as those generated by `pdflatex`, use CID-keyed font subsets, with
    /// two-byte (or multi-byte, but we don't yet handle this) CIDs, custom glyph ID maps, and
    /// `ToUnicode` CMaps for real text extraction.
    CIDKeyed(CMap),
}
