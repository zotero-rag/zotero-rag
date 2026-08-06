//! Functions and structs to handle fonts in PDFs.

use std::borrow::Cow;
use std::collections::HashMap;
use std::hash::Hash;
use std::sync::LazyLock;

use lopdf::{Document, Object};
use ordered_float::OrderedFloat;

use crate::math::{from_cmex, from_cmmi, from_cmsy, from_msbm};
use crate::parse::{PageID, PdfError};

/// Size bound for CMaps, to avoid memory overflow from malformed PDFs.
const MAX_CMAP_ENTRIES: usize = 8192;

/// Fraction of a font's space width used as the `TJ` gap threshold: a negative adjustment whose
/// magnitude exceeds this fraction of the space width is emitted as a space. See the heuristics in
/// the [`DEFAULT_SPACE_WIDTH`] docs.
pub(crate) const SPACE_WIDTH_FRACTION: f32 = 0.4;

/// To detect word boundaries by kerning, we need the width of a space character in the font. If, in
/// a `TJ` array, an adjustment is negative and comparable to the width of a space, we emit one. We
/// use [`SPACE_WIDTH_FRACTION`] of the font's space width, since "how big must a gap be to read as a
/// word break" is font-relative.
///
/// Heuristically, the typical interword space widths are anywhere in 250 (Times) to 333 (Computer
/// Modern Roman 10; CMR9 is 342) to 600 (Courier), and kern magnitudes are under 150. So a threshold
/// in 0.3x - 0.5x (of the space width) should separate the two quite well.
///
/// Since `pdflatex` is more aggressive with its use of adjustment instead of space glyphs, we use
/// 333 (Computer Modern Roman) as the fallback space width in case we don't find it in the font's
/// dictionary.
pub(crate) const DEFAULT_SPACE_WIDTH: f32 = 333.0;

/// A struct to keep track of font size changes. This includes all the metadata you might need
/// about font changes. The primary purpose of this is to track sections, subsections, etc., but
/// the additional metadata here can also be used to chunk text by section.
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(crate) struct FontSizeMarker {
    /// 0-indexed page number. Initialized after the full PDF is parsed.
    pub page_number: usize,
    /// Byte index into the extracted text
    pub byte_index: usize,
    /// The font size in points
    pub font_size: OrderedFloat<f32>,
    /// Font name (e.g., "CMR10", "CMBX12")
    pub font_name: String,
}

/// A type to convert from bytes in math fonts to LaTeX code
type ByteTransformFn = fn(u8) -> std::borrow::Cow<'static, str>;

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
#[derive(Clone, Debug)]
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
    CIDKeyed {
        /// The char-code-to-Unicode mappings parsed from the `ToUnicode` CMap.
        mappings: CMap,
        /// The char code that maps to a space (U+0020), if the CMap defines one.
        space_cid: Option<i64>,
    },
    /// A CID-keyed font with no `ToUnicode` CMap. Text in such fonts cannot be reliably
    /// extracted, so instead of erroring out, the parser skips text set in these fonts and
    /// records how much was skipped.
    Unmappable,
}

/// Given a PDF `Document` reference, a page ID, and a font key (e.g., "F19"), return the font
/// object.
///
/// # Arguments
///
/// * `doc` - The `lopdf::Document` object.
/// * `page_id` - The `lopdf` page ID. Different pages can have different font dictionaries,
///   otherwise operations such as joining PDFs would be more complicated than they already are.
/// * `font_key` - The font key as used in the PDF content stream.
///
/// # Returns
///
/// A `HashMap` with string keys mapping to `lopdf::Object` references.
///
/// # Errors
///
/// * `PdfError::PageFontError` if getting page fonts failed.
/// * `PdfError::FontNotFound` if the font key does not exist.
///
/// # Panics
///
/// * If any of the keys in the font dictionary are not valid UTF-8.
pub(crate) fn get_font<'a>(
    doc: &'a Document,
    page_id: PageID,
    font_key: &str,
) -> Result<HashMap<&'a str, &'a Object>, PdfError> {
    // Get the font dictionary for the page
    let fonts = doc
        .get_page_fonts(page_id)
        .map_err(|_| PdfError::PageFontError)?;

    let font_obj = fonts
        .get(font_key.as_bytes())
        .ok_or(PdfError::FontNotFound(font_key.into()))?;
    let font_hash = font_obj.as_hashmap();

    font_hash
        .iter()
        .map(|(k, v)| {
            let key_str = std::str::from_utf8(k)?;
            Ok((key_str, v))
        })
        .collect::<Result<_, _>>()
}

/// Parse a single `beginbfchar`..`endbfchar` block, assuming the enclosing keywords have been
/// stripped out.
///
/// # Arguments
///
/// * `csrange` - String slice of the block, excluding the `beginbfchar` and `endbfchar` markers.
/// * `font_key` - The key for the current font being parsed, used for logging.
/// * `cmap` - Mutable reference to a map where the parsed mappings will be inserted.
///
/// # Errors
///
/// * `PdfError::EncodingError` in the following cases:
///     * The range contains invalid UTF-8 characters.
///     * A mapping contains invalid UTF-16 code units.
///     * The CMap exceeds the maximum allowed entries (8192)
fn parse_bfchar_block(
    csrange: &str,
    font_key: &str,
    cmap: &mut HashMap<String, String>,
    space_cid: &mut Option<i64>,
) -> Result<(), PdfError> {
    let lines = csrange.split('\n');

    // Within the CMap, each line has the following form:
    //   <001B> <0041>
    // In this case, the 2-byte CID 001B maps to U+0041.
    for line in lines {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 2 {
            log::warn!(
                "In CMap for {font_key}, unexpected line with {} tokens.",
                parts.len()
            );
            continue;
        }

        let cid = parts[0].trim_matches(|c| c == '<' || c == '>');

        // In some cases, `parts[1]` can have multiple Unicode code points. This is
        // sometimes done to handle ligatures, among others. For example, the ligature
        // `ff` is represented as two U+066 code points.
        let code_units: Vec<u16> = parts[1]
            .trim_matches(|c| c == '<' || c == '>')
            .as_bytes()
            .chunks_exact(4)
            .map(|chunk| u16::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16))
            .collect::<Result<_, _>>()
            .map_err(|_| PdfError::InvalidUtf8)?;

        let unicode: String = char::decode_utf16(code_units)
            .map(|r| r.map_err(|e| PdfError::EncodingError(format!("Invalid UTF-16: {e}"))))
            .collect::<Result<_, _>>()?;

        if unicode == " " {
            *space_cid = i64::from_str_radix(cid, 16).ok().or(*space_cid);
        }
        cmap.insert(cid.to_string().to_lowercase(), unicode);
        if cmap.len() > MAX_CMAP_ENTRIES {
            return Err(PdfError::EncodingError(format!(
                "CMap for {font_key} exceeds maximum allowed entries."
            )));
        }
    }

    Ok(())
}

/// Parse a single `beginbfrange`..`endbfrange` block, assuming the enclosing keywords have been
/// stripped out.
///
/// # Arguments
///
/// * `csrange` - String slice of the block, excluding the `beginbfrange` and `endbfrange` markers.
/// * `font_key` - The key for the current font being parsed, used for logging.
/// * `cmap` - Mutable reference to a map where the parsed mappings will be inserted.
///
/// # Errors
///
/// * `PdfError::EncodingError` in the following cases:
///     * The range contains arrays (currently unsupported).
///     * The range contains an invalid UTF-16 CID.
///     * The range contains invalid UTF-8 characters.
///     * Any of the specified ranges contains an end index that overflows, disallowed by the spec.
///     * The CMap contains more than 8192 entries (heuristically chosen).
fn parse_bfrange_block(
    csrange: &str,
    font_key: &str,
    cmap: &mut HashMap<String, String>,
    space_cid: &mut Option<i64>,
) -> Result<(), PdfError> {
    let lines = csrange.split('\n');

    // Within the CMap, each line has the following form:
    //   <000B> <000C> <0028>
    // In this case, we have:
    //   * CID 000B -> U+0028
    //   * CID 000C -> U+0029
    for line in lines {
        if line.contains('[') {
            log::error!("CMaps with `bfrange`s that contain arrays are not yet supported.");
            return Err(PdfError::EncodingError(
                "Unsupported CID map with arrays in range.".into(),
            ));
        }

        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() != 3 {
            log::warn!(
                "In CMap for {font_key}, unexpected line with {} tokens.",
                parts.len()
            );
            continue;
        }

        let start_cid = parts[0].trim_matches(|c| c == '<' || c == '>');
        let end_cid = parts[1].trim_matches(|c| c == '<' || c == '>');

        let start_cid_u16 = u16::from_str_radix(start_cid, 16).map_err(|_| {
            PdfError::EncodingError(format!(
                "In CMap for {font_key}, CID {start_cid} was not valid UTF-16"
            ))
        })?;
        let end_cid_u16 = u16::from_str_radix(end_cid, 16).map_err(|_| {
            PdfError::EncodingError(format!(
                "In CMap for {font_key}, CID {end_cid} was not valid UTF-16"
            ))
        })?;

        // Again, parts[2] can have multiple UTF-16BE code units. In this case, for each
        // consecutive code in the source code range, we increment the last byte of the
        // string, see the PDF standard, ISO 32000-1:2008, §9.10.3 ("ToUnicode CMaps").
        // This means that we don't treat the hex string as one Big Endian integer, and only
        // look at the last byte for the purposes of the range.
        let mut code_units: Vec<u16> = parts[2]
            .trim_matches(|c| c == '<' || c == '>')
            .as_bytes()
            .chunks_exact(4)
            .map(|chunk| u16::from_str_radix(std::str::from_utf8(chunk).unwrap(), 16))
            .collect::<Result<_, _>>()
            .map_err(|_| PdfError::InvalidUtf8)?;

        let original_last = code_units.last().copied();
        for i in start_cid_u16..=end_cid_u16 {
            if let (Some(c), Some(orig)) = (code_units.last_mut(), original_last) {
                // Technically, this is supposed to be *byte* addition, so it's not
                // to-spec, but this is good enough.
                *c = orig.checked_add(i - start_cid_u16).ok_or_else(|| {
                    PdfError::EncodingError(format!(
                        "In CMap for {font_key}, ranged destination overflowed u16"
                    ))
                })?;
            }
            let unicode: String = char::decode_utf16(code_units.iter().copied())
                .map(|r| r.map_err(|e| PdfError::EncodingError(format!("Invalid UTF-16: {e}"))))
                .collect::<Result<_, _>>()?;
            if unicode == " " {
                *space_cid = Some(i64::from(i));
            }
            cmap.insert(format!("{i:04x}"), unicode);

            if cmap.len() > MAX_CMAP_ENTRIES {
                return Err(PdfError::EncodingError(format!(
                    "CMap for {font_key} exceeds maximum allowed entries."
                )));
            }
        }
    }

    Ok(())
}

/// Parse a CMap string. A font key is taken as reference to provide more context in errors.
///
/// Currently, this CMap parser supports multiple `beginbfchar`..`endbfchar` and
/// `beginbfrange`..`endbfrange`s, but does not support more complex use cases such as those
/// involving `begincidrange`..`endcidrange`. Note that per ISO 32000-1:2008, §9.7.5.4(e),
/// `beginrearrangedfont`..`endrearrangedfont` should not be used in embedded CMaps; moreover,
/// `usefont`s should only specify a font number of 0.
///
/// # Arguments
///
/// * `cmap` - The entirety of the CMap to parse.
/// * `font_key` - The key for the current font being parsed, used for logging.
///
/// # Returns
///
/// A tuple of a [`HashMap<String, String>`] mapping CIDs to readable characters, and the char
/// code of the space character (U+0020), if the CMap defines one.
///
/// # Errors
///
/// * `PdfError::EncodingError` in the following cases:
///     * A block ends unexpectedly on "beginbf".
///     * A mapping in a `bfchar` block contains invalid UTF-16 code units.
///     * A `bfrange` block contains arrays (currently unsupported).
///     * A `bfrange` block contains an invalid UTF-16 CID.
///     * The range contains invalid UTF-8 characters.
///     * Any of the specified ranges in a `bfrange` block contains an end index that overflows,
///       disallowed by the spec.
///     * The CMap contains more than 8192 entries (heuristically chosen).
pub(crate) fn parse_cmap(
    cmap: &str,
    font_key: &str,
) -> Result<(HashMap<String, String>, Option<i64>), PdfError> {
    let mut mappings = HashMap::new();
    let mut space_cid = None;
    let bytes = cmap.as_bytes();

    let mut cmap_pos = 0;
    // The spec limits each mapping set to 100 lines, and requires any font needing more than that
    // many entries to split them into batches of at most 100, see Adobe Technical Notes #5014,
    // "Adobe CMap and CIDFont Files Specification", §7.4 ("Operator Details").
    while let Some(bf_pos) = cmap[cmap_pos..].find("beginbf") {
        let offset = "beginbf".len();
        let Some(&discriminant) = bytes.get(cmap_pos + bf_pos + offset) else {
            return Err(PdfError::EncodingError(format!(
                "CMap type for {font_key} could not be determined."
            )));
        };

        if discriminant == b'c' {
            // `beginbfchar`..`endbfchar` section
            let csrange_begin = cmap_pos + bf_pos + offset + "char".len();
            let csrange_end = csrange_begin
                + cmap[csrange_begin..]
                    .find("endbfchar")
                    .ok_or(PdfError::EncodingError(format!(
                        "Deflated ToUnicode CMap for font {font_key} has no `endbfchar`."
                    )))?;
            let csrange = cmap[csrange_begin..csrange_end].trim();
            parse_bfchar_block(csrange, font_key, &mut mappings, &mut space_cid)?;

            cmap_pos = csrange_end + "endbfchar".len();
        } else if discriminant == b'r' {
            // `beginbfrange`..`endbfrange` section
            let csrange_begin = cmap_pos + bf_pos + offset + "range".len();
            let csrange_end = csrange_begin
                + cmap[csrange_begin..]
                    .find("endbfrange")
                    .ok_or(PdfError::EncodingError(format!(
                        "Deflated ToUnicode CMap for font {font_key} has no `endbfrange`."
                    )))?;
            let csrange = cmap[csrange_begin..csrange_end].trim();
            parse_bfrange_block(csrange, font_key, &mut mappings, &mut space_cid)?;

            cmap_pos = csrange_end + "endbfrange".len();
        } else {
            return Err(PdfError::EncodingError(format!(
                "CMap type for {font_key} could not be determined."
            )));
        }
    }

    if mappings.is_empty() {
        return Err(PdfError::EncodingError(format!(
            "Font {font_key} contains an unrecognized CMap kind."
        )));
    }

    Ok((mappings, space_cid))
}

/// Attempt to get the font encoding for a font key on a specific page.
///
/// This function uses heuristics to determine whether a font is likely to be "simple" (i.e.,
/// can be parsed as plain-text), or needs to be translated via a corresponding CMap.
///
/// # Returns
///
/// If the specified font is a CID-keyed font, returns `FontEncoding::CIDKeyed` with a
/// reference to the `ToUnicode` CMap. Otherwise, returns `FontEncoding::Simple`.
///
/// # Errors
///
/// * `PdfError::PageFontError` if getting page fonts failed.
/// * `PdfError::FontNotFound` if the font key does not exist.
/// * `PdfError::EncodingError` if the font dictionary:
///      * Has an /Encoding that could not be read
///      * Has a ToUnicode CMap that could not be read or deflated.
///      * Has a ToUnicode CMap without a `beginbfchar`, `endbfchar`, `beginbfrange`, or
///        `endbfrange` marker.
/// * `PdfError::InternalError` if the font dictionary:
///      * Does not have a /Subtype that could be read.
///      * Has a /ToUnicode reference that is invalid.
/// * `PdfError::InvalidUtf8` if the deflated ToUnicode CMap is not valid UTF-8.
///
/// # Panics
///
/// * If any of the keys in the font dictionary are not valid UTF-8.
pub(crate) fn compute_font_encoding(
    doc: &Document,
    font_obj: &HashMap<&str, &Object>,
    font_key: &str,
) -> Result<FontEncoding, PdfError> {
    // Test 1: if the font has:
    //   /Subtype /Type1
    //   /Subtype /TrueType
    // then it is likely a "simple" font.
    let font_subtype = str::from_utf8(
        font_obj
            .get("Subtype")
            .ok_or(PdfError::MissingSubtype)?
            .as_name()
            .map_err(|_| {
                PdfError::InternalError(format!(
                    "Expected font {font_key}'s Subtype key to be a `name`"
                ))
            })?,
    )
    .map_err(|e| {
        PdfError::InternalError(format!(
            "Font {font_key}'s Subtype value is not valid UTF-8: {e}"
        ))
    })?;

    if ["Type1", "TrueType"].contains(&font_subtype) {
        return Ok(FontEncoding::Simple);
    }

    // Test 2: if the font has:
    //   /Encoding /Identity-H
    //   /Encoding /Identity-V
    // then it is likely a CID-keyed font. However, if it has:
    //   /Encoding /WinAnsiEncoding
    //   /Encoding /MacRomanEncoding
    // it is likely a "simple" font.
    let Some(font_encoding) = font_obj.get("Encoding") else {
        log::warn!(
            "Could not determine font type for {font_key}, assuming simple. This may be wrong."
        );
        return Ok(FontEncoding::Simple);
    };
    let font_encoding = str::from_utf8(font_encoding.as_name().map_err(|_| {
        PdfError::EncodingError(format!(
            "Expected font {font_key}'s Encoding key to be a `name`"
        ))
    })?)
    .map_err(|e| {
        PdfError::EncodingError(format!(
            "Font {font_key}'s Encoding name is not valid UTF-8: {e}"
        ))
    })?;

    if ["WinAnsiEncoding", "MacRomanEncoding"].contains(&font_encoding) {
        Ok(FontEncoding::Simple)
    } else if ["Identity-H", "Identity-V"].contains(&font_encoding) || font_encoding == "Type0" {
        // Test 3: if the font has:
        //   /Subtype /Type0
        // then it is likely a CID-keyed font.
        let Some(to_unicode) = font_obj.get("ToUnicode") else {
            log::debug!(
                "CID-keyed font {font_key} does not have a ToUnicode CMap; text using it will be skipped."
            );
            return Ok(FontEncoding::Unmappable);
        };

        let cmap_ref = to_unicode.as_reference().map_err(|_| {
            PdfError::EncodingError(format!(
                "CID-keyed font {font_key}'s ToUnicode CMap could not be read."
            ))
        })?;

        let decompressed = doc
            .get_object(cmap_ref)
            .map_err(|_| {
                PdfError::InternalError(format!(
                    "Font {font_key}'s ToUnicode CMap points to invalid reference.",
                ))
            })?
            .as_stream()
            .map_err(|_| {
                PdfError::EncodingError(format!(
                    "CID-keyed font {font_key}'s ToUnicode CMap could not be read."
                ))
            })?
            .decompressed_content()
            .map_err(|_| {
                PdfError::EncodingError(format!(
                    "CID-keyed font {font_key}'s ToUnicode CMap could not be deflated."
                ))
            })?;

        let cmap = String::from_utf8(decompressed).map_err(|_| PdfError::InvalidUtf8)?;
        let (mappings, space_cid) = parse_cmap(&cmap, font_key)?;

        Ok(FontEncoding::CIDKeyed {
            mappings,
            space_cid,
        })
    } else {
        // If we got here, then we don't have a good idea; emit a warning and assume Simple.
        log::warn!(
            "No heuristic matched for font {font_key}; assuming simple. This is likely wrong."
        );
        Ok(FontEncoding::Simple)
    }
}

fn expand_cidfont_w(widths: &Object) -> Result<HashMap<i64, f32>, PdfError> {
    let entries = widths
        .as_array()
        .map_err(|e| PdfError::EncodingError(format!("/W was not an array: {e}")))?;

    // Entries of the /W array are in one of two formats:
    // c [w1 w2 ... wn]
    // c_first c_last w
    let mut map = HashMap::new();

    let mut i = 0;
    while i < entries.len() {
        let Ok(cid) = entries[i].as_i64() else {
            return Err(PdfError::EncodingError(format!(
                "Found value {:?} in /W, but expected an integer CID",
                entries[i]
            )));
        };

        match entries.get(i + 1) {
            Some(Object::Array(ws)) => {
                for (j, w) in ws.iter().filter_map(|w| w.as_float().ok()).enumerate() {
                    map.insert(cid + j as i64, w);
                }
                i += 2;
            }
            Some(Object::Integer(end_cid)) => {
                let Some(w) = entries.get(i + 2).and_then(|o| o.as_float().ok()) else {
                    return Err(PdfError::EncodingError(format!(
                        "Found value {:?} in /W, but expected a width",
                        entries.get(i + 2)
                    )));
                };
                for cid in cid..=*end_cid {
                    map.insert(cid, w);
                }
                i += 3;
            }
            _ => i += 1,
        }
    }

    Ok(map)
}

/// Get the width of a space for a given font in text space units.
///
/// Text spacing operators specify widths in text-space units, so this is the unit most useful for
/// parsing purposes. This is particularly useful for computing word boundaries when the PDF
/// creation program uses kerning adjustments instead of emitting a space glyph. For example,
/// Computer Modern has aggressive kern pairs (e.g., "To", "Ta", "AV"), and `pdflatex` emits them as
/// positive `TJ` adjustments (see ISO 32000-2:2020 §9.4.4, "Text space details").
///
/// # Arguments
pub(crate) fn get_space_width(
    doc: &Document,
    font_dict: &HashMap<&str, &Object>,
    font_key: &str,
    font_encoding: Option<&FontEncoding>,
) -> Result<f32, PdfError> {
    let font_type = font_encoding.map_or_else(
        || compute_font_encoding(doc, font_dict, font_key).map(Cow::Owned),
        |encoding| Ok(Cow::Borrowed(encoding)),
    )?;
    let first_char = font_dict
        .get("FirstChar")
        .and_then(|f| f.as_i64().ok())
        .unwrap_or(0_i64);

    match *font_type {
        #[allow(clippy::cast_possible_truncation)]
        #[allow(clippy::cast_sign_loss)]
        FontEncoding::Simple => Ok(font_dict
            .get("Widths")
            .and_then(|w| w.as_array().ok())
            .and_then(|ws| ws.get((32 - first_char) as usize))
            .and_then(|w| w.as_float().ok())
            .unwrap_or(DEFAULT_SPACE_WIDTH)),
        FontEncoding::CIDKeyed { space_cid, .. } => {
            let widths = font_dict.get("W").and_then(|w| {
                expand_cidfont_w(w)
                    .map_err(|e| {
                        log::warn!("Font {font_key}: {e}; falling back to /DW.");
                        e
                    })
                    .ok()
            });

            Ok(widths
                .and_then(|map| space_cid.and_then(|cid| map.get(&cid).copied()))
                .unwrap_or(
                    font_dict
                        .get("DW")
                        .and_then(|v| v.as_float().ok())
                        .unwrap_or(DEFAULT_SPACE_WIDTH),
                ))
        }
        FontEncoding::Unmappable => Ok(DEFAULT_SPACE_WIDTH),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zqa_macros::test_eq;

    #[test]
    fn test_parse_cmap_ranged() {
        // A range that walks from CID 000B to 000C, mapping the last code unit upward from
        // U+0028 ('('). So 000B -> '(' and 000C -> ')'.
        let cmap = "\
1 beginbfrange
<000b> <000c> <0028>
endbfrange";

        let (mappings, _) = parse_cmap(cmap, "F1").expect("ranged CMap should parse");

        assert_eq!(mappings.len(), 2);
        assert_eq!(mappings.get("000b"), Some(&"(".to_string()));
        assert_eq!(mappings.get("000c"), Some(&")".to_string()));
    }

    #[test]
    fn test_parse_cmap_ranged_multi_entry() {
        // Two ranges in a single block, with the start CID and Unicode code points not aligned.
        // The first range maps 0041..=0043 onto 'a'..='c'; the second maps a single CID 0050 to 'Z'.
        let cmap = "\
2 beginbfrange
<0041> <0043> <0061>
<0050> <0050> <005a>
endbfrange";

        let (mappings, _) = parse_cmap(cmap, "F2").expect("ranged CMap should parse");

        assert_eq!(mappings.len(), 4);
        assert_eq!(mappings.get("0041"), Some(&"a".to_string()));
        assert_eq!(mappings.get("0042"), Some(&"b".to_string()));
        assert_eq!(mappings.get("0043"), Some(&"c".to_string()));
        assert_eq!(mappings.get("0050"), Some(&"Z".to_string()));
    }

    #[test]
    fn test_parse_cmap_ranged_rejects_arrays() {
        // Array destinations in a `bfrange` are not yet supported and must surface an error
        // rather than silently dropping the line.
        let cmap = "\
1 beginbfrange
<0001> <0003> [<0041> <0042> <0043>]
endbfrange";

        let err = parse_cmap(cmap, "F3").expect_err("array ranges are unsupported");
        assert!(matches!(err, PdfError::EncodingError(_)));
    }

    #[test]
    fn test_parse_cmap_multiple_bfchar_blocks() {
        // The spec caps each block at 100 entries, so large CMaps split into multiple blocks;
        // mappings from every block must be collected, not just the first.
        let cmap = "\
1 beginbfchar
<0001> <0041>
endbfchar
1 beginbfchar
<0002> <0042>
endbfchar";

        let (mappings, _) = parse_cmap(cmap, "F5").expect("multi-block bfchar CMap should parse");

        test_eq!(mappings.len(), 2);
        test_eq!(mappings.get("0001"), Some(&"A".to_string()));
        test_eq!(mappings.get("0002"), Some(&"B".to_string()));
    }

    #[test]
    fn test_parse_cmap_multiple_bfrange_blocks() {
        // Same multi-block case for ranged mappings: entries from the second block must not be
        // dropped.
        let cmap = "\
1 beginbfrange
<0001> <0002> <0041>
endbfrange
1 beginbfrange
<0003> <0003> <005a>
endbfrange";

        let (mappings, _) = parse_cmap(cmap, "F6").expect("multi-block bfrange CMap should parse");

        test_eq!(mappings.len(), 3);
        test_eq!(mappings.get("0001"), Some(&"A".to_string()));
        test_eq!(mappings.get("0002"), Some(&"B".to_string()));
        test_eq!(mappings.get("0003"), Some(&"Z".to_string()));
    }

    #[test]
    fn test_parse_cmap_mixed_bfchar_and_bfrange_blocks() {
        // A CMap may contain both bfchar and bfrange blocks; mappings from both kinds must be
        // collected.
        let cmap = "\
1 beginbfchar
<0001> <0041>
endbfchar
1 beginbfrange
<0002> <0003> <0042>
endbfrange";

        let (mappings, _) = parse_cmap(cmap, "F7").expect("mixed-kind CMap should parse");

        test_eq!(mappings.len(), 3);
        test_eq!(mappings.get("0001"), Some(&"A".to_string()));
        test_eq!(mappings.get("0002"), Some(&"B".to_string()));
        test_eq!(mappings.get("0003"), Some(&"C".to_string()));
    }

    #[test]
    fn test_parse_cmap_ranged_missing_end_marker() {
        // A `beginbfrange` without its matching `endbfrange` is an error.
        let cmap = "\
1 beginbfrange
<000b> <000c> <0028>";

        let err = parse_cmap(cmap, "F4").expect_err("missing endbfrange should error");
        assert!(matches!(err, PdfError::EncodingError(_)));
    }

    #[test]
    fn test_parse_cmap_records_space_cid() {
        // The space char code must be captured from both bfchar and bfrange blocks. Here the
        // bfrange maps 0001..=0002 onto U+0020..U+0021, so 0001 is the space.
        let cmap = "\
1 beginbfchar
<0003> <0020>
endbfchar";
        let (_, space_cid) = parse_cmap(cmap, "F8").expect("bfchar CMap should parse");
        test_eq!(space_cid, Some(3));

        let cmap = "\
1 beginbfrange
<0001> <0002> <0020>
endbfrange";
        let (_, space_cid) = parse_cmap(cmap, "F9").expect("bfrange CMap should parse");
        test_eq!(space_cid, Some(1));

        // No space mapping at all means no space CID.
        let cmap = "\
1 beginbfchar
<0001> <0041>
endbfchar";
        let (_, space_cid) = parse_cmap(cmap, "F10").expect("CMap without space should parse");
        test_eq!(space_cid, None);
    }

    #[test]
    fn test_expand_cidfont_w_array_form() {
        // Format `c [w1 w2 ... wn]`: consecutive CIDs starting at `c` take the listed widths.
        let widths = Object::Array(vec![
            Object::Integer(1),
            Object::Array(vec![
                Object::Integer(500),
                Object::Real(600.5),
                Object::Integer(250),
            ]),
        ]);

        let map = expand_cidfont_w(&widths).expect("array-form /W should expand");

        test_eq!(map.len(), 3);
        test_eq!(map.get(&1), Some(&500.0));
        test_eq!(map.get(&2), Some(&600.5));
        test_eq!(map.get(&3), Some(&250.0));
    }

    #[test]
    fn test_expand_cidfont_w_range_form_inclusive() {
        // Format `c_first c_last w`: the range includes `c_last` (ISO 32000-2:2020, 9.7.4.3).
        let widths = Object::Array(vec![
            Object::Integer(3),
            Object::Integer(5),
            Object::Integer(250),
        ]);

        let map = expand_cidfont_w(&widths).expect("range-form /W should expand");

        test_eq!(map.len(), 3);
        test_eq!(map.get(&3), Some(&250.0));
        test_eq!(map.get(&4), Some(&250.0));
        test_eq!(map.get(&5), Some(&250.0));
        assert!(!map.contains_key(&6));
    }

    #[test]
    fn test_expand_cidfont_w_mixed_forms() {
        // A /W array usually mixes both formats. CID 0 is a legitimate start CID and must not be
        // dropped, and the parser must resynchronize at each entry boundary.
        let widths = Object::Array(vec![
            Object::Integer(0),
            Object::Array(vec![Object::Integer(100), Object::Integer(200)]),
            Object::Integer(3),
            Object::Integer(5),
            Object::Real(250.0),
        ]);

        let map = expand_cidfont_w(&widths).expect("mixed-form /W should expand");

        test_eq!(map.len(), 5);
        test_eq!(map.get(&0), Some(&100.0));
        test_eq!(map.get(&1), Some(&200.0));
        test_eq!(map.get(&3), Some(&250.0));
        test_eq!(map.get(&4), Some(&250.0));
        test_eq!(map.get(&5), Some(&250.0));
        assert!(!map.contains_key(&2));
    }

    #[test]
    fn test_expand_cidfont_w_rejects_non_array() {
        let err = expand_cidfont_w(&Object::Integer(42)).expect_err("non-array /W should error");
        assert!(matches!(err, PdfError::EncodingError(_)));
    }

    #[test]
    fn test_expand_cidfont_w_rejects_non_integer_head() {
        // Entry heads must be integer CIDs; anything else is malformed and must error rather than
        // desynchronize the parse.
        let widths = Object::Array(vec![Object::Name(b"Foo".to_vec())]);

        let err = expand_cidfont_w(&widths).expect_err("non-integer CID head should error");
        assert!(matches!(err, PdfError::EncodingError(_)));
    }
}
