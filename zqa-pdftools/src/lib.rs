//! This crate provides PDF parsing functionality, and is somewhat tailored to academic PDFs. It
//! handles text and skips images and tables. It also handles commonly-used math expressions,
//! though this feature is not perfect by any means. Note also that due to kerning considerations,
//! the parsed text may contain erroneous spaces.

#![deny(
    unreachable_code,
    unreachable_patterns,
    unsafe_code,
    unused_imports,
    unused_must_use,
    unused_variables
)]
#![warn(
    clippy::all,
    clippy::pedantic,
    missing_docs,
    rust_2018_idioms,
    unreachable_pub,
    unsafe_op_in_unsafe_fn
)]
// These are obnoxious and often wrong anyway
#![allow(clippy::doc_markdown)]
// I disagree with this rule--what would users do knowing panic situations, if it's going to panic
// anyway?
#![allow(clippy::missing_panics_doc)]
// I disagree that this is not readable.
#![allow(clippy::items_after_statements)]
// We will not run into these situations
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

pub mod chunk;
pub(crate) mod edits;
pub(crate) mod fonts;
pub(crate) mod math;
pub mod parse;
pub(crate) mod tokenizer;
