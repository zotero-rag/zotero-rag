//! This crate provides PDF parsing functionality, and is somewhat tailored to academic PDFs. It
//! handles text and skips images and tables. It also handles commonly-used math expressions,
//! though this feature is not perfect by any means. Note also that due to kerning considerations,
//! the parsed text may contain erroneous spaces.

#![deny(
    dead_code,
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

pub(crate) mod math;
pub mod parse;
