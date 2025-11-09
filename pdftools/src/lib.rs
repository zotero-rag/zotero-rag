//! This crate provides PDF parsing functionality, and is somewhat tailored to academic PDFs. It
//! handles text and skips images and tables. It also handles commonly-used math expressions,
//! though this feature is not perfect by any means. Note also that due to kerning considerations,
//! the parsed text may contain erroneous spaces.

#![deny(unused_must_use)]
#![deny(unused_imports)]
#![deny(unused_variables)]
#![deny(dead_code)]
#![deny(unreachable_code)]
#![deny(unreachable_patterns)]
#![deny(unsafe_code)]
#![warn(warnings)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]
#![warn(clippy::all)]
#![warn(clippy::pedantic)]

pub(crate) mod math;
pub mod parse;
