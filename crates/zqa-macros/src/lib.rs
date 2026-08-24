//! Macros useful for improving DX.

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
// I disagree that this is not readable.
#![allow(clippy::items_after_statements)]
// We will not run into these situations
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
// This is necessary to check our macros don't move values
#![allow(clippy::no_effect_underscore_binding)]

pub mod testing;
