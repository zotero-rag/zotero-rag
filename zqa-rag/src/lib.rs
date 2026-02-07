//! A crate for working with LLMs and embedding services.
//!
//! This crate provides a set of tools for working with LLMs and embedding services. It includes
//! functionality for connecting to LLMs, embedding data, and performing vector searches. It also
//! supports tool calling for most LLM providers. The crate is built on top of LanceDB for its simplicity.

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
// Using `let _ =` is clearer than `let () =` for discarding unit values
#![allow(clippy::ignored_unit_patterns)]
// I disagree that this is not readable.
#![allow(clippy::items_after_statements)]
// We will not run into these situations
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation
)]

pub mod capabilities;
pub(crate) mod common;
pub mod config;
pub mod constants;
pub mod embedding;
pub mod llm;
pub mod vector;
