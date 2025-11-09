//! A crate for working with LLMs and embedding services.
//!
//! This crate provides a set of tools for working with LLMs and embedding services. It includes
//! functionality for connecting to LLMs, embedding data, and performing vector searches. It also
//! supports tool calling for most LLM providers. The crate is built on top of LanceDB for its simplicity.

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

pub mod capabilities;
pub(crate) mod common;
pub mod config;
pub mod constants;
pub mod embedding;
pub mod llm;
pub mod vector;
