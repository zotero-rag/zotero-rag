//! A crate for working with LLMs and embedding services.
//!
//! This crate provides a set of tools for working with LLMs and embedding services. It includes
//! functionality for connecting to LLMs, embedding data, and performing vector searches. It also
//! supports tool calling for most LLM providers. The crate is built on top of LanceDB for its simplicity.

#![warn(warnings)]
#![warn(missing_docs)]
#![warn(unreachable_pub)]
#![warn(unsafe_op_in_unsafe_fn)]
#![warn(rust_2018_idioms)]

pub mod capabilities;
pub(crate) mod common;
pub mod config;
pub mod constants;
pub mod embedding;
pub mod llm;
pub mod vector;
