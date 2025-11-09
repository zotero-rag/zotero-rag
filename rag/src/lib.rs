//! A crate for working with LLMs and embedding services.
//!
//! This crate provides a set of tools for working with LLMs and embedding services. It includes
//! functionality for connecting to LLMs, embedding data, and performing vector searches. It also
//! supports tool calling for most LLM providers. The crate is built on top of LanceDB for its simplicity.

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

pub mod capabilities;
pub(crate) mod common;
pub mod config;
pub mod constants;
pub mod embedding;
pub mod llm;
pub mod vector;
