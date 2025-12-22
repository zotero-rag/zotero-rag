#![deny(
    dead_code,
    unreachable_code,
    unreachable_patterns,
    unsafe_code,
    unused_must_use,
    unused_variables
)]
#![warn(clippy::all, clippy::pedantic)]
#![allow(clippy::doc_markdown)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]

pub mod cli;
pub mod common;
pub mod config;
pub mod state;
pub mod utils;

// Re-export commonly used items
pub use utils::arrow::full_library_to_arrow;
