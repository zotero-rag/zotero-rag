# zqa-macros-proc

A procedural macro crate providing `#[retry]`, an attribute for automatically retrying flaky async tests on panic.

## Usage

Place `#[retry(N)]` **before** `#[tokio::test]` so it wraps the fully-decorated test function:

```rust
use zqa_macros_proc::retry;

#[retry(3)]
#[tokio::test(flavor = "multi_thread")]
async fn my_flaky_integration_test() {
    // test body — will be retried up to 3 times on panic
}
```

Works alongside `serial_test::serial`:

```rust
#[retry(3)]
#[tokio::test]
#[serial]
async fn serialized_flaky_test() { ... }
```

## MSRV

Rust **1.91** (edition 2024).
