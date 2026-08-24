# zqa-macros

Testing assertion macros, used by `zqa-rag` and `zqa`.

## Macros

### `test_eq!`

Assert two values are equal. Prints received vs. expected on failure without moving either value.

```rust
use zqa_macros::test_eq;

test_eq!(actual, expected);
```

### `test_contains!`

Assert a container holds a specific item. Works with any type that has a `.contains()` method.

```rust
use zqa_macros::test_contains;

test_contains!(vec![1, 2, 3], 2);
test_contains!(response_text, "expected substring");
```

### `test_contains_all!`

Assert a container holds every item in a list. Stops at and reports the first missing item.

```rust
use zqa_macros::test_contains_all;

test_contains_all!(response_text, ["foo", "bar", "baz"]);
```

### `test_ok!`

Assert a `Result` is `Ok`. Prints the `Err` value on failure without moving it.

```rust
use zqa_macros::test_ok;

let result: Result<i32, &str> = Ok(42);
test_ok!(result);
```

## MSRV

Rust **1.91** (edition 2024).
