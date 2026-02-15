//! Macros useful for testing, that print more info.

/// Check that the two passed arguments are equal.
#[macro_export]
macro_rules! test_eq {
    ($left:expr, $right:expr) => {
        let left_var = &$left;
        let right_var = &$right;
        if left_var != right_var {
            eprintln!("Assertion failed:");
            eprintln!("  received: {:?}", left_var);
            eprintln!("  expected: {:?}", right_var);
            panic!("test_eq failed");
        }
    };
}

/// Check that a container contains a specified element.
///
/// Any container that contains a `.contains()` method can be used as the first argument.
#[macro_export]
macro_rules! test_contains {
    ($left:expr, $right: expr) => {
        let left_val = &$left;
        let right_val = &$right;

        if !left_val.contains(right_val) {
            eprintln!("Assertion failed:");
            eprintln!("   provided vec: {:?}", left_val);
            eprintln!("  expected item: {:?}", right_val);
            panic!("test_vec_contains failed");
        }
    };
}

/// Check that a container contains all of a specified argument.
///
/// Any container that contains a `.contains()` method can be used as the first argument, and any
/// iterable can be used as the second.
#[macro_export]
macro_rules! test_contains_all {
    ($left:expr, $right: expr) => {
        let left_val = &$left;
        let right_val = &$right;

        for val in right_val {
            if !left_val.contains(val) {
                eprintln!("Assertion failed:");
                eprintln!("  provided vec: {:?}", left_val);
                eprintln!("  missing item: {:?}", val);
                panic!("test_vec_contains_all failed");
            }
        }
    };
}

/// Check that a provided [`Result`] value is `Ok`.
#[macro_export]
macro_rules! test_ok {
    ($value:expr) => {
        let result = &$value;
        match result {
            Ok(_) => {}
            Err(e) => {
                panic!("expected {} to be `Ok`, got {e}", stringify!($value));
            }
        }
    };
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_eq_success() {
        test_eq!(2 + 2, 4);
        test_eq!("hello", "hello");
    }

    #[test]
    #[should_panic(expected = "test_eq failed")]
    fn test_eq_failure() {
        test_eq!(2 + 2, 5);
    }

    #[test]
    fn test_eq_no_double_eval() {
        let mut count = 0;
        test_eq!(
            {
                count += 1;
                1
            },
            1
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_contains_vec_success() {
        test_contains!(vec![1, 2, 3], 2);
        test_contains!(vec!["a", "b", "c"], "b");
    }

    #[test]
    fn test_contains_string_success() {
        let x = String::from("CATS");
        // You can't spell cats without TS
        test_contains!(x, "TS");
    }

    #[test]
    #[should_panic(expected = "test_vec_contains failed")]
    fn test_contains_failure() {
        test_contains!(vec![1, 2, 3], 4);
    }

    #[test]
    fn test_contains_no_double_eval() {
        let mut count = 0;
        test_contains!(
            {
                count += 1;
                vec![1, 2, 3]
            },
            2
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_contains_no_move() {
        let v = vec![1, 2, 3];
        let x = 2;

        test_contains!(v, x);

        // Should work if macro doesn't move values
        let _y = v;
        let _z = x;
    }

    #[test]
    fn test_contains_all_success() {
        test_contains_all!(vec![1, 2, 3, 4], vec![2, 3]);
        test_contains_all!(vec!["a", "b", "c"], vec!["a", "c"]);
    }

    #[test]
    fn test_contains_all_empty() {
        test_contains_all!(vec![1, 2, 3], vec![]);
    }

    #[test]
    #[should_panic(expected = "test_vec_contains_all failed")]
    fn test_contains_all_failure() {
        test_contains_all!(vec![1, 2, 3], vec![4, 5]);
    }

    #[test]
    #[should_panic(expected = "test_vec_contains_all failed")]
    fn test_contains_all_partial_failure() {
        test_contains_all!(vec![1, 2, 3], vec![2, 5]);
    }

    #[test]
    fn test_contains_all_no_double_eval() {
        let mut count = 0;
        test_contains_all!(
            {
                count += 1;
                vec![1, 2, 3]
            },
            vec![1, 2]
        );
        assert_eq!(count, 1);
    }

    #[test]
    fn test_contains_all_no_move() {
        let v = vec![1, 2, 3, 4];
        let items = vec![2, 3];

        test_contains_all!(v, items);

        let _y = v;
        let _z = items;
    }

    #[test]
    fn test_ok_success() {
        test_ok!(Ok::<i32, &str>(42));
        test_ok!(Ok::<(), String>(()));
    }

    #[test]
    #[should_panic(expected = "to be `Ok`")]
    fn test_ok_failure() {
        test_ok!(Err::<i32, &str>("something went wrong"));
    }

    #[test]
    fn test_ok_no_double_eval() {
        let mut count = 0;
        test_ok!({
            count += 1;
            Ok::<i32, &str>(42)
        });
        assert_eq!(count, 1);
    }

    #[test]
    fn test_ok_no_move() {
        let result: Result<i32, &str> = Ok(42);
        test_ok!(result);
        let _ = result;
    }
}
