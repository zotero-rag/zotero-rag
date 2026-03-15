//! Proc-macro attribute for retrying async test functions.

use proc_macro::TokenStream;
use quote::quote;
use syn::{ItemFn, LitInt, parse_macro_input};

/// Retry an async test function up to `n` times on panic.
///
/// Each attempt runs the function body inside a fresh
/// `tokio::task::LocalSet` + `spawn_local` pair so that panics from
/// `assert!`, `test_ok!`, etc. are caught and retried rather than
/// immediately failing the test. `!Send` futures (e.g. those using
/// `temp_env`) are supported.
///
/// On the final attempt any panic is re-propagated, failing the test normally.
///
/// This attribute must be placed **before** `#[tokio::test]` so that the
/// retry wrapping is applied first; the remaining attributes pass through
/// into the generated output untouched.
///
/// # Usage
/// ```ignore
/// #[retry(3)]
/// #[tokio::test(flavor = "multi_thread")]
/// #[serial]
/// async fn my_test() {
///     // body at normal indentation — no extra wrapping needed
/// }
/// ```
#[proc_macro_attribute]
pub fn retry(args: TokenStream, input: TokenStream) -> TokenStream {
    let max = parse_macro_input!(args as LitInt);
    match max.base10_parse::<usize>() {
        Ok(val) if val > 0 => (),
        Ok(_) => {
            return syn::Error::new(
                max.span(),
                "#[retry(n)]: number of retries must be positive.",
            )
            .to_compile_error()
            .into();
        }
        Err(e) => return e.to_compile_error().into(),
    };

    let func = parse_macro_input!(input as ItemFn);

    if func.sig.asyncness.is_none() {
        return syn::Error::new_spanned(
            func.sig.fn_token,
            "#[retry] can only be applied to async functions",
        )
        .to_compile_error()
        .into();
    }

    let attrs = &func.attrs;
    let vis = &func.vis;
    let sig = &func.sig;
    let body = &func.block;

    quote! {
        #(#attrs)*
        #vis #sig {
            let __max: usize = #max;
            async {
                for attempt in 1..=__max {
                    let local = ::tokio::task::LocalSet::new();
                    let result = local
                        .run_until(async {
                            ::tokio::task::spawn_local(async #body).await
                        })
                        .await;
                    match result {
                        Ok(value) => return value,
                        Err(_) if attempt < __max => {
                            eprintln!("[retry] attempt {attempt}/{__max} failed, retrying...");
                        }
                        Err(e) => ::std::panic::resume_unwind(e.into_panic()),
                    }
                }

                unreachable!();
            }
            .await;
            unreachable!()
        }
    }
    .into()
}
