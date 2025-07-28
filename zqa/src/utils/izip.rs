// We only need the `izip` macro from the `itertools` crate, so we use its implementation directly:
// See https://docs.rs/itertools/latest/itertools/macro.izip.html
#[macro_export]
/// Create an iterator running multiple iterators in lockstep.
///
/// The `izip!` iterator yields elements until any subiterator
/// returns `None`.
///
/// This is a version of the standard ``.zip()`` that's supporting more than
/// two iterators. The iterator element type is a tuple with one element
/// from each of the input iterators. Just like ``.zip()``, the iteration stops
/// when the shortest of the inputs reaches its end.
///
/// **Note:** The result of this macro is in the general case an iterator
/// composed of repeated `.zip()` and a `.map()`; it has an anonymous type.
/// The special cases of one and two arguments produce the equivalent of
/// `$a.into_iter()` and `$a.into_iter().zip($b)` respectively.
///
/// Prefer this macro `izip!()` over [`multizip`] for the performance benefits
/// of using the standard library `.zip()`.
///
/// ```
/// # // The `izip!` macro is defined locally in this module.
/// # fn main() {
///
/// // iterate over three sequences side-by-side
/// let mut results = [0, 0, 0, 0];
/// let inputs = [3, 7, 9, 6];
///
/// for (r, index, input) in izip!(&mut results, 0..10, &inputs) {
///     *r = index * 10 + input;
/// }
///
/// assert_eq!(results, [0 + 3, 10 + 7, 29, 36]);
/// # }
/// ```
macro_rules! izip {
    // @closure creates a tuple-flattening closure for .map() call. usage:
    // @closure partial_pattern => partial_tuple , rest , of , iterators
    // eg. izip!( @closure ((a, b), c) => (a, b, c) , dd , ee )
    ( @closure $p:pat => $tup:expr ) => {
        |$p| $tup
    };

    // The "b" identifier is a different identifier on each recursion level thanks to hygiene.
    ( @closure $p:pat => ( $($tup:tt)* ) , $_iter:expr $( , $tail:expr )* ) => {
        $crate::izip!(@closure ($p, b) => ( $($tup)*, b ) $( , $tail )*)
    };

    // unary
    ($first:expr $(,)*) => {
        std::iter::IntoIterator::into_iter($first)
    };

    // binary
    ($first:expr, $second:expr $(,)*) => {
        std::iter::Iterator::zip(
            std::iter::IntoIterator::into_iter($first),
            $second,
        )
    };

    // n-ary where n > 2
    ( $first:expr $( , $rest:expr )* $(,)* ) => {
        {
            let iter = std::iter::IntoIterator::into_iter($first);
            $(
                let iter = std::iter::Iterator::zip(iter, $rest);
            )*

            std::iter::Iterator::map(
                iter,
                $crate::izip!(@closure a => (a) $( , $rest )*)
            )
        }
    };
}
