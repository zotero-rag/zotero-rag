//! A global switch for terminal progress bars.
//!
//! Long-running operations (such as embedding generation) render `indicatif` progress bars
//! directly to the terminal. Full-screen frontends (such as zqa's TUI) own the terminal and
//! must suppress this output, since anything drawn outside their frame corrupts the display.
//! Such frontends should call [`set_progress_bars_enabled`] with `false` before invoking any
//! long-running operations.
//!
//! Progress bars are enabled by default.

use std::sync::atomic::{AtomicBool, Ordering};

static PROGRESS_BARS_ENABLED: AtomicBool = AtomicBool::new(true);

/// Enable or disable terminal progress bars globally.
///
/// # Arguments
///
/// * `enabled` - Whether progress bars should be drawn to the terminal.
pub fn set_progress_bars_enabled(enabled: bool) {
    PROGRESS_BARS_ENABLED.store(enabled, Ordering::Relaxed);
}

/// Whether terminal progress bars are currently enabled.
///
/// # Returns
///
/// `true` unless a frontend has disabled progress bars via [`set_progress_bars_enabled`].
#[must_use]
pub fn progress_bars_enabled() -> bool {
    PROGRESS_BARS_ENABLED.load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::{progress_bars_enabled, set_progress_bars_enabled};

    #[test]
    fn test_progress_bars_toggle() {
        assert!(progress_bars_enabled());

        set_progress_bars_enabled(false);
        assert!(!progress_bars_enabled());

        set_progress_bars_enabled(true);
        assert!(progress_bars_enabled());
    }
}
