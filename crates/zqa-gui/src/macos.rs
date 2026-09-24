//! Native material selection for the translucent sidebar.

use gpui_kit::Window;
use objc2_app_kit::{NSView, NSVisualEffectMaterial, NSVisualEffectView};
use raw_window_handle::{HasWindowHandle, RawWindowHandle};

/// Use AppKit's sidebar material for GPUI's window backdrop.
///
/// GPUI's selection material can leave the desktop unblurred on macOS 27. This changes only
/// the material of the existing effect view, leaving its layout and lifecycle with GPUI.
/// Call after opening a blurred window, on GPUI's main thread.
///
/// # Arguments
///
/// * `window` - The newly opened GPUI window with a blurred background.
///
/// # Returns
///
/// Whether a native effect view was found and configured. A false result means the
/// backend's view layout does not expose the expected backdrop.
pub(crate) fn use_sidebar_material(window: &Window) -> bool {
    let Ok(handle) = HasWindowHandle::window_handle(window) else {
        return false;
    };
    let RawWindowHandle::AppKit(handle) = handle.as_raw() else {
        return false;
    };

    // SAFETY: GPUI supplies an NSView pointer whose lifetime is bounded by the borrowed
    // window handle. Window creation calls this on the AppKit main thread; the view is
    // borrowed, not retained or transferred to another thread.
    let view = unsafe { handle.ns_view.cast::<NSView>().as_ref() };
    let Some(content) = view.window().and_then(|window| window.contentView()) else {
        return false;
    };
    let mut configured = false;

    for subview in content.subviews() {
        if let Some(effect) = subview.downcast_ref::<NSVisualEffectView>() {
            effect.setMaterial(NSVisualEffectMaterial::Sidebar);
            configured = true;
        }
    }

    configured
}
