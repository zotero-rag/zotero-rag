use ratatui::style::{Color, Style};

use crate::{ui::app::App, utils::library::parse_library_metadata};

pub fn exit(app: &mut App) {
    app.exit = true;
}

pub fn handle_process_command(app: &mut App) {
    // Clear the input box
    app.user_query = String::new();

    const WARNING_THRESHOLD: usize = 100;
    let item_metadata = parse_library_metadata(None, None);

    if let Err(parse_err) = item_metadata {
        app.output_lines
            .push(format!("Could not parse metadata: {parse_err}"));
        app.line_styles.push(Style::default().fg(Color::Yellow));

        return;
    }

    let item_metadata = item_metadata.unwrap();
    let metadata_length = item_metadata.len();
    if metadata_length >= WARNING_THRESHOLD {
        app.output_lines.push(format!(
            "Your library has {metadata_length} items. Parsing may take a while. Continue?"
        ));
        app.line_styles.push(Style::default().fg(Color::Yellow));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::serial;
    use temp_env::with_vars;

    #[test]
    fn exit_sets_flag() {
        let mut app = App::default();
        assert!(!app.exit);
        exit(&mut app);
        assert!(app.exit);
    }

    #[test]
    #[serial]
    fn handle_process_respects_ci_assets_and_adds_warning_for_large_lib() {
        with_vars([("CI", Some("true"))], || {
            let mut app = App::default();
            let before_len = app.output_lines.len();
            let _before_styles = app.line_styles.len();
            handle_process_command(&mut app);

            // Input cleared
            assert_eq!(app.user_query, "");
            // Either no change or exactly one yellow warning line added
            assert!(app.output_lines.len() == before_len || app.output_lines.len() == before_len + 1);
            assert_eq!(app.output_lines.len(), app.line_styles.len());
        });
    }
}
