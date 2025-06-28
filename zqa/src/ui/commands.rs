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
