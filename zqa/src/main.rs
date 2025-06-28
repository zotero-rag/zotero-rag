use zqa::ui::app::App;

pub fn main() {
    let mut terminal = ratatui::init();
    let _ = App::default().run(&mut terminal);
    ratatui::restore();
}
