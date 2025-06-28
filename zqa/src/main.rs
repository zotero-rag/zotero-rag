use dotenv::dotenv;
use zqa::ui::app::App;

pub fn main() {
    dotenv().ok();

    let mut terminal = ratatui::init();
    let _ = App::default().run(&mut terminal);
    ratatui::restore();
}
