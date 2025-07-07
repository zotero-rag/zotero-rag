use clap::Parser;
use dotenv::dotenv;
use zqa::cli::app::cli;
use zqa::common::Args;
use zqa::ui::app::App;

#[tokio::main]
pub async fn main() {
    dotenv().ok();
    let args = Args::parse();

    if args.tui {
        let mut terminal = ratatui::init();
        let _ = App::default().run(&mut terminal);
        ratatui::restore();

        return;
    }

    cli(args).await;
}
