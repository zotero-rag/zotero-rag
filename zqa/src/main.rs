use clap::Parser;
use dotenv::dotenv;
use zqa::cli::app::cli;
use zqa::ui::app::App;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    /// Whether to use the tui interface
    #[arg(long, default_value_t = false)]
    tui: bool,
}

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

    cli().await;
}
