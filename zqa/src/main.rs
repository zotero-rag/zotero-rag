use std::io::{stderr, stdout};

use clap::Parser;
use dotenv::dotenv;
use ftail::Ftail;
use zqa::cli::app::cli;
use zqa::common::{Args, Context};
use zqa::ui::app::App;

#[tokio::main]
pub async fn main() {
    dotenv().ok();
    let args = Args::parse();

    // Avoid RUST_LOG from interfering by not instantiating the logger if it's disabled.
    if args.log_level != "off" {
        let log_level = match args.log_level.as_str() {
            "debug" => log::LevelFilter::Debug,
            "info" => log::LevelFilter::Info,
            "warn" => log::LevelFilter::Warn,
            "error" => log::LevelFilter::Error,
            _ => log::LevelFilter::Off,
        };
        Ftail::new().console(log_level).init().unwrap();
    }

    if args.tui {
        let mut terminal = ratatui::init();
        let _ = App::default().run(&mut terminal);
        ratatui::restore();

        return;
    }

    let context = Context {
        args,
        out: stdout(),
        err: stderr(),
    };

    if let Err(e) = cli(context).await {
        eprintln!("Error: {e}");
    }
}
