use std::io::{stderr, stdout};

use clap::Parser;
use dotenv::dotenv;
use zqa::cli::app::cli;
use zqa::common::{Args, Context, setup_logger};
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
        setup_logger(log_level).expect("Failed to set up logger.");
    }

    log::debug!(
        "You are running {} version {}.",
        env!("CARGO_CRATE_NAME"),
        env!("CARGO_PKG_VERSION")
    );

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
