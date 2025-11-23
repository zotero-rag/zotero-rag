use std::io::{stderr, stdout};

use clap::Parser;
use dotenv::dotenv;
use zqa::cli::app::cli;
use zqa::common::{Args, Context, setup_logger};
use zqa::config::Config;

#[tokio::main]
pub async fn main() {
    dotenv().ok();

    // Load the configs in priority order: TOML < env < CLI args
    let mut config = Config::default();
    if let Some(xdg_config_dir) = directories::UserDirs::new() {
        let config_path = xdg_config_dir
            .home_dir()
            .join(".config")
            .join("zqa")
            .join("config.toml");

        if config_path.exists() {
            // We want to panic here if the config is invalid; it's early enough that we can
            // reasonably ask a user to fix the underlying issue.
            config = Config::from_file(config_path).unwrap()
        }
    }

    // Overwrite with env
    config.read_env().unwrap();

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

    log::debug!("Loaded configuration: {:#?}", config);

    let context = Context {
        state: Default::default(),
        config,
        args,
        out: stdout(),
        err: stderr(),
    };

    if let Err(e) = cli(context).await {
        eprintln!("Error: {e}");
    }
}
