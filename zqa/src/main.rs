use std::io::{self, stderr, stdout};

use clap::Parser;
use dotenv::dotenv;
#[cfg(not(tarpaulin))]
use zqa::cli::app::cli;
use zqa::common::{Args, Context, setup_logger};
use zqa::config::Config;
use zqa::state::{check_or_create_first_run_file, oobe};

fn load_config() -> Config {
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

    config
}

#[tokio::main]
pub async fn main() {
    dotenv().ok();

    let mut config = load_config();
    let args = Args::parse();

    // Avoid RUST_LOG from interfering by not instantiating the logger if it's disabled.
    setup_logger(args.log_level).expect("Failed to set up logger.");

    log::debug!(
        "You are running {} version {}. Log level: {}",
        env!("CARGO_CRATE_NAME"),
        env!("CARGO_PKG_VERSION"),
        args.log_level
    );

    if args.log_level < log::LevelFilter::Info && args.print_summaries {
        let warning = "`--print-summaries` requires `--log-level=info`. This will have no effect.";

        if args.log_level >= log::LevelFilter::Warn {
            log::warn!("{warning}");
        } else {
            const YELLOW: &str = "\x1b[33m";
            const YELLOW_BOLD: &str = "\x1b[33;1m";
            const RESET: &str = "\x1b[0m";

            eprintln!("{YELLOW_BOLD}warn: {RESET}{YELLOW}{warning}{RESET}");
        }
    }

    log::debug!("Loaded configuration: {:#?}", config);

    let is_first_run = check_or_create_first_run_file()
        .or_else(|e| {
            println!("Error setting up. (R)etry, (I)gnore, (S)how error and ignore, or (Q)uit: ");

            let mut input = String::new();
            io::stdin()
                .read_line(&mut input)
                .expect("Failed to read input");

            // Return whether this is a "first run". Most options are treated as if it is not a first run.
            match input.trim().to_lowercase().as_str() {
                "r" => check_or_create_first_run_file(),
                "i" => Ok(false),
                "s" => {
                    println!("{:?}", e);
                    Ok(false)
                }
                "q" => std::process::exit(1),
                _ => {
                    println!("Invalid input");
                    std::process::exit(1);
                }
            }
        })
        .unwrap();

    if is_first_run {
        if let Err(e) = oobe() {
            eprintln!("Error during setup: {e}");
        } else {
            // Reload possibly different config from OOBE
            config = load_config();
        }
    }

    let context = Context {
        state: Default::default(),
        config,
        args,
        out: stdout(),
        err: stderr(),
    };

    #[cfg(not(tarpaulin))]
    if let Err(e) = cli(context).await {
        eprintln!("Error: {e}");
    }
}
