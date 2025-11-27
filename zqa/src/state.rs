//!  NOTE: This is not state management! The `state` module is actually a way to interact with `XDG_STATE_HOME`.

use std::{fs, io};
use thiserror::Error;

use crate::config::{BaseDirError, Config, get_config_dir};

/// ANSI escape code for dimming text
const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
const RESET: &str = "\x1b[0m";

/// Errors that can occur when interacting with the state directory.
#[derive(Debug, Error)]
pub enum StateErrors {
    #[error("Failed to get home directory.")]
    DirectoryError,
    #[error("Failed to save first run information.")]
    FileWriteError,
    #[error("Serialization error: {0}")]
    SerializationError(#[from] toml::ser::Error),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<BaseDirError> for StateErrors {
    fn from(_value: BaseDirError) -> Self {
        StateErrors::DirectoryError
    }
}

impl From<io::Error> for StateErrors {
    fn from(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::PermissionDenied => StateErrors::FileWriteError,
            _ => StateErrors::Other(err.to_string()),
        }
    }
}

/// Determine if this is the first run of the application. We simply use a blank file's existence to determine this;
/// any contents are ignored. This is also a nice way for users to reset, if for whatever reason they want to.
///
/// # Returns
///
/// `true` if this is the first run of the application (i.e., the `first_run` file does *not* exist); false otherwise.
///
/// # Errors:
///
/// * `StateErrors::DirectoryError` when the state dir could not be obtained.
/// * `StateErrors::FileWriteError` if we do not have permissions to create the `first_run` file.
pub fn check_or_create_first_run_file() -> Result<bool, StateErrors> {
    let base_dir = directories::BaseDirs::new().ok_or(StateErrors::DirectoryError)?;
    let state_dir = base_dir.state_dir().ok_or(StateErrors::DirectoryError)?;
    let first_run_file = state_dir.join("first_run");

    if first_run_file.exists() {
        Ok(false)
    } else {
        fs::File::create(&first_run_file)?;
        Ok(true)
    }
}

/// Read a line of input.
fn read_line() -> String {
    let mut input = String::new();
    io::stdin()
        .read_line(&mut input)
        .expect("Failed to read input");

    input
}

/// Read a character from standard input and return it, handling Enter as a default.
///
/// # Arguments:
///
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters.
fn read_char(default: char, valid_set: &[char]) -> char {
    loop {
        println!("> ");
        let input = read_line();
        let choice = input.chars().next().unwrap_or(default);

        if valid_set.contains(&choice) {
            return choice;
        }
    }
}

/// Read an integer from standard input, and validate that it is within bounds.
///
/// # Arguments:
///
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Lower and upper bounds to accept. Lower bound is inclusive, upper is exclusive.
fn read_number(default: u8, bounds: (u8, u8)) -> u8 {
    loop {
        println!("> ");
        let input = read_line();
        let input = input.trim();
        if input.is_empty() {
            return default;
        }

        let choice = input.parse::<u8>();

        match choice {
            Ok(num) => {
                if bounds.0 <= num && num < bounds.1 {
                    return num;
                } else {
                    println!("Choice must be in [{}, {}).", bounds.0, bounds.1);
                }
            }
            Err(_) => {
                println!("Choice must be in [{}, {}).", bounds.0, bounds.1);
            }
        }
    }
}

/// Set up the out-of-box experience (OOBE) for the application.
///
/// This function uses reasonable defaults for most config attributes. The goal here is not to be meticulous, but
/// rather to get the user to a position where they can start using the application, assuming they have the API
/// keys for the choices they make. For most providers, we use the defaults provided by the `rag` crate; the onus
/// is on that to provide reasonable defaults. Fortunately, I am the author of both :) (and I am reasonable, usually).
///
/// # Errors
///
/// * `StateErrors::DirectoryError` if we could not get the config directory.
/// * `StateErrors::SerializationError` if serialization to JSON failed.
pub fn oobe() -> Result<(), StateErrors> {
    // Here, we don't load the env because that's directory-specific.
    let mut config = Config::default();

    println!("Let's set up your config.");
    println!(
        "{DIM_TEXT}Since this is your first run, you will set up a basic configuration. This should take a few minutes. You can always update these by editing ~/.config/zqa/config.toml directly. For most options, a default will be shown in [square brackets]; if that is okay, you can simply press Enter. You can use uppercase or lowercase.\n{RESET}"
    );

    // TODO: We need smarter defaults where if the user for some reason has a config file, we use that value. We could
    // skip users that have a *valid* config.
    println!("Would you like to set up your configuration?");
    println!("[Y]es");
    println!("(N)o");
    let choice = read_char('n', &['y', 'n']);

    if choice == 'n' {
        return Ok(());
    }

    println!("What model provider do you want to use?");
    println!("[A]nthropic");
    println!("(O)penAI");
    println!("(G)emini");
    println!("Open(R)outer");
    println!();
    let model_provider = read_char('a', &['a', 'o', 'g', 'r']);

    // TODO: Ideally we want to enable the password mode that some shells support.
    println!("Enter your model provider's API key: ");
    let model_api_key = read_line().trim().to_string();
    println!();

    config.model_provider = match model_provider {
        'a' => "anthropic",
        'o' => "openai",
        'g' => "gemini",
        'r' => "openrouter",
        _ => {
            unreachable!("Unknown model provider");
        }
    }
    .into();

    println!("What provider do you want to use for embeddings?");
    println!(
        "{DIM_TEXT}Note that we strongly discourage OpenAI at this time, since we do not chunk PDF texts, and it is very likely to fail. The other providers use truncated texts.{RESET}"
    );
    println!("(C)ohere");
    println!("(G)emini");
    println!("(O)penAI");
    println!("[V]oyage AI");
    println!();
    let embedding_provider = read_char('v', &['c', 'g', 'o', 'v']);

    config.embedding_provider = match embedding_provider {
        'c' => "cohere",
        'g' => "gemini",
        'o' => "openai",
        'v' => "voyageai",
        _ => {
            unreachable!("Unknown model provider");
        }
    }
    .into();

    let embedding_api_key = if embedding_provider == model_provider {
        model_api_key.clone()
    } else {
        println!("Enter your embedding provider's API key: ");
        println!();
        read_line().trim().to_string()
    };

    println!("What provider do you want to use for reranking results?");
    println!("{DIM_TEXT}In general, you will want to use the same provider as embeddings.{RESET}");
    match embedding_provider {
        'c' | 'g' | 'o' => {
            println!("[C]ohere");
            println!("(V)oyage AI");
        }
        'v' => {
            println!("(C)ohere");
            println!("[V]oyage AI");
        }
        _ => {
            unreachable!("Embedding provider was validated.");
        }
    }
    println!();
    let reranker_provider = read_char(embedding_provider, &['c', 'v']);
    config.reranker_provider = match reranker_provider {
        'c' => "cohere",
        'v' => "voyageai",
        _ => unreachable!("Reranker provider was validated."),
    }
    .into();

    let reranker_api_key = if reranker_provider == embedding_provider {
        embedding_api_key.clone()
    } else {
        println!("Enter your reranker provider's API key: ");
        println!();
        read_line().trim().to_string()
    };

    // The 100k and 150k are somewhat napkin math-based, but pretty decent. In practice, embedding providers usually
    // have much higher TPM limits, so we're limited by the generation model provider's TPM. A half-decent estimate
    // for one paper is ~30k-50k tokens. For example, I have an 800k TPM with Claude, and a value of 5 works well,
    // though I haven't tested other values. For most users, the RPM will be far higher than necessary.
    println!("How many requests do you want to send at a time?");
    println!(
        "{DIM_TEXT}A higher number can yield faster results, but can also result in being rate-limited. You should check what tier of API you have access to and check the TPM (tokens per minute) limit to make a choice here. As a rough estimate, your TPM limit divided by 150,000 is a somewhat reasonable estimate.{RESET}"
    );
    let max_concurrent_requests = read_number(5, (1, 20));
    config.max_concurrent_requests = max_concurrent_requests as usize;

    // We can unwrap the provider configs since we initialized via `Default`, which sets them to a `Some(..)`.
    match model_provider {
        'a' => {
            let anthropic_config = config.anthropic.as_mut().unwrap();
            anthropic_config.api_key = Some(model_api_key);
        }
        'o' => {
            let openai_config = config.openai.as_mut().unwrap();
            openai_config.api_key = Some(model_api_key);
        }
        'g' => {
            let gemini_config = config.gemini.as_mut().unwrap();
            gemini_config.api_key = Some(model_api_key);
        }
        'r' => {
            let openrouter_config = config.openrouter.as_mut().unwrap();
            openrouter_config.api_key = Some(model_api_key);
        }
        _ => unreachable!("Model provider was validated."),
    }

    if embedding_provider != model_provider {
        match embedding_provider {
            'c' => {
                let cohere_config = config.cohere.as_mut().unwrap();
                cohere_config.api_key = Some(embedding_api_key);
            }
            'g' => {
                let gemini_config = config.gemini.as_mut().unwrap();
                gemini_config.api_key = Some(embedding_api_key);
            }
            'o' => {
                let openai_config = config.openai.as_mut().unwrap();
                openai_config.api_key = Some(embedding_api_key);
            }
            'v' => {
                let voyage_config = config.voyageai.as_mut().unwrap();
                voyage_config.api_key = Some(embedding_api_key);
            }
            _ => unreachable!("Embedding provider was validated."),
        }
    }

    if reranker_provider != embedding_provider {
        match reranker_provider {
            'c' => {
                let cohere_config = config.cohere.as_mut().unwrap();
                cohere_config.api_key = Some(reranker_api_key);
            }
            'v' => {
                let voyage_config = config.voyageai.as_mut().unwrap();
                voyage_config.api_key = Some(reranker_api_key);
            }
            _ => unreachable!("Reranker provider was validated."),
        }
    }

    let config_dir = get_config_dir()?;
    let config_path = config_dir.join("config.toml");
    fs::write(config_path, toml::to_string_pretty(&config)?)?;

    println!("You've set up your config!");
    println!(
        "{DIM_TEXT}You can change these values any time by editing ~/.config/zqa/config.toml.\n"
    );
    println!(
        "Next, you will likely want to set up your embeddings by typing /process in the prompt that follows. Note that this will take a while! If you don't have time, you can quit now, reopen the CLI later, and run it then.\n"
    );
    println!(
        "Since your API keys are stored in plain-text, make sure to never commit ~/.config/zqa/config.toml without first deleting the keys. The recommended setup is to set the values in the TOML file blank and use .env files where you need them with the keys. As an additional security measure, consider running `chmod 600 ~/.config/zqa/config.toml`.{RESET}"
    );
    Ok(())
}
