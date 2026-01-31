//!  NOTE: This is not state management! The `state` module is actually a way to interact with `XDG_STATE_HOME`.

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use std::{
    fs,
    io::{self, BufRead, Write},
    path::PathBuf,
};
use thiserror::Error;
use zqa_rag::llm::base::ChatHistoryItem;

use crate::config::{BaseDirError, Config, get_config_dir};

/// ANSI escape code for dimming text
const DIM_TEXT: &str = "\x1b[2m";

/// ANSI escape code for resetting text formatting
const RESET: &str = "\x1b[0m";

/// Errors that can occur when interacting with the state directory.
#[derive(PartialEq, Debug, Error)]
pub(crate) enum StateError {
    #[error("Failed to get home directory.")]
    DirectoryError,
    #[error("Failed to save first run information.")]
    FileWriteError,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<toml::ser::Error> for StateError {
    fn from(value: toml::ser::Error) -> Self {
        Self::SerializationError(format!("Serialization error: {value}"))
    }
}

impl From<serde_json::Error> for StateError {
    fn from(value: serde_json::Error) -> Self {
        Self::SerializationError(format!("Serialization error: {value}"))
    }
}

impl From<BaseDirError> for StateError {
    fn from(_value: BaseDirError) -> Self {
        StateError::DirectoryError
    }
}

impl From<io::Error> for StateError {
    fn from(err: io::Error) -> Self {
        match err.kind() {
            io::ErrorKind::PermissionDenied => StateError::FileWriteError,
            _ => StateError::Other(err.to_string()),
        }
    }
}

/// Returns the state directory inside `XDG_STATE_HOME`. On *nix systems, this returns
/// ~/.local/state/zqa. On Windows, this also returns ~/.local/state/zqa, except that `$HOME` is
/// typically C:\Users\<username>.
///
/// # Errors
///
/// * `StateError::DirectoryError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
pub(crate) fn get_state_dir() -> Result<PathBuf, StateError> {
    let base_dir = directories::BaseDirs::new().ok_or(StateError::DirectoryError)?;

    let alt_state_dir = base_dir.home_dir().join(".local").join("state");
    let state_dir = base_dir.state_dir().unwrap_or(&alt_state_dir);

    Ok(state_dir.join("zqa"))
}

/// A chat history that is stored in the user's state directory. This wraps a
/// `Vec<ChatHistoryItem>` under the hood, but also includes some metadata such as when the chat
/// occurred.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct SavedChatHistory {
    /// The chat history
    pub(crate) history: Vec<ChatHistoryItem>,
    /// When this conversation occurred
    pub(crate) date: DateTime<Local>,
    /// A brief title
    pub(crate) title: String,
}

/// Attempt to get all previous conversations if they exist.
///
/// # Returns
///
/// If there are no conversations, or if there was no conversations directory, returns `Ok(None)`. In
/// the latter case, the directory is also created. If conversations could not be loaded (such as
/// due to permissions errors), returns a `StateError`.
///
/// In all other cases, returns `Ok(Some(history))`, where `history` contains reverse
/// chronologically ordered conversation histories.
///
/// # Errors
///
/// * `StateErrors::DirectoryError` when the state dir could not be obtained.
/// * `StateError::FileWriteError` if the conversations directory could not be created.
#[allow(dead_code)]
pub(crate) fn get_conversation_history() -> Result<Option<Vec<SavedChatHistory>>, StateError> {
    let state_dir = get_state_dir()?;
    let conversations_dir = state_dir.join("conversations");

    if !conversations_dir.exists() {
        fs::create_dir_all(&conversations_dir)?;
        return Ok(None);
    }

    let histories = conversations_dir
        .read_dir()?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| fs::read_to_string(entry.path()).ok())
        .filter_map(|s| serde_json::from_str::<SavedChatHistory>(&s).ok())
        .collect::<Vec<_>>();

    Ok(Some(histories))
}

/// Attempt to save a conversation history to the user state directory.
///
/// # Errors
///
/// * `StateErrors::DirectoryError` when the state dir could not be obtained.
/// * `StateError::FileWriteError` if the conversations directory could not be created.
pub(crate) fn save_conversation(conversation: &SavedChatHistory) -> Result<(), StateError> {
    let state_dir = get_state_dir()?;
    let conversations_dir = state_dir.join("conversations");

    if !conversations_dir.exists() {
        fs::create_dir_all(&conversations_dir)?;
    }

    let timestamp_suffix = conversation.date.timestamp_millis().to_string();
    let file_name = format!("conversation_{timestamp_suffix}.json");

    fs::write(
        conversations_dir.join(&file_name),
        serde_json::to_string_pretty(&conversation)?,
    )?;

    Ok(())
}

/// Determine if this is the first run of the application. We simply use a blank file's existence to determine this;
/// any contents are ignored. This is also a nice way for users to reset, if for whatever reason they want to.
///
/// # Returns
///
/// `true` if this is the first run of the application (i.e., the `first_run` file does *not* exist); false otherwise.
///
/// # Errors
///
/// * `StateErrors::DirectoryError` when the state dir could not be obtained.
/// * `StateErrors::FileWriteError` if we do not have permissions to create the `first_run` file.
pub(crate) fn check_or_create_first_run_file() -> Result<bool, StateError> {
    let state_dir = get_state_dir()?;
    let first_run_file = state_dir.join("first_run");

    if first_run_file.exists() {
        Ok(false)
    } else {
        fs::create_dir_all(&state_dir)?;
        fs::File::create(&first_run_file)?;
        Ok(true)
    }
}

/// Read a line of input.
fn read_line<R: BufRead>(input: &mut R) -> String {
    let mut buffer = String::new();
    input.read_line(&mut buffer).expect("Failed to read input");
    buffer
}

/// Read a character from standard input and return it, handling Enter as a default.
///
/// # Arguments:
///
/// * `default` - The default if Enter is pressed.
/// * `valid_set` - The valid set of characters.
fn read_char<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    default: char,
    valid_set: &[char],
) -> char {
    loop {
        write!(output, "> ").ok();
        output.flush().ok();
        let buffer = read_line(input);
        let choice = buffer
            .chars()
            .next()
            .unwrap_or(default)
            .to_ascii_lowercase();

        if valid_set.contains(&choice) {
            return choice;
        }

        if choice == '\n' {
            return default;
        }
    }
}

/// Read an integer from standard input, and validate that it is within bounds.
///
/// # Arguments:
///
/// * `default` - The default value if Enter is pressed.
/// * `bounds` - Lower and upper bounds to accept. Lower bound is inclusive, upper is exclusive.
fn read_number<R: BufRead, W: Write>(
    input: &mut R,
    output: &mut W,
    default: u8,
    bounds: (u8, u8),
) -> u8 {
    loop {
        write!(output, "> ").ok();
        output.flush().ok();
        let buffer = read_line(input);
        let choice_str = buffer.trim();
        if choice_str.is_empty() {
            return default;
        }

        let choice = choice_str.parse::<u8>();

        match choice {
            Ok(num) => {
                if bounds.0 <= num && num < bounds.1 {
                    return num;
                }
                writeln!(output, "Choice must be in [{}, {}).", bounds.0, bounds.1).ok();
            }
            Err(_) => {
                writeln!(output, "Choice must be in [{}, {}).", bounds.0, bounds.1).ok();
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
///
/// # Panics
///
/// If getting the config for the chosen provider fails.
#[allow(clippy::too_many_lines)]
pub(crate) fn oobe<R: BufRead, W: Write>(input: &mut R, output: &mut W) -> Result<(), StateError> {
    // Here, we don't load the env because that's directory-specific.
    let config_dir = get_config_dir()?;
    let config_path = config_dir.join("config.toml");
    let mut config = if config_path.exists() {
        match Config::from_file(&config_path) {
            Ok(c) => c,
            Err(_) => Config::default(),
        }
    } else {
        Config::default()
    };

    writeln!(output, "Let's set up your config.").ok();
    writeln!(
        output,
        "{DIM_TEXT}Since this is your first run, you will set up a basic configuration. This should take a few minutes. You can always update these by editing ~/.config/zqa/config.toml directly. For most options, a default will be shown in [square brackets]; if that is okay, you can simply press Enter. You can use uppercase or lowercase.\n{RESET}"
    ).ok();

    writeln!(output, "Would you like to set up your configuration?").ok();
    let setup_default = if config_path.exists() { 'n' } else { 'y' };
    if setup_default == 'y' {
        writeln!(output, "[Y]es").ok();
        writeln!(output, "(N)o").ok();
    } else {
        writeln!(output, "(Y)es").ok();
        writeln!(output, "[N]o").ok();
    }
    let choice = read_char(input, output, setup_default, &['y', 'n']);

    if choice == 'n' {
        return Ok(());
    }

    writeln!(output, "What model provider do you want to use?").ok();
    let model_default = match config.model_provider.as_str() {
        "anthropic" => 'a',
        "openai" => 'o',
        "gemini" => 'g',
        "openrouter" => 'r',
        _ => 'a',
    };

    writeln!(
        output,
        "{}nthropic",
        if model_default == 'a' { "[A]" } else { "(A)" }
    )
    .ok();
    writeln!(
        output,
        "{}penAI",
        if model_default == 'o' { "[O]" } else { "(O)" }
    )
    .ok();
    writeln!(
        output,
        "{}emini",
        if model_default == 'g' { "[G]" } else { "(G)" }
    )
    .ok();
    writeln!(
        output,
        "Open{}outer",
        if model_default == 'r' { "[R]" } else { "(R)" }
    )
    .ok();
    writeln!(output).ok();
    let model_provider = read_char(input, output, model_default, &['a', 'o', 'g', 'r']);

    // TODO: Ideally we want to enable the password mode that some shells support.
    let existing_api_key = match model_provider {
        'a' => config.anthropic.as_ref().and_then(|c| c.api_key.clone()),
        'o' => config.openai.as_ref().and_then(|c| c.api_key.clone()),
        'g' => config.gemini.as_ref().and_then(|c| c.api_key.clone()),
        'r' => config.openrouter.as_ref().and_then(|c| c.api_key.clone()),
        _ => None,
    };

    if existing_api_key.is_some() {
        writeln!(
            output,
            "Enter your model provider's API key (Press Enter to keep existing): "
        )
        .ok();
    } else {
        writeln!(output, "Enter your model provider's API key: ").ok();
    }
    let line = read_line(input);
    let buffer = line.trim().to_string();
    let model_api_key = if buffer.is_empty() {
        existing_api_key.unwrap_or_default()
    } else {
        buffer
    };
    writeln!(output).ok();

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

    writeln!(output, "What provider do you want to use for embeddings?").ok();
    writeln!(
        output,
        "{DIM_TEXT}Note that we strongly discourage OpenAI at this time, since we do not chunk PDF texts, and it is very likely to fail. The other providers use truncated texts.{RESET}"
    ).ok();

    let embedding_default = match config.embedding_provider.as_str() {
        "cohere" => 'c',
        "gemini" => 'g',
        "openai" => 'o',
        "voyageai" => 'v',
        _ => 'v',
    };

    writeln!(
        output,
        "{}ohere",
        if embedding_default == 'c' {
            "[C]"
        } else {
            "(C)"
        }
    )
    .ok();
    writeln!(
        output,
        "{}emini",
        if embedding_default == 'g' {
            "[G]"
        } else {
            "(G)"
        }
    )
    .ok();
    writeln!(
        output,
        "{}penAI",
        if embedding_default == 'o' {
            "[O]"
        } else {
            "(O)"
        }
    )
    .ok();
    writeln!(
        output,
        "{}oyage AI",
        if embedding_default == 'v' {
            "[V]"
        } else {
            "(V)"
        }
    )
    .ok();
    writeln!(output).ok();
    let embedding_provider = read_char(input, output, embedding_default, &['c', 'g', 'o', 'v']);

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
        let existing_api_key = match embedding_provider {
            'c' => config.cohere.as_ref().and_then(|c| c.api_key.clone()),
            'g' => config.gemini.as_ref().and_then(|c| c.api_key.clone()),
            'o' => config.openai.as_ref().and_then(|c| c.api_key.clone()),
            'v' => config.voyageai.as_ref().and_then(|c| c.api_key.clone()),
            _ => None,
        };

        if existing_api_key.is_some() {
            writeln!(
                output,
                "Enter your embedding provider's API key (Press Enter to keep existing): "
            )
            .ok();
        } else {
            writeln!(output, "Enter your embedding provider's API key: ").ok();
        }
        writeln!(output).ok();
        let line = read_line(input);
        let buffer = line.trim().to_string();
        if buffer.is_empty() {
            existing_api_key.unwrap_or_default()
        } else {
            buffer
        }
    };

    writeln!(
        output,
        "What provider do you want to use for reranking results?"
    )
    .ok();
    writeln!(
        output,
        "{DIM_TEXT}In general, you will want to use the same provider as embeddings.{RESET}"
    )
    .ok();

    let reranker_default = match config.reranker_provider.as_str() {
        "cohere" => 'c',
        "voyageai" => 'v',
        _ => {
            if embedding_provider == 'c' {
                'c'
            } else {
                'v'
            }
        }
    };

    match embedding_provider {
        'c' | 'g' | 'o' => {
            writeln!(
                output,
                "{}ohere",
                if reranker_default == 'c' {
                    "[C]"
                } else {
                    "(C)"
                }
            )
            .ok();
            writeln!(
                output,
                "{}oyage AI",
                if reranker_default == 'v' {
                    "[V]"
                } else {
                    "(V)"
                }
            )
            .ok();
        }
        'v' => {
            writeln!(
                output,
                "{}ohere",
                if reranker_default == 'c' {
                    "[C]"
                } else {
                    "(C)"
                }
            )
            .ok();
            writeln!(
                output,
                "{}oyage AI",
                if reranker_default == 'v' {
                    "[V]"
                } else {
                    "(V)"
                }
            )
            .ok();
        }
        _ => {
            unreachable!("Embedding provider was validated.");
        }
    }
    writeln!(output).ok();
    let reranker_provider = read_char(input, output, reranker_default, &['c', 'v']);
    config.reranker_provider = match reranker_provider {
        'c' => "cohere",
        'v' => "voyageai",
        _ => unreachable!("Reranker provider was validated."),
    }
    .into();

    let reranker_api_key = if reranker_provider == embedding_provider {
        embedding_api_key.clone()
    } else {
        let existing_api_key = match reranker_provider {
            'c' => config.cohere.as_ref().and_then(|c| c.api_key.clone()),
            'v' => config.voyageai.as_ref().and_then(|c| c.api_key.clone()),
            _ => None,
        };

        if existing_api_key.is_some() {
            writeln!(
                output,
                "Enter your reranker provider's API key (Press Enter to keep existing): "
            )
            .ok();
        } else {
            writeln!(output, "Enter your reranker provider's API key: ").ok();
        }
        writeln!(output).ok();
        let line = read_line(input);
        let buffer = line.trim().to_string();
        if buffer.is_empty() {
            existing_api_key.unwrap_or_default()
        } else {
            buffer
        }
    };

    // The 100k and 150k are somewhat napkin math-based, but pretty decent. In practice, embedding providers usually
    // have much higher TPM limits, so we're limited by the generation model provider's TPM. A half-decent estimate
    // for one paper is ~30k-50k tokens. For example, I have an 800k TPM with Claude, and a value of 5 works well,
    // though I haven't tested other values. For most users, the RPM will be far higher than necessary.
    writeln!(
        output,
        "How many requests do you want to send at a time? (Default: {})",
        config.max_concurrent_requests
    )
    .ok();
    writeln!(
        output,
        "{DIM_TEXT}A higher number can yield faster results, but can also result in being rate-limited. You should check what tier of API you have access to and check the TPM (tokens per minute) limit to make a choice here. As a rough estimate, your TPM limit divided by 150,000 is a somewhat reasonable estimate.{RESET}"
    ).ok();
    let max_concurrent_requests =
        read_number(input, output, config.max_concurrent_requests as u8, (1, 20));
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

    writeln!(output, "You've set up your config!").ok();
    writeln!(
        output,
        "{DIM_TEXT}You can change these values any time by editing ~/.config/zqa/config.toml.\n"
    )
    .ok();
    writeln!(
        output,
        "Next, you will likely want to set up your embeddings by typing /process in the prompt that follows. Note that this will take a while! If you don't have time, you can quit now, reopen the CLI later, and run it then.\n"
    ).ok();
    writeln!(
        output,
        "Since your API keys are stored in plain-text, make sure to never commit ~/.config/zqa/config.toml without first deleting the keys. The recommended setup is to set the values in the TOML file blank and use .env files where you need them with the keys. As an additional security measure, consider running `chmod 600 ~/.config/zqa/config.toml`.{RESET}"
    ).ok();
    Ok(())
}

#[cfg(test)]
mod tests {
    use chrono::Local;
    use clap::builder::OsStr;
    use std::fs;
    use std::io::Cursor;
    use std::path::Component;
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, USER_ROLE};

    use crate::config::Config;
    use crate::state::{
        SavedChatHistory, get_config_dir, get_conversation_history, get_state_dir, oobe,
        save_conversation,
    };

    #[test]
    fn test_get_state_dir() {
        let state_dir = get_state_dir();

        assert!(state_dir.is_ok());
        let state_dir = state_dir.unwrap();

        let mut components = state_dir.components();
        assert!(components.next_back() == Some(Component::Normal(&OsStr::from("zqa"))));
        assert!(components.next_back() == Some(Component::Normal(&OsStr::from("state"))));
    }

    #[test]
    fn test_get_conversation_history() {
        let state_dir = get_state_dir().unwrap();

        if state_dir.exists() {
            let _ = fs::remove_dir_all(state_dir);
        }

        assert_eq!(get_conversation_history(), Ok(None));
    }

    #[test]
    fn test_save_conversation_creates_dirs() {
        let state_dir = get_state_dir().unwrap();

        if state_dir.exists() {
            let _ = fs::remove_dir_all(state_dir);
        }

        let conversation = SavedChatHistory {
            date: Local::now(),
            title: "foo".into(),
            history: vec![ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text("Hello!".into())],
            }],
        };

        let result = save_conversation(&conversation);
        let state_dir = get_state_dir().unwrap();
        assert!(result.is_ok());
        assert!(state_dir.exists());
    }

    #[test]
    fn test_get_conversation_history_works() {
        let state_dir = get_state_dir().unwrap();

        if state_dir.exists() {
            let _ = fs::remove_dir_all(state_dir);
        }

        let conversation = SavedChatHistory {
            date: Local::now(),
            title: "foo".into(),
            history: vec![ChatHistoryItem {
                role: USER_ROLE.into(),
                content: vec![ChatHistoryContent::Text("Hello!".into())],
            }],
        };

        let result = save_conversation(&conversation);
        assert!(result.is_ok());

        let conversations = get_conversation_history();
        assert!(conversations.is_ok());

        let conversations = conversations.unwrap();
        assert!(conversations.is_some());

        let conversations = conversations.unwrap();
        assert_eq!(conversations.len(), 1);
        assert_eq!(conversations[0].title, "foo");
        assert_eq!(conversations[0].history.len(), 1);
        assert_eq!(conversations[0].history[0].role, USER_ROLE);
        assert_eq!(conversations[0].history[0].content.len(), 1);
        assert_eq!(
            conversations[0].history[0].content[0],
            ChatHistoryContent::Text("Hello!".into())
        );
    }

    #[test]
    fn test_oobe_setup() {
        // Setup config dir
        let config_dir = get_config_dir().unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.toml");
        if config_path.exists() {
            fs::remove_file(&config_path).unwrap();
        }

        // Simulate input: Yes to setup, default provider (Anthropic), API Key, Default Embedding (Voyage), API Key, Default Reranker (Voyage), API Key, Requests
        let input_str = "y\na\nmy-api-key\nv\nmy-voyage-key\nv\n\n\n";
        let mut input = Cursor::new(input_str);
        let mut output = Vec::new();

        let result = oobe(&mut input, &mut output);
        assert!(result.is_ok());

        let output_str = String::from_utf8(output).unwrap();
        assert!(output_str.contains("Let's set up your config."));
        assert!(output_str.contains("You've set up your config!"));

        let config = Config::from_file(&config_path).unwrap();
        assert_eq!(config.model_provider, "anthropic");
        assert_eq!(config.embedding_provider, "voyageai");
        assert_eq!(
            config.anthropic.unwrap().api_key,
            Some("my-api-key".to_string())
        );
    }

    #[test]
    fn test_oobe_respects_existing_config() {
        // Setup existing config
        let config_dir = get_config_dir().unwrap();
        fs::create_dir_all(&config_dir).unwrap();
        let config_path = config_dir.join("config.toml");
        let mut initial_config = Config::default();
        initial_config.model_provider = "openai".to_string();
        initial_config.openai.as_mut().unwrap().api_key = Some("existing-key".to_string());
        fs::write(
            &config_path,
            toml::to_string_pretty(&initial_config).unwrap(),
        )
        .unwrap();

        // Simulate input: Yes to setup (default N so we type y), then confirm defaults
        // Input: y (setup), Enter (accept OpenAI default), Enter (keep existing key), Enter (accept Voyage default), Enter (accept default key), Enter (accept default reranker), Enter (accept default key), Enter (requests)
        let input_str = "y\n\n\n\n\n\n\n\n";
        let mut input = Cursor::new(input_str);
        let mut output = Vec::new();

        let result = oobe(&mut input, &mut output);
        assert!(result.is_ok());

        let output_str = String::from_utf8(output).unwrap();
        // Check that OpenAI was shown as default [O]
        assert!(output_str.contains("[O]penAI"));
        // Check that it offered to keep existing key
        assert!(output_str.contains("Press Enter to keep existing"));

        let config = Config::from_file(&config_path).unwrap();
        assert_eq!(config.model_provider, "openai");
        assert_eq!(
            config.openai.unwrap().api_key,
            Some("existing-key".to_string())
        );
    }
}
