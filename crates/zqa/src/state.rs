//!  NOTE: This is not state management! The `state` module is actually a way to interact with `XDG_STATE_HOME`.

use std::cmp::Reverse;
use std::fs;
use std::io;
use std::ops::{Add, AddAssign};
use std::path::PathBuf;

use chrono::{DateTime, Local};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio::sync::{mpsc, oneshot};
use zqa_rag::capabilities::{EmbeddingProvider, ModelProvider, RerankerProvider};
use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem};
use zqa_rag::pricing::{ModelUsage, PricingCacheOptions, get_model_pricing};
use zqa_rag::providers::ProviderId;

use crate::common::Context;
use crate::config::{BaseDirError, Config, get_config_dir};
use crate::io::{EngineEvent, StyledSpan, TextStyle};

/// Errors that can occur when interacting with the state directory.
#[derive(PartialEq, Debug, Error)]
pub enum StateError {
    #[error("Failed to get home directory.")]
    DirectoryError,
    #[error("Failed to save first run information.")]
    FileWriteError,
    #[error("Serialization error: {0}")]
    SerializationError(String),
    #[error("Failed to emit event: {0}")]
    EventEmitError(String),
    #[error("Failed to receive a reply: {0}")]
    ChannelRecvError(#[from] oneshot::error::RecvError),
    #[error("Other error: {0}")]
    Other(String),
}

impl From<toml::ser::Error> for StateError {
    fn from(value: toml::ser::Error) -> Self {
        Self::SerializationError(format!("Serialization error: {value}"))
    }
}

impl From<Box<mpsc::error::SendError<EngineEvent>>> for StateError {
    fn from(value: Box<mpsc::error::SendError<EngineEvent>>) -> Self {
        Self::EventEmitError(value.to_string())
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
/// `~/.local/state/zqa`. On Windows, this also returns `~/.local/state/zqa`, except that `$HOME` is
/// typically C:\Users\<username>.
///
/// # Errors
///
/// * `StateError::DirectoryError` if the user's base directory could not be obtained. See
///   `directories::BaseDirError` for when this can occur.
pub(crate) fn get_state_dir() -> Result<PathBuf, StateError> {
    if let Ok(dir) = std::env::var("ZQA_STATE_DIR") {
        return Ok(PathBuf::from(dir));
    }

    let base_dir = directories::BaseDirs::new().ok_or(StateError::DirectoryError)?;

    let alt_state_dir = base_dir.home_dir().join(".local").join("state");
    let state_dir = base_dir.state_dir().unwrap_or(&alt_state_dir);

    Ok(state_dir.join("zqa"))
}

/// For embedding and reranking, where we currently only have characters, estimate the cost if
/// possible and return the usage.
///
/// Cost lookup uses the LiteLLM JSON catalog of models, and uses `{provider_name}/{model}`.
///
/// TODO: We should probably just use the provider-sent usage directly instead of the approximation
/// here.
///
/// # Arguments
///
/// * `provider_name` - The provider name to use in the lookup.
/// * `model` - The model string to use in the lookup.
/// * `chars` - The character count to use in the estimate.
///
/// # Returns
///
/// A [`UsageMetadata`] with estimated cost if the model was found.
pub(crate) async fn get_model_usage(provider_name: &str, model: &str, chars: u64) -> UsageMetadata {
    let mut usage = UsageMetadata {
        #[allow(clippy::cast_possible_truncation)]
        input_tokens: (chars / 4) as u32,
        ..Default::default()
    };

    if let Some(ref p) = get_model_pricing(provider_name, model, None).await {
        #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
        let cost = (p.estimate_cost(usage.into()) * 100.0) as u32;
        usage.estimated_cost += cost;
    }

    usage
}

/// Metadata in a saved conversation.
#[derive(Copy, Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct UsageMetadata {
    /// Input tokens used
    pub input_tokens: u32,
    /// Input tokens read from
    pub input_cache_read: u32,
    /// Input tokens written to cache
    pub input_cache_written: u32,
    /// Output tokens used
    pub output_tokens: u32,
    /// Reasoning tokens used
    pub reasoning_tokens: u32,
    /// Estimated cost so far, in U.S. cents
    pub estimated_cost: u32,
}

impl From<UsageMetadata> for ModelUsage {
    fn from(val: UsageMetadata) -> Self {
        ModelUsage {
            input_tokens: val.input_tokens,
            input_cache_written: val.input_cache_written,
            input_cache_read: val.input_cache_read,
            output_tokens: val.output_tokens,
            reasoning_tokens: val.reasoning_tokens,
        }
    }
}

impl Add<UsageMetadata> for UsageMetadata {
    type Output = UsageMetadata;

    fn add(self, rhs: UsageMetadata) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            input_cache_read: self.input_cache_read + rhs.input_cache_read,
            input_cache_written: self.input_cache_written + rhs.input_cache_written,
            output_tokens: self.output_tokens + rhs.output_tokens,
            estimated_cost: self.estimated_cost + rhs.estimated_cost,
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
        }
    }
}

impl AddAssign<UsageMetadata> for UsageMetadata {
    fn add_assign(&mut self, rhs: UsageMetadata) {
        *self = *self + rhs;
    }
}

impl UsageMetadata {
    pub(crate) async fn from_rag_usage(
        usage: &[ModelUsage],
        provider: ModelProvider,
        model: &str,
    ) -> Self {
        let model_pricing = match get_state_dir() {
            Ok(dir) if dir.exists() || tokio::fs::create_dir_all(&dir).await.is_ok() => {
                get_model_pricing(
                    provider.as_str(),
                    model,
                    Some(PricingCacheOptions {
                        cache_path: dir.join("pricing_cache.json"),
                        ttl: Some(60 * 60 * 24), // 1 day
                    }),
                )
                .await
            }
            _ => None,
        };

        // Prompt-size pricing tiers apply per request, so each request is priced on its own.
        let estimated_cost = model_pricing.map_or(0.0, |p| {
            usage.iter().map(|&request| p.estimate_cost(request)).sum()
        });
        let total: ModelUsage = usage.iter().sum();

        Self {
            input_tokens: total.input_tokens,
            input_cache_read: total.input_cache_read,
            input_cache_written: total.input_cache_written,
            output_tokens: total.output_tokens,
            reasoning_tokens: total.reasoning_tokens,
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            estimated_cost: (estimated_cost * 100.0) as u32,
        }
    }
}

/// A chat history that is stored in the user's state directory. This wraps a
/// `Vec<ChatHistoryItem>` under the hood, but also includes some metadata such as when the chat
/// occurred.
#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct SavedChatHistory {
    /// The chat history
    pub history: Vec<ChatHistoryItem>,
    /// When this conversation occurred
    pub date: DateTime<Local>,
    /// A brief title
    pub title: String,
    /// Tracked usage metadata so far
    #[serde(default)]
    pub usage: UsageMetadata,
}

impl SavedChatHistory {
    /// Return the latest non-empty text message as a conversation preview.
    ///
    /// Reasoning and tool-call content are skipped because they are implementation details rather
    /// than useful conversation summaries.
    ///
    /// # Returns
    ///
    /// The trimmed message text, or `None` when the conversation has no text content.
    #[must_use]
    pub fn preview(&self) -> Option<&str> {
        self.history
            .iter()
            .rev()
            .flat_map(|item| item.content.iter().rev())
            .find_map(|content| match content {
                ChatHistoryContent::Text(text) if !text.trim().is_empty() => Some(text.trim()),
                _ => None,
            })
    }
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
pub fn get_conversation_history() -> Result<Option<Vec<SavedChatHistory>>, StateError> {
    let state_dir = get_state_dir()?;
    let conversations_dir = state_dir.join("conversations");

    if !conversations_dir.exists() {
        fs::create_dir_all(&conversations_dir)?;
        return Ok(None);
    }

    let mut histories = conversations_dir
        .read_dir()?
        .filter_map(std::result::Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
        .filter_map(|entry| fs::read_to_string(entry.path()).ok())
        .filter_map(|s| serde_json::from_str::<SavedChatHistory>(&s).ok())
        .collect::<Vec<_>>();
    histories.sort_by_key(|history| Reverse(history.date));

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

/// The model providers offered during setup, in the order they are shown.
const MODEL_PROVIDERS: [(&str, ModelProvider); 5] = [
    ("Anthropic", ModelProvider::Anthropic),
    ("Ollama", ModelProvider::Ollama),
    ("OpenAI", ModelProvider::OpenAI),
    ("Gemini", ModelProvider::Gemini),
    ("OpenRouter", ModelProvider::OpenRouter),
];

/// The embedding providers offered during setup, in the order they are shown.
const EMBEDDING_PROVIDERS: [(&str, EmbeddingProvider); 4] = [
    ("Cohere", EmbeddingProvider::Cohere),
    ("Gemini", EmbeddingProvider::Gemini),
    ("OpenAI", EmbeddingProvider::OpenAI),
    ("Voyage AI", EmbeddingProvider::VoyageAI),
];

/// The reranking providers offered during setup, in the order they are shown.
const RERANKER_PROVIDERS: [(&str, RerankerProvider); 2] = [
    ("Cohere", RerankerProvider::Cohere),
    ("Voyage AI", RerankerProvider::VoyageAI),
];

/// Ask the user to pick one of `choices` and return the value paired with the chosen label.
///
/// # Arguments
///
/// * `ctx` - The context whose event sender carries the prompt.
/// * `message` - The question shown above the labels.
/// * `choices` - Labels to show, each paired with the value returned if the user picks it.
/// * `default` - The 0-based index picked when the user just presses Enter.
///
/// # Returns
///
/// The value paired with the chosen label.
///
/// # Errors
///
/// * `StateError::EventEmitError` if the prompt could not be sent.
/// * `StateError::ChannelRecvError` if the consumer dropped the prompt without replying, which
///   includes a context with no event sender.
/// * `StateError::Other` if the reply is not an index into `choices`.
async fn choose<T: Copy>(
    ctx: &Context,
    message: &str,
    choices: &[(&str, T)],
    default: usize,
) -> Result<T, StateError> {
    let (reply, answer) = oneshot::channel();

    ctx.emit(EngineEvent::Choose {
        message: message.into(),
        options: choices
            .iter()
            .map(|(label, _)| (*label).to_owned())
            .collect(),
        reply,
        default,
    })
    .await?;

    let index = answer.await?;

    choices
        .get(index)
        .map(|(_, value)| *value)
        .ok_or_else(|| StateError::Other(format!("Invalid choice {index} for {message:?}")))
}

/// Return the API key field in `config` for `provider`, or `None` if `provider` doesn't use a key.
///
/// # Arguments
///
/// * `config` - The config being built.
/// * `provider` - The provider whose key field to return.
///
/// # Returns
///
/// The provider's `api_key` field, or `None` for keyless providers such as Ollama.
///
/// # Panics
///
/// If the config section for `provider` is `None`. [`Config::default`] fills every section, so
/// this can't happen for the config [`oobe`] builds.
fn api_key_slot(config: &mut Config, provider: ProviderId) -> Option<&mut Option<String>> {
    let api_key = match provider {
        ProviderId::Anthropic => config.anthropic.as_mut().map(|c| &mut c.api_key),
        ProviderId::OpenAI => config.openai.as_mut().map(|c| &mut c.api_key),
        ProviderId::OpenRouter => config.openrouter.as_mut().map(|c| &mut c.api_key),
        ProviderId::Gemini => config.gemini.as_mut().map(|c| &mut c.api_key),
        ProviderId::Cohere => config.cohere.as_mut().map(|c| &mut c.api_key),
        ProviderId::VoyageAI => config.voyageai.as_mut().map(|c| &mut c.api_key),
        _ => return None,
    };

    Some(api_key.expect("`Config::default()` fills every provider section"))
}

/// Ask for `provider`'s API key and store it in `config`.
///
/// A provider can fill more than one role, such as Gemini for both generation and embeddings. The
/// user is asked for each provider's key once: `asked` records the providers already asked, and
/// this returns without a prompt for those and for keyless providers. Surrounding whitespace is
/// trimmed from the key.
///
/// # Arguments
///
/// * `ctx` - The context whose event sender carries the prompt.
/// * `config` - The config that receives the key.
/// * `asked` - Providers whose key was already asked for. `provider` is added to it.
/// * `provider` - The provider whose key to ask for.
/// * `role` - The role shown in the prompt, such as "model" or "embedding".
///
/// # Errors
///
/// * `StateError::EventEmitError` if the prompt could not be sent.
/// * `StateError::ChannelRecvError` if the consumer dropped the prompt without replying.
async fn ask_api_key(
    ctx: &Context,
    config: &mut Config,
    asked: &mut Vec<ProviderId>,
    provider: ProviderId,
    role: &str,
) -> Result<(), StateError> {
    if asked.contains(&provider) {
        return Ok(());
    }

    let Some(api_key) = api_key_slot(config, provider) else {
        return Ok(());
    };

    let (reply, answer) = oneshot::channel();

    ctx.emit(EngineEvent::Secret {
        message: format!("Enter your {role} provider's API key: "),
        reply,
    })
    .await?;

    *api_key = Some(answer.await?.trim().to_owned());
    asked.push(provider);

    ctx.emit(EngineEvent::Text {
        message: "\n".into(),
    })
    .await?;

    Ok(())
}

/// Set up the out-of-box experience (OOBE) for the application.
///
/// This function uses reasonable defaults for most config attributes. The goal here is not to be meticulous, but
/// rather to get the user to a position where they can start using the application, assuming they have the API
/// keys for the choices they make. For most providers, we use the defaults provided by the `rag` crate; the onus
/// is on that to provide reasonable defaults. Fortunately, I am the author of both :) (and I am reasonable, usually).
///
/// The questions are sent as prompt events on `ctx`, so a consumer must be rendering events and replying while this
/// runs. The config file is written only after every question is answered; if the user declines setup, nothing is
/// written.
///
/// # Arguments
///
/// * `ctx` - The context whose event sender carries the prompts.
///
/// # Errors
///
/// * `StateError::DirectoryError` if we could not get the config directory.
/// * `StateError::SerializationError` if serialization to TOML failed.
/// * `StateError::EventEmitError` if emitting events fails.
/// * `StateError::ChannelRecvError` if a prompt was dropped without a reply.
/// * `StateError::Other` if an invalid reply is received, or writing the config file fails.
#[allow(clippy::too_many_lines)]
pub(crate) async fn oobe(ctx: &Context) -> Result<(), StateError> {
    // Here, we don't load the env because that's directory-specific.
    let mut config = Config::default();

    let (reply, answer) = oneshot::channel();
    ctx.emit_multiple([
        EngineEvent::StyledText {
            spans: vec![
                StyledSpan {
                    text: "Let's set up your config.\n".into(),
                    style: TextStyle::Plain,
                },
                StyledSpan {
                    text: "Since this is your first run, you will set up a basic configuration. This should take a few minutes. You can always update these by editing ~/.config/zqa/config.toml directly. For most options, a default is shown; if that is okay, you can simply press Enter.\n\n".into(),
                    style: TextStyle::Secondary,
                },
            ],
        },
        // TODO: We need smarter defaults where if the user for some reason has a config file, we use that value. We
        // could skip users that have a *valid* config.
        EngineEvent::Confirm {
            message: "Would you like to set up your configuration?".into(),
            default: true,
            reply,
        },
    ])
    .await?;

    if !answer.await? {
        return Ok(());
    }

    let mut asked = Vec::new();

    config.model_provider = choose(
        ctx,
        "What model provider do you want to use?",
        &MODEL_PROVIDERS,
        0,
    )
    .await?;
    let model_id = ProviderId::from(&config.model_provider);
    ask_api_key(ctx, &mut config, &mut asked, model_id, "model").await?;

    config.embedding_provider = choose(
        ctx,
        "What provider do you want to use for embeddings?",
        &EMBEDDING_PROVIDERS,
        3,
    )
    .await?;
    let embedding_id = ProviderId::from(&config.embedding_provider);
    ask_api_key(ctx, &mut config, &mut asked, embedding_id, "embedding").await?;

    ctx.emit(EngineEvent::StyledText {
        spans: vec![StyledSpan {
            text: "In general, you will want to use the same provider as embeddings.\n".into(),
            style: TextStyle::Secondary,
        }],
    })
    .await?;

    let reranker_default = usize::from(config.embedding_provider == EmbeddingProvider::VoyageAI);
    let reranker_provider = choose(
        ctx,
        "What provider do you want to use for reranking results?",
        &RERANKER_PROVIDERS,
        reranker_default,
    )
    .await?;
    config.reranker_provider = Some(reranker_provider);
    ask_api_key(
        ctx,
        &mut config,
        &mut asked,
        ProviderId::from(&reranker_provider),
        "reranker",
    )
    .await?;

    // The 100k and 150k are somewhat napkin math-based, but pretty decent. In practice, embedding providers usually
    // have much higher TPM limits, so we're limited by the generation model provider's TPM. A half-decent estimate
    // for one paper is ~30k-50k tokens. For example, I have an 800k TPM with Claude, and a value of 5 works well,
    // though I haven't tested other values. For most users, the RPM will be far higher than necessary.
    ctx.emit(EngineEvent::StyledText {
        spans: vec![
            StyledSpan {
                text: "How many requests do you want to send at a time? [5]\n".into(),
                style: TextStyle::Plain,
            },
            StyledSpan {
                text: "A higher number can yield faster results, but can also result in being rate-limited. You should check what tier of API you have access to and check the TPM (tokens per minute) limit to make a choice here. As a rough estimate, your TPM limit divided by 150,000 is a somewhat reasonable estimate.\n".into(),
                style: TextStyle::Secondary,
            },
        ],
    })
    .await?;

    config.max_concurrent_requests = loop {
        let (reply, answer) = oneshot::channel();
        ctx.emit(EngineEvent::Line {
            message: None,
            reply,
        })
        .await?;

        let answer = answer.await?;
        let answer = answer.trim();
        if answer.is_empty() {
            break 5;
        }

        if let Ok(requests) = answer.parse::<usize>()
            && (1..20).contains(&requests)
        {
            break requests;
        }

        ctx.emit(EngineEvent::Warning {
            message: "Enter a number from 1 to 19.\n".into(),
        })
        .await?;
    };

    let config_dir = get_config_dir()?;
    let config_path = config_dir.join("config.toml");
    fs::create_dir_all(&config_dir)?;
    fs::write(config_path, toml::to_string_pretty(&config)?)?;

    ctx.emit(EngineEvent::StyledText {
        spans: vec![
            StyledSpan {
                text: "You've set up your config!\n".into(),
                style: TextStyle::Plain,
            },
            StyledSpan {
                text: "You can change these values any time by editing ~/.config/zqa/config.toml.\n\nNext, you will likely want to set up your embeddings by typing /process in the prompt that follows. Note that this will take a while! If you don't have time, you can quit now, reopen the CLI later, and run it then.\n\nSince your API keys are stored in plain-text, make sure to never commit ~/.config/zqa/config.toml without first deleting the keys. The recommended setup is to set the values in the TOML file blank and use .env files where you need them with the keys. As an additional security measure, consider running `chmod 600 ~/.config/zqa/config.toml`.\n\n".into(),
                style: TextStyle::Secondary,
            },
        ],
    })
    .await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::Component;

    use chrono::{Duration, Local};
    use clap::builder::OsStr;
    use serial_test::serial;
    use zqa_macros::test_ok;
    use zqa_rag::llm::base::{ChatHistoryContent, ChatHistoryItem, MessageRole};

    use crate::common::test_support::{create_test_context, respond_to_events};
    use crate::io::EngineEvent;
    use crate::state::{
        SavedChatHistory, UsageMetadata, get_conversation_history, get_state_dir, oobe,
        save_conversation,
    };

    #[test]
    #[serial]
    fn test_get_state_dir() {
        let state_dir = get_state_dir();

        test_ok!(state_dir);
        let state_dir = state_dir.unwrap();

        let mut components = state_dir.components();
        assert_eq!(
            components.next_back(),
            Some(Component::Normal(&OsStr::from("zqa")))
        );
        assert_eq!(
            components.next_back(),
            Some(Component::Normal(&OsStr::from("state")))
        );
    }

    #[test]
    #[serial]
    fn test_get_conversation_history() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let state_dir = get_state_dir().unwrap();

            if state_dir.exists() {
                let _ = fs::remove_dir_all(&state_dir);
            }

            assert_eq!(get_conversation_history(), Ok(None));
        });
    }

    #[test]
    #[serial]
    fn test_save_conversation_creates_dirs() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let conversation = SavedChatHistory {
                date: Local::now(),
                title: "foo".into(),
                history: vec![ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("Hello!".into())],
                }],
                usage: UsageMetadata::default(),
            };

            let result = save_conversation(&conversation);
            let state_dir = get_state_dir().unwrap();
            test_ok!(result);
            assert!(state_dir.exists());
        });
    }

    #[test]
    #[serial]
    fn test_get_conversation_history_works() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let conversation = SavedChatHistory {
                date: Local::now(),
                title: "foo".into(),
                history: vec![ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("Hello!".into())],
                }],
                usage: UsageMetadata {
                    input_tokens: 500,
                    input_cache_written: 0,
                    input_cache_read: 100,
                    output_tokens: 1000,
                    reasoning_tokens: 0,
                    estimated_cost: 5,
                },
            };

            let result = save_conversation(&conversation);
            test_ok!(result);

            let conversations = get_conversation_history();
            test_ok!(conversations);

            let conversations = conversations.unwrap();
            assert!(conversations.is_some());

            let conversations = conversations.unwrap();
            assert_eq!(conversations.len(), 1);
            assert_eq!(conversations[0].title, "foo");
            assert_eq!(conversations[0].history.len(), 1);
            assert_eq!(conversations[0].history[0].role, MessageRole::User);
            assert_eq!(conversations[0].history[0].content.len(), 1);
            assert_eq!(
                conversations[0].history[0].content[0],
                ChatHistoryContent::Text("Hello!".into())
            );
            assert_eq!(conversations[0].usage.input_tokens, 500);
            assert_eq!(conversations[0].usage.input_cache_read, 100);
            assert_eq!(conversations[0].usage.output_tokens, 1000);
            assert_eq!(conversations[0].usage.estimated_cost, 5);
        });
    }

    #[test]
    fn saved_conversation_preview_uses_latest_text() {
        let conversation = SavedChatHistory {
            date: Local::now(),
            title: "Preview test".into(),
            history: vec![
                ChatHistoryItem {
                    role: MessageRole::User,
                    content: vec![ChatHistoryContent::Text("Initial question".into())],
                },
                ChatHistoryItem {
                    role: MessageRole::Assistant,
                    content: vec![
                        ChatHistoryContent::Text("  Latest response  ".into()),
                        ChatHistoryContent::Reasoning("Internal reasoning".into()),
                    ],
                },
            ],
            usage: UsageMetadata::default(),
        };

        assert_eq!(conversation.preview(), Some("Latest response"));
    }

    #[test]
    #[serial]
    fn conversation_history_is_newest_first() {
        let temp_dir = tempfile::tempdir().unwrap();
        temp_env::with_var("ZQA_STATE_DIR", Some(temp_dir.path()), || {
            let now = Local::now();
            for (title, date) in [
                ("Older conversation", now - Duration::hours(1)),
                ("Newer conversation", now),
            ] {
                save_conversation(&SavedChatHistory {
                    date,
                    title: title.into(),
                    history: Vec::new(),
                    usage: UsageMetadata::default(),
                })
                .unwrap();
            }

            let conversations = get_conversation_history().unwrap().unwrap();
            let titles = conversations
                .iter()
                .map(|conversation| conversation.title.as_str())
                .collect::<Vec<_>>();

            assert_eq!(titles, ["Newer conversation", "Older conversation"]);
        });
    }

    #[tokio::test]
    #[serial]
    async fn test_oobe_interactive() {
        // Create temp directories
        let temp_home = std::env::temp_dir().join("zqa_test_home");
        let temp_config = std::env::temp_dir().join("zqa_test_config");
        fs::create_dir_all(&temp_home).unwrap();
        fs::create_dir_all(&temp_config).unwrap();

        temp_env::async_with_vars(
            [
                ("XDG_CONFIG_HOME", Some(temp_config.to_str().unwrap())),
                ("HOME", Some(temp_home.to_str().unwrap())),
            ],
            async {
                // Anthropic for generation, then Voyage AI for embeddings. The reranker takes its
                // default, which follows the embedding provider, so Voyage's key is asked once.
                let mut choices = [Some("Anthropic"), Some("Voyage AI"), None].into_iter();
                let mut secrets = ["sk-test-model\n", "  sk-test-embedding  "].into_iter();
                // The first answer is out of range, so the question is asked again.
                let mut lines = ["abc\n", "10\n"].into_iter();

                let mut ctx = create_test_context(Vec::new());
                let result = respond_to_events(
                    &mut ctx,
                    async |ctx| oobe(ctx).await,
                    |event| match event {
                        EngineEvent::Confirm { reply, .. } => reply.send(true).unwrap(),
                        EngineEvent::Choose {
                            options,
                            default,
                            reply,
                            ..
                        } => {
                            let choice = choices.next().unwrap();
                            let index = choice.map_or(default, |label| {
                                options.iter().position(|o| o == label).unwrap()
                            });
                            reply.send(index).unwrap();
                        }
                        EngineEvent::Secret { reply, .. } => {
                            reply.send(secrets.next().unwrap().into()).unwrap();
                        }
                        EngineEvent::Line { reply, .. } => {
                            reply.send(lines.next().unwrap().into()).unwrap();
                        }
                        _ => {}
                    },
                )
                .await;
                result.unwrap();

                assert_eq!(choices.next(), None);
                assert_eq!(secrets.next(), None);
                assert_eq!(lines.next(), None);

                // get_config_dir depends on directories crate which depends on HOME/XDG_CONFIG_HOME
                // We mocked them.
                let config_dir = crate::config::get_config_dir().unwrap();
                let config_path = config_dir.join("config.toml");
                assert!(config_path.exists());

                let content = fs::read_to_string(config_path).unwrap();
                assert!(content.contains("model_provider = \"anthropic\""));
                assert!(content.contains("embedding_provider = \"voyageai\""));
                assert!(content.contains("reranker_provider = \"voyageai\""));
                assert!(content.contains("api_key = \"sk-test-model\"\n"));
                assert!(content.contains("api_key = \"sk-test-embedding\"\n"));
                assert!(content.contains("max_concurrent_requests = 10"));
            },
        )
        .await;

        // Cleanup
        let _ = fs::remove_dir_all(temp_home);
        let _ = fs::remove_dir_all(temp_config);
    }
}
