//! Command handlers for document-related operations.

use std::{io::Write, path::Path, sync::Arc};

use zqa_rag::{llm::tools::Tool, providers::registry::provider_registry};

use crate::{
    cli::{commands::DocsCommand, errors::CLIError},
    common::Context,
    tools::documents::{DocumentsToolFactory, parse_user_document},
};

/// Given a path to a file, attempt to return a relative path, or return the full canonical path
/// string. This returns a relative path if either `path` is relative, or if the specified `path` is
/// in the cwd or a subdirectory thereof.
///
/// # Arguments
///
/// * `path` - Path reference to the file.
///
/// # Errors
///
/// * `CLIError::IOError` if the path could not be canonicalized.
pub(crate) fn get_document_session_key(path: &Path) -> Result<String, CLIError> {
    if path.is_relative() {
        return Ok(path.to_string_lossy().into_owned());
    }

    let canonical_path = path.canonicalize()?;

    if let Ok(cwd) = std::env::current_dir()
        && let Ok(canonical_cwd) = cwd.canonicalize()
        && let Ok(relative_path) = canonical_path.strip_prefix(&canonical_cwd)
    {
        return Ok(relative_path.to_string_lossy().into_owned());
    }

    Ok(canonical_path.file_name().map_or_else(
        || canonical_path.to_string_lossy().into_owned(),
        |name| name.to_string_lossy().into_owned(),
    ))
}

/// Get tools to work with user-imported documents.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// A list of tools from [`crate::tools::documents::DocumentsToolFactory`].
///
/// # Errors
///
/// * `LLMError::InvalidProviderError` if the provider is not supported
pub(super) fn get_user_document_tools<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
) -> Result<Vec<Box<dyn Tool>>, CLIError> {
    let imports = ctx.state.imports.clone();
    if imports.read()?.is_empty() {
        return Ok(Vec::new());
    }

    let embedding_config = ctx.config.get_embedding_config().ok_or_else(|| {
        CLIError::ConfigError(
            "No embedding config found; mentioned documents won't be imported.".into(),
        )
    })?;
    let reranker_config = ctx.config.get_reranker_config();

    let client = ctx
        .config
        .get_generation_config()
        .map(|c| provider_registry().create_llm(&c))
        .transpose()?;

    Ok(
        DocumentsToolFactory::new(&imports, embedding_config, reranker_config, client)
            .build_tools(),
    )
}

/// Extract file paths referenced as `@path.pdf` or `@"path with spaces.pdf"`.
///
/// # Arguments
///
/// * `query` - User query
///
/// # Returns
///
/// A list of file paths from the user's query.
pub(crate) fn get_document_mentions(query: &str) -> Vec<String> {
    let mut mentions = Vec::new();
    let mut cursor = 0;

    while let Some(at_offset) = query[cursor..].find('@') {
        let at_idx = cursor + at_offset;
        if at_idx > 0
            && query[..at_idx]
                .chars()
                .last()
                .is_some_and(|c| !c.is_whitespace())
        {
            cursor = at_idx + 1;
            continue;
        }

        let rest = &query[at_idx + 1..];
        // If the user has such a pathological filename that we'd need to look at escaping...
        // maybe PEBCAK.
        if let Some(rest) = rest.strip_prefix('"')
            && let Some(end_quote_idx) = rest.find('"')
        {
            let mention = &rest[..end_quote_idx];
            if !mention.is_empty() {
                mentions.push(mention.to_string());
            }

            // at_idx + 1 + 1 for the stripped " prefix.
            cursor = at_idx + 2 + end_quote_idx + 1;
        } else {
            let end_idx = rest.find(char::is_whitespace).unwrap_or(rest.len());
            let mention = &rest[..end_idx];
            if !mention.is_empty() {
                mentions.push(mention.to_string());
            }
            cursor = at_idx + 1 + end_idx;
        }
    }

    mentions
}

/// Import a document specified by a user-specified path into the current context.
///
/// # Arguments
///
/// * `ctx` - A `Context` object that contains CLI args and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
/// * `path` - A [`Path`] reference to the file.
///
/// # Returns
///
/// The key in the `ctx.state.imports` map.
///
/// # Errors
///
/// * `CLIError::IOError` if the path could not be canonicalized.
/// * `CLIError::ConfigError` if importing failed; this is usually due to some config error.
pub(super) fn import_document<O: Write, E: Write>(
    ctx: &mut Context<O, E>,
    path: &Path,
) -> Result<String, CLIError> {
    let key = get_document_session_key(path)?;
    let imports = Arc::clone(&ctx.state.imports);

    if imports.read()?.contains_key(&key) {
        return Ok(key);
    }

    // Do expensive work before taking the write lock.
    let document = Arc::new(
        parse_user_document(path)
            .map_err(|e| CLIError::ConfigError(format!("Failed to import {key}: {e}")))?,
    );

    imports.write()?.insert(key.clone(), document);

    Ok(key)
}

/// Execute a `/docs` subcommand against the current session's imported documents.
///
/// # Arguments
///
/// * `subcmd` - The parsed document subcommand to execute.
/// * `ctx` - A `Context` object that contains CLI state and objects that implement
///   [`std::io::Write`] for `stdout` and `stderr`.
///
/// # Returns
///
/// `Ok(())` if the subcommand completed successfully.
///
/// # Errors
///
/// Returns a [`CLIError`] if document state could not be accessed or output could not be written.
pub(crate) fn handle_docs_cmd<O, E>(
    subcmd: DocsCommand,
    ctx: &mut Context<O, E>,
) -> Result<(), CLIError>
where
    O: Write,
    E: Write,
{
    match subcmd {
        DocsCommand::Clear => {
            let mut imports = ctx.state.imports.write()?;
            imports.clear();
        }
        DocsCommand::Remove { key } => {
            let mut imports = ctx.state.imports.write()?;
            imports.remove(&key);
        }
        DocsCommand::List => {
            let imports = ctx.state.imports.read()?;
            let keys: Vec<String> = imports.keys().cloned().collect();

            if keys.is_empty() {
                writeln!(
                    &mut ctx.out,
                    "No documents currently in state. Use @ in a message to add some."
                )?;
                return Ok(());
            }

            writeln!(&mut ctx.out, "Available documents:")?;
            for key in &keys {
                writeln!(&mut ctx.out, "{key}")?;
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use zqa_macros::{test_eq, test_ok};

    use super::{get_document_mentions, get_document_session_key};

    #[test]
    fn test_get_document_mentions_unquoted() {
        let mentions = get_document_mentions("Compare @symbols.pdf with @subtables.pdf");
        test_eq!(mentions, vec!["symbols.pdf", "subtables.pdf"]);
    }

    #[test]
    fn test_get_document_mentions_quoted_spaces() {
        let mentions = get_document_mentions("Summarize @\"image 1.pdf\" for me");
        test_eq!(mentions, vec!["image 1.pdf"]);
    }

    #[test]
    fn test_get_document_mentions_ignores_email_like_text() {
        let mentions = get_document_mentions("email me at test@example.com about @symbols.pdf");
        test_eq!(mentions, vec!["symbols.pdf"]);
    }

    #[test]
    fn test_get_document_session_key_preserves_relative_input() {
        let original = std::path::Path::new("papers/image 1.pdf");
        let key = get_document_session_key(original);

        test_ok!(key);
        let key = key.unwrap();

        test_eq!(key, "papers/image 1.pdf");
    }
}
