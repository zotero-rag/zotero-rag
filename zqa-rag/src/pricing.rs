//! Provides pricing information for the various providers regardless of their capabilities. Values
//! are based on the [LiteLLM
//! repo](https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json), which
//! is cached. This module makes a few assumptions for caching behavior:
//!
//! * First, it assumes that generally speaking, model prices won't change. Historically this has
//!   been true, at least for mainstream models.
//! * Second, it assumes that any new model is added relatively quickly to this JSON file. We of
//!   course, do not directly control this, but we need *some* way to get pricing info.
//!
//! There isn't a real point in keeping our own table of this since (a) it's a chore to keep up to
//! date, and I don't want to be stuck updating this until the heat death of the universe (b) the
//! only time pricing matters is if the user already has an Internet connection, so we can also
//! grab this file.

use std::{
    ops::{Add, AddAssign},
    path::PathBuf,
    time::Duration,
};

use serde::{Deserialize, Serialize};
use tempfile::NamedTempFile;
use tokio::sync::OnceCell;

const LITELLM_PRICING_URL: &str =
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json";

static PRICING_CACHE: OnceCell<serde_json::Value> = OnceCell::const_new();

/// Usage information for a model. This does not include pricing estimates; for that information,
/// use [`ModelPricing::estimate_cost`].
///
/// One thing to note is that the Gemini API doesn't distinguish between cache reads and writes in
/// its response, see [docs]. For that case, we arbitrarily populate this value in
/// `input_cache_read` and set `input_cache_written` to 0.
///
/// [docs]: https://ai.google.dev/api/generate-content#UsageMetadata
#[derive(Debug, Default, Copy, Clone, Serialize, Deserialize)]
pub struct ModelUsage {
    /// Total input tokens used, including cache writes and reads. See
    /// the [OTel semantic
    /// conventions](https://github.com/open-telemetry/semantic-conventions-genai/blob/63f8200eee093730ce845d26ce2aafb621b0807e/docs/gen-ai/gen-ai-spans.md).
    pub input_tokens: u32,
    /// Input tokens written to cache
    pub input_cache_written: u32,
    /// Input tokens read from the cache
    pub input_cache_read: u32,
    /// Output tokens
    pub output_tokens: u32,
    /// Reasoning tokens generated
    pub reasoning_tokens: u32,
}

impl Add<ModelUsage> for ModelUsage {
    type Output = ModelUsage;

    fn add(self, rhs: ModelUsage) -> Self::Output {
        Self {
            input_tokens: self.input_tokens + rhs.input_tokens,
            input_cache_read: self.input_cache_read + rhs.input_cache_read,
            input_cache_written: self.input_cache_written + rhs.input_cache_written,
            output_tokens: self.output_tokens + rhs.output_tokens,
            reasoning_tokens: self.reasoning_tokens + rhs.reasoning_tokens,
        }
    }
}

impl AddAssign<ModelUsage> for ModelUsage {
    fn add_assign(&mut self, rhs: ModelUsage) {
        *self = *self + rhs;
    }
}

/// Per-token pricing for an AI model, in USD per token.
///
/// Pricing values sourced from official provider pages; see
/// <https://github.com/BerriAI/litellm/blob/main/model_prices_and_context_window.json>
/// for a regularly updated cross-provider reference.
#[derive(Debug, Copy, Clone)]
pub struct ModelPricing {
    /// Cost per one input (prompt) token, in USD.
    pub input_cost_per_token: f64,
    /// Cost per one input (prompt) token written to a cache with a 5m ttl.
    pub cache_write_cost_per_token: f64,
    /// Cost per one cached input (prompt) token, in USD.
    pub cached_input_cost_per_token: f64,
    /// Cost per one output (completion) token, in USD.
    pub output_cost_per_token: f64,
}

impl ModelPricing {
    /// Estimate the total USD cost for a given token usage.
    ///
    /// # Arguments
    ///
    /// * `input_tokens`  - Number of input (prompt) tokens consumed.
    /// * `output_tokens` - Number of output (completion) tokens produced.
    ///
    /// # Returns
    ///
    /// Estimated cost in USD.
    #[must_use]
    pub fn estimate_cost(&self, usage: ModelUsage) -> f64 {
        [
            (
                usage
                    .input_tokens
                    .saturating_sub(usage.input_cache_written)
                    .saturating_sub(usage.input_cache_read),
                self.input_cost_per_token,
            ),
            (usage.input_cache_written, self.cache_write_cost_per_token),
            (usage.input_cache_read, self.cached_input_cost_per_token),
            (usage.output_tokens, self.output_cost_per_token),
        ]
        .map(|(n, c)| f64::from(n) * c)
        .iter()
        .sum()
    }
}

/// Cache options for updating pricing information.
pub struct PricingCacheOptions {
    /// Path to the file where the file is stored. This does *not* need to actually exist on disk;
    /// if it doesn't, this will be used as the location to save the file. It is, of course,
    /// assumed that we have permissions to read from/write to this file, and that the directories
    /// leading to this filename exist. In particular, `get_model_pricing` will *not* perform a
    /// `mkdir -p` before creating this file.
    pub cache_path: PathBuf,
    /// Time-to-live for the cache (in s), before it is refreshed. If `None`, this is equivalent to
    /// infinity (i.e., it is *never* refreshed). This is likely to be a bad idea, but I won't stop
    /// you from shooting yourself in the foot (as long as you do it explicitly--this struct does
    /// not provide a `Default` impl).
    ///
    /// Technically, setting this to `None` conflicts with the case where `cache_path` itself
    /// doesn't exist; in this case, the file is still fetched, i.e., the `ttl` aims to end up in a
    /// situation where the file exists, however stale. If you do not want any caching, you should
    /// instead set `cache_opts` itself to `None` in `get_model_pricing`.
    pub ttl: Option<usize>,
}

impl PricingCacheOptions {
    /// Determines if the cache file should be fetched from the LiteLLM repo. In any of the failure
    /// cases, this returns `false`, since an attempt to fetch would fail. However, the fact that
    /// this returns any value at all should not be taken as indication that the file itself
    /// exists; callers should still verify that `cache_path` exists after calling this.
    ///
    /// This function returning `true` also does not mean the file exists and can be written to;
    /// that would introduce a class of [TOCTOU] bugs. All that said, this function returning true
    /// indicates that *all* of the following are true:
    ///
    /// * `self.cache_path` is a path that ends with a JSON filename.
    /// * `self.cache_path` definitely exists or definitely doesn't; more precisely, it is not in a
    ///   condition whose existence can neither be confirmed nor denied. See [`std::fs::exists`]
    ///   for details.
    /// * One of the following is true:
    ///     * `self.cache_path` does not exist *at the time of checking*.
    ///     * `self.cache_path` exists *at the time of checking*, and `self.ttl` contains a value
    ///       greater than the time elapsed since the file's last modified time (mtime).
    ///
    /// [TOCTOU]: https://doc.rust-lang.org/stable/std/fs/index.html#time-of-check-to-time-of-use-toctou
    pub(crate) fn should_fetch(&self) -> bool {
        let extension = self.cache_path.extension();

        if extension.is_none_or(|ext| !ext.eq_ignore_ascii_case("json")) {
            // The file extension is not a JSON, so we act like a coward and assume this
            // isn't our cache to maintain; i.e., we should *not* attempt to fetch and replace
            // this file.
            return false;
        }

        match self.cache_path.try_exists() {
            Err(_) => return false,
            Ok(false) => return true,
            _ => {}
        }

        let Some(ttl) = self.ttl else {
            return false;
        };

        self.cache_path
            .metadata()
            .and_then(|meta| meta.modified())
            .ok()
            .and_then(|mtime| mtime.elapsed().ok())
            .is_some_and(|age| age > Duration::from_secs(ttl as u64))
    }
}

/// Look up pricing for a `(provider, model)` pair.
///
/// # Arguments
///
/// * `provider` - The name of the provider. Must be one of the values in any of the `*Provider`
///   enums in [`crate::capabilities`].
/// * `model` - The model used.
/// * `cache_opts` - An optional set of options if a cache should be used. If `None`, a cache is
///   not used. If `Some(opts)`, then `opts.cache_path` is used as the file for the cache, and if
///   the file is out-of-date (based on `opts.ttl`) or does not exist, it is fetched.
///
/// # Returns
///
/// * `None` when pricing is unknown (unknown future model, etc.).
/// * `Some` with both prices set to `0.0` for local/free providers (Ollama).
pub async fn get_model_pricing(
    provider: &str,
    model: &str,
    cache_opts: Option<PricingCacheOptions>,
) -> Option<ModelPricing> {
    // Keep the `NamedTempFile` alive so the underlying file is not deleted before we read it.
    let _tempfile;
    let (path, should_fetch) = match cache_opts {
        None => {
            let f = NamedTempFile::new().ok()?;
            let p = f.path().to_path_buf();
            _tempfile = Some(f);
            (p, true)
        }
        Some(opt) => {
            let fetch = opt.should_fetch();
            (opt.cache_path, fetch)
        }
    };

    if should_fetch {
        let fetched = match reqwest::get(LITELLM_PRICING_URL).await {
            Ok(resp) => resp.bytes().await,
            Err(e) => Err(e),
        };
        match fetched {
            Ok(bytes) => {
                if let Err(e) = tokio::fs::write(&path, &bytes).await {
                    log::warn!("Failed to write pricing cache file: {e}");
                }
            }
            Err(e) => {
                log::warn!("Failed to fetch pricing file, using cache if available: {e}");
            }
        }
    }

    if provider == "ollama" {
        return Some(ModelPricing {
            input_cost_per_token: 0.0,
            cache_write_cost_per_token: 0.0,
            cached_input_cost_per_token: 0.0,
            output_cost_per_token: 0.0,
        });
    }

    let json = PRICING_CACHE
        .get_or_try_init(|| async {
            let content = tokio::fs::read_to_string(&path)
                .await
                .map_err(|e| e.to_string())?;
            serde_json::from_str::<serde_json::Value>(&content).map_err(|e| e.to_string())
        })
        .await
        .ok()?;

    // LiteLLM keys are `model` for most models; fall back to `provider/name`, which is the case
    // for OpenRouter.
    let fallback_key = format!("{provider}/{model}");
    let entry = json.get(model).or_else(|| json.get(&fallback_key))?;

    let input_cost = entry.get("input_cost_per_token")?.as_f64()?;
    let input_cache_read_cost = entry
        .get("cache_read_input_token_cost")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let input_cache_write_cost = entry
        .get("cache_creation_input_token_cost")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(0.0);
    let output_cost = entry.get("output_cost_per_token")?.as_f64()?;

    Some(ModelPricing {
        input_cost_per_token: input_cost,
        cached_input_cost_per_token: input_cache_read_cost,
        cache_write_cost_per_token: input_cache_write_cost,
        output_cost_per_token: output_cost,
    })
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use tempfile::NamedTempFile;

    use super::*;

    #[test]
    fn test_estimate_cost_basic() {
        let usage = ModelUsage {
            input_tokens: 1000,
            input_cache_written: 0,
            input_cache_read: 0,
            output_tokens: 500,
            reasoning_tokens: 0,
        };
        let pricing = ModelPricing {
            input_cost_per_token: 0.000_003,
            output_cost_per_token: 0.000_015,
            cached_input_cost_per_token: 0.000_003_75,
            cache_write_cost_per_token: 0.000_000_3,
        };
        let cost = pricing.estimate_cost(usage);
        let expected = 0.0105;

        // Relative tolerance
        assert!((cost - expected).abs() < expected * f64::EPSILON * 10.0);
    }

    #[test]
    fn test_estimate_cost_zero_tokens() {
        let pricing = ModelPricing {
            input_cost_per_token: 0.000_003,
            output_cost_per_token: 0.000_015,
            cached_input_cost_per_token: 0.000_003_75,
            cache_write_cost_per_token: 0.000_000_3,
        };

        assert!(pricing.estimate_cost(ModelUsage::default()) < f64::EPSILON);
    }

    #[test]
    fn test_estimate_cost_output_only() {
        let pricing = ModelPricing {
            input_cost_per_token: 0.000_001,
            output_cost_per_token: 0.000_002,
            cached_input_cost_per_token: 0.000_003_75,
            cache_write_cost_per_token: 0.000_000_3,
        };
        let usage = ModelUsage {
            input_tokens: 0,
            input_cache_written: 0,
            input_cache_read: 0,
            output_tokens: 1000,
            reasoning_tokens: 0,
        };

        let cost = pricing.estimate_cost(usage);
        let expected = 0.002;

        assert!((cost - expected).abs() < expected * f64::EPSILON * 10.0);
    }

    #[test]
    fn test_estimate_cost_all_fields() {
        let pricing = ModelPricing {
            input_cost_per_token: 0.000_003,
            output_cost_per_token: 0.000_015,
            cached_input_cost_per_token: 0.000_003_75,
            cache_write_cost_per_token: 0.000_000_3,
        };
        let usage = ModelUsage {
            input_tokens: 2000,
            input_cache_written: 500,
            input_cache_read: 500,
            output_tokens: 1000,
            reasoning_tokens: 0,
        };

        let cost = pricing.estimate_cost(usage);
        let expected = 0.020_025;

        assert!((cost - expected).abs() < expected * f64::EPSILON * 10.0);
    }

    #[test]
    fn test_estimate_cost_with_mostly_cached_tokens() {
        let pricing = ModelPricing {
            input_cost_per_token: 0.000_003,
            output_cost_per_token: 0.000_015,
            cached_input_cost_per_token: 0.000_003_75,
            cache_write_cost_per_token: 0.000_000_3,
        };
        let usage = ModelUsage {
            input_tokens: 2000,
            input_cache_written: 500,
            input_cache_read: 1200,
            output_tokens: 1000,
            reasoning_tokens: 0,
        };

        let cost = pricing.estimate_cost(usage);
        let expected = 0.02055;

        assert!((cost - expected).abs() < expected * f64::EPSILON * 10.0);
    }

    #[test]
    fn test_should_fetch_no_extension_returns_false() {
        let opts = PricingCacheOptions {
            cache_path: PathBuf::from("/tmp/pricingcache_no_ext"),
            ttl: Some(3600),
        };

        assert!(!opts.should_fetch());
    }

    #[test]
    fn test_should_fetch_non_json_extension_returns_false() {
        let opts = PricingCacheOptions {
            cache_path: PathBuf::from("/tmp/pricingcache.txt"),
            ttl: Some(3600),
        };

        assert!(!opts.should_fetch());
    }

    #[test]
    fn test_should_fetch_nonexistent_file_returns_true() {
        let opts = PricingCacheOptions {
            cache_path: PathBuf::from("/tmp/zqa_pricing_does_not_exist_abc123.json"),
            ttl: Some(3600),
        };

        assert!(opts.should_fetch());
    }

    #[test]
    fn test_should_fetch_no_ttl_existing_file_returns_false() {
        let mut f = NamedTempFile::with_suffix(".json").unwrap();
        f.write_all(b"{}").unwrap();
        let opts = PricingCacheOptions {
            cache_path: f.path().to_path_buf(),
            ttl: None,
        };

        assert!(!opts.should_fetch());
    }

    #[test]
    fn test_should_fetch_fresh_file_returns_false() {
        let mut f = NamedTempFile::with_suffix(".json").unwrap();
        f.write_all(b"{}").unwrap();
        let opts = PricingCacheOptions {
            cache_path: f.path().to_path_buf(),
            ttl: Some(usize::MAX),
        };

        assert!(!opts.should_fetch());
    }

    const SAMPLE_JSON: &str = r#"{
        "gpt-5.4": {
            "input_cost_per_token": 0.0000025,
            "output_cost_per_token": 0.00001
        },
        "openrouter/mistral-7b": {
            "input_cost_per_token": 0.0000001,
            "output_cost_per_token": 0.0000002
        }
    }"#;

    /// Write `json` to a fresh `.json` tempfile and return cache options that will
    /// never trigger a re-fetch (ttl = usize::MAX on a file that was just written).
    fn make_cache(json: &str) -> (NamedTempFile, PricingCacheOptions) {
        let mut f = NamedTempFile::with_suffix(".json").unwrap();
        f.write_all(json.as_bytes()).unwrap();
        let path = f.path().to_path_buf();
        let opts = PricingCacheOptions {
            cache_path: path,
            ttl: Some(usize::MAX),
        };
        (f, opts)
    }

    #[tokio::test]
    async fn test_get_pricing_ollama_returns_zero() {
        let (_, opts) = make_cache(SAMPLE_JSON);
        let p = get_model_pricing("ollama", "llama3.2", Some(opts))
            .await
            .unwrap();

        assert!(p.input_cost_per_token < f64::EPSILON);
        assert!(p.output_cost_per_token < f64::EPSILON);
    }

    #[tokio::test]
    async fn test_get_pricing_known_model() {
        // NOTE: This cannot be changed to `_`! That would clean up the tempfile.
        let (_f, opts) = make_cache(SAMPLE_JSON);
        let p = get_model_pricing("openai", "gpt-5.4", Some(opts))
            .await
            .unwrap();
        let expected_input_cost = 0.000_002_5;
        let expected_output_cost = 0.00001;

        assert!(
            (p.input_cost_per_token - expected_input_cost).abs()
                < expected_input_cost * f64::EPSILON * 10.0
        );
        assert!(
            (p.output_cost_per_token - expected_output_cost).abs()
                < expected_output_cost * f64::EPSILON * 10.0
        );
    }

    #[tokio::test]
    async fn test_get_pricing_fallback_provider_slash_model_key() {
        let (_f, opts) = make_cache(SAMPLE_JSON);
        let p = get_model_pricing("openrouter", "mistral-7b", Some(opts))
            .await
            .unwrap();
        let expected_input_cost = 0.000_000_1;
        let expected_output_cost = 0.000_000_2;

        assert!(
            (p.input_cost_per_token - expected_input_cost).abs()
                < expected_input_cost * f64::EPSILON * 10.0
        );
        assert!(
            (p.output_cost_per_token - expected_output_cost).abs()
                < expected_output_cost * f64::EPSILON * 10.0
        );
    }

    #[tokio::test]
    async fn test_get_pricing_unknown_model_returns_none() {
        let (_f, opts) = make_cache(SAMPLE_JSON);
        let result = get_model_pricing("openai", "foobar", Some(opts)).await;

        assert!(result.is_none());
    }

    #[tokio::test]
    async fn test_get_pricing_estimate_round_trip() {
        let (_f, opts) = make_cache(SAMPLE_JSON);
        let p = get_model_pricing("openai", "gpt-5.4", Some(opts))
            .await
            .unwrap();
        let usage = ModelUsage {
            input_tokens: 100,
            input_cache_written: 0,
            input_cache_read: 0,
            output_tokens: 50,
            reasoning_tokens: 0,
        };

        let cost = p.estimate_cost(usage);
        let expected = 100.0 * 0.000_002_5 + 50.0 * 0.00001;

        assert!((cost - expected).abs() < expected * f64::EPSILON * 10.0);
    }
}
