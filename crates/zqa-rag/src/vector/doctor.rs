//! Similar to `rag::vector::checkhealth`, this provides utilities to aid end-users in
//! troubleshooting issues, providing suggestions where possible.

use crate::vector::checkhealth::{HealthCheckError, HealthCheckable};

/// A problem found by [`doctor`], and how to fix it if there is a known fix.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Diagnosis {
    /// What the health check observed, such as "database storage does not exist.".
    pub symptom: String,
    /// A suggested fix, if there is one.
    pub help: Option<String>,
}

impl Diagnosis {
    /// Build a diagnosis.
    ///
    /// # Arguments
    ///
    /// * `symptom` - What the health check observed.
    /// * `help` - A suggested fix, if there is one.
    ///
    /// # Returns
    ///
    /// The diagnosis.
    fn new(symptom: &str, help: Option<&str>) -> Self {
        Self {
            symptom: symptom.into(),
            help: help.map(Into::into),
        }
    }
}

/// Run health checks on a vector store, and suggest fixes for the problems they found. This does
/// not run those fixes, so the caller keeps control over them (e.g., the caller may want to first
/// print some message or ask for confirmation before proceeding). Some suggestions assume the
/// end-user knows what "/index" means. That may change later, but while this crate is tailored to
/// `zqa`, it is a very low priority.
///
/// Diagnosis stops at the first problem that later checks depend on, such as missing storage or
/// an inaccessible table, so the list may not cover every check.
///
/// # Arguments
///
/// * `backend` - The vector store backend to diagnose.
///
/// # Returns
///
/// The problems found, in the order the checks ran. An empty list means no problems were found.
///
/// # Errors
///
/// Returns [`HealthCheckError::InvalidState`] if the health check result is in a state it should
/// never be in, which likely means a bug in the backend's health check.
pub async fn doctor<T: HealthCheckable>(backend: &T) -> Result<Vec<Diagnosis>, HealthCheckError> {
    let healthcheck_results = backend.health_check().await;

    if !healthcheck_results.storage_exists {
        return Ok(vec![Diagnosis::new(
            "database storage does not exist.",
            Some("maybe you are not in the right directory?"),
        )]);
    }

    let tbl_accessible =
        healthcheck_results
            .table_accessible
            .ok_or(HealthCheckError::InvalidState(
            "Invalid healthcheck result: if storage exists, `table_accessible` cannot be `None`."
                .into(),
        ))?;

    if tbl_accessible.is_err() {
        // Usually, there isn't much we can do here
        return Ok(vec![Diagnosis::new(
            "the database table is not accessible.",
            Some("check that the `data/` directory actually contains the DB and is not corrupted."),
        )]);
    }

    let Some(row_count) = healthcheck_results.num_rows else {
        return Ok(vec![Diagnosis::new(
            "row count check was skipped; the database may not support this operation.",
            None,
        )]);
    };

    let mut diagnoses = Vec::new();

    if row_count.is_err() {
        diagnoses.push(Diagnosis::new(
            "row count cannot be obtained.",
            Some("this is usually transient; if this persists, your database may be corrupted."),
        ));
    }

    let Some(index_info) = healthcheck_results.index_info else {
        return Ok(diagnoses);
    };

    if let Ok(indices) = index_info {
        if indices.is_empty()
            && let Ok(row_count) = row_count
            && row_count > 10000
        {
            diagnoses.push(Diagnosis::new(
                "there were no indices with > 10k rows.",
                Some("run /index to create indices."),
            ));
        }
    } else {
        diagnoses.push(Diagnosis::new(
            "index information could not be obtained",
            Some("this is usually transient; if this persists, your database may be corrupted."),
        ));
    }

    Ok(diagnoses)
}

#[cfg(test)]
mod tests {
    use std::env;
    use std::sync::Arc;

    use arrow_array::{RecordBatch, StringArray};
    use dotenv::dotenv;
    use zqa_macros::test_ok;

    use super::doctor;
    use crate::config::VoyageAIConfig;
    use crate::constants::{
        DEFAULT_VOYAGE_EMBEDDING_DIM, DEFAULT_VOYAGE_EMBEDDING_MODEL, DEFAULT_VOYAGE_RERANK_MODEL,
    };
    use crate::embedding::common::EmbeddingProviderConfig;
    use crate::vector::backends::backend::VectorBackend;
    use crate::vector::backends::lance::LanceBackend;

    /// Builds a [`LanceBackend`] pointing at an isolated temp directory, along with the schema it
    /// was constructed with. The returned [`tempfile::TempDir`] guard must be kept alive for the
    /// test's duration. Because each backend points at its own URI, these tests need no
    /// `#[serial]`.
    fn temp_backend() -> (tempfile::TempDir, LanceBackend, Arc<arrow_schema::Schema>) {
        dotenv().ok();

        let dir = tempfile::tempdir().unwrap();
        let uri = dir
            .path()
            .join("lancedb-table")
            .to_str()
            .unwrap()
            .to_string();

        let schema = Arc::new(arrow_schema::Schema::new(vec![
            arrow_schema::Field::new("pdf_text", arrow_schema::DataType::Utf8, false),
            arrow_schema::Field::new("title", arrow_schema::DataType::Utf8, false),
        ]));

        let backend = LanceBackend::new(
            EmbeddingProviderConfig::VoyageAI(VoyageAIConfig {
                embedding_model: DEFAULT_VOYAGE_EMBEDDING_MODEL.into(),
                embedding_dims: DEFAULT_VOYAGE_EMBEDDING_DIM as usize,
                api_key: env::var("VOYAGE_AI_API_KEY").unwrap_or_default(),
                reranker: DEFAULT_VOYAGE_RERANK_MODEL.into(),
                max_concurrent_requests: crate::constants::DEFAULT_MAX_CONCURRENT_REQUESTS,
                max_retries: crate::constants::DEFAULT_MAX_RETRIES,
            }),
            Arc::clone(&schema),
            "pdf_text".into(),
        )
        .with_uri(&uri);

        (dir, backend, schema)
    }

    /// When the database does not exist, `doctor` should report that symptom and short-circuit
    /// without running the remaining checks (and without any network access).
    #[tokio::test]
    async fn test_doctor_reports_missing_database() {
        let (_dir, backend, _) = temp_backend();

        let result = doctor(&backend).await;
        test_ok!(result);

        // The early return means the later checks are never reached.
        let diagnoses = result.unwrap();
        assert_eq!(diagnoses.len(), 1);
        assert!(
            diagnoses[0]
                .symptom
                .contains("database storage does not exist")
        );
        assert!(diagnoses[0].help.is_some());
    }

    /// Builds a small, healthy database in an isolated temp directory and checks that `doctor` runs
    /// every check through to completion without finding a problem.
    #[tokio::test]
    async fn test_doctor_healthy_database_completes() {
        let (_dir, backend, schema) = temp_backend();

        let record_batch = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(StringArray::from(vec!["Hello world", "Test document"])),
                Arc::new(StringArray::from(vec!["doc1.pdf", "doc2.pdf"])),
            ],
        )
        .unwrap();

        backend
            .insert_items(vec![record_batch], None)
            .await
            .unwrap();

        let result = doctor(&backend).await;
        test_ok!(result);
        assert_eq!(result.unwrap(), []);
    }
}
