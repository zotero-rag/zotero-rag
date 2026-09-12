//! Exercise enabled debug logs through the public LLM API and a local HTTP endpoint.

use std::fmt;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::Mutex;
use std::thread;
use std::time::Duration;

use log::{LevelFilter, Log, Metadata, Record};
use zqa_rag::config::{LLMClientConfig, OllamaConfig};
use zqa_rag::llm::base::ChatRequest;
use zqa_rag::llm::errors::LLMError;
use zqa_rag::llm::factory::get_client_with_config;
use zqa_rag::logging::preview;

struct CaptureLog(Mutex<Vec<String>>);

impl Log for CaptureLog {
    fn enabled(&self, metadata: &Metadata<'_>) -> bool {
        metadata.target().starts_with("zqa_rag")
    }

    fn log(&self, record: &Record<'_>) {
        if self.enabled(record.metadata()) {
            self.0.lock().unwrap().push(record.args().to_string());
        }
    }

    fn flush(&self) {}
}

static LOG: CaptureLog = CaptureLog(Mutex::new(Vec::new()));

struct MustNotFormat;

impl fmt::Display for MustNotFormat {
    fn fmt(&self, _: &mut fmt::Formatter<'_>) -> fmt::Result {
        panic!("Disabled debug logging must not format previews");
    }
}

/// Serve a rate limit followed by a large error, returning both received JSON requests.
fn serve_responses(listener: TcpListener, error_body: &str) -> Vec<serde_json::Value> {
    let mut requests = Vec::new();
    for (status, extra_headers, body) in [
        (
            "429 Too Many Requests",
            "Retry-After: 0\r\n",
            "rate limited",
        ),
        ("400 Bad Request", "", error_body),
    ] {
        let (mut stream, _) = listener.accept().unwrap();
        stream
            .set_read_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        stream
            .set_write_timeout(Some(Duration::from_secs(10)))
            .unwrap();
        let mut reader = BufReader::new(&mut stream);
        let mut headers = String::new();
        loop {
            let mut line = String::new();
            assert_ne!(reader.read_line(&mut line).unwrap(), 0);
            if line == "\r\n" {
                break;
            }
            headers.push_str(&line);
        }
        assert!(headers.contains("fake-query-key"));
        assert!(headers.contains("x-api-key: ollama"));
        let length: usize = headers
            .lines()
            .find_map(|line| {
                let (name, value) = line.split_once(':')?;
                name.eq_ignore_ascii_case("content-length")
                    .then(|| value.trim().parse().unwrap())
            })
            .unwrap();
        let mut bytes = vec![0; length];
        reader.read_exact(&mut bytes).unwrap();
        requests.push(serde_json::from_slice(&bytes).unwrap());
        write!(stream, "HTTP/1.1 {status}\r\nContent-Length: {}\r\n{extra_headers}Connection: close\r\n\r\n{body}", body.len()).unwrap();
    }
    requests
}

#[tokio::test]
async fn debug_logs_are_bounded_without_changing_http_payloads() {
    log::set_logger(&LOG).unwrap();
    log::set_max_level(LevelFilter::Off);
    log::debug!("{}", preview(MustNotFormat));
    log::set_max_level(LevelFilter::Debug);

    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let error_body = format!(
        "provider error: {}END_OF_ERROR",
        "paper passage ".repeat(10_000)
    );
    let server_body = error_body.clone();
    let server = thread::spawn(move || serve_responses(listener, &server_body));
    let client = get_client_with_config(&LLMClientConfig::Ollama(OllamaConfig {
        base_url: format!("http://fake-user:fake-password@{address}/?key=fake-query-key"),
        ..Default::default()
    }))
    .unwrap();
    let message = format!(
        "PRIVATE_REQUEST {}END_OF_REQUEST",
        "whole paper ".repeat(10_000)
    );
    let request = ChatRequest {
        message: message.clone(),
        ..Default::default()
    };
    let error = client.send_message(&request).await.unwrap_err();
    let requests = server.join().unwrap();
    assert_eq!(requests.len(), 2);
    for request in requests {
        assert_eq!(request["messages"][0]["content"][0]["text"], message);
    }
    assert!(matches!(error, LLMError::HttpStatusError(body) if body == error_body));

    let logs = LOG.0.lock().unwrap();
    let combined = logs.join("\n");
    assert!(combined.contains("Rate limited on attempt 1"));
    assert!(combined.contains("attempt 2 returned 400"));
    assert!(combined.contains("Generation turn 1 failed"));
    assert!(combined.contains("truncated;"));
    for secret in [
        "fake-user",
        "fake-password",
        "fake-query-key",
        "x-api-key",
        "PRIVATE_REQUEST",
        "END_OF_ERROR",
    ] {
        assert!(
            !combined.contains(secret),
            "Unexpected content in logs: {secret}"
        );
    }
    assert!(logs.iter().all(|line| line.len() < 800));
    for line in logs.iter() {
        println!("{line}");
    }
}
