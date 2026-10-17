// Each integration test binary compiles its own copy of these helpers.
#![allow(dead_code)]

use axum::{
    body::Body,
    http::{header, HeaderMap, Method, Request, StatusCode},
    Router,
};
use http_body_util::BodyExt;
use jev_local::{Backend, Config};
use serde_json::{json, Value};
use std::time::Duration;
use tower::ServiceExt;

pub fn config(backend: Backend) -> Config {
    Config {
        backend,
        // Offline tests must never accidentally contact an external provider.
        laya_url: "http://127.0.0.1:1".into(),
        jev_url: "http://127.0.0.1:1".into(),
        jev_key: None,
        gateway_key: None,
        benchmark_dataset_dir: None,
        registry: None,
    }
}

pub fn fixture() -> Value {
    serde_json::from_str(include_str!("../../examples/door.json")).unwrap()
}

pub struct TestResponse {
    pub status: StatusCode,
    pub headers: HeaderMap,
    pub body: Vec<u8>,
}

impl TestResponse {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).expect("response must contain valid JSON")
    }

    pub fn assert_json_status(&self, expected: StatusCode) {
        assert_eq!(
            self.status,
            expected,
            "unexpected status; response: {}",
            String::from_utf8_lossy(&self.body)
        );
        assert!(
            self.headers
                .get(header::CONTENT_TYPE)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.split(';').next() == Some("application/json")),
            "expected JSON content type"
        );
    }
}

pub async fn send(router: &Router, request: Request<Body>) -> TestResponse {
    tokio::time::timeout(Duration::from_secs(130), async {
        let response = router.clone().oneshot(request).await.unwrap();
        let (parts, body) = response.into_parts();
        TestResponse {
            status: parts.status,
            headers: parts.headers,
            body: body.collect().await.unwrap().to_bytes().to_vec(),
        }
    })
    .await
    .expect("route request exceeded 130 seconds (gateway upstream timeout is 120 seconds)")
}

pub async fn request(
    router: &Router,
    method: Method,
    path: &str,
    body: Option<Value>,
    bearer: Option<&str>,
) -> TestResponse {
    let mut builder = Request::builder().method(method).uri(path);
    if let Some(bearer) = bearer {
        builder = builder.header(header::AUTHORIZATION, format!("Bearer {bearer}"));
    }
    let body = match body {
        Some(body) => {
            builder = builder.header(header::CONTENT_TYPE, "application/json");
            Body::from(serde_json::to_vec(&body).unwrap())
        }
        None => Body::empty(),
    };
    send(router, builder.body(body).unwrap()).await
}

pub fn assert_models(body: &Value) {
    let models = body["data"]
        .as_array()
        .expect("models.data must be an array");
    for alias in ["jev-latest", "jev-preview"] {
        assert!(
            models.iter().any(|model| model["id"] == alias),
            "model alias {alias} is missing"
        );
    }
}

fn number_in_range(value: &Value, min: f64, max: f64) {
    let number = value.as_f64().expect("expected a numeric value");
    assert!(
        number.is_finite() && (min..=max).contains(&number),
        "expected {min}..={max}, got {number}"
    );
}

fn assert_probabilities(probabilities: &Value, keys: &[&str]) {
    let probabilities = probabilities.as_object().expect("probabilities object");
    assert_eq!(probabilities.len(), keys.len());
    let mut total = 0.0;
    for key in keys {
        let probability = probabilities.get(*key).expect("missing probability key");
        number_in_range(probability, 0.0, 1.0);
        total += probability.as_f64().unwrap();
    }
    assert!((total - 1.0).abs() < 0.01, "probabilities sum to {total}");
}

// Shared contract checks deliberately allow provider-specific extra fields,
// resolved model IDs, token counts, and different predictions.
pub fn assert_inference(body: &Value) {
    assert!(body["model"].as_str().is_some_and(|s| !s.is_empty()));
    let answers = body["answers"].as_object().expect("answers object");
    assert_eq!(answers.len(), 3);

    let vote = &answers["vote"];
    assert_eq!(vote["type"], "choice");
    assert!(["yes", "no", "irrelevant"].contains(&vote["choice"].as_str().unwrap()));
    number_in_range(&vote["confidence"], 0.0, 1.0);
    assert_probabilities(&vote["probabilities"], &["yes", "no", "irrelevant"]);

    let tension = &answers["tension"];
    assert_eq!(tension["type"], "score");
    number_in_range(&tension["score"], 0.0, 2.0);
    number_in_range(&tension["confidence"], 0.0, 1.0);
    assert_probabilities(&tension["probabilities"], &["0", "1", "2"]);
    assert_eq!(
        tension["legend"],
        json!({"0":"calm", "1":"concerned", "2":"alarmed"})
    );

    let urgency = &answers["urgency"];
    assert_eq!(urgency["type"], "noul");
    number_in_range(&urgency["noul"], 0.0, 1.0);
    for key in ["input_tokens", "output_tokens"] {
        assert!(
            body["usage"][key].as_u64().is_some(),
            "missing token count {key}"
        );
    }
}
