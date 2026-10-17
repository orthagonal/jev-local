mod common;

use axum::{
    http::{header, HeaderMap, Method, StatusCode},
    routing::post,
    Json, Router,
};
use common::*;
use jev_local::{app, Backend};
use serde_json::{json, Value};
use std::{
    sync::{
        atomic::{AtomicUsize, Ordering},
        Arc, Mutex,
    },
    time::Duration,
};
use tokio::{net::TcpListener, task::JoinHandle};

fn inline_dataset() -> Value {
    json!({
        "id":"test-scenario", "version":"1", "description":"Known answers for metric verification",
        "cases":[{"id":"case-a", "request":{
            "state":"test", "questions":{
                "choice":{"type":"choice", "instructions":"Pick", "criteria":{"a":"A", "b":"B"}},
                "score":{"type":"score", "instructions":"Rate", "criteria":["low", "high"]},
                "noul":{"type":"noul", "instructions":"Yes?"}
            }
        }, "expected":{"choice":"a", "score":1, "noul":true}}]
    })
}

fn prediction() -> Value {
    json!({"model":"fixture-model", "answers":{
        "choice":{"type":"choice", "choice":"a", "probabilities":{"a":0.8,"b":0.2}, "confidence":0.01},
        "score":{"type":"score", "score":0.75, "probabilities":{"0":0.25,"1":0.75}, "confidence":0.99},
        "noul":{"type":"noul", "noul":0.9}
    }})
}

struct Upstream {
    url: String,
    calls: Arc<Mutex<Vec<(HeaderMap, Value)>>>,
    max_concurrency: Arc<AtomicUsize>,
    task: JoinHandle<()>,
}

impl Upstream {
    async fn start(body: Value, status: StatusCode, delay: Duration) -> Self {
        let calls = Arc::new(Mutex::new(Vec::new()));
        let active = Arc::new(AtomicUsize::new(0));
        let max_concurrency = Arc::new(AtomicUsize::new(0));
        let captured = calls.clone();
        let peak = max_concurrency.clone();
        let router = Router::new().route(
            "/v1/systemone",
            post(move |headers: HeaderMap, Json(request): Json<Value>| {
                let captured = captured.clone();
                let active = active.clone();
                let peak = peak.clone();
                let body = body.clone();
                async move {
                    captured.lock().unwrap().push((headers, request));
                    let current = active.fetch_add(1, Ordering::SeqCst) + 1;
                    peak.fetch_max(current, Ordering::SeqCst);
                    tokio::time::sleep(delay).await;
                    active.fetch_sub(1, Ordering::SeqCst);
                    (status, Json(body))
                }
            }),
        );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self {
            url,
            calls,
            max_concurrency,
            task,
        }
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn catalogs_and_fetches_switchable_builtin_datasets() {
    let router = app(config(Backend::Mock));
    let response = request(&router, Method::GET, "/v1/benchmarks/datasets", None, None).await;
    response.assert_json_status(StatusCode::OK);
    let catalog = response.json();
    let data = catalog["data"].as_array().unwrap();
    assert_eq!(data.len(), 2);
    for id in ["support-triage", "sentiment"] {
        assert!(data.iter().any(|entry| entry["id"] == id));
        let response = request(
            &router,
            Method::GET,
            &format!("/v1/benchmarks/datasets/{id}"),
            None,
            None,
        )
        .await;
        response.assert_json_status(StatusCode::OK);
        assert!(!response.json()["cases"].as_array().unwrap().is_empty());
        let response = request(
            &router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({"dataset":id,"providers":["mock"],"options":{"warmup":0}})),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::OK);
        let report = response.json();
        assert_eq!(report["dataset"]["id"], id);
        assert_eq!(report["results"][0]["status"], "ok");
        assert!(report["results"][0]["quality"]["calibration"]["brier_score"].is_number());
    }
}

#[tokio::test]
async fn all_benchmark_routes_require_gateway_auth() {
    let mut cfg = config(Backend::Mock);
    cfg.gateway_key = Some("gateway-secret".into());
    let router = app(cfg);
    for (method, path, body) in [
        (Method::GET, "/v1/benchmarks/datasets", None),
        (Method::GET, "/v1/benchmarks/datasets/sentiment", None),
        (
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({"dataset":"sentiment","providers":["mock"]})),
        ),
    ] {
        for token in [None, Some("wrong"), Some("gateway-secret")] {
            let response = request(&router, method.clone(), path, body.clone(), token).await;
            response.assert_json_status(if token == Some("gateway-secret") {
                StatusCode::OK
            } else {
                StatusCode::UNAUTHORIZED
            });
        }
    }
}

#[tokio::test]
async fn compares_both_providers_with_known_metrics_and_excludes_warmup() {
    let upstream = Upstream::start(prediction(), StatusCode::OK, Duration::from_millis(20)).await;
    let mut cfg = config(Backend::Mock); // Benchmark selection is independent of the normal route backend.
    cfg.laya_url = upstream.url.clone();
    cfg.jev_url = upstream.url.clone();
    cfg.jev_key = Some("hosted-secret".into());
    cfg.gateway_key = Some("gateway-secret".into());
    let response = request(
        &app(cfg),
        Method::POST,
        "/v1/benchmarks/compare",
        Some(json!({
            "dataset":inline_dataset(),
            "models":{"laya":"english","jev":"jev-latest"},
            "options":{"repetitions":4,"concurrency":2,"warmup":1,"include_details":true}
        })),
        Some("gateway-secret"),
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let report = response.json();
    for result in report["results"].as_array().unwrap() {
        assert_eq!(result["status"], "ok");
        assert_eq!(result["warmup"]["requests"], 1);
        assert_eq!(result["requests"]["attempted"], 4);
        assert_eq!(result["requests"]["successful"], 4);
        assert_eq!(result["quality"]["scored_decisions"], 12);
        assert_eq!(result["quality"]["accuracy"], 1.0);
        assert_eq!(result["quality"]["score_mae"], 0.25);
        let calibration = &result["quality"]["calibration"];
        let brier = (0.08 + 0.125 + 0.02) / 3.0;
        assert!((calibration["brier_score"].as_f64().unwrap() - brier).abs() < 1e-12);
        assert!(
            (calibration["negative_log_likelihood"].as_f64().unwrap()
                + (0.8f64.ln() + 0.75f64.ln() + 0.9f64.ln()) / 3.0)
                .abs()
                < 1e-12
        );
        assert!(
            (calibration["expected_calibration_error"].as_f64().unwrap()
                - (0.2 + 0.25 + 0.1) / 3.0)
                .abs()
                < 1e-12
        );
        assert!(
            result["latency"]["successful_requests"]["min_ms"]
                .as_f64()
                .unwrap()
                >= 20.0
        );
        let throughput = &result["throughput"];
        let seconds = throughput["duration_ms"].as_f64().unwrap() / 1000.0;
        assert!(
            (throughput["successful_requests_per_second"]
                .as_f64()
                .unwrap()
                - 4.0 / seconds)
                .abs()
                < 1e-9
        );
        assert_eq!(result["details"].as_array().unwrap().len(), 4);
        assert_eq!(result["observed_models"], json!(["fixture-model"]));
    }
    assert_eq!(upstream.max_concurrency.load(Ordering::SeqCst), 2);
    let calls = upstream.calls.lock().unwrap();
    assert_eq!(calls.len(), 10); // Four measured plus one warmup for each provider.
    for (index, (headers, payload)) in calls.iter().enumerate() {
        assert_eq!(payload["state"], "test");
        assert!(payload.get("expected").is_none());
        assert!(payload.get("cases").is_none());
        if index < 5 {
            assert!(!headers.contains_key(header::AUTHORIZATION));
            assert_eq!(payload["model"], "english");
        } else {
            assert_eq!(headers[header::AUTHORIZATION], "Bearer hosted-secret");
            assert_eq!(payload["model"], "jev-latest");
        }
    }
}

#[tokio::test]
async fn http_errors_and_malformed_predictions_count_as_failures() {
    for (status, body) in [
        (StatusCode::TOO_MANY_REQUESTS, json!({"error":"limited"})),
        (StatusCode::OK, json!({"model":"broken","answers":{}})),
    ] {
        let upstream = Upstream::start(body, status, Duration::ZERO).await;
        let mut cfg = config(Backend::Mock);
        cfg.laya_url = upstream.url.clone();
        let response = request(
            &app(cfg),
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({
                "dataset":inline_dataset(),"providers":["laya"],"options":{"warmup":0}
            })),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::OK);
        let report = response.json();
        let result = &report["results"][0];
        assert_eq!(result["status"], "failed");
        assert_eq!(result["requests"]["failed"], 1);
        assert_eq!(result["quality"]["accuracy"], 0.0);
        assert_eq!(result["quality"]["scored_decisions"], 0);
        assert!(result["quality"]["calibration"].is_null());
        assert!(result["latency"]["successful_requests"].is_null());
        assert_eq!(result["throughput"]["successful_requests_per_second"], 0.0);
        assert_eq!(result["failures"][0]["http_status"], status.as_u16());
    }
}

#[tokio::test]
async fn timeouts_are_reported_and_dont_fall_back_to_mock() {
    let upstream = Upstream::start(prediction(), StatusCode::OK, Duration::from_millis(200)).await;
    let mut cfg = config(Backend::Mock);
    cfg.laya_url = upstream.url.clone();
    let response = request(
        &app(cfg),
        Method::POST,
        "/v1/benchmarks/compare",
        Some(json!({
            "dataset":inline_dataset(),"providers":["laya"],"options":{"warmup":0,"timeout_ms":30}
        })),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let report = response.json();
    assert_eq!(report["results"][0]["requests"]["failed"], 1);
    assert!(report["results"][0]["failures"][0]["error"]
        .as_str()
        .unwrap()
        .contains("timed out"));
}

#[tokio::test]
async fn invalid_inputs_are_rejected_before_upstream_calls() {
    let upstream = Upstream::start(prediction(), StatusCode::OK, Duration::ZERO).await;
    let mut cfg = config(Backend::Mock);
    cfg.laya_url = upstream.url.clone();
    let router = app(cfg);
    for payload in [
        json!({"dataset":"sentiment"}), // Jev key is required for the default comparison.
        json!({"dataset":"sentiment","providers":[]}),
        json!({"dataset":"sentiment","providers":["laya","laya"]}),
        json!({"dataset":"../secret","providers":["laya"]}),
        json!({"dataset":"not-found","providers":["laya"]}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"concurrency":0}}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"concurrency":33}}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"repetitions":0}}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"warmup":11}}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"calibration_bins":0}}),
        json!({"dataset":"sentiment","providers":["laya"],"options":{"timeout_ms":0}}),
        json!({"dataset":"sentiment","providers":["laya"],"models":{"jev":"jev-latest"}}),
    ] {
        let response = request(
            &router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(payload),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    }
    for label in [json!("missing"), json!(null)] {
        let mut dataset = inline_dataset();
        dataset["cases"][0]["expected"]["choice"] = label;
        let response = request(
            &router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({"dataset":dataset,"providers":["laya"]})),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    }
    for label in [json!(2), json!(0.5), json!(-1)] {
        let mut dataset = inline_dataset();
        dataset["cases"][0]["expected"]["score"] = label;
        let response = request(
            &router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({"dataset":dataset,"providers":["laya"]})),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    }
    let mut dataset = inline_dataset();
    dataset["cases"][0]["expected"]
        .as_object_mut()
        .unwrap()
        .remove("noul");
    let response = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(json!({"dataset":dataset,"providers":["laya"]})),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    assert!(upstream.calls.lock().unwrap().is_empty());
}

#[tokio::test]
async fn overlapping_runs_are_rejected_and_gate_is_released_after_completion() {
    let upstream = Upstream::start(prediction(), StatusCode::OK, Duration::from_millis(150)).await;
    let mut cfg = config(Backend::Mock);
    cfg.laya_url = upstream.url.clone();
    let router = app(cfg);
    let payload = json!({"dataset":inline_dataset(),"providers":["laya"],"options":{"warmup":0}});
    let first_router = router.clone();
    let first_payload = payload.clone();
    let first = tokio::spawn(async move {
        request(
            &first_router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(first_payload),
            None,
        )
        .await
    });
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if !upstream.calls.lock().unwrap().is_empty() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(2)).await;
        }
    })
    .await
    .unwrap();
    let second = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(payload.clone()),
        None,
    )
    .await;
    second.assert_json_status(StatusCode::CONFLICT);
    first.await.unwrap().assert_json_status(StatusCode::OK);
    let third = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(payload),
        None,
    )
    .await;
    third.assert_json_status(StatusCode::OK);
}

#[tokio::test]
async fn partial_runs_keep_failed_decisions_in_the_accuracy_denominator() {
    let upstream = Upstream::start(prediction(), StatusCode::OK, Duration::ZERO).await;
    let mut cfg = config(Backend::Mock);
    cfg.laya_url = upstream.url.clone();
    let mut dataset = inline_dataset();
    let mut missing = dataset["cases"][0].clone();
    missing["id"] = json!("case-b");
    let question = missing["request"]["questions"]
        .as_object_mut()
        .unwrap()
        .remove("noul")
        .unwrap();
    missing["request"]["questions"]["missing"] = question;
    missing["expected"].as_object_mut().unwrap().remove("noul");
    missing["expected"]["missing"] = json!(true);
    dataset["cases"].as_array_mut().unwrap().push(missing);
    let response = request(
        &app(cfg),
        Method::POST,
        "/v1/benchmarks/compare",
        Some(json!({
            "dataset":dataset,"providers":["laya"],"options":{"warmup":0}
        })),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let report = response.json();
    let result = &report["results"][0];
    assert_eq!(result["status"], "partial");
    assert_eq!(result["requests"]["successful"], 1);
    assert_eq!(result["requests"]["failed"], 1);
    assert_eq!(result["quality"]["expected_decisions"], 6);
    assert_eq!(result["quality"]["scored_decisions"], 3);
    assert_eq!(result["quality"]["accuracy"], 0.5);
    assert_eq!(result["quality"]["accuracy_on_success"], 1.0);
    assert_eq!(result["by_type"]["choice"]["accuracy"], 0.5);
    assert_eq!(result["by_type"]["noul"]["accuracy"], 0.5);
    assert_eq!(result["by_type"]["score"]["accuracy"], 0.5);
    assert!(result["details"].is_null());
}

struct DatasetDirectory(std::path::PathBuf);
impl Drop for DatasetDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(self.0.join("test-scenario.json"));
        let _ = std::fs::remove_dir(&self.0);
    }
}

#[tokio::test]
async fn loads_custom_dataset_files_without_rebuilding_the_gateway() {
    let root = std::env::temp_dir().join(format!(
        "jev-benchmark-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&root).unwrap();
    let directory = DatasetDirectory(root);
    let file = directory.0.join("test-scenario.json");
    std::fs::write(&file, serde_json::to_vec(&inline_dataset()).unwrap()).unwrap();
    let mut cfg = config(Backend::Mock);
    cfg.benchmark_dataset_dir = Some(directory.0.clone());
    let router = app(cfg);
    let response = request(&router, Method::GET, "/v1/benchmarks/datasets", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["data"].as_array().unwrap().len(), 3);
    let payload = json!({"dataset":"test-scenario","providers":["mock"],"options":{"warmup":0}});
    let response = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(payload.clone()),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["dataset"]["version"], "1");
    let mut updated = inline_dataset();
    updated["version"] = json!("2");
    std::fs::write(&file, serde_json::to_vec(&updated).unwrap()).unwrap();
    let response = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(payload),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["dataset"]["version"], "2");
    updated["id"] = json!("different-name");
    std::fs::write(&file, serde_json::to_vec(&updated).unwrap()).unwrap();
    let response = request(
        &router,
        Method::GET,
        "/v1/benchmarks/datasets/test-scenario",
        None,
        None,
    )
    .await;
    response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
}
