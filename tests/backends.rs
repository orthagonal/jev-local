mod common;

use axum::{
    body::Body,
    http::{header, HeaderMap, Method, Request, StatusCode},
    routing::{get, post},
    Json, Router,
};
use common::*;
use jev_local::{app, try_app, Backend};
use serde_json::{json, Value};
use std::sync::{Arc, Mutex};
use tokio::{net::TcpListener, task::JoinHandle};

fn configured(value: Value) -> jev_local::Config {
    let mut cfg = config(Backend::Mock);
    cfg.registry = Some(serde_json::from_value(value).unwrap());
    cfg
}

fn registry(profiles: Value, default: &str) -> jev_local::Config {
    configured(json!({"default_backend":default,"backends":profiles}))
}

struct Upstream {
    url: String,
    requests: Arc<Mutex<Vec<Value>>>,
    authorizations: Arc<Mutex<Vec<Option<String>>>>,
    task: JoinHandle<()>,
}
impl Upstream {
    async fn start(native: bool) -> Self {
        let requests = Arc::new(Mutex::new(Vec::new()));
        let received = requests.clone();
        let authorizations = Arc::new(Mutex::new(Vec::new()));
        let captured_auth = authorizations.clone();
        let router = Router::new()
            .route(
                if native { "/predict" } else { "/v1/systemone" },
                post(move |headers: HeaderMap, Json(body): Json<Value>| {
                    let received = received.clone();
                    let captured_auth = captured_auth.clone();
                    async move {
                        captured_auth.lock().unwrap().push(
                            headers
                                .get(header::AUTHORIZATION)
                                .map(|v| v.to_str().unwrap().to_owned()),
                        );
                        received.lock().unwrap().push(body.clone());
                        if native {
                            Json(json!({"model":"native-checkpoint","probabilities":[0.1,0.9]}))
                        } else {
                            let mut result = jev_local::mock_decide(&body);
                            result["model"] = json!("resolved-checkpoint");
                            result["answers"]["vote"]["confidence"] = json!(0.99);
                            result["provider_extension"] = json!("not part of the common contract");
                            Json(result)
                        }
                    }
                }),
            )
            .route(
                "/v1/models",
                get(|| async { Json(json!({"data":[{"id":"native-checkpoint"}]})) }),
            );
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self {
            url,
            requests,
            authorizations,
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
async fn same_request_works_across_named_servers_with_aliases_and_normalization() {
    let upstream = Upstream::start(false).await;
    let cfg = registry(
        json!({
            "kev-small":{"base_url":upstream.url,"default_model":"kev-latest"},
            "jevany":{"base_url":upstream.url,"default_model":"jevany-latest"}
        }),
        "kev-small",
    );
    let router = app(cfg);
    for (index, id) in ["kev-small", "jevany"].iter().enumerate() {
        let response = request(
            &router,
            Method::POST,
            &format!("/v1/backends/{id}/systemone"),
            Some(fixture()),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::OK);
        assert_eq!(response.headers["x-decision-backend"], *id);
        assert_eq!(response.headers["x-decision-simulated"], "false");
        let body = response.json();
        assert_inference(&body);
        assert_eq!(body["model"], "resolved-checkpoint");
        assert!(body["answers"]["vote"]["confidence"].as_f64().unwrap() < 1e-10);
        assert_eq!(
            response.headers["x-decision-confidence-definition"],
            "normalized-entropy"
        );
        assert_eq!(body.as_object().unwrap().len(), 3);
        assert!(body.get("provider_extension").is_none());
        assert_eq!(
            upstream.requests.lock().unwrap()[index]["model"],
            if index == 0 {
                "kev-latest"
            } else {
                "jevany-latest"
            }
        );
    }
    let response = request(&router, Method::GET, "/v1/models", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_models(&response.json());
    let response = request(
        &router,
        Method::GET,
        "/v1/backends/kev-small/models",
        None,
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["data"][0]["id"], "native-checkpoint");
}

#[tokio::test]
async fn selection_header_and_default_profile_use_the_same_api_payload() {
    let router = app(registry(
        json!({
            "a":{"adapter":"simulation","default_model":"sim-a"},
            "b":{"adapter":"simulation","default_model":"sim-b"}
        }),
        "a",
    ));
    let response = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await;
    assert_eq!(response.json()["model"], "sim-a");
    let req = Request::builder()
        .method(Method::POST)
        .uri("/v1/systemone")
        .header(header::CONTENT_TYPE, "application/json")
        .header("x-decision-backend", "b")
        .body(Body::from(fixture().to_string()))
        .unwrap();
    let response = send(&router, req).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["model"], "sim-b");
}

#[tokio::test]
async fn simulation_probabilities_delays_and_periodic_errors_work_through_benchmarks() {
    let cfg = registry(
        json!({"custom-sim":{"adapter":"simulation","default_model":"simulated-model",
        "simulation":{"latency_ms":15,"fail_every":3,"probabilities":{
            "vote":{"yes":0.0,"no":1.0,"irrelevant":0.0},
            "tension":{"0":0.0,"1":0.0,"2":1.0},
            "urgency":{"false":0.0,"true":1.0}
        }}}}),
        "custom-sim",
    );
    let router = app(cfg);
    let dataset = json!({"id":"door","version":"1","cases":[{"id":"one","request":fixture(),"expected":{"vote":"no","tension":2,"urgency":true}}]});
    let response = request(&router,Method::POST,"/v1/benchmarks/compare",Some(json!({"dataset":dataset,"providers":["custom-sim"],"options":{"repetitions":3,"warmup":0,"include_details":true}})),None).await;
    response.assert_json_status(StatusCode::OK);
    let report = response.json();
    let result = &report["results"][0];
    assert_eq!(result["simulated"], true);
    assert_eq!(result["status"], "partial");
    assert_eq!(result["requests"]["successful"], 2);
    assert_eq!(result["requests"]["failed"], 1);
    assert_eq!(result["quality"]["accuracy_on_success"], 1.0);
    assert!((result["quality"]["accuracy"].as_f64().unwrap() - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(result["quality"]["calibration"]["brier_score"], 0.0);
    assert!(
        result["latency"]["all_requests"]["min_ms"]
            .as_f64()
            .unwrap()
            >= 15.0
    );
    assert_eq!(result["failures"][0]["http_status"], 503);
    let response = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.headers["x-decision-simulated"], "true");
    assert_eq!(response.json()["answers"]["tension"]["score"], 2.0);
}

#[tokio::test]
async fn fixtures_match_original_request_and_return_recorded_answers() {
    let mut recorded = jev_local::mock_decide(&fixture());
    recorded["model"] = json!("recorded-model");
    let cfg = registry(
        json!({"recorded":{"adapter":"simulation","default_model":"resolved-alias",
            "simulation":{"fixtures":[{"match_request":{"model":"jev-latest","state":"Don't open that door!"},"response":recorded}]}
        }}),
        "recorded",
    );
    let router = app(cfg);
    let response = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["model"], "recorded-model");
    let mut other = fixture();
    other["state"] = json!("unmatched");
    let response = request(&router, Method::POST, "/v1/systemone", Some(other), None).await;
    assert_eq!(response.json()["model"], "resolved-alias");
}

#[tokio::test]
async fn discovery_and_explicit_routes_are_authenticated_and_do_not_expose_credentials() {
    let mut cfg = registry(
        json!({"private":{"adapter":"simulation","api_key_env":"JEV_LOCAL_TEST_UNSET_SECRET","require_api_key":true}}),
        "private",
    );
    cfg.gateway_key = Some("gateway-secret".into());
    let router = app(cfg);
    for path in [
        "/v1/backends",
        "/v1/backends/private",
        "/v1/backends/private/models",
    ] {
        request(&router, Method::GET, path, None, None)
            .await
            .assert_json_status(StatusCode::UNAUTHORIZED);
    }
    request(
        &router,
        Method::POST,
        "/v1/backends/private/systemone",
        Some(fixture()),
        None,
    )
    .await
    .assert_json_status(StatusCode::UNAUTHORIZED);
    let response = request(
        &router,
        Method::GET,
        "/v1/backends",
        None,
        Some("gateway-secret"),
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let data = &response.json()["data"][0];
    assert_eq!(data["configuration_ready"], false);
    assert!(data.get("api_key_env").is_none());
    assert!(data.get("base_url").is_none());
    assert!(!String::from_utf8_lossy(&response.body).contains("gateway-secret"));
    request(
        &router,
        Method::POST,
        "/v1/backends/private/systemone",
        Some(fixture()),
        Some("gateway-secret"),
    )
    .await
    .assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn jevlike_bridge_maps_options_without_losing_question_ids() {
    let upstream = Upstream::start(true).await;
    let router = app(registry(
        json!({"native":{
            "adapter":"jevlike","base_url":upstream.url,"inference_path":"/predict","default_model":"jevlike",
            "capabilities":{"question_types":["choice"]}
        }}),
        "native",
    ));
    let body: Value = serde_json::from_str(include_str!("../examples/choice.json")).unwrap();
    let response = request(&router, Method::POST, "/v1/systemone", Some(body), None).await;
    response.assert_json_status(StatusCode::OK);
    let result = response.json();
    assert_eq!(result["model"], "native-checkpoint");
    assert_eq!(result["answers"]["team"]["choice"], "technical");
    assert_eq!(
        result["answers"]["team"]["probabilities"],
        json!({"billing":0.1,"technical":0.9})
    );
    assert_eq!(response.headers["x-decision-usage-source"], "unavailable");
    {
        let calls = upstream.requests.lock().unwrap();
        assert_eq!(calls.len(), 1);
        assert_eq!(
            calls[0]["options"],
            json!([
                "billing: Payments and refunds",
                "technical: Bugs and outages"
            ])
        );
        assert!(calls[0]["context"]
            .as_str()
            .unwrap()
            .ends_with("Which team should handle this request?"));
    }
    request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await
    .assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(upstream.requests.lock().unwrap().len(), 1);
}

#[tokio::test]
async fn capability_and_missing_backend_checks_happen_before_benchmark_inference() {
    let upstream = Upstream::start(false).await;
    let router = app(registry(
        json!({
            "full":{"base_url":upstream.url},
            "choice-only":{"base_url":upstream.url,"capabilities":{"question_types":["choice"]}}
        }),
        "full",
    ));
    for providers in [json!(["full", "choice-only"]), json!(["full", "unknown"])] {
        request(
            &router,
            Method::POST,
            "/v1/benchmarks/compare",
            Some(json!({"dataset":"support-triage","providers":providers})),
            None,
        )
        .await
        .assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
    }
    assert!(upstream.requests.lock().unwrap().is_empty());
}

#[tokio::test]
async fn simulation_timeout_and_invalid_recordings_fail_without_real_inference() {
    for profile in [
        json!({"adapter":"simulation","timeout_ms":5,"simulation":{"latency_ms":50}}),
        json!({"adapter":"simulation","simulation":{"fixtures":[{"match_request":{},"response":{"model":"broken","answers":{}}}]}}),
    ] {
        let router = app(registry(json!({"broken":profile}), "broken"));
        request(
            &router,
            Method::POST,
            "/v1/systemone",
            Some(fixture()),
            None,
        )
        .await
        .assert_json_status(StatusCode::BAD_GATEWAY);
    }
}

#[test]
fn invalid_configurations_fail_at_startup() {
    for profile in [
        json!({"base_url":"https://secret:password@example.com"}),
        json!({"base_url":"file:///tmp/model"}),
        json!({"base_url":"https://example.com","inference_path":"//other-host/predict"}),
        json!({"adapter":"simulation","simulation":{"fail_every":0}}),
        json!({"adapter":"simulation","simulation":{"probabilities":{"q":{"a":0.9,"b":0.9}}}}),
        json!({"adapter":"jevlike","base_url":"http://127.0.0.1:8011"}),
    ] {
        assert!(try_app(registry(json!({"bad":profile}), "bad")).is_err());
    }
    assert!(try_app(registry(
        json!({"good":{"adapter":"simulation"}}),
        "missing"
    ))
    .is_err());
}

#[tokio::test]
async fn shipped_profiles_support_a_complete_offline_benchmark() {
    let cfg = configured(serde_json::from_str(include_str!("../examples/backends.json")).unwrap());
    let router = app(cfg);
    let response = request(&router, Method::GET, "/v1/backends", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["data"].as_array().unwrap().len(), 8);
    let payload: Value =
        serde_json::from_str(include_str!("../examples/benchmark-simulations.json")).unwrap();
    let response = request(
        &router,
        Method::POST,
        "/v1/benchmarks/compare",
        Some(payload),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let report = response.json();
    assert!(report["results"]
        .as_array()
        .unwrap()
        .iter()
        .all(|r| r["simulated"] == true));
    assert_eq!(report["results"][1]["requests"]["failed"], 2);
    let fixture_cfg = configured(
        serde_json::from_str(include_str!("../examples/backends-fixtures.json")).unwrap(),
    );
    let body: Value = serde_json::from_str(include_str!("../examples/choice.json")).unwrap();
    let response = request(
        &app(fixture_cfg),
        Method::POST,
        "/v1/systemone",
        Some(body),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["model"], "recorded-refund");
}

struct GatewayProcess(std::process::Child);
impl Drop for GatewayProcess {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}
struct ConfigFile(std::path::PathBuf);
impl Drop for ConfigFile {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
    }
}

#[tokio::test]
async fn cli_loads_profile_files_and_resolves_secrets_in_the_server_process() {
    use std::process::{Command, Stdio};
    use std::time::Duration;
    let upstream = Upstream::start(false).await;
    let file = ConfigFile(std::env::temp_dir().join(format!(
            "jev-backends-{}-{}.json",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        )));
    std::fs::write(&file.0,json!({"default_backend":"named-real","backends":{
        "named-real":{"base_url":upstream.url,"default_model":"server-model","api_key_env":"DECISION_TEST_CHILD_KEY","require_api_key":true},
        "offline":{"adapter":"simulation","default_model":"offline"}
    }}).to_string()).unwrap();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    drop(listener);
    let mut command = Command::new(env!("CARGO_BIN_EXE_jev-local"));
    command
        .arg("--bind")
        .arg(address.to_string())
        .arg("--backends-file")
        .arg(&file.0)
        .env("DECISION_TEST_CHILD_KEY", "test-upstream-secret")
        .env_remove("JEV_LOCAL_PROVIDER")
        .env_remove("JEV_LOCAL_API_KEY")
        .env_remove("BENCHMARK_DATASET_DIR")
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        command.creation_flags(0x08000000); // CREATE_NO_WINDOW
    }
    let mut child = GatewayProcess(command.spawn().unwrap());
    let base = format!("http://{address}");
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    tokio::time::timeout(Duration::from_secs(10), async {
        loop {
            assert!(
                child.0.try_wait().unwrap().is_none(),
                "gateway exited during startup"
            );
            if client.get(format!("{base}/health")).send().await.is_ok() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
    })
    .await
    .unwrap();
    let response = client
        .post(format!("{base}/v1/systemone"))
        .json(&fixture())
        .send()
        .await
        .unwrap();
    assert_eq!(response.status(), StatusCode::OK);
    assert_eq!(response.headers()["x-decision-backend"], "named-real");
    let body: Value = response.json().await.unwrap();
    assert_inference(&body);
    assert_eq!(
        upstream.requests.lock().unwrap()[0]["model"],
        "server-model"
    );
    assert_eq!(
        upstream.authorizations.lock().unwrap()[0].as_deref(),
        Some("Bearer test-upstream-secret")
    );
    let metadata: Value = client
        .get(format!("{base}/v1/backends"))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert!(!metadata.to_string().contains("test-upstream-secret"));
    let result = client.post(format!("{base}/v1/benchmarks/compare"))
        .json(&json!({"dataset":"support-triage","providers":["named-real","offline"],"options":{"warmup":0,"include_details":true}}))
        .send().await.unwrap();
    assert_eq!(result.status(), StatusCode::OK);
    let report: Value = result.json().await.unwrap();
    assert_eq!(report["results"][0]["status"], "ok");
    assert_eq!(report["results"][0]["simulated"], false);
    assert_eq!(report["results"][1]["simulated"], true);
    assert!(upstream
        .authorizations
        .lock()
        .unwrap()
        .iter()
        .all(|v| v.as_deref() == Some("Bearer test-upstream-secret")));
}
