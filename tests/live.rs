mod common;

use axum::http::{Method, StatusCode};
use common::*;
use jev_local::{app, Backend, Config};
use std::env;

fn setting(name: &str) -> Option<String> {
    match env::var(name) {
        Ok(value) => Some(value.trim().to_owned()).filter(|value| !value.is_empty()),
        Err(env::VarError::NotPresent) => None,
        Err(env::VarError::NotUnicode(_)) => panic!("{name} must contain valid Unicode"),
    }
}

fn test_enabled(name: &str) -> bool {
    match setting(name).as_deref() {
        None | Some("0") => false,
        Some(value) if value.eq_ignore_ascii_case("false") => false,
        Some("1") => true,
        Some(value) if value.eq_ignore_ascii_case("true") => true,
        Some(_) => panic!("{name} must be true, false, 1, or 0"),
    }
}

async fn check_live_routes(mut cfg: Config, choice_only: bool) {
    // Exercise gateway authentication without depending on a developer's local key.
    cfg.gateway_key = Some("live-test-gateway-secret".into());
    let router = app(cfg);
    let response = request(&router, Method::GET, "/health", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json()["status"], "ok");

    let response = request(
        &router,
        Method::GET,
        "/v1/models",
        None,
        Some("live-test-gateway-secret"),
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_models(&response.json());

    // One inference call per provider covers choice, score, and noul together.
    let response = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(if choice_only {
            serde_json::from_str(include_str!("../examples/choice.json")).unwrap()
        } else {
            fixture()
        }),
        Some("live-test-gateway-secret"),
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    if choice_only {
        let body = response.json();
        assert!(
            ["billing", "technical"].contains(&body["answers"]["team"]["choice"].as_str().unwrap())
        );
        assert_eq!(body["answers"]["team"]["type"], "choice");
        let sum: f64 = ["billing", "technical"]
            .iter()
            .map(|key| {
                let p = body["answers"]["team"]["probabilities"][key]
                    .as_f64()
                    .unwrap();
                assert!((0.0..=1.0).contains(&p));
                p
            })
            .sum();
        assert!((sum - 1.0).abs() < 0.02);
    } else {
        assert_inference(&response.json());
    }
}

#[tokio::test]
async fn laya_live_routes() {
    if !test_enabled("TEST_LAYA") {
        eprintln!("SKIP laya_live_routes: set TEST_LAYA=true to run real Laya inference");
        return;
    }
    eprintln!("RUN laya_live_routes: real inference through the gateway");
    let mut cfg = config(Backend::Laya);
    cfg.laya_url = setting("LAYA_BASE_URL").unwrap_or_else(|| "http://127.0.0.1:8000".into());
    if setting("LAYA_API_KEY").is_some() {
        let mut registry = jev_local::backends::RegistryConfig::legacy(&cfg);
        registry.backends.get_mut("laya").unwrap().api_key_env = Some("LAYA_API_KEY".into());
        cfg.registry = Some(registry);
    }
    check_live_routes(cfg, false).await;
}

#[tokio::test]
async fn jev_live_routes() {
    let Some(key) = setting("JEV_API_KEY") else {
        eprintln!("SKIP jev_live_routes: JEV_API_KEY is not set or is empty");
        return;
    };
    eprintln!("RUN jev_live_routes: real hosted inference through the gateway");
    let mut cfg = config(Backend::Jev);
    cfg.jev_url = setting("JEV_BASE_URL").unwrap_or_else(|| "https://jev-api.com".into());
    cfg.jev_key = Some(key);
    check_live_routes(cfg, false).await;
}

async fn check_additional_provider(id: &str, flag: &str, url_env: &str, default_url: &str) {
    if !test_enabled(flag) {
        eprintln!("SKIP {id}_live_routes: set {flag}=true to run real inference");
        return;
    }
    eprintln!("RUN {id}_live_routes: real inference through the shared adapter");
    let mut cfg = config(Backend::Named(id.into()));
    let mut registry = jev_local::backends::RegistryConfig::legacy(&cfg);
    registry.backends.get_mut(id).unwrap().base_url =
        Some(setting(url_env).unwrap_or_else(|| default_url.into()));
    cfg.registry = Some(registry);
    check_live_routes(cfg, id == "jevlike").await;
}

#[tokio::test]
async fn kev_live_routes() {
    check_additional_provider("kev", "TEST_KEV", "KEV_BASE_URL", "http://127.0.0.1:8009").await;
}

#[tokio::test]
async fn jevany_live_routes() {
    check_additional_provider(
        "jevany",
        "TEST_JEVANY",
        "JEVANY_BASE_URL",
        "http://127.0.0.1:8008",
    )
    .await;
}

#[tokio::test]
async fn jevlike_live_routes() {
    check_additional_provider(
        "jevlike",
        "TEST_JEVLIKE",
        "JEVLIKE_BASE_URL",
        "http://127.0.0.1:8011",
    )
    .await;
}
