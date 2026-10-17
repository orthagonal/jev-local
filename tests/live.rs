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

fn test_laya_enabled() -> bool {
    match setting("TEST_LAYA").as_deref() {
        None | Some("0") => false,
        Some(value) if value.eq_ignore_ascii_case("false") => false,
        Some("1") => true,
        Some(value) if value.eq_ignore_ascii_case("true") => true,
        Some(_) => panic!("TEST_LAYA must be true, false, 1, or 0"),
    }
}

async fn check_live_routes(mut cfg: Config) {
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
        Some(fixture()),
        Some("live-test-gateway-secret"),
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_inference(&response.json());
}

#[tokio::test]
async fn laya_live_routes() {
    if !test_laya_enabled() {
        eprintln!("SKIP laya_live_routes: set TEST_LAYA=true to run real Laya inference");
        return;
    }
    eprintln!("RUN laya_live_routes: real inference through the gateway");
    let mut cfg = config(Backend::Laya);
    cfg.laya_url = setting("LAYA_BASE_URL").unwrap_or_else(|| "http://127.0.0.1:8000".into());
    check_live_routes(cfg).await;
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
    check_live_routes(cfg).await;
}
