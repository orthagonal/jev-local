mod common;

use axum::{
    body::Body,
    http::{header, Method, Request, StatusCode},
};
use common::*;
use jev_local::{app, Backend};
use serde_json::json;

#[tokio::test]
async fn health_is_public_even_when_gateway_auth_is_enabled() {
    let mut cfg = config(Backend::Mock);
    cfg.gateway_key = Some("gateway-secret".into());
    let response = request(&app(cfg), Method::GET, "/health", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.json(), json!({"status": "ok"}));
}

#[tokio::test]
async fn mock_models_expose_compatibility_aliases() {
    let response = request(
        &app(config(Backend::Mock)),
        Method::GET,
        "/v1/models",
        None,
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    assert_models(&response.json());
}

#[tokio::test]
async fn mock_inference_returns_exact_deterministic_answers() {
    let router = app(config(Backend::Mock));
    let response = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await;
    response.assert_json_status(StatusCode::OK);
    let body = response.json();
    assert_inference(&body);
    let uniform = 1.0 / 3.0;
    assert_eq!(
        body,
        json!({
            "model": "jev-latest",
            "answers": {
                "vote": {"type": "choice", "choice": "irrelevant", "confidence": uniform,
                    "probabilities": {"yes": uniform, "no": uniform, "irrelevant": uniform}},
                "tension": {"type": "score", "score": 1.0, "confidence": uniform,
                    "legend": {"0": "calm", "1": "concerned", "2": "alarmed"},
                    "probabilities": {"0": uniform, "1": uniform, "2": uniform}},
                "urgency": {"type": "noul", "noul": 0.5}
            },
            "usage": {"input_tokens": 0, "output_tokens": 0}
        })
    );
    let repeated = request(
        &router,
        Method::POST,
        "/v1/systemone",
        Some(fixture()),
        None,
    )
    .await;
    assert_eq!(repeated.body, response.body);
}

#[tokio::test]
async fn mock_defaults_model_and_preserves_explicit_alias() {
    let router = app(config(Backend::Mock));
    for model in [None, Some("jev-preview")] {
        let mut payload = fixture();
        payload.as_object_mut().unwrap().remove("model");
        if let Some(model) = model {
            payload["model"] = json!(model);
        }
        let response = request(&router, Method::POST, "/v1/systemone", Some(payload), None).await;
        response.assert_json_status(StatusCode::OK);
        assert_eq!(response.json()["model"], model.unwrap_or("jev-latest"));
    }
}

#[tokio::test]
async fn protected_routes_require_the_gateway_bearer_token() {
    let mut cfg = config(Backend::Mock);
    cfg.gateway_key = Some("gateway-secret".into());
    let router = app(cfg);
    for (method, path, payload) in [
        (Method::GET, "/v1/models", None),
        (Method::POST, "/v1/systemone", Some(fixture())),
    ] {
        for token in [None, Some("wrong-token"), Some("gateway-secret")] {
            let response = request(&router, method.clone(), path, payload.clone(), token).await;
            let expected = if token == Some("gateway-secret") {
                StatusCode::OK
            } else {
                StatusCode::UNAUTHORIZED
            };
            response.assert_json_status(expected);
            if expected == StatusCode::UNAUTHORIZED {
                assert_eq!(
                    response.json(),
                    json!({"error": {"message": "Invalid bearer token"}})
                );
            }
        }
    }
}

#[tokio::test]
async fn invalid_payloads_return_validation_errors() {
    let router = app(config(Backend::Mock));
    let cases = [
        (json!([]), "Request must be a JSON object"),
        (json!({"questions": {}}), "state is required"),
        (json!({"state": null, "questions": {}}), "state is required"),
        (json!({"state": "test"}), "questions must be an object"),
        (
            json!({"state": "test", "questions": []}),
            "questions must be an object",
        ),
        (
            json!({"state": "test", "questions": {}}),
            "questions cannot be empty",
        ),
        (
            json!({"state": "test", "questions": {"q": {}}}),
            "q: missing type",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "unknown"}}}),
            "q: unsupported question type unknown",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "noul"}}}),
            "q: missing instructions",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": []}}}),
            "q: choice criteria must be an object",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "choice", "instructions": "Pick", "criteria": {"a": "A"}}}}),
            "q: choice requires two or more options",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": {}}}}),
            "q: score criteria must be an array",
        ),
        (
            json!({"state": "test", "questions": {"q": {"type": "score", "instructions": "Rate", "criteria": ["low"]}}}),
            "q: score requires two or more levels",
        ),
    ];
    for (payload, message) in cases {
        let response = request(&router, Method::POST, "/v1/systemone", Some(payload), None).await;
        response.assert_json_status(StatusCode::UNPROCESSABLE_ENTITY);
        assert_eq!(response.json(), json!({"error": {"message": message}}));
    }
}

#[tokio::test]
async fn malformed_json_and_missing_content_type_are_rejected() {
    let router = app(config(Backend::Mock));
    for (content_type, body, expected) in [
        (Some("application/json"), "{", StatusCode::BAD_REQUEST),
        (None, "{}", StatusCode::UNSUPPORTED_MEDIA_TYPE),
        (Some("text/plain"), "{}", StatusCode::UNSUPPORTED_MEDIA_TYPE),
    ] {
        let mut builder = Request::builder().method(Method::POST).uri("/v1/systemone");
        if let Some(content_type) = content_type {
            builder = builder.header(header::CONTENT_TYPE, content_type);
        }
        let response = send(&router, builder.body(Body::from(body)).unwrap()).await;
        assert_eq!(response.status, expected);
    }
}

#[tokio::test]
async fn unknown_routes_and_unsupported_methods_are_rejected() {
    let router = app(config(Backend::Mock));
    for (method, path, expected) in [
        (Method::GET, "/missing", StatusCode::NOT_FOUND),
        (Method::GET, "/v1/systemone", StatusCode::METHOD_NOT_ALLOWED),
        (Method::POST, "/v1/models", StatusCode::METHOD_NOT_ALLOWED),
        (Method::POST, "/health", StatusCode::METHOD_NOT_ALLOWED),
    ] {
        let response = request(&router, method, path, None, None).await;
        assert_eq!(response.status, expected, "{path}");
    }
}
