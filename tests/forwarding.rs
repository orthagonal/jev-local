mod common;

use axum::{
    body::{to_bytes, Body},
    http::{header, HeaderMap, Method, Request, StatusCode, Uri},
    response::Response,
    Router,
};
use common::*;
use jev_local::{app, Backend};
use serde_json::Value;
use std::time::Duration;
use tokio::{net::TcpListener, sync::mpsc, task::JoinHandle};

struct CapturedRequest {
    method: Method,
    uri: Uri,
    headers: HeaderMap,
    body: Vec<u8>,
}

struct Upstream {
    url: String,
    requests: mpsc::UnboundedReceiver<CapturedRequest>,
    task: JoinHandle<()>,
}

impl Upstream {
    async fn start(status: StatusCode, content_type: Option<&str>, body: &[u8]) -> Self {
        let (sender, requests) = mpsc::unbounded_channel();
        let body = body.to_vec();
        let content_type = content_type.map(str::to_owned);
        let router = Router::new().fallback(move |request: Request<Body>| {
            let sender = sender.clone();
            let body = body.clone();
            let content_type = content_type.clone();
            async move {
                let (parts, request_body) = request.into_parts();
                sender
                    .send(CapturedRequest {
                        method: parts.method,
                        uri: parts.uri,
                        headers: parts.headers,
                        body: to_bytes(request_body, 2 * 1024 * 1024)
                            .await
                            .unwrap()
                            .to_vec(),
                    })
                    .unwrap();
                let mut response = Response::builder().status(status);
                if let Some(content_type) = content_type {
                    response = response.header(header::CONTENT_TYPE, content_type);
                }
                response.body(Body::from(body)).unwrap()
            }
        });
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let task = tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });
        Self {
            url,
            requests,
            task,
        }
    }

    async fn received(&mut self) -> CapturedRequest {
        tokio::time::timeout(Duration::from_secs(5), self.requests.recv())
            .await
            .expect("upstream did not receive a request")
            .expect("upstream request channel closed")
    }
}

impl Drop for Upstream {
    fn drop(&mut self) {
        self.task.abort();
    }
}

#[tokio::test]
async fn inference_forwarding_preserves_payload_and_uses_provider_credentials() {
    for backend in [Backend::Laya, Backend::Jev] {
        let upstream_body = br#"{"provider":"fixture","answers":{}}"#;
        let mut upstream =
            Upstream::start(StatusCode::OK, Some("application/json"), upstream_body).await;
        let mut cfg = config(backend.clone());
        // Trailing slash joining is also exercised here.
        cfg.laya_url = format!("{}/", upstream.url);
        cfg.jev_url = format!("{}/", upstream.url);
        cfg.jev_key = Some("hosted-secret".into());
        cfg.gateway_key = Some("gateway-secret".into());
        let response = request(
            &app(cfg),
            Method::POST,
            "/v1/systemone",
            Some(fixture()),
            Some("gateway-secret"),
        )
        .await;
        response.assert_json_status(StatusCode::OK);
        assert_eq!(response.body, upstream_body);

        let captured = upstream.received().await;
        assert_eq!(captured.method, Method::POST);
        assert_eq!(captured.uri.path(), "/v1/systemone");
        assert_eq!(captured.uri.query(), None);
        assert_eq!(captured.headers[header::CONTENT_TYPE], "application/json");
        assert_eq!(
            serde_json::from_slice::<Value>(&captured.body).unwrap(),
            fixture()
        );
        match backend {
            Backend::Jev => assert_eq!(
                captured.headers[header::AUTHORIZATION],
                "Bearer hosted-secret"
            ),
            Backend::Laya => assert!(!captured.headers.contains_key(header::AUTHORIZATION)),
            Backend::Mock => unreachable!(),
        }
    }
}

#[tokio::test]
async fn jev_models_forwards_get_with_hosted_credentials_and_no_body() {
    let body = br#"{"data":[{"id":"upstream-model"}]}"#;
    let mut upstream = Upstream::start(StatusCode::OK, Some("application/json"), body).await;
    let mut cfg = config(Backend::Jev);
    cfg.jev_url = upstream.url.clone();
    cfg.jev_key = Some("hosted-secret".into());
    let response = request(&app(cfg), Method::GET, "/v1/models", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_eq!(response.body, body);
    let captured = upstream.received().await;
    assert_eq!(captured.method, Method::GET);
    assert_eq!(captured.uri.path(), "/v1/models");
    assert_eq!(
        captured.headers[header::AUTHORIZATION],
        "Bearer hosted-secret"
    );
    assert!(captured.body.is_empty());
}

#[tokio::test]
async fn laya_models_are_local_and_do_not_call_upstream() {
    let mut upstream = Upstream::start(
        StatusCode::INTERNAL_SERVER_ERROR,
        None,
        b"must not be called",
    )
    .await;
    let mut cfg = config(Backend::Laya);
    cfg.laya_url = upstream.url.clone();
    let response = request(&app(cfg), Method::GET, "/v1/models", None, None).await;
    response.assert_json_status(StatusCode::OK);
    assert_models(&response.json());
    assert!(matches!(
        upstream.requests.try_recv(),
        Err(mpsc::error::TryRecvError::Empty)
    ));
}

#[tokio::test]
async fn upstream_status_content_type_and_raw_body_are_preserved() {
    for backend in [Backend::Laya, Backend::Jev] {
        for (status, content_type, body) in [
            (
                StatusCode::CREATED,
                Some("application/octet-stream"),
                &b"\x00\xfffixture\n"[..],
            ),
            (
                StatusCode::UNAUTHORIZED,
                Some("application/json"),
                &b"{\"error\":\"invalid upstream key\"}"[..],
            ),
            (
                StatusCode::TOO_MANY_REQUESTS,
                Some("application/problem+json"),
                &b"{\"error\":\"rate limited\"}"[..],
            ),
            (
                StatusCode::SERVICE_UNAVAILABLE,
                Some("text/plain; charset=utf-8"),
                &b"provider unavailable\n"[..],
            ),
            (StatusCode::BAD_GATEWAY, None, &b"no content type"[..]),
        ] {
            let upstream = Upstream::start(status, content_type, body).await;
            let mut cfg = config(backend.clone());
            cfg.laya_url = upstream.url.clone();
            cfg.jev_url = upstream.url.clone();
            cfg.jev_key = Some("hosted-secret".into());
            let response = request(
                &app(cfg),
                Method::POST,
                "/v1/systemone",
                Some(fixture()),
                None,
            )
            .await;
            assert_eq!(response.status, status);
            assert_eq!(
                response
                    .headers
                    .get(header::CONTENT_TYPE)
                    .map(|v| v.to_str().unwrap()),
                content_type
            );
            assert_eq!(response.body, body);
        }
    }
}

#[tokio::test]
async fn rejected_gateway_requests_do_not_reach_either_upstream() {
    for backend in [Backend::Laya, Backend::Jev] {
        let mut upstream = Upstream::start(StatusCode::OK, None, b"must not be called").await;
        let mut cfg = config(backend);
        cfg.laya_url = upstream.url.clone();
        cfg.jev_url = upstream.url.clone();
        cfg.gateway_key = Some("gateway-secret".into());
        let router = app(cfg);
        for (path, method, payload, token, expected) in [
            (
                "/v1/systemone",
                Method::POST,
                Some(fixture()),
                None,
                StatusCode::UNAUTHORIZED,
            ),
            (
                "/v1/models",
                Method::GET,
                None,
                Some("wrong"),
                StatusCode::UNAUTHORIZED,
            ),
            (
                "/v1/systemone",
                Method::POST,
                Some(serde_json::json!({})),
                Some("gateway-secret"),
                StatusCode::UNPROCESSABLE_ENTITY,
            ),
        ] {
            let response = request(&router, method, path, payload, token).await;
            response.assert_json_status(expected);
        }
        assert!(matches!(
            upstream.requests.try_recv(),
            Err(mpsc::error::TryRecvError::Empty)
        ));
    }
}

#[tokio::test]
async fn unreachable_upstreams_return_bad_gateway() {
    for backend in [Backend::Laya, Backend::Jev] {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        drop(listener);
        let mut cfg = config(backend);
        cfg.laya_url = url.clone();
        cfg.jev_url = url;
        let response = request(
            &app(cfg),
            Method::POST,
            "/v1/systemone",
            Some(fixture()),
            None,
        )
        .await;
        response.assert_json_status(StatusCode::BAD_GATEWAY);
        assert!(response.json()["error"]["message"]
            .as_str()
            .unwrap()
            .starts_with("upstream unavailable:"));
    }
}
