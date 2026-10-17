use axum::{body::Body, extract::State, http::{header, HeaderMap, HeaderValue, StatusCode}, routing::{get, post}, Json, Router};
use serde_json::{json, Map, Value};
use std::{sync::Arc, time::Duration};

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend { Mock, Laya, Jev }
impl Backend {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s { "mock" => Ok(Self::Mock), "laya" => Ok(Self::Laya), "jev" => Ok(Self::Jev), _ => Err(format!("unknown provider: {s}")) }
    }
}

#[derive(Clone)]
pub struct Config {
    pub backend: Backend,
    pub laya_url: String,
    pub jev_url: String,
    pub jev_key: Option<String>,
    pub gateway_key: Option<String>,
}

#[derive(Clone)]
pub struct AppState { pub cfg: Config, pub client: reqwest::Client }

pub fn app(cfg: Config) -> Router {
    let client = reqwest::Client::builder().timeout(Duration::from_secs(120)).build().expect("client");
    Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models))
        .with_state(Arc::new(AppState { cfg, client }))
}

fn error(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error":{"message":message}})))
}
fn authorized(cfg: &Config, headers: &HeaderMap) -> bool {
    match &cfg.gateway_key {
        None => true,
        Some(secret) => headers.get(header::AUTHORIZATION).and_then(|x| x.to_str().ok()) == Some(format!("Bearer {secret}").as_str())
    }
}

async fn models(State(st): State<Arc<AppState>>, headers: HeaderMap) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) { return Err(error(StatusCode::UNAUTHORIZED,"Invalid bearer token")); }
    if st.cfg.backend == Backend::Jev { return forward(&st, "/v1/models", reqwest::Method::GET, None).await; }
    let models = json!({"data":[{"id":"jev-latest","name":"Local Laya-compatible model","description":"Jev-compatible interface backed by local Laya or deterministic mock","release_date":null},{"id":"jev-preview","name":"Local Laya-compatible model","description":"Compatibility alias","release_date":null}]});
    Ok((StatusCode::OK, Json(models)).into_response())
}

use axum::response::IntoResponse;
async fn systemone(State(st): State<Arc<AppState>>, headers: HeaderMap, Json(body): Json<Value>) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) { return Err(error(StatusCode::UNAUTHORIZED,"Invalid bearer token")); }
    validate(&body).map_err(|e| error(StatusCode::UNPROCESSABLE_ENTITY,&e))?;
    match st.cfg.backend {
        Backend::Mock => Ok((StatusCode::OK, Json(mock_decide(&body))).into_response()),
        Backend::Laya => forward(&st, "/v1/systemone", reqwest::Method::POST, Some(body)).await,
        Backend::Jev => forward(&st, "/v1/systemone", reqwest::Method::POST, Some(body)).await,
    }
}

fn validate(v: &Value) -> Result<(), String> {
    let obj = v.as_object().ok_or("Request must be a JSON object")?;
    if !obj.get("state").is_some_and(|s| !s.is_null()) { return Err("state is required".into()); }
    let questions = obj.get("questions").and_then(Value::as_object).ok_or("questions must be an object")?;
    if questions.is_empty() { return Err("questions cannot be empty".into()); }
    for (name, q) in questions {
        let ty = q.get("type").and_then(Value::as_str).ok_or(format!("{name}: missing type"))?;
        if !["choice","score","noul"].contains(&ty) { return Err(format!("{name}: unsupported question type {ty}")); }
        if q.get("instructions").is_none() { return Err(format!("{name}: missing instructions")); }
        if ty == "choice" {
            let cs = q.get("criteria").and_then(Value::as_object).ok_or(format!("{name}: choice criteria must be an object"))?;
            if cs.len() < 2 { return Err(format!("{name}: choice requires two or more options")); }
        }
        if ty == "score" {
            let cs = q.get("criteria").and_then(Value::as_array).ok_or(format!("{name}: score criteria must be an array"))?;
            if cs.len() < 2 { return Err(format!("{name}: score requires two or more levels")); }
        }
    }
    Ok(())
}

async fn forward(st: &AppState, path: &str, method: reqwest::Method, payload: Option<Value>) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    let (base, key) = match st.cfg.backend {
        Backend::Laya => (&st.cfg.laya_url, None),
        Backend::Jev => (&st.cfg.jev_url, st.cfg.jev_key.as_deref()),
        Backend::Mock => unreachable!(),
    };
    let url = format!("{}{}", base.trim_end_matches('/'), path);
    let mut req = st.client.request(method, &url);
    if let Some(key) = key { req = req.bearer_auth(key); }
    if let Some(payload) = payload { req = req.json(&payload); }
    let upstream = req.send().await.map_err(|e| error(StatusCode::BAD_GATEWAY, &format!("upstream unavailable: {e}")))?;
    let code = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
    let content_type = upstream.headers().get(header::CONTENT_TYPE).cloned();
    let bytes = upstream.bytes().await.map_err(|e| error(StatusCode::BAD_GATEWAY,&format!("upstream body error: {e}")))?;
    let mut response = axum::response::Response::builder().status(code);
    if let Some(ct) = content_type { response = response.header(header::CONTENT_TYPE, ct); }
    response.body(Body::from(bytes)).map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR,&e.to_string()))
}

pub fn mock_decide(v: &Value) -> Value {
    let mut answers = Map::new();
    let questions = v["questions"].as_object().expect("validated request");
    for (name, q) in questions {
        let answer = match q["type"].as_str().unwrap_or("") {
            "choice" => {
                let criteria = q["criteria"].as_object().unwrap();
                let n = criteria.len() as f64;
                let probabilities: Map<String, Value> = criteria.keys().map(|key| (key.clone(), json!(1.0/n))).collect();
                json!({"type":"choice","choice":criteria.keys().next().unwrap(),"confidence":1.0/n,"probabilities":probabilities})
            },
            "score" => {
                let arr = q["criteria"].as_array().unwrap();
                let n = arr.len() as f64;
                let legend: Map<String,Value> = arr.iter().enumerate().map(|(i,x)| (i.to_string(),x.clone())).collect();
                let probs: Map<String,Value> = arr.iter().enumerate().map(|(i,_)| (i.to_string(),json!(1.0/n))).collect();
                json!({"type":"score","score":(n-1.0)/2.0,"confidence":1.0/n,"legend":legend,"probabilities":probs})
            },
            _ => json!({"type":"noul","noul":0.5})
        };
        answers.insert(name.clone(),answer);
    }
    json!({"model":v.get("model").and_then(Value::as_str).unwrap_or("jev-latest"),"answers":answers,"usage":{"input_tokens":0,"output_tokens":0}})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_all_decision_types() {
        let body=json!({"state":"test","questions":{"x":{"type":"choice","instructions":"Pick","criteria":{"a":"A","b":"B"}},"y":{"type":"score","instructions":"Rate","criteria":["low","high"]},"z":{"type":"noul","instructions":"Yes?"}}});
        assert!(validate(&body).is_ok());
        let result=mock_decide(&body);
        assert_eq!(result["answers"]["z"]["noul"],0.5);
        assert_eq!(result["answers"]["x"]["probabilities"]["a"],0.5);
        assert_eq!(result["answers"]["y"]["score"],0.5);
    }
    #[test]
    fn rejects_invalid_questions() {
        assert!(validate(&json!({"state":"test","questions":{}})).is_err());
    }
}
