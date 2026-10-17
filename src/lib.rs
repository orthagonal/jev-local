use axum::{
    extract::State,
    http::{header, HeaderMap, StatusCode},
    routing::{get, post},
    Json, Router,
};
use serde_json::{json, Map, Value};
use std::{sync::Arc, time::Duration};

pub mod backends;
pub mod benchmark;

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Backend {
    Mock,
    Laya,
    Jev,
    Named(String),
}
impl Backend {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "mock" => Ok(Self::Mock),
            "laya" => Ok(Self::Laya),
            "jev" => Ok(Self::Jev),
            _ if backends::valid_id(s) => Ok(Self::Named(s.into())),
            _ => Err(format!("invalid provider: {s}")),
        }
    }
    pub fn id(&self) -> &str {
        match self {
            Self::Mock => "mock",
            Self::Laya => "laya",
            Self::Jev => "jev",
            Self::Named(id) => id,
        }
    }
}

#[derive(Clone)]
pub struct Config {
    pub backend: Backend,
    pub laya_url: String,
    pub jev_url: String,
    pub jev_key: Option<String>,
    pub gateway_key: Option<String>,
    pub benchmark_dataset_dir: Option<std::path::PathBuf>,
    pub registry: Option<backends::RegistryConfig>,
}

#[derive(Clone)]
pub struct AppState {
    pub cfg: Config,
    pub client: reqwest::Client,
    pub benchmark_gate: Arc<tokio::sync::Semaphore>,
    pub registry: Arc<backends::Registry>,
}

pub fn app(cfg: Config) -> Router {
    try_app(cfg).expect("valid gateway configuration")
}

pub fn try_app(cfg: Config) -> Result<Router, String> {
    let registry = Arc::new(backends::Registry::build(&cfg)?);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(120))
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .map_err(|_| "cannot build HTTP client".to_owned())?;
    Ok(Router::new()
        .route("/health", get(|| async { Json(json!({"status": "ok"})) }))
        .route("/v1/systemone", post(systemone))
        .route("/v1/models", get(models))
        .route("/v1/backends", get(backend_catalog))
        .route("/v1/backends/:id", get(backend_details))
        .route("/v1/backends/:id/models", get(backend_models))
        .route("/v1/backends/:id/systemone", post(backend_inference))
        .route("/v1/benchmarks/datasets", get(benchmark::datasets))
        .route("/v1/benchmarks/datasets/:id", get(benchmark::dataset))
        .route("/v1/benchmarks/compare", post(benchmark::compare))
        .with_state(Arc::new(AppState {
            cfg,
            client,
            registry,
            benchmark_gate: Arc::new(tokio::sync::Semaphore::new(1)),
        })))
}

fn error(status: StatusCode, message: &str) -> (StatusCode, Json<Value>) {
    (status, Json(json!({"error":{"message":message}})))
}
fn authorized(cfg: &Config, headers: &HeaderMap) -> bool {
    match &cfg.gateway_key {
        None => true,
        Some(secret) => {
            headers
                .get(header::AUTHORIZATION)
                .and_then(|x| x.to_str().ok())
                == Some(format!("Bearer {secret}").as_str())
        }
    }
}

async fn models(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    let backend = selected(&st, &headers)?;
    let response = if st.cfg.registry.is_some()
        || backend.profile.response_mode == backends::ResponseMode::Normalized
    {
        backend.catalog_models()
    } else {
        backend.models(&st.client).await.map_err(backend_error)?
    };
    Ok(response.into_response(backend))
}

async fn systemone(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    let backend = selected(&st, &headers)?;
    Ok(backend
        .infer(&st.client, body, None, backend.profile.timeout_ms)
        .await
        .map_err(backend_error)?
        .into_response(backend))
}

fn backend_error(err: backends::BackendError) -> (StatusCode, Json<Value>) {
    error(err.status, &err.message)
}

fn selected<'a>(
    st: &'a AppState,
    headers: &HeaderMap,
) -> Result<&'a backends::Backend, (StatusCode, Json<Value>)> {
    let id = headers
        .get("x-decision-backend")
        .map(|v| v.to_str())
        .transpose()
        .map_err(|_| error(StatusCode::UNPROCESSABLE_ENTITY, "Invalid backend header"))?
        .unwrap_or(&st.registry.default_backend);
    st.registry.get(id).map_err(backend_error)
}

async fn backend_catalog(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    Ok(Json(
        json!({"default_backend":st.registry.default_backend,"data":st.registry.backends.values().map(backends::Backend::metadata).collect::<Vec<_>>()}),
    ))
}

async fn backend_details(
    State(st): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    headers: HeaderMap,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    Ok(Json(
        st.registry.get(&id).map_err(backend_error)?.metadata(),
    ))
}

async fn backend_models(
    State(st): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    headers: HeaderMap,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    let backend = st.registry.get(&id).map_err(backend_error)?;
    Ok(backend
        .models(&st.client)
        .await
        .map_err(backend_error)?
        .into_response(backend))
}

async fn backend_inference(
    State(st): State<Arc<AppState>>,
    axum::extract::Path(id): axum::extract::Path<String>,
    headers: HeaderMap,
    Json(body): Json<Value>,
) -> Result<axum::response::Response, (StatusCode, Json<Value>)> {
    if !authorized(&st.cfg, &headers) {
        return Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"));
    }
    let backend = st.registry.get(&id).map_err(backend_error)?;
    Ok(backend
        .infer(&st.client, body, None, backend.profile.timeout_ms)
        .await
        .map_err(backend_error)?
        .into_response(backend))
}

fn validate(v: &Value) -> Result<(), String> {
    let obj = v.as_object().ok_or("Request must be a JSON object")?;
    if !obj.get("state").is_some_and(|s| !s.is_null()) {
        return Err("state is required".into());
    }
    let questions = obj
        .get("questions")
        .and_then(Value::as_object)
        .ok_or("questions must be an object")?;
    if questions.is_empty() {
        return Err("questions cannot be empty".into());
    }
    for (name, q) in questions {
        let ty = q
            .get("type")
            .and_then(Value::as_str)
            .ok_or(format!("{name}: missing type"))?;
        if !["choice", "score", "noul"].contains(&ty) {
            return Err(format!("{name}: unsupported question type {ty}"));
        }
        if q.get("instructions").is_none() {
            return Err(format!("{name}: missing instructions"));
        }
        if ty == "choice" {
            let cs = q
                .get("criteria")
                .and_then(Value::as_object)
                .ok_or(format!("{name}: choice criteria must be an object"))?;
            if cs.len() < 2 {
                return Err(format!("{name}: choice requires two or more options"));
            }
        }
        if ty == "score" {
            let cs = q
                .get("criteria")
                .and_then(Value::as_array)
                .ok_or(format!("{name}: score criteria must be an array"))?;
            if cs.len() < 2 {
                return Err(format!("{name}: score requires two or more levels"));
            }
        }
    }
    Ok(())
}

pub fn mock_decide(v: &Value) -> Value {
    let mut answers = Map::new();
    let questions = v["questions"].as_object().expect("validated request");
    for (name, q) in questions {
        let answer = match q["type"].as_str().unwrap_or("") {
            "choice" => {
                let criteria = q["criteria"].as_object().unwrap();
                let n = criteria.len() as f64;
                let probabilities: Map<String, Value> = criteria
                    .keys()
                    .map(|key| (key.clone(), json!(1.0 / n)))
                    .collect();
                json!({"type":"choice","choice":criteria.keys().next().unwrap(),"confidence":1.0/n,"probabilities":probabilities})
            }
            "score" => {
                let arr = q["criteria"].as_array().unwrap();
                let n = arr.len() as f64;
                let legend: Map<String, Value> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, x)| (i.to_string(), x.clone()))
                    .collect();
                let probs: Map<String, Value> = arr
                    .iter()
                    .enumerate()
                    .map(|(i, _)| (i.to_string(), json!(1.0 / n)))
                    .collect();
                json!({"type":"score","score":(n-1.0)/2.0,"confidence":1.0/n,"legend":legend,"probabilities":probs})
            }
            _ => json!({"type":"noul","noul":0.5}),
        };
        answers.insert(name.clone(), answer);
    }
    json!({"model":v.get("model").and_then(Value::as_str).unwrap_or("jev-latest"),"answers":answers,"usage":{"input_tokens":0,"output_tokens":0}})
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn validates_all_decision_types() {
        let body = json!({"state":"test","questions":{"x":{"type":"choice","instructions":"Pick","criteria":{"a":"A","b":"B"}},"y":{"type":"score","instructions":"Rate","criteria":["low","high"]},"z":{"type":"noul","instructions":"Yes?"}}});
        assert!(validate(&body).is_ok());
        let result = mock_decide(&body);
        assert_eq!(result["answers"]["z"]["noul"], 0.5);
        assert_eq!(result["answers"]["x"]["probabilities"]["a"], 0.5);
        assert_eq!(result["answers"]["y"]["score"], 0.5);
    }
    #[test]
    fn rejects_invalid_questions() {
        assert!(validate(&json!({"state":"test","questions":{}})).is_err());
    }
}
