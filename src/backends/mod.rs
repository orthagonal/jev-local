//! Configured decision backends shared by the API and benchmarks.
mod normalize;
mod simulation;

use axum::{
    body::Body,
    http::{header, HeaderValue, StatusCode},
    response::Response,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::atomic::AtomicU64,
    time::Duration,
};

pub use simulation::{Fixture, Simulation};

#[derive(Clone, Copy, Debug, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum Adapter {
    Systemone,
    Jevlike,
    Simulation,
}

#[derive(Clone, Copy, Debug, Default, Deserialize, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ResponseMode {
    #[default]
    Normalized,
    Passthrough,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Capabilities {
    pub question_types: Vec<String>,
    pub max_questions: usize,
    pub max_choices: usize,
    pub max_score_levels: usize,
}

impl Default for Capabilities {
    fn default() -> Self {
        Self {
            question_types: vec!["choice".into(), "score".into(), "noul".into()],
            max_questions: 64,
            max_choices: 255,
            max_score_levels: 32,
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Profile {
    pub adapter: Adapter,
    pub base_url: Option<String>,
    pub api_key_env: Option<String>,
    pub require_api_key: bool,
    pub default_model: Option<String>,
    pub model_aliases: BTreeMap<String, String>,
    pub inference_path: String,
    pub models_path: Option<String>,
    pub timeout_ms: u64,
    pub capabilities: Capabilities,
    pub response_mode: ResponseMode,
    pub simulation: Simulation,
}

impl Default for Profile {
    fn default() -> Self {
        Self {
            adapter: Adapter::Systemone,
            base_url: None,
            api_key_env: None,
            require_api_key: false,
            default_model: None,
            model_aliases: BTreeMap::new(),
            inference_path: "/v1/systemone".into(),
            models_path: Some("/v1/models".into()),
            timeout_ms: 120_000,
            capabilities: Capabilities::default(),
            response_mode: ResponseMode::Normalized,
            simulation: Simulation::default(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RegistryConfig {
    pub default_backend: String,
    pub backends: BTreeMap<String, Profile>,
}

pub fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id.len() <= 64
        && id
            .bytes()
            .all(|c| c.is_ascii_alphanumeric() || c == b'-' || c == b'_')
}

impl RegistryConfig {
    pub fn legacy(cfg: &crate::Config) -> Self {
        let mut mock = Profile {
            adapter: Adapter::Simulation,
            response_mode: ResponseMode::Passthrough,
            default_model: Some("jev-latest".into()),
            models_path: None,
            ..Profile::default()
        };
        mock.model_aliases
            .insert("jev-preview".into(), "jev-preview".into());
        let laya = Profile {
            base_url: Some(cfg.laya_url.clone()),
            response_mode: ResponseMode::Passthrough,
            models_path: None,
            ..Profile::default()
        };
        let jev = Profile {
            base_url: Some(cfg.jev_url.clone()),
            response_mode: ResponseMode::Passthrough,
            require_api_key: true,
            ..Profile::default()
        };
        let kev = Profile {
            base_url: Some("http://127.0.0.1:8009".into()),
            api_key_env: Some("KEV_API_KEY".into()),
            default_model: Some("kev-latest".into()),
            ..Profile::default()
        };
        let jevany = Profile {
            base_url: Some("http://127.0.0.1:8008".into()),
            default_model: Some("jevany-latest".into()),
            ..Profile::default()
        };
        let jevlike = Profile {
            adapter: Adapter::Jevlike,
            base_url: Some("http://127.0.0.1:8011".into()),
            api_key_env: Some("JEVLIKE_API_KEY".into()),
            default_model: Some("jevlike".into()),
            inference_path: "/predict".into(),
            capabilities: Capabilities {
                question_types: vec!["choice".into()],
                ..Capabilities::default()
            },
            ..Profile::default()
        };
        Self {
            default_backend: cfg.backend.id().into(),
            backends: BTreeMap::from([
                ("mock".into(), mock),
                ("laya".into(), laya),
                ("jev".into(), jev),
                ("kev".into(), kev),
                ("jevany".into(), jevany),
                ("jevlike".into(), jevlike),
            ]),
        }
    }
}

pub struct Registry {
    pub default_backend: String,
    pub backends: BTreeMap<String, Backend>,
}

pub struct Backend {
    pub id: String,
    pub profile: Profile,
    key: Option<String>,
    counter: AtomicU64,
}

#[derive(Debug)]
pub struct BackendError {
    pub status: StatusCode,
    pub upstream_status: Option<u16>,
    pub message: String,
}

impl BackendError {
    fn invalid(message: impl Into<String>) -> Self {
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            upstream_status: None,
            message: message.into(),
        }
    }
    fn upstream(message: impl Into<String>, status: Option<u16>) -> Self {
        Self {
            status: StatusCode::BAD_GATEWAY,
            upstream_status: status,
            message: message.into(),
        }
    }
}

pub struct BackendResponse {
    pub status: StatusCode,
    pub content_type: Option<HeaderValue>,
    pub body: Vec<u8>,
    pub metadata: Value,
}

impl BackendResponse {
    fn json(status: StatusCode, body: &Value) -> Self {
        Self {
            status,
            content_type: Some(HeaderValue::from_static("application/json")),
            body: serde_json::to_vec(body).expect("JSON response"),
            metadata: Value::Null,
        }
    }
    pub fn into_response(self, backend: &Backend) -> Response {
        let mut response = Response::builder()
            .status(self.status)
            .header("x-decision-backend", &backend.id)
            .header(
                "x-decision-simulated",
                if backend.simulated() { "true" } else { "false" },
            );
        if let Some(content_type) = self.content_type {
            response = response.header(header::CONTENT_TYPE, content_type);
        }
        if self.metadata["confidence_definition"].is_string() {
            response = response.header("x-decision-confidence-definition", "normalized-entropy");
        }
        let usage_source = if backend.simulated() {
            Some("simulated")
        } else {
            self.metadata["usage_source"].as_str()
        };
        if let Some(source) =
            usage_source.filter(|s| ["simulated", "reported", "unavailable"].contains(s))
        {
            response = response.header("x-decision-usage-source", source);
        }
        response
            .body(Body::from(self.body))
            .expect("validated response headers")
    }
}

impl Registry {
    pub fn build(cfg: &crate::Config) -> Result<Self, String> {
        let config = cfg
            .registry
            .clone()
            .unwrap_or_else(|| RegistryConfig::legacy(cfg));
        if !config.backends.contains_key(&config.default_backend) {
            return Err("default_backend must name a registered backend".into());
        }
        if config.backends.is_empty() || config.backends.len() > 64 {
            return Err("register 1..64 backends".into());
        }
        let mut backends = BTreeMap::new();
        for (id, profile) in config.backends {
            profile
                .validate()
                .map_err(|e| format!("backend {id}: {e}"))?;
            if !valid_id(&id) {
                return Err("backend ids must use 1..64 letters/digits/_/-".into());
            }
            let mut key = match &profile.api_key_env {
                Some(name) => std::env::var(name).ok().filter(|v| !v.trim().is_empty()),
                None => None,
            };
            if (cfg.registry.is_none() && id == "jev")
                || (profile.api_key_env.as_deref() == Some("JEV_API_KEY") && cfg.jev_key.is_some())
            {
                key = cfg.jev_key.clone().filter(|v| !v.trim().is_empty());
            }
            if let Some(key) = &key {
                HeaderValue::from_str(&format!("Bearer {key}"))
                    .map_err(|_| format!("backend {id}: invalid credential header"))?;
            }
            backends.insert(
                id.clone(),
                Backend {
                    id,
                    profile,
                    key,
                    counter: AtomicU64::new(0),
                },
            );
        }
        Ok(Self {
            default_backend: config.default_backend,
            backends,
        })
    }

    pub fn get(&self, id: &str) -> Result<&Backend, BackendError> {
        self.backends
            .get(id)
            .ok_or_else(|| BackendError::invalid(format!("unknown backend: {id}")))
    }
}

impl Profile {
    fn validate(&self) -> Result<(), String> {
        if !(1..=120_000).contains(&self.timeout_ms) {
            return Err("timeout_ms must be 1..120000".into());
        }
        if self.adapter != Adapter::Simulation {
            let url = reqwest::Url::parse(self.base_url.as_deref().ok_or("base_url is required")?)
                .map_err(|_| "invalid base_url")?;
            if !["http", "https"].contains(&url.scheme())
                || url.host_str().is_none()
                || !url.username().is_empty()
                || url.password().is_some()
                || url.query().is_some()
                || url.fragment().is_some()
            {
                return Err(
                    "base_url must be HTTP(S), without credentials, query, or fragment".into(),
                );
            }
        }
        for path in std::iter::once(&self.inference_path).chain(self.models_path.iter()) {
            if !path.starts_with('/') || path.starts_with("//") || path.contains(['?', '#']) {
                return Err(
                    "endpoint paths must start with / and contain no query or fragment".into(),
                );
            }
        }
        if self
            .default_model
            .as_ref()
            .is_some_and(|m| m.trim().is_empty())
            || self
                .model_aliases
                .iter()
                .any(|(a, b)| a.trim().is_empty() || b.trim().is_empty())
        {
            return Err("model names and aliases must be nonempty".into());
        }
        if self.api_key_env.as_ref().is_some_and(|name| {
            name.is_empty()
                || name
                    .bytes()
                    .any(|c| !c.is_ascii_alphanumeric() && c != b'_')
        }) {
            return Err("invalid api_key_env name".into());
        }
        let caps = &self.capabilities;
        if caps.question_types.is_empty()
            || caps
                .question_types
                .iter()
                .any(|t| !["choice", "score", "noul"].contains(&t.as_str()))
            || caps.question_types.iter().collect::<BTreeSet<_>>().len()
                != caps.question_types.len()
            || !(1..=64).contains(&caps.max_questions)
            || !(2..=4096).contains(&caps.max_choices)
            || !(2..=4096).contains(&caps.max_score_levels)
        {
            return Err("invalid capability types or limits".into());
        }
        if self.adapter == Adapter::Jevlike && caps.question_types != ["choice"] {
            return Err("jevlike currently supports choice questions only".into());
        }
        self.simulation.validate()?;
        Ok(())
    }
}

impl Backend {
    pub fn simulated(&self) -> bool {
        self.profile.adapter == Adapter::Simulation
    }

    pub fn check(&self, payload: &Value) -> Result<(), BackendError> {
        crate::validate(payload).map_err(BackendError::invalid)?;
        if self.profile.require_api_key && self.key.is_none() {
            return Err(BackendError::invalid(format!(
                "backend {} requires a configured API key",
                self.id
            )));
        }
        let questions = payload["questions"].as_object().unwrap();
        let caps = &self.profile.capabilities;
        if questions.len() > caps.max_questions {
            return Err(BackendError::invalid("too many questions for backend"));
        }
        for (id, q) in questions {
            if !caps.question_types.iter().any(|t| q["type"] == *t) {
                return Err(BackendError::invalid(format!(
                    "backend {} does not support question type {} ({id})",
                    self.id, q["type"]
                )));
            }
            if q["type"] == "choice" && q["criteria"].as_object().unwrap().len() > caps.max_choices
                || q["type"] == "score"
                    && q["criteria"].as_array().unwrap().len() > caps.max_score_levels
            {
                return Err(BackendError::invalid("criteria exceed backend limits"));
            }
        }
        Ok(())
    }

    fn payload(
        &self,
        mut payload: Value,
        model_override: Option<&str>,
    ) -> Result<Value, BackendError> {
        let requested = model_override.map(str::to_owned).or_else(|| {
            payload
                .get("model")
                .and_then(Value::as_str)
                .map(str::to_owned)
        });
        if payload.get("model").is_some_and(|m| !m.is_string()) {
            return Err(BackendError::invalid("model must be a string"));
        }
        if requested
            .as_ref()
            .is_some_and(|model| model.trim().is_empty())
        {
            return Err(BackendError::invalid("model must be nonempty"));
        }
        if requested.as_deref() == Some("decision-default")
            && self.profile.default_model.is_none()
            && !self.profile.model_aliases.contains_key("decision-default")
        {
            payload.as_object_mut().unwrap().remove("model");
            return Ok(payload);
        }
        let model = match requested.as_deref() {
            Some("decision-default" | "jev-latest" | "jev-preview")
                if self.profile.default_model.is_some() =>
            {
                self.profile
                    .model_aliases
                    .get(requested.as_deref().unwrap())
                    .or(self.profile.default_model.as_ref())
            }
            Some(model) => self.profile.model_aliases.get(model),
            None => self.profile.default_model.as_ref(),
        }
        .cloned()
        .or(requested);
        if let Some(model) = model {
            payload["model"] = json!(model);
        }
        Ok(payload)
    }

    pub fn metadata(&self) -> Value {
        json!({"id":self.id, "adapter":self.profile.adapter, "simulated":self.simulated(),
            "credential_configured":self.key.is_some(), "requires_api_key":self.profile.require_api_key,
            "configuration_ready":!self.profile.require_api_key || self.key.is_some(),
            "default_model":self.profile.default_model,"model_aliases":self.profile.model_aliases,
            "capabilities":self.profile.capabilities,"response_mode":self.profile.response_mode,
            "timeout_ms":self.profile.timeout_ms})
    }

    pub async fn infer(
        &self,
        client: &reqwest::Client,
        payload: Value,
        model_override: Option<&str>,
        timeout_ms: u64,
    ) -> Result<BackendResponse, BackendError> {
        self.check(&payload)?;
        let original = payload.clone();
        let payload = self.payload(payload, model_override)?;
        tokio::time::timeout(Duration::from_millis(timeout_ms.min(self.profile.timeout_ms)), async {
            let response = match self.profile.adapter {
                Adapter::Simulation => self.simulate(&original, &payload).await?,
                Adapter::Systemone => self.http(client, &self.profile.inference_path, Some(&payload)).await?,
                Adapter::Jevlike => self.jevlike(client, &payload).await?,
            };
            if self.profile.response_mode == ResponseMode::Passthrough { return Ok(response); }
            if !response.status.is_success() {
                return Ok(BackendResponse::json(response.status, &json!({"error":{"message":format!("backend {} returned HTTP {}",self.id,response.status.as_u16())}})));
            }
            let body: Value = serde_json::from_slice(&response.body).map_err(|_| BackendError::upstream("upstream returned invalid JSON", Some(response.status.as_u16())))?;
            let mut body = normalize::response(&payload, &body).map_err(|e| BackendError::upstream(format!("invalid inference response: {e}"), Some(response.status.as_u16())))?;
            let metadata = body.as_object_mut().unwrap().remove("backend_metadata").unwrap();
            let mut output = BackendResponse::json(response.status,&body);
            output.metadata = metadata;
            Ok(output)
        }).await.map_err(|_| BackendError::upstream("upstream request timed out", None))?
    }

    async fn http(
        &self,
        client: &reqwest::Client,
        path: &str,
        payload: Option<&Value>,
    ) -> Result<BackendResponse, BackendError> {
        let url = format!(
            "{}{}",
            self.profile
                .base_url
                .as_deref()
                .unwrap()
                .trim_end_matches('/'),
            path
        );
        let mut request = if let Some(body) = payload {
            client.post(url).json(body)
        } else {
            client.get(url)
        };
        if let Some(key) = &self.key {
            request = request.bearer_auth(key);
        }
        let mut response = request
            .send()
            .await
            .map_err(|_| BackendError::upstream("upstream unavailable: transport error", None))?;
        let status = response.status();
        let content_type = response.headers().get(header::CONTENT_TYPE).cloned();
        let mut body = Vec::new();
        while let Some(chunk) = response.chunk().await.map_err(|_| {
            BackendError::upstream("upstream body read failed", Some(status.as_u16()))
        })? {
            if body.len() + chunk.len() > 4 * 1024 * 1024 {
                return Err(BackendError::upstream(
                    "upstream response exceeds 4 MiB",
                    Some(status.as_u16()),
                ));
            }
            body.extend_from_slice(&chunk);
        }
        Ok(BackendResponse {
            status,
            content_type,
            body,
            metadata: Value::Null,
        })
    }

    pub async fn models(&self, client: &reqwest::Client) -> Result<BackendResponse, BackendError> {
        if self.profile.require_api_key && self.key.is_none() {
            return Err(BackendError::invalid(
                "backend requires a configured API key",
            ));
        }
        if !self.simulated() {
            if let Some(path) = &self.profile.models_path {
                return tokio::time::timeout(
                    Duration::from_millis(self.profile.timeout_ms),
                    self.http(client, path, None),
                )
                .await
                .map_err(|_| BackendError::upstream("upstream request timed out", None))?;
            }
        }
        Ok(self.catalog_models())
    }

    pub fn catalog_models(&self) -> BackendResponse {
        let mut ids: BTreeSet<String> = self.profile.model_aliases.keys().cloned().collect();
        ids.extend([
            "decision-default".into(),
            "jev-latest".into(),
            "jev-preview".into(),
        ]);
        if let Some(model) = &self.profile.default_model {
            ids.insert(model.clone());
        }
        let data:Vec<_> = ids.iter().map(|id| json!({"id":id,"name":self.id,"description":"Gateway model alias","release_date":null})).collect();
        BackendResponse::json(StatusCode::OK, &json!({"data":data}))
    }

    async fn jevlike(
        &self,
        client: &reqwest::Client,
        payload: &Value,
    ) -> Result<BackendResponse, BackendError> {
        let mut answers = serde_json::Map::new();
        let mut model = self
            .profile
            .default_model
            .clone()
            .unwrap_or_else(|| "jevlike".into());
        for (id, q) in payload["questions"].as_object().unwrap() {
            let criteria = q["criteria"].as_object().unwrap();
            let context = if let Some(s) = payload["state"].as_str() {
                s.to_owned()
            } else {
                payload["state"].to_string()
            };
            let instructions = q["instructions"]
                .as_str()
                .map(str::to_owned)
                .unwrap_or_else(|| q["instructions"].to_string());
            let options: Vec<_> = criteria
                .iter()
                .map(|(key, value)| {
                    format!(
                        "{key}: {}",
                        value
                            .as_str()
                            .map(str::to_owned)
                            .unwrap_or_else(|| value.to_string())
                    )
                })
                .collect();
            let response = self.http(client, &self.profile.inference_path,Some(&json!({"context":format!("{context}\n\n{instructions}"),"options":options,"model":payload.get("model")}))).await?;
            if !response.status.is_success() {
                return Ok(response);
            }
            let native: Value = serde_json::from_slice(&response.body).map_err(|_| {
                BackendError::upstream("invalid jevlike JSON", Some(response.status.as_u16()))
            })?;
            let probabilities = native["probabilities"]
                .as_array()
                .filter(|v| v.len() == criteria.len())
                .ok_or_else(|| {
                    BackendError::upstream(
                        "jevlike probability count differs from options",
                        Some(response.status.as_u16()),
                    )
                })?;
            let distribution: serde_json::Map<String, Value> = criteria
                .keys()
                .cloned()
                .zip(probabilities.iter().cloned())
                .collect();
            let distribution = normalize::distribution(
                &Value::Object(distribution),
                &criteria.keys().cloned().collect::<Vec<_>>(),
            )
            .map_err(|e| BackendError::upstream(e, Some(response.status.as_u16())))?;
            let chosen = distribution
                .iter()
                .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                .unwrap()
                .0;
            answers.insert(id.clone(),json!({"type":"choice","choice":chosen,"probabilities":distribution,"confidence":normalize::certainty(&distribution)}));
            if let Some(name) = native["model"].as_str() {
                model = name.into();
            }
        }
        Ok(BackendResponse::json(
            StatusCode::OK,
            &json!({"model":model,"answers":answers,"usage":{"input_tokens":0,"output_tokens":0},"backend_metadata":{"usage_source":"unavailable"}}),
        ))
    }
}
