//! Labeled dataset evaluation over the same HTTP inference endpoints as the gateway.
pub mod dataset;
mod metrics;
mod runner;

use crate::{authorized, error, AppState};
use axum::{
    extract::{Path, State},
    http::{HeaderMap, StatusCode},
    Json,
};
use dataset::{Dataset, Selection};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
};

type RouteError = (StatusCode, Json<Value>);

pub type Provider = String;

#[derive(Clone, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Options {
    pub repetitions: usize,
    pub concurrency: usize,
    pub warmup: usize,
    pub timeout_ms: u64,
    pub calibration_bins: usize,
    pub include_details: bool,
}

impl Default for Options {
    fn default() -> Self {
        Self {
            repetitions: 1,
            concurrency: 1,
            warmup: 1,
            timeout_ms: 120_000,
            calibration_bins: 10,
            include_details: false,
        }
    }
}

fn default_providers() -> Vec<Provider> {
    vec!["laya".into(), "jev".into()]
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompareRequest {
    pub dataset: Selection,
    #[serde(default = "default_providers")]
    pub providers: Vec<Provider>,
    #[serde(default)]
    pub options: Options,
    #[serde(default)]
    pub models: BTreeMap<Provider, String>,
}

fn check_auth(st: &AppState, headers: &HeaderMap) -> Result<(), RouteError> {
    if authorized(&st.cfg, headers) {
        Ok(())
    } else {
        Err(error(StatusCode::UNAUTHORIZED, "Invalid bearer token"))
    }
}

fn invalid(message: &str) -> RouteError {
    error(StatusCode::UNPROCESSABLE_ENTITY, message)
}

pub async fn datasets(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
) -> Result<Json<Value>, RouteError> {
    check_auth(&st, &headers)?;
    let data = dataset::catalog(st.cfg.benchmark_dataset_dir.as_deref())
        .await
        .map_err(|e| invalid(&e))?;
    Ok(Json(json!({"data": data})))
}

pub async fn dataset(
    State(st): State<Arc<AppState>>,
    Path(id): Path<String>,
    headers: HeaderMap,
) -> Result<Json<Dataset>, RouteError> {
    check_auth(&st, &headers)?;
    let dataset = dataset::load(&id, st.cfg.benchmark_dataset_dir.as_deref())
        .await
        .map_err(|e| invalid(&e))?;
    Ok(Json(dataset))
}

pub async fn compare(
    State(st): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(request): Json<CompareRequest>,
) -> Result<Json<Value>, RouteError> {
    check_auth(&st, &headers)?;
    let options = &request.options;
    if request.providers.is_empty()
        || request.providers.iter().collect::<BTreeSet<_>>().len() != request.providers.len()
    {
        return Err(invalid(
            "providers must be a nonempty list without duplicates",
        ));
    }
    if !(1..=20).contains(&options.repetitions)
        || !(1..=32).contains(&options.concurrency)
        || options.warmup > 10
        || !(1..=120_000).contains(&options.timeout_ms)
        || !(1..=50).contains(&options.calibration_bins)
    {
        return Err(invalid("options require repetitions 1..20, concurrency 1..32, warmup 0..10, timeout_ms 1..120000, calibration_bins 1..50"));
    }
    if request
        .models
        .iter()
        .any(|(provider, model)| !request.providers.contains(provider) || model.trim().is_empty())
    {
        return Err(invalid(
            "model overrides must be nonempty strings for selected providers",
        ));
    }
    let dataset = match request.dataset {
        Selection::Named(id) => dataset::load(&id, st.cfg.benchmark_dataset_dir.as_deref())
            .await
            .map_err(|e| invalid(&e))?,
        Selection::Inline(data) => {
            data.validate().map_err(|e| invalid(&e))?;
            data
        }
    };
    let questions: usize = dataset.question_counts().values().sum();
    for provider in &request.providers {
        let backend = st.registry.get(provider).map_err(|e| invalid(&e.message))?;
        for case in &dataset.cases {
            backend.check(&case.request).map_err(|e| {
                invalid(&format!(
                    "backend {provider}, case {}: {}",
                    case.id, e.message
                ))
            })?;
        }
    }
    if dataset.cases.len() * options.repetitions + options.warmup > 10_000
        || questions * options.repetitions > 100_000
    {
        return Err(invalid(
            "benchmark exceeds 10000 requests or 100000 measured decisions per provider",
        ));
    }
    // Overlapping benchmark runs would distort latency and throughput comparisons.
    let _permit = st
        .benchmark_gate
        .clone()
        .try_acquire_owned()
        .map_err(|_| error(StatusCode::CONFLICT, "A benchmark is already running"))?;
    let dataset = Arc::new(dataset);
    let started_at_unix_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let started = std::time::Instant::now();
    let mut results = Vec::new();
    for provider in &request.providers {
        results.push(
            runner::run(
                st.clone(),
                dataset.clone(),
                provider.clone(),
                options,
                request.models.get(provider).cloned(),
            )
            .await
            .map_err(|e| error(StatusCode::INTERNAL_SERVER_ERROR, &e))?,
        );
    }
    Ok(Json(json!({
        "schema_version": 1,
        "started_at_unix_ms": started_at_unix_ms,
        "total_duration_ms": started.elapsed().as_secs_f64() * 1000.0,
        "dataset": dataset.metadata(),
        "options": options,
        "models": request.models,
        "provider_order": request.providers,
        "methodology": {
            "execution": "providers run sequentially; bounded concurrency within each provider",
            "accuracy": "micro average over labeled questions; failed requests count as incorrect",
            "calibration": "successful requests only; returned probabilities, not provider confidence",
            "brier_score": "mean sum of squared categorical probability errors; range 0..2, including binary noul",
            "latency": "client-observed adapter time including transport, decoding, and normalization; warmup excluded; nearest-rank percentiles",
            "throughput": "completed requests divided by measured provider wall time; warmup excluded"
        },
        "results": results
    })))
}
