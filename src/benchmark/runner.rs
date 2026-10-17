use super::{
    dataset::Dataset,
    metrics::{self, QuestionScore},
    Options, Provider,
};
use crate::AppState;
use serde::Serialize;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    sync::Arc,
    time::Instant,
};
use tokio::task::JoinSet;

#[derive(Serialize)]
struct Attempt {
    case_id: String,
    repetition: usize,
    latency_ms: f64,
    http_status: Option<u16>,
    error: Option<String>,
    model: Option<String>,
    questions: Vec<QuestionScore>,
    backend_metadata: Value,
}

struct CallError {
    status: Option<u16>,
    message: String,
}

async fn infer(
    st: &AppState,
    provider: &str,
    payload: Value,
    model_override: Option<&str>,
    timeout_ms: u64,
) -> Result<(Value, u16, Value), CallError> {
    let backend = st.registry.get(provider).map_err(|e| CallError {
        status: e.upstream_status,
        message: e.message,
    })?;
    let response = backend
        .infer(&st.client, payload, model_override, timeout_ms)
        .await
        .map_err(|e| CallError {
            status: e.upstream_status,
            message: e.message,
        })?;
    if !response.status.is_success() {
        return Err(CallError {
            status: Some(response.status.as_u16()),
            message: format!("upstream returned HTTP {}", response.status),
        });
    }
    let body = serde_json::from_slice(&response.body).map_err(|_| CallError {
        status: Some(response.status.as_u16()),
        message: "upstream returned invalid JSON".into(),
    })?;
    Ok((body, response.status.as_u16(), response.metadata))
}

async fn attempt(
    st: Arc<AppState>,
    dataset: Arc<Dataset>,
    provider: Provider,
    index: usize,
    timeout_ms: u64,
    model: Option<String>,
) -> Attempt {
    let case = &dataset.cases[index % dataset.cases.len()];
    let payload = case.request.clone();
    let started = Instant::now();
    let outcome = infer(&st, &provider, payload, model.as_deref(), timeout_ms).await;
    let mut result = Attempt {
        case_id: case.id.clone(),
        repetition: index / dataset.cases.len() + 1,
        latency_ms: started.elapsed().as_secs_f64() * 1000.0,
        http_status: None,
        error: None,
        model: None,
        questions: Vec::new(),
        backend_metadata: Value::Null,
    };
    match outcome {
        Ok((body, status, metadata)) => {
            result.backend_metadata = metadata;
            result.http_status = Some(status);
            match metrics::score(case, &body) {
                Ok(questions) => {
                    result.questions = questions;
                    result.model = body["model"].as_str().map(str::to_owned);
                }
                Err(error) => result.error = Some(format!("invalid inference response: {error}")),
            }
        }
        Err(error) => {
            result.http_status = error.status;
            result.error = Some(error.message);
        }
    }
    result
}

fn launch(
    tasks: &mut JoinSet<Attempt>,
    st: &Arc<AppState>,
    dataset: &Arc<Dataset>,
    provider: Provider,
    index: usize,
    options: &Options,
    model: &Option<String>,
) {
    tasks.spawn(attempt(
        st.clone(),
        dataset.clone(),
        provider,
        index,
        options.timeout_ms,
        model.clone(),
    ));
}

pub async fn run(
    st: Arc<AppState>,
    dataset: Arc<Dataset>,
    provider: Provider,
    options: &Options,
    model: Option<String>,
) -> Result<Value, String> {
    let mut warmup_errors = Vec::new();
    for index in 0..options.warmup {
        let result = attempt(
            st.clone(),
            dataset.clone(),
            provider.clone(),
            index,
            options.timeout_ms,
            model.clone(),
        )
        .await;
        if result.error.is_some() {
            warmup_errors.push(result);
        }
    }
    let count = dataset.cases.len() * options.repetitions;
    let started = Instant::now();
    let mut tasks = JoinSet::new();
    let mut next = count.min(options.concurrency);
    for index in 0..next {
        launch(
            &mut tasks,
            &st,
            &dataset,
            provider.clone(),
            index,
            options,
            &model,
        );
    }
    let mut attempts = Vec::with_capacity(count);
    while let Some(outcome) = tasks.join_next().await {
        attempts.push(outcome.map_err(|_| "benchmark worker failed".to_owned())?);
        if next < count {
            launch(
                &mut tasks,
                &st,
                &dataset,
                provider.clone(),
                next,
                options,
                &model,
            );
            next += 1;
        }
    }
    let seconds = started.elapsed().as_secs_f64().max(1e-9);
    attempts.sort_by(|a, b| (a.repetition, &a.case_id).cmp(&(b.repetition, &b.case_id)));
    let successes = attempts.iter().filter(|a| a.error.is_none()).count();
    let scores: Vec<_> = attempts.iter().flat_map(|a| &a.questions).collect();
    let counts = dataset.question_counts();
    let expected = counts.values().sum::<usize>() * options.repetitions;
    let mut by_type = BTreeMap::new();
    for (kind, count) in &counts {
        let subset: Vec<_> = scores.iter().copied().filter(|s| &s.kind == kind).collect();
        by_type.insert(
            kind,
            metrics::quality(
                &subset,
                count * options.repetitions,
                options.calibration_bins,
            ),
        );
    }
    let models: BTreeSet<_> = attempts.iter().filter_map(|a| a.model.as_deref()).collect();
    let failures: Vec<_> = attempts.iter().filter(|a| a.error.is_some()).collect();
    Ok(json!({
        "provider": provider,
        "backend": st.registry.get(&provider).map_err(|e|e.message)?.metadata(),
        "simulated": st.registry.get(&provider).map_err(|e|e.message)?.simulated(),
        "status": if successes == count { "ok" } else if successes == 0 { "failed" } else { "partial" },
        "observed_models": models,
        "warmup": {"requests": options.warmup, "failed_requests": warmup_errors.len(), "failures": warmup_errors},
        "requests": {"attempted": count, "successful": successes, "failed": count - successes, "error_rate": (count - successes) as f64 / count as f64},
        "quality": metrics::quality(&scores, expected, options.calibration_bins),
        "by_type": by_type,
        "latency": {
            "all_requests": metrics::latency(attempts.iter().map(|a| a.latency_ms).collect()),
            "successful_requests": metrics::latency(attempts.iter().filter(|a| a.error.is_none()).map(|a| a.latency_ms).collect())
        },
        "throughput": {
            "duration_ms": seconds * 1000.0,
            "attempted_requests_per_second": count as f64 / seconds,
            "successful_requests_per_second": successes as f64 / seconds,
            "scored_decisions_per_second": scores.len() as f64 / seconds
        },
        "failures": failures,
        "details": if options.include_details { Some(&attempts) } else { None }
    }))
}
