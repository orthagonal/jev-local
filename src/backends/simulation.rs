use super::{normalize, Backend, BackendError, BackendResponse};
use axum::http::StatusCode;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Value};
use std::{collections::BTreeMap, sync::atomic::Ordering, time::Duration};

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default, deny_unknown_fields)]
pub struct Simulation {
    pub latency_ms: u64,
    pub status: u16,
    pub fail_every: Option<u64>,
    pub failure_status: u16,
    pub probabilities: BTreeMap<String, BTreeMap<String, f64>>,
    pub fixtures: Vec<Fixture>,
}

impl Default for Simulation {
    fn default() -> Self {
        Self {
            latency_ms: 0,
            status: 200,
            fail_every: None,
            failure_status: 503,
            probabilities: BTreeMap::new(),
            fixtures: Vec::new(),
        }
    }
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Fixture {
    pub match_request: Value,
    pub response: Value,
    pub status: Option<u16>,
    pub latency_ms: Option<u64>,
}

fn valid_status(status: u16) -> bool {
    status == 200 || (400..=599).contains(&status)
}

impl Simulation {
    pub(super) fn validate(&self) -> Result<(), String> {
        if self.latency_ms > 120_000
            || !valid_status(self.status)
            || !(400..=599).contains(&self.failure_status)
            || self.fail_every == Some(0)
        {
            return Err("invalid simulation latency, HTTP status, or fail_every".into());
        }
        for probabilities in self.probabilities.values() {
            if probabilities.is_empty()
                || probabilities
                    .values()
                    .any(|p| !p.is_finite() || !(0.0..=1.0).contains(p))
                || (probabilities.values().sum::<f64>() - 1.0).abs() > 0.02
            {
                return Err(
                    "simulation probabilities must be finite, in 0..1, and sum to one".into(),
                );
            }
        }
        for fixture in &self.fixtures {
            if !fixture.match_request.is_object()
                || !fixture.response.is_object()
                || fixture.status.is_some_and(|s| !valid_status(s))
                || fixture.latency_ms.is_some_and(|v| v > 120_000)
            {
                return Err("invalid simulation fixture".into());
            }
        }
        Ok(())
    }
}

fn matches(pattern: &Value, request: &Value) -> bool {
    match (pattern, request) {
        (Value::Object(pattern), Value::Object(request)) => pattern
            .iter()
            .all(|(key, value)| request.get(key).is_some_and(|r| matches(value, r))),
        _ => pattern == request,
    }
}

impl Backend {
    pub(super) async fn simulate(
        &self,
        original: &Value,
        payload: &Value,
    ) -> Result<BackendResponse, BackendError> {
        let settings = &self.profile.simulation;
        let count = self.counter.fetch_add(1, Ordering::SeqCst) + 1;
        let fixture = settings
            .fixtures
            .iter()
            .find(|f| matches(&f.match_request, original));
        let latency = fixture
            .and_then(|f| f.latency_ms)
            .unwrap_or(settings.latency_ms);
        if latency > 0 {
            tokio::time::sleep(Duration::from_millis(latency)).await;
        }
        let status = fixture.and_then(|f| f.status).unwrap_or_else(|| {
            if settings.fail_every.is_some_and(|n| count.is_multiple_of(n)) {
                settings.failure_status
            } else {
                settings.status
            }
        });
        let status = StatusCode::from_u16(status).unwrap();
        if let Some(fixture) = fixture {
            return Ok(BackendResponse::json(status, &fixture.response));
        }
        if !status.is_success() {
            return Ok(BackendResponse::json(
                status,
                &json!({"error":{"message":"Simulated backend failure"}}),
            ));
        }
        let mut response = crate::mock_decide(payload);
        for (id, q) in payload["questions"].as_object().unwrap() {
            let Some(configured) = settings.probabilities.get(id) else {
                continue;
            };
            let kind = q["type"].as_str().unwrap();
            let keys: Vec<String> = match kind {
                "choice" => q["criteria"].as_object().unwrap().keys().cloned().collect(),
                "score" => (0..q["criteria"].as_array().unwrap().len())
                    .map(|i| i.to_string())
                    .collect(),
                "noul" => vec!["false".into(), "true".into()],
                _ => unreachable!(),
            };
            let p = normalize::distribution(&json!(configured), &keys)
                .map_err(BackendError::invalid)?;
            let answer = &mut response["answers"][id];
            match kind {
                "choice" => {
                    let choice = p
                        .iter()
                        .max_by(|a, b| a.1.total_cmp(b.1).then_with(|| b.0.cmp(a.0)))
                        .unwrap()
                        .0;
                    *answer = json!({"type":"choice","choice":choice,"confidence":p[choice],"probabilities":p});
                }
                "score" => {
                    let score: f64 = p
                        .iter()
                        .map(|(key, p)| key.parse::<f64>().unwrap() * p)
                        .sum();
                    let legend: Map<String, Value> = q["criteria"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .enumerate()
                        .map(|(i, v)| (i.to_string(), v.clone()))
                        .collect();
                    *answer = json!({"type":"score","score":score,"confidence":p.values().copied().fold(0.0,f64::max),"probabilities":p,"legend":legend});
                }
                "noul" => *answer = json!({"type":"noul","noul":p["true"]}),
                _ => unreachable!(),
            }
        }
        Ok(BackendResponse::json(status, &response))
    }
}
