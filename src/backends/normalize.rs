use serde_json::{json, Map, Value};
use std::collections::BTreeMap;

pub(super) fn distribution(
    value: &Value,
    keys: &[String],
) -> Result<BTreeMap<String, f64>, String> {
    let object = value
        .as_object()
        .ok_or("missing probability distribution")?;
    if object.len() != keys.len() {
        return Err("probability keys differ from criteria".into());
    }
    let mut output = BTreeMap::new();
    for key in keys {
        let p = object
            .get(key)
            .and_then(Value::as_f64)
            .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
            .ok_or("invalid or missing probability")?;
        output.insert(key.clone(), p);
    }
    let sum: f64 = output.values().sum();
    // Upstreams commonly round each of up to 255 values to four decimals.
    if sum <= 0.0 || (sum - 1.0).abs() > 0.02 {
        return Err("probabilities must sum to one (tolerance 0.02)".into());
    }
    for p in output.values_mut() {
        *p /= sum;
    }
    Ok(output)
}

pub(super) fn certainty(probabilities: &BTreeMap<String, f64>) -> f64 {
    let entropy: f64 = probabilities
        .values()
        .filter(|p| **p > 0.0)
        .map(|p| -p * p.ln())
        .sum();
    (1.0 - entropy / (probabilities.len() as f64).ln()).clamp(0.0, 1.0)
}

pub(super) fn response(request: &Value, source: &Value) -> Result<Value, String> {
    let model = source["model"]
        .as_str()
        .filter(|m| !m.is_empty())
        .ok_or("missing response model")?;
    let mut answers = Map::new();
    let mut confidences = Map::new();
    for (id, q) in request["questions"].as_object().unwrap() {
        let raw = &source["answers"][id];
        let kind = q["type"].as_str().unwrap();
        if raw["type"] != kind {
            return Err(format!("question {id}: missing answer or wrong type"));
        }
        if raw["confidence"].as_f64().is_some_and(f64::is_finite) {
            confidences.insert(id.clone(), raw["confidence"].clone());
        }
        let answer = match kind {
            "choice" => {
                let keys: Vec<_> = q["criteria"].as_object().unwrap().keys().cloned().collect();
                let p = distribution(&raw["probabilities"], &keys)?;
                let choice = raw["choice"]
                    .as_str()
                    .filter(|key| p.contains_key(*key))
                    .ok_or("choice not in criteria")?;
                json!({"type":"choice","choice":choice,"confidence":certainty(&p),"probabilities":p})
            }
            "score" => {
                let levels = q["criteria"].as_array().unwrap();
                let keys: Vec<_> = (0..levels.len()).map(|i| i.to_string()).collect();
                let p = distribution(&raw["probabilities"], &keys)?;
                let score = raw["score"]
                    .as_f64()
                    .filter(|s| s.is_finite() && (0.0..=(levels.len() - 1) as f64).contains(s))
                    .ok_or("invalid score")?;
                let legend: Map<String, Value> = levels
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i.to_string(), v.clone()))
                    .collect();
                json!({"type":"score","score":score,"confidence":certainty(&p),"probabilities":p,"legend":legend})
            }
            "noul" => {
                let p = raw["noul"]
                    .as_f64()
                    .filter(|p| p.is_finite() && (0.0..=1.0).contains(p))
                    .ok_or("invalid noul probability")?;
                json!({"type":"noul","noul":p})
            }
            _ => unreachable!(),
        };
        answers.insert(id.clone(), answer);
    }
    let usage_reported = source["usage"]["input_tokens"].as_u64().is_some()
        && source["usage"]["output_tokens"].as_u64().is_some();
    if source.get("usage").is_some() && !usage_reported {
        return Err("invalid usage token counts".into());
    }
    let usage = if usage_reported {
        json!({"input_tokens":source["usage"]["input_tokens"],"output_tokens":source["usage"]["output_tokens"]})
    } else {
        json!({"input_tokens":0,"output_tokens":0})
    };
    let usage_source = source["backend_metadata"]["usage_source"]
        .as_str()
        .unwrap_or(if usage_reported {
            "reported"
        } else {
            "unavailable"
        });
    Ok(json!({"model":model,"answers":answers,"usage":usage,
        "backend_metadata":{"confidence_definition":"1 - normalized entropy","source_confidences":confidences,"usage_source":usage_source}}))
}
