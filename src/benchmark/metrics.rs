use super::dataset::Case;
use serde::Serialize;
use serde_json::{json, Value};

#[derive(Debug, Serialize)]
pub struct QuestionScore {
    pub question: String,
    pub kind: String,
    pub expected: Value,
    pub predicted: Value,
    pub correct: bool,
    pub confidence: f64,
    pub brier_score: f64,
    pub negative_log_likelihood: f64,
    pub absolute_error: Option<f64>,
}

fn probability(value: &Value) -> Result<f64, String> {
    value
        .as_f64()
        .filter(|v| v.is_finite() && (0.0..=1.0).contains(v))
        .ok_or_else(|| "invalid or missing probability".into())
}

fn distribution(answer: &Value, keys: &[String]) -> Result<Vec<f64>, String> {
    let values = answer["probabilities"]
        .as_object()
        .ok_or("missing probabilities")?;
    if values.len() != keys.len() {
        return Err("probability keys do not match criteria".into());
    }
    let mut probabilities = keys
        .iter()
        .map(|key| probability(&answer["probabilities"][key]))
        .collect::<Result<Vec<_>, _>>()?;
    let sum: f64 = probabilities.iter().sum();
    if (sum - 1.0).abs() > 0.01 {
        return Err("probabilities must sum to one (tolerance 0.01)".into());
    }
    // Accommodate rounded API output while using a true distribution for metrics.
    for p in &mut probabilities {
        *p /= sum;
    }
    Ok(probabilities)
}

pub fn score(case: &Case, response: &Value) -> Result<Vec<QuestionScore>, String> {
    if !response["model"].as_str().is_some_and(|s| !s.is_empty()) {
        return Err("missing response model id".into());
    }
    let mut scores = Vec::new();
    for (id, question) in case.request["questions"].as_object().unwrap() {
        let answer = &response["answers"][id];
        let kind = question["type"].as_str().unwrap();
        if answer["type"] != kind {
            return Err(format!("question {id}: missing answer or wrong type"));
        }
        let expected = &case.expected[id];
        let (probabilities, target, predicted_index, predicted, absolute_error) = match kind {
            "choice" => {
                let keys: Vec<String> = question["criteria"]
                    .as_object()
                    .unwrap()
                    .keys()
                    .cloned()
                    .collect();
                let probabilities = distribution(answer, &keys)?;
                let predicted = answer["choice"].as_str().ok_or("missing choice")?;
                let index = keys
                    .iter()
                    .position(|key| key == predicted)
                    .ok_or("choice is not a criterion")?;
                let target = keys
                    .iter()
                    .position(|key| expected == key.as_str())
                    .unwrap();
                (probabilities, target, index, json!(predicted), None)
            }
            "score" => {
                let n = question["criteria"].as_array().unwrap().len();
                let keys = (0..n).map(|i| i.to_string()).collect::<Vec<_>>();
                let probabilities = distribution(answer, &keys)?;
                let value = answer["score"]
                    .as_f64()
                    .filter(|v| v.is_finite() && (0.0..=(n - 1) as f64).contains(v))
                    .ok_or("score outside criteria range")?;
                let mut index = 0;
                for i in 1..n {
                    if probabilities[i] > probabilities[index] {
                        index = i;
                    }
                }
                let target = expected.as_u64().unwrap() as usize;
                (
                    probabilities,
                    target,
                    index,
                    json!(index),
                    Some((value - target as f64).abs()),
                )
            }
            "noul" => {
                let p = probability(&answer["noul"])?;
                let predicted = p >= 0.5;
                (
                    vec![1.0 - p, p],
                    usize::from(expected.as_bool().unwrap()),
                    usize::from(predicted),
                    json!(predicted),
                    None,
                )
            }
            _ => unreachable!(),
        };
        let brier_score = probabilities
            .iter()
            .enumerate()
            .map(|(i, p)| (p - if i == target { 1.0 } else { 0.0 }).powi(2))
            .sum();
        scores.push(QuestionScore {
            question: id.clone(),
            kind: kind.into(),
            expected: expected.clone(),
            predicted,
            correct: predicted_index == target,
            confidence: probabilities[predicted_index],
            brier_score,
            negative_log_likelihood: -probabilities[target].max(1e-15).ln(),
            absolute_error,
        });
    }
    Ok(scores)
}

#[derive(Debug, Serialize)]
pub struct CalibrationBin {
    pub lower: f64,
    pub upper: f64,
    pub count: usize,
    pub accuracy: Option<f64>,
    pub mean_confidence: Option<f64>,
}

#[derive(Debug, Serialize)]
pub struct Calibration {
    pub brier_score: f64,
    pub negative_log_likelihood: f64,
    pub expected_calibration_error: f64,
    pub bins: Vec<CalibrationBin>,
}

#[derive(Debug, Serialize)]
pub struct Quality {
    pub expected_decisions: usize,
    pub scored_decisions: usize,
    pub correct_decisions: usize,
    pub accuracy: f64,
    pub accuracy_on_success: Option<f64>,
    pub calibration: Option<Calibration>,
    pub score_mae: Option<f64>,
}

pub fn quality(scores: &[&QuestionScore], expected: usize, bin_count: usize) -> Quality {
    let n = scores.len();
    let correct = scores.iter().filter(|s| s.correct).count();
    let mut calibration = None;
    if n > 0 {
        let mut counts = vec![0usize; bin_count];
        let mut confidence = vec![0.0; bin_count];
        let mut corrects = vec![0usize; bin_count];
        for score in scores {
            let index = ((score.confidence * bin_count as f64) as usize).min(bin_count - 1);
            counts[index] += 1;
            confidence[index] += score.confidence;
            corrects[index] += usize::from(score.correct);
        }
        let mut ece = 0.0;
        let bins = (0..bin_count)
            .map(|i| {
                ece += (corrects[i] as f64 - confidence[i]).abs() / n as f64;
                CalibrationBin {
                    lower: i as f64 / bin_count as f64,
                    upper: (i + 1) as f64 / bin_count as f64,
                    count: counts[i],
                    accuracy: (counts[i] > 0).then(|| corrects[i] as f64 / counts[i] as f64),
                    mean_confidence: (counts[i] > 0).then(|| confidence[i] / counts[i] as f64),
                }
            })
            .collect();
        calibration = Some(Calibration {
            brier_score: scores.iter().map(|s| s.brier_score).sum::<f64>() / n as f64,
            negative_log_likelihood: scores
                .iter()
                .map(|s| s.negative_log_likelihood)
                .sum::<f64>()
                / n as f64,
            expected_calibration_error: ece,
            bins,
        });
    }
    let errors: Vec<f64> = scores.iter().filter_map(|s| s.absolute_error).collect();
    Quality {
        expected_decisions: expected,
        scored_decisions: n,
        correct_decisions: correct,
        accuracy: if expected > 0 {
            correct as f64 / expected as f64
        } else {
            0.0
        },
        accuracy_on_success: (n > 0).then(|| correct as f64 / n as f64),
        calibration,
        score_mae: (!errors.is_empty()).then(|| errors.iter().sum::<f64>() / errors.len() as f64),
    }
}

#[derive(Debug, Serialize)]
pub struct Latency {
    pub min_ms: f64,
    pub mean_ms: f64,
    pub p50_ms: f64,
    pub p95_ms: f64,
    pub p99_ms: f64,
    pub max_ms: f64,
}

pub fn latency(mut values: Vec<f64>) -> Option<Latency> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let percentile = |p: f64| values[((p * values.len() as f64).ceil() as usize).saturating_sub(1)];
    Some(Latency {
        min_ms: values[0],
        mean_ms: values.iter().sum::<f64>() / values.len() as f64,
        p50_ms: percentile(0.5),
        p95_ms: percentile(0.95),
        p99_ms: percentile(0.99),
        max_ms: *values.last().unwrap(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn binary_case() -> Case {
        serde_json::from_value(json!({"id":"one", "request":{"state":"test", "questions": {
            "q":{"type":"noul", "instructions":"Yes?"}}}, "expected":{"q":true}}))
        .unwrap()
    }

    #[test]
    fn calibration_uses_probability_and_has_known_values() {
        let scores = score(&binary_case(), &json!({"model":"test", "answers":{"q":{"type":"noul", "noul":0.8, "confidence":0.01}}})).unwrap();
        let summary = quality(&scores.iter().collect::<Vec<_>>(), 2, 10);
        assert_eq!(summary.accuracy, 0.5); // One additional failed decision.
        assert_eq!(summary.accuracy_on_success, Some(1.0));
        let calibration = summary.calibration.unwrap();
        assert!((calibration.brier_score - 0.08).abs() < 1e-12);
        assert!((calibration.negative_log_likelihood + 0.8f64.ln()).abs() < 1e-12);
        assert!((calibration.expected_calibration_error - 0.2).abs() < 1e-12);
        assert_eq!(calibration.bins[8].count, 1);
    }

    #[test]
    fn perfect_and_impossible_predictions_have_finite_metrics() {
        for (p, expected_brier, expected_accuracy) in [(1.0, 0.0, 1.0), (0.0, 2.0, 0.0)] {
            let scores = score(
                &binary_case(),
                &json!({"model":"test", "answers":{"q":{"type":"noul", "noul":p}}}),
            )
            .unwrap();
            let summary = quality(&scores.iter().collect::<Vec<_>>(), 1, 10);
            assert_eq!(summary.accuracy, expected_accuracy);
            let calibration = summary.calibration.unwrap();
            assert_eq!(calibration.brier_score, expected_brier);
            assert!(calibration.negative_log_likelihood.is_finite());
            assert_eq!(calibration.bins[9].count, 1);
        }
    }

    #[test]
    fn invalid_distributions_fail_instead_of_improving_the_score() {
        let case: Case = serde_json::from_value(json!({"id":"one", "request":{"state":"test", "questions": {
            "q":{"type":"choice", "instructions":"Pick", "criteria":{"a":"A", "b":"B"}}}}, "expected":{"q":"a"}})).unwrap();
        for probabilities in [
            json!({"a":0.9,"b":0.9}),
            json!({"a":1.0}),
            json!({"a":-0.2,"b":1.2}),
        ] {
            assert!(score(&case, &json!({"model":"test", "answers":{"q":{"type":"choice", "choice":"a", "probabilities":probabilities}}})).is_err());
        }
        assert!(quality(&[], 1, 10).calibration.is_none());
    }

    #[test]
    fn latency_uses_nearest_rank_percentiles() {
        let latency = latency((1..=100).map(f64::from).rev().collect()).unwrap();
        assert_eq!(latency.mean_ms, 50.5);
        assert_eq!(latency.p50_ms, 50.0);
        assert_eq!(latency.p95_ms, 95.0);
        assert_eq!(latency.p99_ms, 99.0);
    }
}
