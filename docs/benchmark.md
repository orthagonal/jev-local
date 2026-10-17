# Model Comparison Benchmark

`POST /v1/benchmarks/compare` evaluates labeled examples against Laya and Jev.
The response contains separate accuracy, calibration, latency, throughput, and
failure results for each provider, plus the dataset version, run options, provider
order, start timestamp, and total wall time. Save this JSON to compare runs.

## Run a comparison

Start Laya as described in the main README, and set `JEV_API_KEY` in the shell
that starts the gateway. The benchmark uses `LAYA_BASE_URL` and `JEV_BASE_URL`
from the gateway configuration. It works independently of the provider chosen
for ordinary `/v1/systemone` calls, so the gateway can remain in mock mode:

```powershell
cargo run --release -- --provider mock
```

From another PowerShell terminal:

```powershell
$body = Get-Content examples/benchmark.json -Raw
$report = Invoke-RestMethod -Method Post `
    -Uri http://127.0.0.1:8080/v1/benchmarks/compare `
    -ContentType application/json -Body $body
$report | ConvertTo-Json -Depth 40 | Set-Content benchmark-report.json
$report.results | Select-Object provider, status, quality, latency, throughput
```

All benchmark routes use the gateway's existing `JEV_LOCAL_API_KEY`
authentication when configured. Add `-Headers @{Authorization="Bearer <gateway-key>"}`
to these requests when needed. The hosted Jev key stays on the server.

The default providers are `["laya", "jev"]`. Selecting Jev without a configured
nonempty API key returns `422` before running either provider. Each enabled Jev
request, including warmup and repetitions, may consume account credits. The
`TEST_LAYA` flag only controls automated live tests; the benchmark route uses the
providers selected in its request.

For an offline trial, change `providers` to `["mock"]`. To evaluate only Laya,
use `["laya"]`. Mock results exercise the benchmark and have no inference quality
meaning.

## Choose a scenario

```powershell
# List available datasets and their case/question counts.
Invoke-RestMethod http://127.0.0.1:8080/v1/benchmarks/datasets

# Inspect a dataset, including requests and expected labels.
Invoke-RestMethod http://127.0.0.1:8080/v1/benchmarks/datasets/sentiment

# Select the sentiment dataset instead of support-triage.
$request = Get-Content examples/benchmark.json -Raw | ConvertFrom-Json
$request.dataset = "sentiment"
$body = $request | ConvertTo-Json -Depth 40
Invoke-RestMethod -Method Post `
    -Uri http://127.0.0.1:8080/v1/benchmarks/compare `
    -ContentType application/json -Body $body
```

The built-in `support-triage` scenario exercises choice and noul; `sentiment`
exercises ordinal scores. These are tiny hand-labeled examples to verify the
workflow. Use a larger, representative labeled dataset to assess model quality.

## Bring your own dataset

Set `dataset` in the comparison request to an inline object:

```json
{
  "dataset": {
    "id": "refund-routing",
    "version": "1",
    "description": "Support tickets labeled by our team",
    "cases": [
      {
        "id": "duplicate-charge",
        "request": {
          "state": "I was charged twice. Please refund the duplicate payment.",
          "questions": {
            "team": {
              "type": "choice",
              "instructions": "Which team should handle this request?",
              "criteria": {"billing": "Payments and refunds", "technical": "Bugs and outages"}
            }
          }
        },
        "expected": {"team": "billing"}
      }
    ]
  },
  "providers": ["laya", "jev"],
  "options": {"warmup": 1, "repetitions": 3, "concurrency": 1}
}
```

For a reusable named dataset, save only the dataset object to
`my-datasets/refund-routing.json`, then start the gateway with:

```powershell
cargo run --release -- --benchmark-dataset-dir ./my-datasets
# Equivalent configuration: BENCHMARK_DATASET_DIR=./my-datasets
```

Now select it with `"dataset": "refund-routing"`. Files are read at request time,
so changing a file requires no rebuild or restart. Bump `version` when changing
examples or labels. Filename and dataset `id` must match; IDs and case IDs use
1..64 letters, digits, underscores, or dashes. Custom filenames cannot shadow
the built-in IDs. Dataset files are limited to 4 MiB; inline requests use the
gateway's default 2 MiB JSON body limit.

Each case contains an ordinary inference `request` and exactly one expected label
per question. Only `request` is sent upstream; expected labels are never sent.
Choice labels are criterion keys, noul labels are booleans, and score labels are
integer level indices starting at zero. Cases must have unique IDs. A dataset
contains 1..1000 cases, with at most 64 questions per case, 255 choice options,
and 32 score levels. Upstream limits still apply.

Use optional `models` overrides when selecting provider-specific model IDs:

```json
{"dataset":"sentiment", "models":{"laya":"english", "jev":"jev-latest"}}
```

Without an override, each dataset request's `model` is preserved; omitting it
leaves the upstream to choose its default. `observed_models` records the model
IDs returned by successful measured requests.

## Run options

| Option | Default | Range / meaning |
|---|---|---|
| `repetitions` | 1 | 1..20 complete passes over the dataset |
| `concurrency` | 1 | 1..32 requests in flight within one provider |
| `warmup` | 1 | 0..10 extra requests, cycling through cases from the beginning |
| `timeout_ms` | 120000 | 1..120000 per upstream request, including response reading |
| `calibration_bins` | 10 | 1..50 bins of equal width for ECE |
| `include_details` | false | Include predicted labels and per-question scores for each measured case |

Each provider is limited to 10000 total requests including warmup, and 100000
measured labeled decisions. All input validation happens before inference.
Warmup failures are listed separately; warmup requests never contribute to
measured quality, latency, or throughput.

This route returns the completed report synchronously. Set your HTTP client's
timeout to accommodate the full run. Each provider runs sequentially in the
requested order, with bounded concurrency within its run. A second benchmark
on the same gateway returns `409` while one is running. Run on an otherwise idle
gateway and keep concurrency and dataset versions fixed when comparing results.
Use concurrency 1 for serial latency, then increase it for a throughput scenario.
Network time is included, so these numbers describe the deployed services rather
than isolated model execution. Requests use the same shared HTTP client and
connection pool as ordinary gateway traffic. Preload Laya before running if its
first model load exceeds the per-request timeout.

## Metric definitions

All quality values are micro averages over labeled questions, with an additional
breakdown in `by_type`. Repetitions contribute repeated observations, not new
independent examples.

| Result field | Definition |
|---|---|
| `quality.accuracy` | Correct decisions / all expected decisions; failed requests count as incorrect |
| `quality.accuracy_on_success` | Correct decisions / scored decisions from valid successful requests |
| `quality.score_mae` | Mean absolute difference between returned numeric score and expected level index |
| `quality.calibration.brier_score` | Mean sum of squared errors between the categorical probabilities and the one-hot label; range 0..2, including noul represented as `[1-p, p]` |
| `quality.calibration.negative_log_likelihood` | Mean `-ln(p_true)`; probabilities clipped to at least `1e-15` for the log |
| `quality.calibration.expected_calibration_error` | Sum over bins of `count / total * abs(accuracy - mean_confidence)` |
| `latency.all_requests` | Minimum, mean, p50, p95, p99, maximum milliseconds for all measured attempts |
| `latency.successful_requests` | Same latency statistics, restricted to valid successful responses |
| `throughput.attempted_requests_per_second` | Measured completed attempts / provider wall time |
| `throughput.successful_requests_per_second` | Valid successful requests / provider wall time |
| `throughput.scored_decisions_per_second` | Scored questions / provider wall time |

Choice accuracy uses the returned choice. Score accuracy uses the level with the
highest probability, choosing the lowest index on ties; MAE uses the returned
numeric score. Noul predicts true at probability >= 0.5. Calibration uses the
probability assigned to the predicted answer, never the providers' `confidence`
fields, which have different definitions. Probabilities must be finite, in
0..1, contain exactly the expected keys, and sum to one within 0.01. Rounded
distributions within that tolerance are normalized before scoring. ECE bins
are `[lower, upper)`, with probability 1 included in the last bin.

Latency runs from immediately before the upstream request until its response
has been read and JSON decoded; quality scoring is excluded. For HTTP errors it
includes reading the error body; for connection/timeout errors it ends when that
error is observed. Percentiles use nearest rank. Throughput measures the wall
time to finish the measured workload, including scheduling and quality scoring.

Provider results have status `ok`, `partial`, or `failed`. HTTP errors, timeouts,
invalid JSON, missing answers, and invalid probabilities fail the entire request.
Failures include the case ID, repetition, HTTP status when available, latency,
and an error description. There is no retry or fallback to mock. Compare failure
rates and scored coverage alongside calibration: calibration and MAE exclude
failed requests, and are `null` if there are no valid observations. A completed
benchmark returns HTTP `200` even if a provider fails; inspect each result's
status and failure counts.

Definitions follow the [Jev response contract](https://jev-api.com/docs/api),
[Laya HTTP API](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md),
[Guo et al.'s ECE formulation](https://arxiv.org/abs/1706.04599), and
[categorical Brier score](https://scikit-learn.org/stable/modules/model_evaluation.html#brier-score-loss).

## Verify the implementation without inference

```powershell
cargo test --lib --test benchmark --test routes --test forwarding
```

These tests use local controlled upstream servers, with known probability
distributions, errors, delays, and concurrency. `cargo test --all-targets` also
includes the existing conditional live route tests.
