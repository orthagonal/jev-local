# jev-local

A Rust server for decision models: one System One HTTP API for real inference,
simulation, and model comparison. Switch the backend in server configuration
while keeping your application's endpoint and request format.

Supports hosted Jev, Laya, KEV, JevAny, a jevlike serving bridge, and arbitrary
System One servers. Register multiple model sizes or deployments with distinct
profile IDs. Simulation profiles provide deterministic probabilities, recorded
responses, delays, and errors without a model or API key.

**Status: prototype.** The gateway provides a common interface. Predictions,
calibration, resource requirements, and upstream limits remain model-specific.

## Start without a model

Requires Rust stable and Cargo:

```powershell
cargo run --release -- --backends-file examples/backends.json
```

In another PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/backends
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/systemone `
    -ContentType application/json -Body (Get-Content examples/door.json -Raw)
```

The default profile is an offline uniform simulation. Add
`-Headers @{"X-Decision-Backend"="simulated-door"}` to select configured
probabilities with the same request. These fixtures are for application testing.

## Use a real backend

Start the relevant model server, then select its profile:

```powershell
cargo run --release -- --backends-file examples/backends.json --provider kev
```

Or edit `default_backend` in your profile file. Existing clients keep calling
`POST /v1/systemone` with `state`, `questions`, and optional `model`. Stable model
aliases map to the backend's model. Individual calls can use a selection header
or `/v1/backends/<id>/systemone`.

| Backend | Adapter | Default service | Question types |
|---|---|---|---|
| Jev | `systemone` | `https://jev-api.com` | choice, score, noul |
| Laya | `systemone` | localhost:8000 | choice, score, noul |
| KEV (`jaredpalmer/kev`) | `systemone` | localhost:8009 | choice, score, noul |
| JevAny | `systemone` | localhost:8008 | choice, score, noul |
| jevlike bridge | `jevlike` | localhost:8011 | choice |
| Simulation | `simulation` | In process | choice, score, noul |

Hosted Jev requires `JEV_API_KEY` in the gateway's environment. Credentials are
referenced by profile configuration. Add another System One server through
configuration without editing Rust. Real runtimes run separately; the gateway
does not distribute or automatically download weights.

[Backend setup, configuration, response semantics, and simulations](docs/backends.md).

## Compare models or simulations

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/benchmarks/compare `
    -ContentType application/json `
    -Body (Get-Content examples/benchmark-simulations.json -Raw)
```

Choose any registered IDs in `providers`. Reports include accuracy, Brier score,
log loss, ECE, latency percentiles, throughput, failures, and simulation status.
Use named datasets or supply labeled cases. Warmup, repetitions, concurrency,
and model selectors are configurable.

`examples/benchmark-models.json` compares Laya, Jev, KEV, and JevAny when their
services are configured. `examples/benchmark.json` retains the original Laya/Jev
comparison. Starter datasets demonstrate the workflow; use representative
labeled data for meaningful accuracy comparisons.

[Benchmark setup, dataset format, and metric definitions](docs/benchmark.md).

## Routes

| Route | Purpose |
|---|---|
| `GET /health` | Gateway liveness |
| `POST /v1/systemone` | Inference or simulation |
| `GET /v1/models` | Selected backend's model aliases |
| `GET /v1/backends` | Registered profiles and capabilities |
| `GET /v1/backends/<id>` | Profile metadata |
| `POST /v1/backends/<id>/systemone` | Select a profile explicitly |
| `GET /v1/backends/<id>/models` | Underlying model listing |
| `GET /v1/benchmarks/datasets` | Dataset catalog |
| `GET /v1/benchmarks/datasets/<id>` | Dataset requests and labels |
| `POST /v1/benchmarks/compare` | Comparison report |

Set `JEV_LOCAL_API_KEY` to require bearer authentication for every route except
`/health`. The server defaults to loopback on port 8080. Backend selection uses
server-registered URLs and credentials. Response headers
`X-Decision-Simulated` and `X-Decision-Backend` identify the source.

## Build and test

```powershell
cargo build --release
# Offline checks, regardless of live credentials:
cargo test --lib --test routes --test forwarding --test benchmark --test backends
# Bridge protocol checks: Python 3.10+, no model dependencies:
python -m unittest discover -s scripts -p test_serve_jevlike.py -v
# All checks, including enabled live providers:
cargo test --all-targets -- --nocapture
```

| Live test | Runs when | Optional service override |
|---|---|---|
| Laya | `TEST_LAYA=true` | `LAYA_BASE_URL` |
| Jev | `JEV_API_KEY` is nonempty | `JEV_BASE_URL` |
| KEV | `TEST_KEV=true` | `KEV_BASE_URL` |
| JevAny | `TEST_JEVANY=true` | `JEVANY_BASE_URL` |
| jevlike bridge | `TEST_JEVLIKE=true` | `JEVLIKE_BASE_URL` |

Start model servers before enabling live tests. Disabled tests print `SKIP`;
Rust's standard harness labels early returns `ok`, so use `--nocapture` to see
which actually ran. Enabled provider failures fail tests. Tests read process
environment, not `.env` files. Jev calls may consume credits.

## Existing CLI compatibility

The original commands still work without a profile file:

```powershell
cargo run --release -- --provider mock
cargo run --release -- --provider laya --laya-url http://127.0.0.1:8000
# Set JEV_API_KEY in this shell first:
cargo run --release -- --provider jev
```

Those modes retain legacy response behavior. Profile files default to normalized
responses with consistent confidence semantics. Reevaluate thresholds when
changing models or confidence definitions. Built-in profiles also include KEV,
JevAny, and jevlike at their default local ports.

This repository is independent of the model providers. Upstream links and
setup instructions are in [the backend guide](docs/backends.md).
