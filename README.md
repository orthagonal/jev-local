# jev-local

**A Test Environment for Jev-S1 request/response interface.** A lightweight HTTP mock of Jev interface written in Rust and backed by Laya-S1. Test locally on your own GPU, then switch to paid hosted Jev without changing your application's endpoint or payloads!

**Status: v0.1 prototype.** Compatible **route and successful response shape**, not a certified byte-for-byte clone of every Jev behavior. HTTP request validation, errors, model metadata, and upstream limits differ. Local Laya predictions are not identical to Jev predictions.

## Requirements

- Rust stable toolchain (install via https://rustup.rs), Cargo
- For actual local AI classification: Python 3.10+ and `laya[serve]` (installed separately, as Laya provides its own mature Jev-compatible server)
- For hosted Jev: an API key

## Build

```sh
cargo build --release
cargo test
```

On Windows, executable path is `target\\release\\jev-local.exe`; on macOS/Linux it is `target/release/jev-local`.

## Automated route tests

The tests build the real gateway router in-process; you do not need to start the
gateway executable. Mock route tests and controlled upstream HTTP tests always
run. Real inference tests are additional and enabled by the environment:

| Tests | Enabled when | Upstream |
|---|---|---|
| Mock routes and forwarding | Always | None; forwarding tests start temporary local HTTP servers |
| Laya live routes | `TEST_LAYA=true` (also accepts `1`) | `LAYA_BASE_URL`, default `http://127.0.0.1:8000` |
| Jev live routes | `JEV_API_KEY` is nonempty | `JEV_BASE_URL`, default `https://jev-api.com` |

```powershell
# Run all enabled tests; show RUN/SKIP messages for live providers.
cargo test --all-targets -- --nocapture

# Also enable Laya. Start laya-serve separately before running this.
$env:TEST_LAYA = "true"
$env:LAYA_BASE_URL = "http://127.0.0.1:8000"
cargo test --all-targets -- --nocapture

# Jev tests run automatically if JEV_API_KEY is already set in this shell.
# To run only the conditional live tests:
cargo test --test live -- --nocapture

# Explicitly run only offline tests, regardless of environment variables:
cargo test --lib --test routes --test forwarding --test benchmark
```

In Bash, enable Laya with `TEST_LAYA=true cargo test --all-targets -- --nocapture`.
Both live providers can run in the same test invocation. Tests read process
environment variables, not `.env` files or gateway command-line arguments.

Each enabled live provider checks `/health`, `/v1/models`, and one
`POST /v1/systemone` containing all three question types. They check the response
contract, probability ranges and totals, score bounds, and token counts; they do
not require identical predictions across providers. Jev calls use your configured
account and may consume credits. The gateway returns its own model metadata in
Laya mode, so only the inference POST contacts Laya.

Disabled live tests return early and print `SKIP` with the reason. Rust's standard
test harness labels these returns `ok`; use `--nocapture` to see whether inference
actually ran. Once enabled, provider failures fail the test without falling back
to mock. The gateway allows 120 seconds per upstream request; preload/warm Laya
before testing if its initial model download would take longer. Laya must accept
unauthenticated requests because this gateway does not forward a Laya API key.

Test sources are in `tests/routes.rs`, `tests/forwarding.rs`, and `tests/live.rs`,
with shared fixtures and contract checks in `tests/common/mod.rs`. The response
checks follow the [Jev API reference](https://jev-api.com/docs/api) and
[Laya HTTP API](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md).

## Model Comparison Benchmark

Run `POST /v1/benchmarks/compare` with a labeled dataset to compare Laya and Jev
on accuracy, calibration (Brier score, log loss, and ECE), latency percentiles,
and throughput. Select a built-in scenario by name, load custom dataset files,
or supply an inline dataset. The response reports errors and coverage alongside
scores, with configurable warmup, repetitions, concurrency, and model IDs.

Start the gateway with Laya's URL and your Jev key configured, then post
[`examples/benchmark.json`](examples/benchmark.json) to the comparison route.
The benchmark's providers are independent of the ordinary inference provider.
List scenarios at `GET /v1/benchmarks/datasets`.

See [benchmark setup, dataset format, and metric definitions](docs/benchmark.md).
The included starter datasets demonstrate the workflow; they are too small to
draw model quality conclusions.

## 1. Zero-cost mock mode (no model needed)

```sh
cargo run --release -- --provider mock
```

In another terminal:

```sh
curl -s http://127.0.0.1:8080/v1/systemone -H 'content-type: application/json' --data-binary @examples/door.json
```

Mock mode intentionally returns **uniform probabilities**, not a meaningful classification. It is suitable for transport integration and deterministic contract tests. Do not use mock results to evaluate model quality.

## 2. Real local Laya mode (free inference)

Install/start upstream Laya in a separate terminal:

```sh
python -m venv .venv
# Windows PowerShell: .\\.venv\\Scripts\\python.exe -m pip install 'laya[serve]'
# Unix: .venv/bin/python -m pip install 'laya[serve]'
# Activate the environment, or call the executable inside .venv directly:
# Windows: .\\.venv\\Scripts\\laya-serve.exe
# Unix: .venv/bin/laya-serve
```

It listens at `http://localhost:8000`; first real prediction may download model weights. Then start gateway:

```sh
cargo run --release -- --provider laya --laya-url http://127.0.0.1:8000
```

Call the **same** curl command as mock mode. All inference happens in the local Laya service; this Rust prototype is a gateway, **not native Rust Candle inference**. Upstream Laya also natively offers the Jev endpoint, so a gateway is optional for local-only use. This project is useful for stable routing, future policy, observability and switching providers.

## 3. Hosted Jev passthrough

```sh
# PowerShell: $env:JEV_API_KEY='your-key'
# bash: export JEV_API_KEY='your-key'
cargo run --release -- --provider jev
```

The gateway forwards POST `/v1/systemone` and GET `/v1/models` to the configured hosted endpoint with the server-side secret. Responses preserve the upstream HTTP status, content-type and body. The gateway does not send your API key to callers.

## Optional local gateway authentication

Set `JEV_LOCAL_API_KEY` in environment. Clients must send `Authorization: Bearer <key>` to the gateway. Keep the service bound to loopback unless you configure network security and TLS separately. Do not expose raw Laya or the gateway publicly without appropriate protection.

## Current limitations

- `choice`, `score`, `noul` requests: supported through mock or upstream forwarding.
- GET `/v1/models` in mock/Laya mode provides local compatibility metadata, **not** an exact replica of hosted Jev model metadata.
- Only a subset of documented Jev request validation is implemented; upstream providers may differ on edge cases.
- Requests forwarded to Laya may produce Laya-specific errors, which are passed through unchanged.
- Native Candle-based Laya inference is **not included in v0.1**. The Rust HTTP gateway uses upstream Laya's existing Jev-compatible server.
- No batching, rate-limiting, websocket or shadow comparison in v0.1.
- Model weights are not redistributed.

## Architecture

```
Your app (any language) --> jev-local (Rust, :8080)
                              |-- mock (uniform fixtures)
                              |-- laya (localhost:8000, real local model)
                              `-- jev  (hosted paid endpoint)
```

## Sources / upstream

- Jev wire contract: https://jev-api.com/docs/api
- Laya source and self-hosted Jev-compatible API: https://github.com/NandhaKishorM/laya
- Laya HTTP docs: https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md

This repository is independently developed and is not affiliated with or endorsed by Jev or Laya maintainers.
