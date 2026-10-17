# Decision backends and simulations

One System One API serves inference, simulation, discovery, and benchmarks. A
backend is a named profile: register multiple sizes, checkpoints, or deployments
of the same model family. Change the default profile to switch existing clients.

## Start offline

```powershell
cargo run --release -- --backends-file examples/backends.json
```

From another PowerShell terminal:

```powershell
Invoke-RestMethod http://127.0.0.1:8080/v1/backends
$body = Get-Content examples/door.json -Raw
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/systemone `
    -ContentType application/json -Body $body
# Same payload, configured nonuniform simulated probabilities:
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8080/v1/systemone `
    -Headers @{"X-Decision-Backend"="simulated-door"} `
    -ContentType application/json -Body $body
```

No model downloads, Python environment, or hosted keys are needed for simulation.
Simulation predicts configured values, not a real model's accuracy or speed.

## Selection and discovery

`POST /v1/systemone` uses the registry's `default_backend`. `--provider <id>` or
`JEV_LOCAL_PROVIDER` overrides that default. For individual calls, use
`X-Decision-Backend: <id>` or `POST /v1/backends/<id>/systemone`. The explicit path
takes precedence over a header. Selection uses only server-registered profiles;
callers cannot supply upstream URLs or keys.

`GET /v1/backends` lists profiles; `GET /v1/backends/<id>` returns capabilities,
simulation status, aliases, adapter, and credential readiness. `configuration_ready`
checks required credentials, not network reachability or model loading. Discovery
omits credentials and upstream URLs. `GET /v1/models` lists gateway aliases for
the selected configured profile; `GET /v1/backends/<id>/models` gets actual upstream
metadata, or local aliases when `models_path` is null. Every route except `/health`
uses gateway bearer authentication when `JEV_LOCAL_API_KEY` is set.

## Profile configuration

`--backends-file <path>` / `DECISION_BACKENDS_FILE` loads a JSON registry once at
startup, replacing the built-in registry. Restart after changing profiles or
credential environment variables. The file is limited to 4 MiB.

```json
{
  "default_backend": "kev-small",
  "backends": {
    "kev-small": {
      "adapter": "systemone", "base_url": "http://127.0.0.1:8009",
      "api_key_env": "KEV_API_KEY", "default_model": "kev-latest"
    },
    "kev-large": {
      "adapter": "systemone", "base_url": "http://127.0.0.1:8010",
      "default_model": "kev-latest"
    },
    "offline": {"adapter": "simulation", "default_model": "offline"}
  }
}
```

| Field | Default / meaning |
|---|---|
| `adapter` | `systemone`, `jevlike`, or `simulation` |
| `base_url` | Required HTTP(S) service URL for real adapters |
| `api_key_env` | Optional variable holding the upstream bearer key |
| `require_api_key` | `false`; true rejects calls until the key is configured |
| `default_model` | Model for omitted or stable alias selectors |
| `model_aliases` | Application selector to upstream selector map |
| `inference_path` | `/v1/systemone`; `/predict` for jevlike |
| `models_path` | `/v1/models`; null uses local aliases |
| `timeout_ms` | `120000`; range 1..120000 for the entire adapter call |
| `response_mode` | `normalized`; `passthrough` preserves upstream status/content type/body |
| `capabilities` | Supported question types and gateway admission limits |
| `simulation` | Probabilities, fixtures, latency, and failure behavior |

Credentials are environment references, not literal config fields. Missing
optional keys allow open local servers; missing required keys return `422`. A
file owns profile URLs; legacy `--laya-url` and `--jev-url` apply to the built-in
registry. `--jev-api-key` also overrides file profiles referencing `JEV_API_KEY`.

Register 1..64 profiles with IDs of 1..64 letters, digits, underscores, or dashes.
Unknown fields, invalid defaults, unsafe URLs, invalid capabilities, and invalid
simulation settings fail startup. HTTP redirects are not followed. Replies are
limited to 4 MiB.

Capabilities default to all three question types, 64 questions, 255 choice options,
and 32 score levels. Override limits to fit each deployment:

```json
{"capabilities":{"question_types":["choice"],"max_questions":8,"max_choices":20}}
```

These are configured admission limits; upstream token windows and validation
still apply. Choice and score each require at least two options/levels. The
jevlike adapter requires `question_types: ["choice"]`. Unsupported requests fail
before inference. Benchmarks preflight every profile against every case, so use
a choice-only dataset when including jevlike.

## Model aliases and response contract

`decision-default`, `jev-latest`, and `jev-preview` map to the selected profile's
`default_model` when configured. Explicit `model_aliases` take priority. Other IDs
pass through unchanged; aliases are resolved once. With no default model, an
omitted selector lets the upstream choose and `decision-default` is removed before
forwarding. Responses identify the model actually returned by the server.

Normalized inference bodies contain only `model`, `answers`, and `usage`. Choice
keys, noul probability, score value, and question IDs retain their System One
meaning. Distributions must match criteria, be finite/in 0..1, and sum to one
within 0.02; rounded distributions are normalized. Choice/score `confidence`
consistently means `1 - H(p)/ln(K)`, clamped to 0..1: concentration, not a measured
accuracy probability. Original provider confidence values appear in benchmark
details under `backend_metadata.source_confidences`.

Headers carry provenance without extending the inference JSON envelope:

```text
X-Decision-Backend: kev-small
X-Decision-Simulated: false
X-Decision-Confidence-Definition: normalized-entropy
X-Decision-Usage-Source: reported
```

If token counts are unavailable, compatibility fields are zero and usage source
is `unavailable`; simulation uses `simulated`. Normalized HTTP errors preserve
upstream status with a common JSON error envelope. Invalid successful replies
and transport errors return `502`. No retry or automatic simulation fallback occurs.

Without a profile file, original mock/Laya/Jev modes retain their legacy response
behavior, including upstream confidence and passthrough bodies. Built-in KEV and
JevAny normalize responses. Use profile files for a common contract across all
backends, and reevaluate thresholds when changing models or confidence semantics.

## Simulation

The simulation adapter supports every question type in process. Defaults are
uniform probabilities, deterministic first choice, middle numeric score, and
noul 0.5. Override probabilities by question ID:

```json
{
  "adapter": "simulation",
  "default_model": "simulated-support",
  "simulation": {
    "latency_ms": 30, "fail_every": 5, "failure_status": 503,
    "probabilities": {
      "team": {"billing":0.9, "technical":0.1},
      "urgent": {"false":0.2, "true":0.8},
      "priority": {"0":0.1, "1":0.3, "2":0.6}
    }
  }
}
```

Option keys must match the current question. Unconfigured question IDs remain
uniform. `status` defaults to 200; set 400..599 for constant failure. `fail_every`
fails each Nth request with `failure_status` (default 503). The shared counter
includes API calls and benchmark warmup, resets on restart, and assigns slots in
arrival order; under concurrency the failing case depends on scheduling.

For recorded answers and request-specific errors, start
`--backends-file examples/backends-fixtures.json` and send `examples/choice.json`.
Each fixture has `match_request`, `response`, optional `status`, and optional
`latency_ms`. First match wins. Objects match recursively by subset against the
original request before alias mapping; scalars/arrays match exactly. An empty
match object is a catch-all. Missing fixture status still uses the profile's
status/failure schedule. Unmatched requests use generated probabilities.
Successful recordings are normalized and validated; bad recordings return `502`.
Explicit passthrough mode preserves recorded JSON fields and supports invalid-response-shape
testing. All simulation output is marked in headers, discovery, and benchmarks.

## Real model setup

Real servers run in separate environments. The gateway does not install model
dependencies or download weights. Start each needed runtime, then select its
profile. Use separate ports for multiple checkpoints.

### Laya

Install `laya[serve]` in its Python environment and run `laya-serve` on port 8000.
The example profile supports `LAYA_API_KEY` and uses `models_path: null` for local
model aliases. [Upstream HTTP docs](https://github.com/NandhaKishorM/laya/blob/main/docs/http-api.md).

### KEV (`jaredpalmer/kev`)

In a separate checkout using its supported Python 3.12/3.13 environment:

```sh
git clone https://github.com/jaredpalmer/kev.git
cd kev
uv sync --extra serve
uv run --extra serve python -m kev.serve --run jaredpalmer/kev-4b --port 8009
```

Select `kev`. If the server requires `KEV_API_KEY`, set it in the gateway's shell
too. Other KEV projects can use `systemone` profiles when compatible; check model
selectors and limits for the particular project. [Upstream server](https://github.com/jaredpalmer/kev/blob/main/kev/serve.py).

### JevAny

Install [SimpleJev/JevAny](https://github.com/SimpleJev/JevAny) in its own Python
environment according to its hardware guide. Its documented serving command is:

```sh
python -m pip install -e '.[serve,multimodal]'
jevany serve --checkpoint SimpleJev/JevAny-Qwen3.5-4B-LoRA --device cuda --dtype bf16 --port 8008
```

Select `jevany` with default model `jevany-latest`. Alias mapping keeps existing
`jev-latest` requests working. [API reference](https://github.com/SimpleJev/JevAny/blob/main/docs/API.md).

### jevlike

Install [vinnylarouge/jevlike](https://github.com/vinnylarouge/jevlike) in a separate
environment and train or obtain a checkpoint. Use that environment's Python to
run this repository's bridge:

```powershell
python scripts/serve_jevlike.py --checkpoint C:/models/synthetic.pt --port 8011
```

It loads the checkpoint once, serializes inference, and serves `/predict`,
`/health`, and `/v1/models`. Options include `--model-id`, `--device`, `--host`,
`--port`, and `--api-key-env` (default `JEVLIKE_API_KEY`). It accepts its loaded
model ID or `jevlike` as selectors. Choose the `jevlike` profile and send
`examples/choice.json`.

The Rust adapter makes one native call per question, preserves key/index mapping,
renders state as text/JSON, appends instructions after a blank line, and formats
options as `name: description`. This rendering should be evaluated on your labels.
Native token usage is unavailable. The checkpoint's truncation settings still
apply; the upstream starter uses short byte-based inputs. Its scorer has no
native score/noul API, so this adapter advertises choice only.

## Verification

```powershell
cargo test --lib --test routes --test forwarding --test benchmark --test backends
python -m unittest discover -s scripts -p test_serve_jevlike.py -v
```

Optional live tests use `TEST_LAYA`, `TEST_KEV`, `TEST_JEVANY`, and `TEST_JEVLIKE`
(`true` or `1`), plus corresponding `*_BASE_URL` overrides. Jev runs when
`JEV_API_KEY` is nonempty. Start servers before enabling these tests; tests never
download or launch models themselves. Use `cargo test --test live -- --nocapture`
to see RUN/SKIP messages.
