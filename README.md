# Model Observatory

**APIs are not invariants.**

> This tool detects **behavior drift**, not only weight updates. Determinism varies by provider; drift classification is tiered.

Model Observatory is a multi-provider monitoring system that detects infrastructure drift, behavioral drift, and experimental contamination across LLM APIs. It provides the governance layer that independent AI researchers need but don't have: verifiable model-state provenance for experiments that depend on API stability.

## The Problem

If you run experiments across multiple LLM providers, you have an unacknowledged contamination vector:

- **Silent weight updates.** Providers update model weights without changing the model ID. Your experiment on "GPT-5.2" today may not be the same "GPT-5.2" as yesterday. Your results are contaminated and you have no way to know.
- **API registry drift.** Models appear, disappear, change pricing, change context windows. If your experiment design assumes a specific model is available with specific capabilities, that assumption can silently break.
- **Behavioral instability.** Even at temperature=0, LLM APIs are not deterministic. GPU floating-point non-associativity, MoE batch contention, and infrastructure variance mean that binary hash comparison of model responses produces false positives on nearly every poll. Without a tolerance model, your drift detector becomes a false-positive factory that trains operators to ignore it.

Most AI labs have internal infrastructure to detect this. The open-source research ecosystem does not.

Model Observatory fills that gap.

## What This Is / What This Isn't

**This is** a behavioral provenance system for LLM API research. It monitors whether the models you're calling today behave the same as the models you called yesterday — and gives you a verifiable record when they don't.

**This is not** a weight-diffing tool. It cannot tell you *what* changed inside a model. It tells you *that* something changed, *when* it changed, and *how confident* it is — using tiered canonicalization and confirmation sampling, not binary hash comparison.

If you run multi-provider LLM experiments and need to know that your experimental conditions haven't been silently contaminated, this is the tool.

## How This Differs from LangSmith (and Why You Might Use Both)

Model Observatory and LangSmith operate at different layers.

**LangSmith** is primarily *application observability*: tracing and debugging the behavior of LLM-powered apps (agents, chains, tool calls, datasets/evals) on specific runs.

**Model Observatory** is *ecosystem observability*: it monitors the underlying model layer across providers — detecting model availability changes, metadata drift, and behavioral drift — and provides **epoch + manifest + baseline lockfiles** so research runs can be proven not to be silently contaminated by changing APIs.

In practice, they complement each other:

- Use **LangSmith** to understand *why your app behaved the way it did*.
- Use **Model Observatory** to know whether *the models your app depends on changed* (and when to invalidate/re-run baselines).

## Quickstart

```bash
# Clone
git clone https://github.com/hexalot-institute/model-observatory.git
cd model-observatory

# Configure (set at least one provider key)
cp .env.example .env
# Edit .env — add your API keys. Ollama needs no key if running locally.

# Run a single poll
python model_observatory.py --providers ollama

# View the TUI dashboard
python observatory_tui.py

# Run contract checks
make ci
```

## Architecture

```
                    +------------------------------------------+
                    |          EXPERIMENT INVARIANTS            |
                    | Canonical baselines | Epoch-locked state  |
                    +------------------------------------------+
                    |          BEHAVIORAL DRIFT                 |
                    | Three-state detector | MATCH/VARIANT/DRIFT|
                    +------------------------------------------+
                    |          INFRASTRUCTURE DRIFT              |
                    | Semantic fingerprints | Epoch snapshots    |
                    +------------------------------------------+
                    |          SINGLE-WRITER CORE                |
                    | Immutable PollResults | Epoch commit order |
                    +------------------------------------------+
                    |          PROVIDER ADAPTERS                 |
                    |  OpenAI | Anthropic | Google | xAI | ...  |
                    +------------------------------------------+
```

### Core: Single-Writer Pattern

Worker threads poll providers in parallel and return immutable `PollResult` dataclasses. No worker writes to any state file. The main thread is the sole writer, committing results in a strict epoch ordering:

```
snapshots -> manifest -> registry -> benchmarks -> events -> epoch pointer -> webhooks
```

Crash at any step leaves the previous epoch authoritative. No partial state.

### Layer 1: Infrastructure Drift

Polls model registries across all configured providers using **semantic fingerprinting** (stable fields only — model ID, context length, capabilities, pricing). Model list reordering, float representation differences, and Unicode normalization variants do not trigger false alerts.

Events: `MODEL_ADDED`, `MODEL_REMOVED`, `METADATA_CHANGED`, `CANARY_RESULT`, `POLL_ERROR`

### Layer 2: Behavioral Drift (Three-State Detector)

LLM APIs are not deterministic at temperature=0. The observatory uses a **three-state classification model** instead of binary hash comparison:

- **MATCH** — Response canonicalizes to the active baseline. No action.
- **VARIANT** — Different surface form, same semantic content. Logged, not alerted.
- **DRIFT** — Confirmed behavioral change after persistence thresholds are met.

**The system never emits `BEHAVIOR_DRIFT` from a single observation.** Drift requires:
- **R=3** confirmation samples on mismatch (immediate re-sampling)
- **K=3** consecutive polls showing the new value, OR
- **T=0.6** frequency threshold in a rolling window of W=20 observations

Benchmark prompts are tiered by sensitivity with tier-specific canonicalization:

| Tier | Type | Example | Canonical Form |
|------|------|---------|----------------|
| 1 | Echo | "Return exactly: OK" | `"ok"` |
| 1 | Arithmetic | "What is 2+2?" | `"4"` |
| 1 | Sequence | "Next: 2,4,6,?" | `"8"` |
| 2 | Reasoning | "P(3 from 10)?" | `"3/10"` |
| 3 | JSON | Structured output | Canonical JSON |
| 4 | Tool-call | Function call format | Sorted args |

Drift severity scales with tier: Tier 1 is INFO, Tier 4 is CRITICAL.

### Layer 3: Experiment Invariants

Lock the current model ecosystem state before starting an experiment:

```bash
# Recommended: run fresh benchmarks, then lock with canonical baselines
python model_observatory.py --lock-experiment my_experiment --benchmark-before-lock

# Check if invariants still hold
python model_observatory.py --check-experiment my_experiment
```

Lockfiles capture:
- Epoch ID and manifest hash (specific point-in-time, not "latest")
- Provider registry hash (semantic fingerprints)
- Model ID list hash
- **Canonical baseline values** for all benchmarked models (not binary hashes)
- Benchmark coverage count and staleness warnings

During invariant checks, **variants do not invalidate the lock** — only confirmed DRIFT (per the three-state thresholds) triggers a violation.

## Supported Providers

| Provider | Type | Auth |
|----------|------|------|
| OpenAI | Direct | API key |
| Anthropic | Direct | API key |
| Google (Gemini) | Direct | API key |
| xAI (Grok) | Direct | API key |
| Mistral | Direct | API key |
| DeepSeek | Direct | API key |
| OpenRouter | Aggregator | API key |
| Venice AI | Aggregator | API key |
| Volink | Aggregator | API key |
| Zhipu (GLM) | Direct | API key |
| Moonshot (Kimi) | Direct | API key |
| Ollama | Local | None |

Adding a new provider: inherit `ProviderWatcher`, implement `poll_models()`, `list_model_ids_and_fingerprints()`, `canary_call()`, and `benchmark_call()`.

## Quick Start

```bash
# Clone
git clone https://github.com/hexalot-institute/model-observatory.git
cd model-observatory

# Configure
cp .env.example .env
# Edit .env — add API keys for providers you use

# Install
pip install requests python-dotenv

# First run — polls all configured providers
python model_observatory.py

# Check what's configured
python model_observatory.py --list-providers
```

## Usage

### Polling

```bash
# Poll all configured providers (parallel execution)
python model_observatory.py

# Poll specific providers
python model_observatory.py --providers openai,anthropic

# Also run canary calls on changed models
python model_observatory.py --canary-changed
```

### Behavioral Drift Detection

```bash
# Run deterministic benchmarks on all known models
python model_observatory.py --benchmark

# Benchmark specific models only
python model_observatory.py --benchmark-models gpt-5.2,claude-opus-4-6
```

### Safe Experiment Workflow

The recommended workflow for locking an experiment:

```bash
# 1. Run a poll to establish current state
python model_observatory.py --benchmark

# 2. Lock with fresh benchmark baselines (canonical values, not hashes)
python model_observatory.py --lock-experiment my_experiment \
  --benchmark-before-lock \
  --lock-notes "Phase 2 trial, GPT-5.2 + Claude Opus"

# 3. On every subsequent poll, invariants are checked automatically
python model_observatory.py --benchmark

# 4. Manually check at any time
python model_observatory.py --check-experiment my_experiment
```

Output when ecosystem has drifted:
```
DRIFT DETECTED: Experiment 'my_experiment' invariants violated: BENCHMARK_DRIFT(openai/gpt-5.2)
```

Output for v1 lockfiles (pre-canonical):
```
CLEAN: Experiment 'old_experiment' invariants hold.
  Behavioral: UNLOCKED (v1 lockfile — re-lock with --benchmark-before-lock)
```

### Capability Registry

```bash
# Show all tracked model capabilities
python model_observatory.py --show-registry

# Query with filters
python model_observatory.py --query-registry "context>200000,reliability>0.9,cost<=mid"
```

### Routing Oracle

```bash
# Route to best model matching criteria
python routing_oracle.py --route "context>200000,reliability>0.9,cost<=mid,prefer=latency"

# Check model health
python routing_oracle.py --healthy openrouter/gpt-5.2

# Full model status
python routing_oracle.py --status openrouter/gpt-5.2

# List all models sorted by attribute
python routing_oracle.py --list-by reliability

# Generate routing recommendations
python routing_oracle.py --policy default
```

### Terminal Dashboard

```bash
# Launch TUI dashboard
python observatory_tui.py

# Auto-refresh mode
python observatory_tui.py --watch

# Keys: 1-5 switch panels, j/k scroll, Tab cycle, r refresh, q quit
```

### Webhook Notifications

Copy `webhooks.example.json` to `state/webhooks.json` and configure:

```json
{
  "webhooks": [
    {
      "type": "discord",
      "url": "https://discord.com/api/webhooks/YOUR_ID/YOUR_TOKEN",
      "min_severity": "WARN"
    }
  ],
  "cooldown_seconds": 300
}
```

Severity levels:
- **INFO** — model added, canary result, benchmark baseline, variant detected
- **WARN** — model removed, poll error, benchmark missing, state corruption
- **CRITICAL** — confirmed behavioral drift (Tier 4), experiment invariant violation

CRITICAL always fires immediately. INFO/WARN respect cooldown to prevent spam.

## v1 to v2 Migration

If you have an existing v1 state directory, the observatory migrates automatically on first v2 run:

1. **Backup** — v1 state copied to `state/v1_backup_<timestamp>/`
2. **Epoch 0** — Created from existing `*_snapshot_latest.json` files
3. **Fingerprint rebaseline** — Single `MIGRATION_REBASELINE` event (not per-model flood)
4. **Benchmark baselines** — Marked `NEEDS_REBASELINE` (v1 binary hashes cannot be converted to canonical values; first v2 benchmark run populates them)
5. **Lockfiles** — v1 locks with hash-only behavioral data are marked `UNLOCKED_BEHAVIORAL`

Migration is idempotent. Re-running does not duplicate epochs or re-emit events.

## Scheduling

### systemd timer (recommended)

```bash
# Create service
cat > ~/.config/systemd/user/model-observatory.service << 'EOF'
[Unit]
Description=Model Observatory poll
After=network-online.target

[Service]
Type=oneshot
WorkingDirectory=/path/to/model-observatory
ExecStart=/usr/bin/python3 model_observatory.py --canary-changed --benchmark
StandardOutput=append:observatory.log
StandardError=append:observatory.log
EOF

# Create 12-hour timer
cat > ~/.config/systemd/user/model-observatory.timer << 'EOF'
[Unit]
Description=Run Model Observatory every 12 hours

[Timer]
OnBootSec=5min
OnUnitActiveSec=12h
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl --user daemon-reload
systemctl --user enable --now model-observatory.timer
```

### Cron alternative

```
0 */12 * * * cd /path/to/model-observatory && python3 model_observatory.py --benchmark >> observatory.log 2>&1
```

The observatory uses a PID lockfile (`fcntl.flock`) to prevent overlapping cron instances. If a previous run is still executing, the new invocation exits cleanly.

## Silent Model Drift as a Contamination Vector

This section explains the research motivation behind Model Observatory.

LLM-based research increasingly depends on API-accessed models. Papers report results against model identifiers ("GPT-4", "Claude 3 Opus") as if these are fixed targets. They are not.

**The contamination mechanism:**

1. Researcher designs experiment against model X at time T1.
2. Provider silently updates model X weights at time T2.
3. Researcher collects data at T3 > T2 under the same model identifier.
4. Results mix pre-update and post-update behavior.
5. Paper reports results as if model X was a single, stable system.

This is not hypothetical. Providers routinely update models without version bumps. The research community has no standard mechanism to detect or document this.

**What Model Observatory provides:**

- **Detection**: Three-state behavioral classification (MATCH/VARIANT/DRIFT) catches confirmed changes while filtering out the non-deterministic noise that makes binary comparison useless.
- **Provenance**: Experiment lockfiles create a verifiable, epoch-specific snapshot of the model ecosystem with canonical baselines. Nondeterministic surface variants don't invalidate your lock; confirmed behavioral changes do.
- **Documentation**: The JSONL event log creates an auditable timeline of every model change, behavioral shift, and invariant violation.

For independent researchers without institutional infrastructure, this is the difference between defensible results and hope.

## Integration

### As a library

```python
from routing_oracle import RoutingOracle

oracle = RoutingOracle("./state")

# Route to best model for a task
model = oracle.route(min_context=200000, min_reliability=0.9, prefer="latency")
if model:
    print(f"Use: {model.key}")

# Check health before API call
if oracle.is_healthy("openrouter/gpt-5.2"):
    # proceed with call
    pass
```

### Adding a provider

```python
class MyProviderWatcher(ProviderWatcher):
    provider_name = "myprovider"

    def __init__(self, api_key: str):
        self.api_key = api_key
        # ...

    def poll_models(self) -> Dict[str, Any]:
        # Return raw model registry payload
        ...

    def list_model_ids_and_fingerprints(self, raw) -> Dict[str, str]:
        # Return {model_id: sha256_fingerprint}
        ...

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        # Return (callable, latency_ms, notes)
        ...

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        # Return (ok, response_text, latency_ms)
        ...
```

Add to `PROVIDER_REGISTRY` in `model_observatory.py` and set the env var.

## License

MIT. See [LICENSE](LICENSE).

Built by [Hexalot Institute of Cognitive Architecture](https://github.com/hexalot-institute).
