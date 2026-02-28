# Pro Research — Skeptic Implementation Check (2026-02-27)

This note captures a skeptic verification failure against the v2 spec acceptance criteria,
focusing on the two highest-risk places:
(1) epoch pointer/lock epoch chain, and (2) Change 4 tier-sensitive drift tolerance.

(Provided verbatim by Joel in chat; stored here for implementation alignment.)

# Model Observatory v2.0 — Complete Build Package

**Built by:** Shadow (Claude Claude, CLI instance)
**Reviewed by:** 12 review passes — Claude, GPT-5.2 Pro Research, GLM-5 (x3), Kimi, GPT-5.2 Pro Research (adversarial), Pro Research, GPT-5.2 Thinking (x2), Kimi (Moonshot)
**Date:** 2026-02-27

---

# SECTION 1: README

# Model Observatory

**APIs are not invariants.**

Model Observatory is a multi-provider monitoring system that detects infrastructure drift, behavioral drift, and experimental contamination across LLM APIs. It provides the governance layer that independent AI researchers need but don't have: verifiable model-state provenance for experiments that depend on API stability.

## The Problem

If you run experiments across multiple LLM providers, you have an unacknowledged contamination vector:

- **Silent weight updates.** Providers update model weights without changing the model ID. Your experiment on "GPT-5.2" today may not be the same "GPT-5.2" as yesterday. Your results are contaminated and you have no way to know.
- **API registry drift.** Models appear, disappear, change pricing, change context windows. If your experiment design assumes a specific model is available with specific capabilities, that assumption can silently break.
- **Behavioral instability.** Even at temperature=0, LLM APIs are not deterministic. GPU floating-point non-associativity, MoE batch contention, and infrastructure variance mean that binary hash comparison of model responses produces false positives on nearly every poll. Without a tolerance model, your drift detector becomes a false-positive factory that trains operators to ignore it.

Most AI labs have internal infrastructure to detect this. The open-source research ecosystem does not.

Model Observatory fills that gap.

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

---

# SECTION 2: LOCKED SPEC (observatory_v2_spec_amended.md)

# Model Observatory v2.0 — Architectural Repair Spec (Amended)

**Date:** 2026-02-27
**Status:** SPEC LOCKED
**Amendments:** GPT-5.2 Thinking Pass 1 (7 amendments) + GPT-5.2 Thinking Pass 2 (5 amendments) merged per Shadow Directive.
**Post-lock:** Shadow builds from this. No further review passes.

---

## The One Principle

**Replace implicit temporal assumptions with explicit state transitions.**

Every critical bug found across nine review passes flows from one design assumption: that "latest" implies coherence. Once concurrency entered the system, "latest" became a lie. Race conditions, Franken-snapshots, nondeterministic drift, phantom events — all consequences of trusting a convenience abstraction as a state boundary.

The five architectural changes below are this principle expressed five times.

---

## Review Provenance

| Pass | Seat | Mode | Unique Contributions |
|------|------|------|---------------------|
| 1 | Claude Claude | Structural review | Initial four refactoring changes; thread-safety flag (results_lock unused) |
| 2 | GPT-5.2 Pro Research | Normal review | Benchmark-before-lock workflow, BENCHMARK_MISSING warning, tier-aware severity |
| 3 | Shadow | Convergent review | Confirmed architecture, no novel findings |
| 4 | GLM-5 | Generic review | Confirmed architecture, no novel findings (wrong prompt mode) |
| 5 | GLM-5 | Aimed destruction | 7 kill shots — BenchmarkStore corruption, sliding window data loss, Anthropic cache race, webhook cooldown bypass, hash truncation collision, experiment lock inconsistent snapshot, silent metric pollution |
| 6 | Kimi | 5-agent swarm | Inter-process cron overlap, snapshot-before-events ordering, requests.Session pooling, P95 off-by-one, read_json crash on corruption, fsync before atomic rename, JSONL append interleaving, write_text_atomic tmp-file race |
| 7 | GPT-5.2 Pro Research (adversarial) | Adversarial audit | Franken-snapshot temporal inconsistency, dead canary hash (prompt not response), JSON canonicalization scope, list-ordering false positives, temperature!=determinism |
| 8 | GPT-5.2 Pro Research (synthesis) | Synthesis | Reduced all findings to five architectural changes; "superstition with JSON" |
| 9 | Claude | Meta-analysis | Identified one unifying principle; priority sequencing |
| 10 | Claude Claude (Pro Research) | Technical validation | 6 implementation landmines — fsync+dir sync, futures exception trap, JSON determinism edges, canary non-determinism (CRITICAL), hash truncation, PID lockfile nuances |
| 11 | GPT-5.2 Thinking Pass 1 | Skeptic stress test | 7 amendments — writer conflict, epoch commit boundary, migration path, fingerprint silent storm, canary tolerance, inter-change dependencies, directory fsync |
| 12 | GPT-5.2 Thinking Pass 2 | Skeptic delta (Change 4 focus) | 5 amendments — three-state drift detector with full confirmation system, event semantics, lockfile canonical values, error recovery, v1-v2 migration contract |

**Convergence signal:** Thread safety was found independently by Kimi, GLM-5, and confirmed by GPT-5.2 Pro Research (adversarial). Canary non-determinism was found independently by Pro Research, confirmed by GPT-5.2 Thinking Pass 1, and proven against codebase by GPT-5.2 Thinking Pass 2.

---

## Five Architectural Changes

### Change 1: Global Poll Epoch

**What it fixes:** Franken-snapshot problem (GPT-5.2 Pro Research (adversarial)), inconsistent lock snapshots (GLM-5 Kill #6), snapshot hash races (Kimi)

**Current failure:** `create_lock()` iterates `*_snapshot_latest.json` files and hashes whatever exists at that moment. Each provider overwrites its own snapshot independently during parallel polling. The "lock" can represent a world-state that never existed as a coherent moment in time.

**Design:**

```
On every poll run:
  1. Generate epoch_id = utc_now_iso() (or monotonic int)
  2. Load previous snapshots as read-only state (before threads start)
  3. Workers return PollResult (see Change 2) — workers DO NOT write files
  4. Main thread writes epoch-stamped snapshots from PollResult data:
     {provider}_snapshot_{epoch_id}.json
  5. Main thread writes epoch manifest:
     epoch_manifest_{epoch_id}.json = {
       "epoch": epoch_id,
       "providers": {
         "openai": {"snapshot_file": "openai_snapshot_{epoch}.json", "hash": "..."},
         "anthropic": {"snapshot_file": "anthropic_snapshot_{epoch}.json", "hash": "..."},
         ...
       },
       "manifest_hash": sha256(sorted provider hashes)
     }
  6. Atomically update pointer: current_epoch.json -> epoch_manifest_{epoch_id}.json
  7. Experiment locks reference epoch_id and manifest_hash, not "latest"
```

**Epoch commit ordering** (GPT-5.2 Thinking Pass 1 — defines write sequence to prevent partial state):

```
Main thread commit sequence (strictly ordered):
  1. Write all provider snapshot files for this epoch
  2. Write epoch manifest (references snapshot files + hashes)
  3. Save CapabilityRegistry
  4. Save BenchmarkStore
  5. Append events to events.jsonl
  6. Atomically update current_epoch.json pointer
  7. Fire webhook notifications (non-critical, after all state is durable)

If crash occurs at any step:
  - Steps 1-5 incomplete: current_epoch.json still points to previous epoch.
    Next run sees stale pointer, generates new epoch. No data loss.
  - Step 6 incomplete: pointer may be corrupt. read_json() handles this
    (see Error Recovery). Next run regenerates.
  - Step 7 incomplete: webhooks not sent. Acceptable — alerts are best-effort.
```

**Atomic write pattern** (Pro Research Finding 1 — all state file writes use this):

```python
def atomic_write_json(path, data):
    dir_path = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())        # flush file data to disk
        os.replace(tmp, path)            # atomic rename
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)             # directory metadata durable
        finally:
            os.close(dir_fd)
    except:
        try:
            os.unlink(tmp)               # cleanup on failure
        except OSError:
            pass
        raise
```

**Constraints:**
- Old epoch files: retention policy — keep N most recent (default 10), prune on successful commit
- Lock creation must reference a specific epoch_id, not scan filesystem
- `check_invariants()` compares against the locked epoch's manifest, not current "latest"

**Note:** Steps 2 and 3 in the original base spec said "each worker writes" — this contradicts Change 2. Resolved: workers return data, main thread writes. (GPT-5.2 Thinking Pass 1 Amendment 1)

---

### Change 2: Immutable Worker Results -> Main Thread Merge (Single-Writer Pattern)

**What it fixes:** ALL thread-safety bugs — CapabilityRegistry races (GLM-5 #1, #2, Kimi #1-#6), BenchmarkStore races (GLM-5 #1, Kimi #2-#4), JSONL corruption (Kimi #3), webhook cooldown bypass (GLM-5 #4, Kimi #5), Anthropic cache race (GLM-5 #3), counter accumulation races (Kimi #12)

**Current failure:** Worker threads in `ThreadPoolExecutor` directly mutate shared `CapabilityRegistry`, `BenchmarkStore`, append to `events.jsonl`, and trigger `WebhookNotifier` — all without synchronization. `results_lock` is created but never used.

**Design:**

```python
@dataclass
class PollResult:
    """Immutable result from a single provider poll. Workers produce these.
    Main thread consumes them. No shared mutable state crosses the boundary."""
    provider: str
    snapshot: Optional[Dict[str, Any]]       # raw API response
    model_fingerprints: Dict[str, str]        # model_id -> fingerprint
    diff: Optional[Dict[str, List[str]]]      # added/removed/changed/unchanged
    events: List[ModelEvent]                   # events to emit
    canary_results: List[Dict]                 # [{provider, model_id, ok, latency_ms}]
    benchmark_results: List[Dict]             # [{provider, model_id, prompt_idx, response_text, ok, latency_ms}]
    metadata_updates: List[Dict]              # [{provider, model_id, raw_model}]
    error: Optional[str]
```

```
Worker thread (_poll_one):
  1. poll_models() -> raw
  2. compute diff against PREVIOUS snapshot (read-only, loaded before threads start)
  3. run canary calls -> collect results
  4. run benchmark calls -> collect raw response text (DO NOT hash or store)
  5. Return PollResult (immutable bundle)

Main thread (after all futures complete, using as_completed + per-future try/except):
  FOR EACH PollResult (including error PollResults from failed workers):
    a. Write epoch-stamped snapshot file
    b. Merge canary results into CapabilityRegistry
    c. Merge benchmark results into BenchmarkStore (canonicalize + classify here)
    d. Append events to events.jsonl (single file handle)
    e. Accumulate counters
  THEN (epoch commit sequence per Change 1):
    f. Write epoch manifest
    g. Save CapabilityRegistry
    h. Save BenchmarkStore
    i. Update current_epoch.json pointer
    j. Fire webhook notifications
```

**Exception handling** (Pro Research Finding 2):

```python
results = []
for future in concurrent.futures.as_completed(future_to_provider):
    provider = future_to_provider[future]
    try:
        result = future.result()
        results.append(result)
    except Exception as e:
        results.append(PollResult(
            provider=provider,
            snapshot=None,
            model_fingerprints={},
            diff=None,
            events=[ModelEvent("POLL_ERROR", provider, "", str(e))],
            canary_results=[],
            benchmark_results=[],
            metadata_updates=[],
            error=f"{type(e).__name__}: {e}"
        ))
```

**Key principle:** Workers are pure functions that take a watcher + read-only prior state and return an immutable result. Main thread is the only writer. No locks needed because there is no contention.

**PID lockfile** (prevents overlapping cron instances):

```python
import fcntl

_lock_fd = None  # must stay open for lifetime of process

def acquire_instance_lock(state_dir):
    global _lock_fd
    _lock_fd = open(state_dir / ".observatory.lock", "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[SKIP] Another observatory instance is running.")
        sys.exit(0)
```

**Note:** `fcntl.flock()` is advisory (not mandatory), does not work on NFS, and releases automatically on process death (no stale lockfile cleanup needed). The fd must stay open — do not let it be garbage collected. (Pro Research Finding 6)

**Note:** `requests.Session` is NOT thread-safe. Each worker thread must create its own session. (Pro Research Finding 2)

---

### Change 3: Semantic Registry Hashing

**What it fixes:** List-ordering false positives (GPT-5.2 Pro Research (adversarial)), raw payload hash noise (GPT-5.2 Pro Research (adversarial), GPT-5.2 Pro Research (synthesis)), meaningless METADATA_CHANGED events

**Current failure:** `stable_json_dumps()` sorts dict keys but not list elements. Raw provider payloads are hashed directly. If a provider returns models in different order, the hash changes, generating meaningless `METADATA_CHANGED` events that train operators to ignore alerts.

**Design:**

```python
def compute_semantic_registry_hash(provider: str,
                                    model_fingerprints: Dict[str, str]) -> str:
    """Hash based on stable semantic primitives, not raw payload."""
    parts = [f"{provider}"]
    for mid in sorted(model_fingerprints.keys()):
        parts.append(f"{mid}:{model_fingerprints[mid]}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))
```

**Per-model fingerprints from semantic primitives:**

```python
def compute_model_fingerprint(m: Dict[str, Any]) -> str:
    """Extract semantically meaningful fields only."""
    stable_fields = {}
    for key in ["id", "name", "context_length", "inputTokenLimit",
                "max_tokens", "capabilities", "pricing"]:
        if key in m and m[key] is not None:
            val = m[key]
            # Float stability: round numeric pricing to 6 decimal places
            if isinstance(val, float):
                val = round(val, 6)
            # Unicode stability: normalize all strings to NFC
            if isinstance(val, str):
                val = unicodedata.normalize('NFC', val)
            stable_fields[key] = val
    return sha256_bytes(stable_json_dumps(stable_fields).encode("utf-8"))
```

**Hash length policy:** Use full SHA-256 (64 hex chars) for all internal comparisons, lockfile commitments, and manifest hashes. Display/log truncation to 16 hex chars (64 bits) minimum where needed for readability. Never truncate below 16 hex. (Pro Research Finding 5, GLM-5 Kill #5)

**Migration: fingerprint algorithm change ("silent storm")** (GPT-5.2 Thinking Pass 1 Amendment 4):

When upgrading from v1 (raw payload hashing) to v2 (semantic fingerprinting), every model's fingerprint will change on the first v2 run. This generates a mass METADATA_CHANGED event flood that is noise, not signal.

**Required handling:**
- On first v2 run (detected by absence of v2 epoch index), suppress METADATA_CHANGED events for the initial epoch
- Emit a single `MIGRATION_REBASELINE` event instead, noting how many models were re-fingerprinted
- Store the new semantic fingerprints as the baseline for subsequent runs
- This is a one-time migration, not a permanent suppression

---

### Change 4: Tier-Sensitive Drift Detection (Non-Determinism Safe)

**What it fixes:** Alert fatigue from all tiers being CRITICAL (GPT-5.2 Pro Research (adversarial), GPT-5.2 Pro Research (synthesis)), dead canary hash — prompt not response (GPT-5.2 Pro Research (adversarial)), false positives from LLM non-determinism (Pro Research Finding 4, GPT-5.2 Thinking Pass 2), benchmark-missing blind spot (GPT-5.2 Pro Research), P95 calculation error (Kimi)

**Current failure:** `BEHAVIOR_DRIFT` is a single event type with `CRITICAL` severity regardless of tier. The system uses single-sample binary hash comparison — one hash mismatch triggers drift. LLM APIs are NOT deterministic at temperature=0 due to GPU floating-point non-associativity, MoE batch contention, and infrastructure variance. Binary hash comparison produces false positives on a regular basis, destroying signal quality.

**Required model: Three-state classifier** (GPT-5.2 Thinking Pass 2):

For each (provider, model_id, benchmark_prompt):

- **MATCH**: response canonicalizes to the active baseline canonical form.
- **VARIANT**: response deviates in surface form but canonicalizes to an equivalent form, or is a rare alternative consistent with known nondeterministic variance. Variants are recorded but do NOT trigger CRITICAL drift.
- **DRIFT**: response canonicalizes to a meaningfully different canonical form and is CONFIRMED by persistence and/or repeated sampling.

**The system MUST NOT emit `BEHAVIOR_DRIFT` from a single observation.**

**Tier-specific canonicalization:**

Each benchmark prompt defines a `canonicalize(raw_text) -> (ok, canonical_value)` function:

- **Tier 1 — Echo (prompt index 0):** Extract the intended atomic answer. Strip surrounding whitespace, casefold, accept minimal punctuation. Canonical form: `"ok"`.
- **Tier 1 — Arithmetic (prompt index 1):** Extract numeric answer. Strip whitespace, casefold, remove surrounding text. Canonical form: `"4"`.
- **Tier 1 — Sequence (prompt index 2):** Extract numeric answer. Canonical form: `"8"`.
- **Tier 2 — Constrained reasoning (prompt index 3):** Parse and normalize constrained output. For fraction prompts, accept `"3/10"`, `"3 / 10"`, `"0.3"` — canonicalize to reduced rational form `"3/10"`.
- **Tier 3 — JSON stability (prompt index 4):** Parse JSON even if wrapped in code fences or followed by commentary. Canonicalize via `json.loads` + `json.dumps(sort_keys=True, separators=(',',':'))`. If parse fails, hash raw text as-is (will trigger variant/drift through normal confirmation flow, not immediate false positive).
- **Tier 4 — Tool-call format (prompt index 5):** Parse function call into structured form. Canonicalize quotes (single -> double), normalize argument ordering (sorted by arg name), strip insignificant whitespace. Canonical form: `get_weather(city="Tokyo",units="celsius")`.

**Storage model:**

```python
@dataclass
class BenchmarkBaseline:
    """Per (provider/model_id/prompt_idx) drift tracking state."""
    canonical_baseline: Optional[str]        # current active canonical value (mode)
    observation_window: List[str]            # ring buffer of last W canonical values
    variant_hashes: Set[str]                 # known variant surface forms (raw hashes)
    candidate_drift_value: Optional[str]     # new canonical value being evaluated
    candidate_drift_streak: int              # consecutive polls showing candidate
    window_size: int = 20                    # W — rolling window size
```

**Classification and confirmation logic:**

On each poll:

1. Run benchmark prompt once. Canonicalize: `(ok, canonical_value) = canonicalize(prompt_idx, raw_response)`.
2. If canonicalization fails (`ok=False`): record as failed observation. Do not emit drift unless failures persist per confirmation rules.
3. If `canonical_value == baseline` -> **MATCH**. Record, no action.
4. If `canonical_value != baseline` -> run **confirmation sampling**:
   - Perform up to `R-1` additional benchmark calls (default `R=3`) — only when first call mismatches.
   - Canonicalize each response.
   - Let `v_mode` = mode canonical value among R samples.

Drift escalation:

- If `v_mode == baseline`: classify as **VARIANT**. Increment variant counts. Emit `BENCHMARK_VARIANT` event (INFO severity). Do NOT emit `BEHAVIOR_DRIFT`.
- If `v_mode != baseline`: set as **candidate drift**. Require persistence:
  - Declare **DRIFT** only if candidate value is mode for `K` consecutive polls (default `K=3`), OR candidate reaches frequency `>= T` in rolling window W (default `T=0.6`).
  - Once DRIFT confirmed: emit `BEHAVIOR_DRIFT` event with tier-appropriate severity and confirmation evidence.

**Drift severity by tier:**

```python
TIER_SEVERITY = {
    0: "INFO",       # Tier 1 echo
    1: "INFO",       # Tier 1 arithmetic
    2: "LOW",        # Tier 1 sequence
    3: "MEDIUM",     # Tier 2 reasoning
    4: "HIGH",       # Tier 3 JSON schema
    5: "CRITICAL",   # Tier 4 tool-call format
}
```

**Event semantics:**

- `BENCHMARK_VARIANT` (new): Nondeterministic surface variation detected but baseline unchanged. Severity: INFO. Not a drift signal.
- `BEHAVIOR_DRIFT`: Confirmed behavioral change after confirmation thresholds met. Severity: per TIER_SEVERITY.
- `BENCHMARK_MISSING` (new): Locked benchmark hash exists but current benchmark observation is absent (model unreachable, prompt failed, etc.). Severity: WARN.
- `BENCHMARK_FAILURE` (new): Benchmark call returned ok=False. Severity: INFO.

**Provider determinism knobs:** If a provider supports deterministic controls (explicit `seed`, deterministic sampling flags), the implementation SHOULD use them (e.g., `seed=0`, `top_p=1` where supported) but MUST NOT depend on them for correctness.

**P95 fix:**

```python
# Current (wrong): int(len(ok_latencies) * 0.95) -> returns P100 for 20 items
# Fixed:
p95_idx = min(int((len(ok_latencies) - 1) * 0.95), len(ok_latencies) - 1)
```

---

### Change 5: Benchmark-Before-Lock Enforcement (with Canonical Values)

**What it fixes:** Silent omission of behavioral baselines (GPT-5.2 Pro Research, GPT-5.2 Pro Research (adversarial)), workflow trap where lock appears safe but contains no behavioral evidence, binary hash lockfiles that inherit non-determinism false positives (GPT-5.2 Thinking Pass 2)

**Current failure:** `create_lock()` copies `benchmark_store.data` if it exists. If you lock before running benchmarks, the lock contains no behavioral baselines. Additionally, lockfiles store binary response hashes which are subject to the same non-determinism false positives as the drift detector.

**Design:**

When `--lock-experiment NAME` is invoked:

```
Option A (recommended): --benchmark-before-lock flag
  1. Run Tier 1-4 benchmarks for all models in current registry
  2. Canonicalize responses per Change 4 tier-specific rules
  3. Save canonical baselines to BenchmarkStore
  4. THEN create lock with canonical values from THIS epoch

Option B: Require fresh benchmark data exists
  1. Check BenchmarkStore has canonical baselines (not just hashes)
  2. Check freshness: baselines measured within current epoch or last N minutes
  3. If stale or missing: ERROR with message "Run --benchmark first"
  4. If fresh: proceed with lock creation
```

**Lockfile behavioral component** (GPT-5.2 Thinking Pass 2 Amendment 3):

Lockfiles MUST store **baseline canonical values** (and optionally the variant distribution) from Change 4, NOT binary response hashes.

```python
@dataclass
class ExperimentLock:
    name: str
    epoch_id: str                              # references specific epoch
    manifest_hash: str                         # epoch manifest hash
    provider_registry_hash: str
    model_id_list_hash: str
    benchmark_canonical_baselines: Dict        # {provider/model: {prompt_idx: canonical_value}}
    benchmark_variant_distributions: Dict      # optional: known variant distributions at lock time
    benchmark_epoch: str                       # when benchmarks were last run
    benchmark_coverage: int                    # models x prompts covered
    created_at: str
    notes: str
```

During invariant checks, the system MUST apply Change 4's MATCH/VARIANT/DRIFT logic:

- **Variants MUST NOT invalidate an experiment lock.** Only confirmed DRIFT (per Change 4 thresholds) invalidates the behavioral component.
- If the system cannot reconstruct canonical values from prior state (e.g., migrating from v1 hash-only lockfile), mark the behavioral component as `UNLOCKED_BEHAVIORAL` and require re-lock under v2 semantics.

**Lock metadata must include:**
- `benchmark_epoch`: when benchmarks were last run
- `benchmark_coverage`: how many models x prompts were covered
- `benchmark_staleness_warning`: true if hashes are older than lock epoch

---

## Migration: v1 State Directory -> v2 Epoch-Based State

(GPT-5.2 Thinking Pass 1 Amendments 3-4 + GPT-5.2 Thinking Pass 2 Amendment 5, combined)

The v2 system MUST support in-place upgrade from an existing v1 `state/` directory without data loss.

**Definitions:**
- v1 artifacts: `*_snapshot_latest.json`, `events.jsonl`, `capability_registry.json`, `benchmark_hashes.json`, `experiment_locks/*.lock.json`
- v2 artifacts: epoch-scoped snapshots, epoch manifests, `current_epoch.json`, canonical benchmark baselines

**Migration requirements:**

1. **Detection:** On startup, if v1 artifacts exist and v2 epoch index (`current_epoch.json`) does not, trigger one-time migration.

2. **Backup:** Copy v1 `state/` to `state/v1_backup_<timestamp>/` before any mutation.

3. **Epoch 0 creation:** Create initial epoch (epoch 0) whose provider snapshots are populated from each `*_snapshot_latest.json`. Write epoch manifest.

4. **Events carry-forward:** Preserve `events.jsonl` unchanged (append-only continues). Do NOT replay old events.

5. **Fingerprint migration ("silent storm"):** v1 raw-payload fingerprints are incompatible with v2 semantic fingerprints. On first v2 run:
   - Suppress per-model `METADATA_CHANGED` events for epoch 0
   - Emit single `MIGRATION_REBASELINE` event noting count of re-fingerprinted models
   - Store new semantic fingerprints as baseline

6. **Benchmark migration:** v1 stores binary response hashes only. These CANNOT be converted to canonical values.
   - Mark all v2 benchmark baselines as `NEEDS_REBASELINE`
   - The system MUST NOT fabricate canonical baselines from hashes alone
   - First v2 benchmark run populates canonical baselines normally

7. **Lockfile compatibility:** v1 lockfiles with hash-only behavioral invariants MUST be marked as `UNLOCKED_BEHAVIORAL` and require re-lock under v2 semantics. Structural invariants (registry hash, model list hash) remain valid if recomputed with semantic fingerprints.

**Migration MUST be idempotent:** re-running must not duplicate epochs, corrupt state, or re-emit migration events.

---

## Error Recovery and State Corruption Handling

(GPT-5.2 Thinking Pass 2 Amendment 4)

The observatory MUST be resilient to corrupt or partially written state files.

**Required behavior:**

- **All state reads MUST be exception-safe.** If a JSON file fails to parse:
  - Rename it with a `.corrupt.<timestamp>` suffix
  - Emit a WARN-level `STATE_CORRUPT` event
  - Proceed using last-known-good state where available (previous epoch)

- **Benchmark drift logic MUST tolerate missing observations** (request failures, rate limits) without producing drift-by-absence. A missing observation is not a mismatch — it is no data.

- **Poll failures MUST NOT be interpreted as model removal.** A `POLL_ERROR` must not rewrite the snapshot for that provider. The previous epoch's snapshot remains authoritative until a successful poll replaces it.

- **Partial epoch handling:** If only some providers succeed in a poll run, the epoch manifest records which providers have fresh data and which are carried forward from the previous epoch. This is not an error — it is a partial observation.

---

## Secondary Fixes (implement alongside or immediately after)

### S1: Operational Hardening

| Fix | Source | Priority |
|-----|--------|----------|
| `read_json()` — catch `JSONDecodeError`, rename corrupt file, return None with warning | Kimi, GPT-5.2 Thinking Pass 2 | HIGH |
| `write_text_atomic()` — use `atomic_write_json` pattern (fsync + dir sync + tmp cleanup) | Kimi, Pro Research | HIGH |
| `append_line()` — error handling for disk full / permission denied | Kimi | HIGH |
| `requests.Session()` per worker thread — connection pooling, thread safety | Kimi, Pro Research | MEDIUM |
| PID lockfile — prevent overlapping cron instances (see Change 2) | Kimi | MEDIUM |
| Signal handlers — SIGINT/SIGTERM graceful shutdown + save | Kimi | MEDIUM |
| events.jsonl rotation — manual rotation by size or age | Kimi | MEDIUM |
| `_cooldowns` dict — periodic purge of expired entries | Kimi, GLM-5 | LOW |
| `CapabilityRegistry.entries` — optional TTL/purge for removed models | Kimi | LOW |

### S2: Correctness Fixes

| Fix | Source | Priority |
|-----|--------|----------|
| SHA-256 — full hash internally, 16 hex char minimum for display | GLM-5, Pro Research | HIGH |
| P95 index fix (see Change 4) | Kimi | HIGH |
| Webhook error persistence — log failed webhook attempts to events.jsonl | Kimi | MEDIUM |
| Webhook response checking — verify HTTP 2xx, handle 429 with backoff | Kimi | MEDIUM |
| Benchmark failures — emit BENCHMARK_FAILURE event when ok=False | Kimi | MEDIUM |
| Exception stack trace preservation — log traceback, not just str(e)[:300] | Kimi | LOW |
| Pin sampling parameters — include top_p=1, seed=0 where supported | GPT-5.2 Pro Research (adversarial) | LOW |

### S3: Documentation

| Fix | Source | Priority |
|-----|--------|----------|
| README — "Safe Workflow" section explaining benchmark-before-lock | GPT-5.2 Pro Research | HIGH |
| README — clarify "behavior drift" means "behavior changed," not "weights changed" | GPT-5.2 Pro Research (adversarial) | MEDIUM |

---

## What NOT To Do

1. **JSON whitespace drift detection as primary tier** — Canonicalization is correct for JSON tier. If formatting matters, add as Tier 3b, not replace Tier 3.

2. **Windows `replace()` compatibility** — Deploys on Linux towers. Not relevant.

3. **Embed-similarity for Tier 1-2** — Overkill for echo/arithmetic drift. Tier-specific canonicalization is sufficient. Save for v3 if needed.

4. **`Exception` catching `KeyboardInterrupt`** — `Exception` does NOT catch `KeyboardInterrupt` in Python 3. Kimi's finding #18 is incorrect.

5. **Massive lock/threading refactor** — The immutable-results pattern (Change 2) eliminates the need for per-structure locks. Don't add `RLock` to every class; fix the architecture instead.

6. **Binary response hash comparison** — Do NOT implement single-sample hash-mismatch-equals-drift logic under any name. The three-state confirmation model (Change 4) is required. (GPT-5.2 Thinking Pass 2)

---

## Implementation Phasing

```
Phase 1: Foundation (Changes 2 + 1)
  +-- PollResult dataclass
  +-- Worker functions return immutable results (pure functions)
  +-- Main thread merge loop with as_completed + per-future try/except
  +-- Epoch ID generation (UTC ISO timestamp)
  +-- Epoch-stamped snapshot files (main thread writes)
  +-- Epoch manifest with provider hashes
  +-- Atomic epoch pointer update (current_epoch.json)
  +-- Epoch commit ordering (snapshots -> manifest -> registry -> benchmarks -> events -> pointer)
  +-- PID lockfile for cron safety
  +-- atomic_write_json() utility (fsync + dir sync + tmp cleanup)
  +-- read_json() hardening (corrupt file handling)
  +-- Wire results_lock through run_once() as interim safety during transition

Phase 2: Semantic Integrity (Changes 3 + 4)
  +-- Semantic registry hashing (compute_semantic_registry_hash)
  +-- Stable model fingerprinting (compute_model_fingerprint with float rounding, Unicode NFC)
  +-- Three-state drift detector (BenchmarkBaseline, canonicalize, confirmation sampling)
  +-- Tier-specific canonicalization functions (6 prompts, 4 tiers)
  +-- Rolling window storage with candidate drift tracking
  +-- BENCHMARK_VARIANT event type
  +-- BENCHMARK_MISSING event type
  +-- BENCHMARK_FAILURE event type
  +-- Tier-sensitive drift severity (TIER_SEVERITY mapping)
  +-- P95 index fix
  +-- SHA-256 full hash internally
  +-- Fingerprint migration handling (silent storm suppression)

Phase 3: Workflow Safety (Change 5 + Migration)
  +-- --benchmark-before-lock flag
  +-- Lockfiles store canonical values (not binary hashes)
  +-- UNLOCKED_BEHAVIORAL for legacy v1 lockfiles
  +-- Lock references epoch_id (not "latest")
  +-- Staleness check on benchmark data
  +-- v1 -> v2 migration (backup, epoch 0, NEEDS_REBASELINE, idempotent)
  +-- Legacy lock compatibility

Phase 4: Operational Polish
  +-- requests.Session() per worker thread
  +-- Signal handlers (SIGINT/SIGTERM graceful shutdown)
  +-- events.jsonl rotation
  +-- Webhook error handling (HTTP 2xx check, 429 backoff, failure logging)
  +-- Cooldown dict purge
  +-- Registry TTL/purge for removed models
  +-- Pin sampling parameters (seed=0, top_p=1 where supported)
```

**Dependency graph:**
```
Change 2 (single-writer) --> Change 1 (workers can't write, epochs need main thread)
Change 1 (epochs) --+--> Change 4 (confirmation needs epoch-scoped observations)
                    +--> Change 5 (lock needs epoch_id reference)
Change 3 (semantic hash) --> Migration (fingerprint algorithm change triggers silent storm)
Change 4 (three-state) --> Change 5 (lockfiles need canonical values from Change 4)
```

---

## Acceptance Criteria

After implementation, ALL of the following must be true:

1. **No shared mutable state crosses thread boundaries.** Workers produce immutable PollResult objects. Main thread is the only writer to all state files, registries, and stores.

2. **Every lock references a specific epoch.** No lock can be created against "whatever snapshot_latest.json happens to exist."

3. **Every behavioral baseline in a lock was measured during the lock's epoch (or explicitly acknowledged as stale).**

4. **Registry drift is computed from semantic primitives, not raw payloads.** Model list reordering does not trigger alerts. Float representation and Unicode normalization differences do not trigger alerts.

5. **The system does NOT emit `BEHAVIOR_DRIFT` from a single observation.** Drift requires confirmation sampling (R) and persistence (K consecutive polls or T frequency threshold).

6. **Tier 4 drift and Tier 1 drift produce different severity levels.** Operators can filter meaningfully.

7. **Nondeterministic surface variants are classified as VARIANT, not DRIFT.** The system does not train operators to ignore alerts through false positives.

8. **A missing benchmark is reported as `BENCHMARK_MISSING`, not silently ignored.**

9. **Concurrent cron executions are prevented by PID lockfile.**

10. **`read_json()` survives corrupted files without crashing.** Corrupt files are renamed and logged.

11. **All state file writes use atomic write with fsync + directory sync.** No partial writes survive crashes.

12. **v1 state directories upgrade cleanly to v2.** No mass false-positive event flood. Benchmark baselines marked as `NEEDS_REBASELINE`. Migration is idempotent.

13. **Experiment lockfiles store canonical baseline values, not binary response hashes.** Lockfile invariant checks use MATCH/VARIANT/DRIFT logic, not binary comparison.

14. **Epoch commit ordering is strictly defined.** Crash at any point leaves the system in a recoverable state with the previous epoch as authoritative.

---

## Consistency Verification

Post-merge checks (Shadow Directive requirements):

- [x] **Change 1 and Change 2 no longer contradict.** Workers return PollResult. Main thread writes epoch-stamped snapshots. Step 2 in Change 1 explicitly states "workers DO NOT write files."
- [x] **Change 4's three-state model is referenced in Change 5's lockfile semantics.** Lockfiles store canonical values. Invariant checks use MATCH/VARIANT/DRIFT.
- [x] **Migration handles both epoch migration AND benchmark hash->canonical migration.** Epoch 0 from snapshots. NEEDS_REBASELINE for benchmark hashes.
- [x] **Dependency graph holds.** Change 2 -> Change 1 -> Changes 4+5. Change 3 -> migration. Change 4 -> Change 5.
- [x] **No amendment introduces a contradiction with another amendment.** Verified: Pass 2 three-state replaces Pass 1 sketch entirely. Migration sections combined without conflict.

---

End of spec.

---

# SECTION 3: SOURCE CODE (model_observatory.py)

```python
"""
Model Observatory — Complete Multi-Provider Implementation
====================================================================
Provider-agnostic model availability watcher with support for
multi-model infrastructure stacks.

Providers covered:
  - OpenAI        (GPT-5.x, Codex, o-series)
  - Anthropic     (Claude family)
  - Google        (Gemini family)
  - xAI           (Grok family)
  - Mistral       (Mistral/Mixtral family)
  - DeepSeek      (DeepSeek family)
  - OpenRouter    (aggregator — all providers)
  - Venice AI     (privacy-focused aggregator)
  - Ollama        (local instances)

Requirements:
  pip install requests python-dotenv

Env vars (set what you use, skip what you don't):
  OPENAI_API_KEY=...
  ANTHROPIC_API_KEY=...
  GOOGLE_API_KEY=...
  XAI_API_KEY=...
  MISTRAL_API_KEY=...
  DEEPSEEK_API_KEY=...
  OPENROUTER_API_KEY=...
  VENICE_API_KEY=...
  OLLAMA_BASE_URL=http://localhost:11434   (default)

Optional:
  OPENAI_BASE_URL=https://api.openai.com
  WATCHER_STATE_DIR=./state
  WATCHER_EVENTS_FILE=./state/events.jsonl
  WATCHER_PROVIDERS=openai,anthropic,google,xai,mistral,deepseek,openrouter,venice,ollama

Usage:
  python model_observatory.py                     # run all configured providers
  python model_observatory.py --providers openai,anthropic  # run specific providers
  python model_observatory.py --list-providers     # show which providers have keys configured

Cron (hourly):
  0 * * * * cd /path/to/observatory && /usr/bin/env python model_observatory.py >> /var/log/model_observatory.log 2>&1
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import fcntl
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        """Fallback: no-op if python-dotenv is not installed."""
        pass


# =============================================================================
# Utilities
# =============================================================================

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)

def mkdirp(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def read_json(path: Path) -> Optional[Any]:
    """Read a JSON file with corruption handling.
    On parse failure: renames corrupt file, prints warning, returns None."""
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        # Rename corrupt file so it doesn't block future runs
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        corrupt_path = path.with_suffix(f".corrupt.{ts}")
        try:
            path.rename(corrupt_path)
            print(f"  [WARN] Corrupt JSON: {path.name} -> {corrupt_path.name}: {e}")
        except OSError:
            print(f"  [WARN] Corrupt JSON: {path.name}: {e} (could not rename)")
        return None
    except OSError as e:
        print(f"  [WARN] Cannot read {path.name}: {e}")
        return None


def atomic_write_json(path: Path, data: Any, indent: int = 2) -> None:
    """Write JSON atomically with fsync + directory sync.
    Survives process crash and power loss."""
    dir_path = str(path.parent) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f, indent=indent, sort_keys=True, ensure_ascii=False)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
        # Sync directory metadata
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_text_atomic(path: Path, text: str) -> None:
    """Write text atomically with fsync + directory sync."""
    dir_path = str(path.parent) or "."
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, str(path))
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def append_line(path: Path, line: str) -> None:
    """Append a line to a file with error handling."""
    try:
        with path.open("a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except OSError as e:
        print(f"  [WARN] Cannot append to {path.name}: {e}")


# =============================================================================
# Normalized Event Schema
# =============================================================================

@dataclass
class ModelEvent:
    timestamp: str
    provider: str
    model_id: str
    event_type: str  # MODEL_ADDED | MODEL_REMOVED | METADATA_CHANGED | CANARY_RESULT | BEHAVIOR_DRIFT | BENCHMARK_RESULT
    callable: Optional[bool] = None
    canary_latency_ms: Optional[int] = None
    notes: Optional[str] = None
    raw_snapshot_hash: Optional[str] = None
    details: Optional[Dict[str, Any]] = None

    def to_jsonl(self) -> str:
        return stable_json_dumps(dataclasses.asdict(self))


# =============================================================================
# PollResult — immutable worker output (Change 2: single-writer pattern)
# =============================================================================

@dataclass
class PollResult:
    """Immutable result from a single provider poll.
    Workers produce these. Main thread consumes them.
    No shared mutable state crosses the boundary."""
    provider: str
    snapshot: Optional[Dict[str, Any]]
    raw_hash: Optional[str]
    model_fingerprints: Dict[str, str]
    diff: Optional[Dict[str, List[str]]]
    events: List[ModelEvent]
    canary_results: List[Dict[str, Any]]
    benchmark_results: List[Dict[str, Any]]
    metadata_updates: List[Dict[str, Any]]
    error: Optional[str] = None


# =============================================================================
# Provider Watcher Interface
# =============================================================================

class ProviderWatcher:
    provider_name: str = "unknown"

    def poll_models(self) -> Dict[str, Any]:
        """Return raw model registry payload."""
        raise NotImplementedError

    def list_model_ids_and_fingerprints(self, raw: Dict[str, Any]) -> Dict[str, str]:
        """Return mapping: model_id -> fingerprint string."""
        raise NotImplementedError

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        """Try a tiny request. Returns: (callable, latency_ms, notes)"""
        raise NotImplementedError

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        """Deterministic benchmark. Returns: (ok, normalized_response_text, latency_ms).
        Default implementation uses canary_call — override for providers with
        different response extraction."""
        ok, latency_ms, notes = self.canary_call(model_id)
        return ok, notes, latency_ms


# =============================================================================
# Deterministic Benchmark — tier-specific canonicalization + three-state drift
# =============================================================================

BENCHMARK_PROMPTS = [
    # Tier 1: Gross drift — basic determinism
    "Return exactly: OK",
    "2+2=?",
    "Complete this sequence: 1, 1, 2, 3, 5, __",
    # Tier 2: Structured reasoning — catches verbosity/chain-of-thought shifts
    "A box contains 3 red balls and 2 blue balls. You draw 2 without replacement. "
    "What is P(both red)? Reply with ONLY the fraction, nothing else.",
    # Tier 3: JSON output stability — catches formatting drift
    'Output exactly this JSON, no explanation: {"name":"test","values":[1,2,3],"nested":{"ok":true}}',
    # Tier 4: Tool-call format — catches function calling schema drift
    "You have a function: get_weather(city: str, units: str). "
    'Call it for Tokyo in celsius. Reply ONLY with the function call in this format: '
    'get_weather(city="Tokyo", units="celsius")',
]

# Drift severity by tier (Change 4)
TIER_SEVERITY = {
    0: "INFO",       # Tier 1 echo
    1: "INFO",       # Tier 1 arithmetic
    2: "LOW",        # Tier 1 sequence
    3: "MEDIUM",     # Tier 2 reasoning
    4: "HIGH",       # Tier 3 JSON schema
    5: "CRITICAL",   # Tier 4 tool-call format
}

# Confirmation thresholds
CONFIRM_SAMPLES = 3      # R — resamples on mismatch
CONFIRM_STREAK = 3       # K — consecutive polls for drift declaration
CONFIRM_FREQ = 0.6       # T — frequency threshold in rolling window
BASELINE_WINDOW = 20     # W — rolling window size


import re  # noqa: E402 — needed for tool-call canonicalization


def canonicalize_benchmark(prompt_index: int, response_text: str) -> Tuple[bool, str]:
    """Tier-specific canonicalization. Returns (ok, canonical_value).

    ok=False means canonicalization failed (unparseable response).
    The canonical value is what gets compared for drift detection.
    """
    text = response_text.strip()
    if not text:
        return False, ""

    if prompt_index == 0:
        # Tier 1 echo: extract atomic answer. Canonical: "ok"
        t = text.lower().strip().rstrip(".,!?;:")
        t = " ".join(t.split())
        # Accept variants like "OK", "Ok.", "ok!", "Sure, OK"
        if "ok" in t:
            return True, "ok"
        return True, t  # different content — will be caught by drift logic

    elif prompt_index == 1:
        # Tier 1 arithmetic: extract numeric. Canonical: "4"
        t = text.lower().strip().rstrip(".,!?;:")
        # Extract first number-like token
        nums = re.findall(r'\b\d+\b', t)
        if nums and nums[0] == "4":
            return True, "4"
        if "four" in t:
            return True, "4"
        return True, " ".join(t.split())

    elif prompt_index == 2:
        # Tier 1 sequence: extract numeric. Canonical: "8"
        t = text.lower().strip().rstrip(".,!?;:")
        nums = re.findall(r'\b\d+\b', t)
        if nums and nums[0] == "8":
            return True, "8"
        if "eight" in t:
            return True, "8"
        return True, " ".join(t.split())

    elif prompt_index == 3:
        # Tier 2 reasoning: fraction. Canonical: reduced rational "3/10"
        t = text.strip().rstrip(".,!?;:")
        # Try to find a fraction-like pattern
        frac_match = re.search(r'(\d+)\s*/\s*(\d+)', t)
        if frac_match:
            num, den = int(frac_match.group(1)), int(frac_match.group(2))
            from math import gcd
            g = gcd(num, den)
            return True, f"{num // g}/{den // g}"
        # Try decimal
        dec_match = re.search(r'0?\.(\d+)', t)
        if dec_match:
            dec_val = float("0." + dec_match.group(1))
            if abs(dec_val - 0.3) < 0.01:
                return True, "3/10"
        return True, " ".join(t.lower().split())

    elif prompt_index == 4:
        # Tier 3 JSON: parse even from code fences, canonicalize
        t = text.strip()
        # Strip code fences if present
        if t.startswith("```"):
            lines = t.split("\n")
            # Remove first line (```json or ```) and last line (```)
            inner_lines = []
            started = False
            for line in lines:
                if not started and line.strip().startswith("```"):
                    started = True
                    continue
                if started and line.strip() == "```":
                    break
                if started:
                    inner_lines.append(line)
            if inner_lines:
                t = "\n".join(inner_lines).strip()
        try:
            parsed = json.loads(t)
            canonical = json.dumps(parsed, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            return True, canonical
        except (json.JSONDecodeError, TypeError):
            # Try to extract first JSON object from text
            brace_start = t.find("{")
            if brace_start >= 0:
                depth = 0
                for i, ch in enumerate(t[brace_start:], brace_start):
                    if ch == "{":
                        depth += 1
                    elif ch == "}":
                        depth -= 1
                        if depth == 0:
                            try:
                                parsed = json.loads(t[brace_start:i + 1])
                                return True, json.dumps(parsed, sort_keys=True,
                                                        separators=(",", ":"), ensure_ascii=False)
                            except (json.JSONDecodeError, TypeError):
                                break
            return False, t

    elif prompt_index == 5:
        # Tier 4 tool-call: parse into canonical form
        t = text.strip()
        # Try to parse get_weather(...) call
        call_match = re.search(r'get_weather\s*\((.*?)\)', t, re.DOTALL)
        if call_match:
            args_str = call_match.group(1).strip()
            # Parse key=value pairs
            args_dict = {}
            for arg_match in re.finditer(r'(\w+)\s*=\s*["\']([^"\']*)["\']', args_str):
                args_dict[arg_match.group(1)] = arg_match.group(2)
            if args_dict:
                # Canonical: sorted args, double quotes
                parts = [f'{k}="{v}"' for k, v in sorted(args_dict.items())]
                return True, f"get_weather({','.join(parts)})"
        return True, t  # can't parse — pass through, drift logic handles it

    return True, text


@dataclass
class BenchmarkBaseline:
    """Per (provider/model_id/prompt_idx) drift tracking state."""
    canonical_baseline: Optional[str] = None
    observation_window: List[str] = field(default_factory=list)
    variant_hashes: Set[str] = field(default_factory=set)
    candidate_drift_value: Optional[str] = None
    candidate_drift_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_baseline": self.canonical_baseline,
            "observation_window": self.observation_window,
            "variant_hashes": list(self.variant_hashes),
            "candidate_drift_value": self.candidate_drift_value,
            "candidate_drift_streak": self.candidate_drift_streak,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BenchmarkBaseline':
        return cls(
            canonical_baseline=d.get("canonical_baseline"),
            observation_window=d.get("observation_window", []),
            variant_hashes=set(d.get("variant_hashes", [])),
            candidate_drift_value=d.get("candidate_drift_value"),
            candidate_drift_streak=d.get("candidate_drift_streak", 0),
        )


class BenchmarkStore:
    """Three-state drift detector with tier-specific canonicalization.

    States per (model, prompt):
      MATCH   — canonical value matches baseline
      VARIANT — different surface form, same semantic content (or rare nondeterministic variant)
      DRIFT   — confirmed behavioral change after persistence thresholds met

    The system MUST NOT emit BEHAVIOR_DRIFT from a single observation.
    """

    def __init__(self, store_path: Path):
        self.path = store_path
        self.baselines: Dict[str, Dict[str, BenchmarkBaseline]] = {}
        # Legacy compat: also maintain simple hash data for experiment lockfiles
        self.data: Dict[str, Dict[str, str]] = {}
        self._load()

    def _load(self) -> None:
        loaded = read_json(self.path)
        if not isinstance(loaded, dict):
            return
        # Load v2 baselines if present
        if "_v2_baselines" in loaded:
            for model_key, prompts in loaded["_v2_baselines"].items():
                self.baselines[model_key] = {}
                for prompt_key, baseline_data in prompts.items():
                    self.baselines[model_key][prompt_key] = BenchmarkBaseline.from_dict(baseline_data)
        # Load legacy hash data
        for key, val in loaded.items():
            if key.startswith("_"):
                continue
            if isinstance(val, dict):
                self.data[key] = val

    def save(self) -> None:
        out: Dict[str, Any] = {}
        # Save legacy hash data
        for key, val in self.data.items():
            out[key] = val
        # Save v2 baselines
        v2: Dict[str, Any] = {}
        for model_key, prompts in self.baselines.items():
            v2[model_key] = {}
            for prompt_key, baseline in prompts.items():
                v2[model_key][prompt_key] = baseline.to_dict()
        out["_v2_baselines"] = v2
        write_text_atomic(self.path, json.dumps(out, indent=2, sort_keys=True))

    def _key(self, provider: str, model_id: str) -> str:
        return f"{provider}/{model_id}"

    def classify_observation(self, provider: str, model_id: str,
                             prompt_index: int, canonical_value: str,
                             raw_hash: str) -> str:
        """Classify a single canonicalized observation.

        Returns: "NEW", "MATCH", "VARIANT", or "DRIFT"
        """
        key = self._key(provider, model_id)
        prompt_key = str(prompt_index)

        if key not in self.baselines:
            self.baselines[key] = {}
        if prompt_key not in self.baselines[key]:
            self.baselines[key][prompt_key] = BenchmarkBaseline()

        baseline = self.baselines[key][prompt_key]

        # Also update legacy hash data
        if key not in self.data:
            self.data[key] = {}
        legacy_prompt_hash = sha256_bytes(BENCHMARK_PROMPTS[prompt_index].encode("utf-8"))[:16]
        self.data[key][legacy_prompt_hash] = sha256_bytes(canonical_value.encode("utf-8"))

        # First observation — establish baseline
        if baseline.canonical_baseline is None:
            baseline.canonical_baseline = canonical_value
            baseline.observation_window.append(canonical_value)
            baseline.variant_hashes.add(raw_hash)
            return "NEW"

        # Add to observation window (ring buffer)
        baseline.observation_window.append(canonical_value)
        if len(baseline.observation_window) > BASELINE_WINDOW:
            baseline.observation_window = baseline.observation_window[-BASELINE_WINDOW:]

        # MATCH: canonical value equals baseline
        if canonical_value == baseline.canonical_baseline:
            baseline.variant_hashes.add(raw_hash)
            baseline.candidate_drift_value = None
            baseline.candidate_drift_streak = 0
            return "MATCH"

        # Mismatch — track candidate drift
        if canonical_value == baseline.candidate_drift_value:
            baseline.candidate_drift_streak += 1
        else:
            baseline.candidate_drift_value = canonical_value
            baseline.candidate_drift_streak = 1

        # Check drift confirmation: K consecutive polls
        if baseline.candidate_drift_streak >= CONFIRM_STREAK:
            # Confirmed drift — update baseline
            baseline.canonical_baseline = canonical_value
            baseline.variant_hashes = {raw_hash}
            baseline.candidate_drift_value = None
            baseline.candidate_drift_streak = 0
            return "DRIFT"

        # Check drift confirmation: T frequency in window
        window_count = baseline.observation_window.count(canonical_value)
        window_freq = window_count / len(baseline.observation_window) if baseline.observation_window else 0
        if window_freq >= CONFIRM_FREQ and len(baseline.observation_window) >= CONFIRM_STREAK:
            baseline.canonical_baseline = canonical_value
            baseline.variant_hashes = {raw_hash}
            baseline.candidate_drift_value = None
            baseline.candidate_drift_streak = 0
            return "DRIFT"

        # Not confirmed — it's a VARIANT
        return "VARIANT"

    def check_and_update(self, provider: str, model_id: str,
                         prompt_index: int, prompt: str,
                         response_text: str) -> Optional[str]:
        """Backward-compatible interface. Canonicalizes then classifies.
        Returns drift type or None if stable."""
        ok, canonical = canonicalize_benchmark(prompt_index, response_text)
        raw_hash = sha256_bytes(response_text.encode("utf-8"))

        if not ok:
            # Canonicalization failed — record but don't trigger drift
            return None

        result = self.classify_observation(provider, model_id, prompt_index, canonical, raw_hash)
        if result == "NEW":
            return "NEW"
        elif result == "DRIFT":
            return "BEHAVIOR_DRIFT"
        elif result == "VARIANT":
            return "VARIANT"
        return None  # MATCH


# =============================================================================
# Semantic Model Fingerprinting (Change 3)
# =============================================================================

# Fields considered semantically meaningful for fingerprinting
FINGERPRINT_FIELDS = ["id", "name", "context_length", "inputTokenLimit",
                      "max_tokens", "capabilities", "pricing"]


def compute_model_fingerprint(m: Dict[str, Any]) -> str:
    """Compute a stable fingerprint from semantically meaningful fields only.
    Float-stable, Unicode-normalized, order-independent."""
    stable_fields: Dict[str, Any] = {}
    for key in FINGERPRINT_FIELDS:
        if key in m and m[key] is not None:
            val = m[key]
            if isinstance(val, float):
                val = round(val, 6)
            if isinstance(val, str):
                val = unicodedata.normalize('NFC', val)
            stable_fields[key] = val
    return sha256_bytes(stable_json_dumps(stable_fields).encode("utf-8"))


def compute_semantic_registry_hash(provider: str,
                                    model_fingerprints: Dict[str, str]) -> str:
    """Hash based on stable semantic primitives, not raw payload.
    Model list reordering does not change the hash."""
    parts = [f"{provider}"]
    for mid in sorted(model_fingerprints.keys()):
        parts.append(f"{mid}:{model_fingerprints[mid]}")
    return sha256_bytes("\n".join(parts).encode("utf-8"))


# =============================================================================
# OpenAI Watcher
# =============================================================================

def _extract_chat_response(r_json: Dict[str, Any]) -> str:
    """Extract text from OpenAI-compatible chat completions response."""
    choices = r_json.get("choices", [])
    if choices:
        msg = choices[0].get("message", {})
        return msg.get("content", "")
    return ""


def _openai_compat_benchmark(url: str, headers: Dict[str, str],
                              model_id: str, prompt: str,
                              timeout_s: int = 30) -> Tuple[bool, str, int]:
    """Generic benchmark call for any OpenAI-compatible chat endpoint.
    Returns (ok, response_text, latency_ms)."""
    t0 = time.time()
    try:
        payload = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 32,
            "temperature": 0,
        }
        r = requests.post(url, headers=headers, data=json.dumps(payload), timeout=timeout_s)
        latency_ms = int((time.time() - t0) * 1000)
        if r.status_code == 200:
            text = _extract_chat_response(r.json())
            return True, text, latency_ms
        return False, "", latency_ms
    except requests.RequestException:
        return False, "", int((time.time() - t0) * 1000)


class OpenAICompatWatcher(ProviderWatcher):
    """Base class for all providers using OpenAI-compatible API format.

    Covers: OpenAI, xAI, Mistral, DeepSeek, OpenRouter, Venice, Volink,
    Zhipu, Moonshot — any endpoint that follows the /models + /chat/completions
    convention with Bearer auth.  Provider-specific differences are handled
    via constructor parameters (base_url, models_path, chat_path).
    """

    def __init__(self, provider_name: str, api_key: str, base_url: str,
                 models_path: str = "/v1/models",
                 chat_path: str = "/v1/chat/completions",
                 timeout_s: int = 30):
        self.provider_name = provider_name
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.models_path = models_path
        self.chat_path = chat_path
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def poll_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}{self.models_path}"
        r = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def list_model_ids_and_fingerprints(self, raw: Dict[str, Any]) -> Dict[str, str]:
        data = raw.get("data", raw.get("models", []))
        if isinstance(data, dict):
            data = [data]
        out: Dict[str, str] = {}
        for m in data:
            if isinstance(m, dict):
                mid = m.get("id") or m.get("model_id")
            else:
                mid = str(m)
            if not mid:
                continue
            fp = sha256_bytes(stable_json_dumps(m).encode("utf-8"))
            out[mid] = fp
        return out

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        t0 = time.time()
        try:
            url = f"{self.base_url}{self.chat_path}"
            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": "Return exactly: OK"}],
                "max_tokens": 16,
                "temperature": 0,
            }
            r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return True, latency_ms, "chat_ok"
            try:
                err = r.json()
            except Exception:
                err = {"status_code": r.status_code, "text": r.text[:300]}
            return False, latency_ms, f"chat_fail:{stable_json_dumps(err)[:300]}"
        except requests.RequestException as e:
            latency_ms = int((time.time() - t0) * 1000)
            return False, latency_ms, f"request_error:{str(e)[:200]}"

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        url = f"{self.base_url}{self.chat_path}"
        return _openai_compat_benchmark(url, self._headers(), model_id, prompt, self.timeout_s)


class OpenAIWatcher(OpenAICompatWatcher):
    """OpenAI watcher — extends base with Responses API fallback for canary calls."""

    def __init__(self, api_key: str, base_url: str = "https://api.openai.com", timeout_s: int = 30):
        super().__init__("openai", api_key, base_url, timeout_s=timeout_s)

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        # Attempt Responses API first
        t0 = time.time()
        try:
            url = f"{self.base_url}/v1/responses"
            payload = {
                "model": model_id,
                "input": "Return exactly: OK",
                "max_output_tokens": 16,
            }
            r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return True, latency_ms, "responses_ok"
        except requests.RequestException:
            pass
        # Fallback: standard Chat Completions
        return super().canary_call(model_id)


# =============================================================================
# Anthropic Watcher
# =============================================================================
# Anthropic doesn't have a /v1/models list endpoint (as of Feb 2026).
# We maintain a known-models registry and canary-test each one.
# When they add a list endpoint, swap poll_models to hit it.

class AnthropicWatcher(ProviderWatcher):
    provider_name = "anthropic"

    # Known model IDs — seed list, updated from API + cache
    KNOWN_MODELS = [
        "claude-opus-4-6",
        "claude-opus-4-5-20250929",
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20251001",
        # Legacy (may still be callable)
        "claude-3-5-sonnet-20241022",
        "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ]

    # Class-level cache of last successful API response
    _cached_api_response: Optional[Dict[str, Any]] = None

    def __init__(self, api_key: str, timeout_s: int = 30):
        self.api_key = api_key
        self.base_url = "https://api.anthropic.com"
        self.timeout_s = timeout_s

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

    def poll_models(self) -> Dict[str, Any]:
        """
        Try the /v1/models endpoint first.
        On success: cache the response for future fallback.
        On failure: use cached response if available, else known-models list.
        """
        try:
            url = f"{self.base_url}/v1/models"
            r = requests.get(url, headers=self._headers(), timeout=self.timeout_s)
            if r.status_code == 200:
                result = r.json()
                AnthropicWatcher._cached_api_response = result
                return result
        except requests.RequestException:
            pass

        # Fallback 1: cached API response from last successful call
        if AnthropicWatcher._cached_api_response:
            cached = AnthropicWatcher._cached_api_response.copy()
            cached["_fallback"] = "cached_api"
            return cached

        # Fallback 2: seed list
        return {
            "data": [{"id": mid, "type": "model", "source": "known_list"} for mid in self.KNOWN_MODELS],
            "_fallback": "seed_list",
        }

    def list_model_ids_and_fingerprints(self, raw: Dict[str, Any]) -> Dict[str, str]:
        data = raw.get("data", [])
        out: Dict[str, str] = {}
        for m in data:
            mid = m.get("id")
            if not mid:
                continue
            fp = sha256_bytes(stable_json_dumps(m).encode("utf-8"))
            out[mid] = fp
        return out

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/v1/messages"
            payload = {
                "model": model_id,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": "Return exactly: OK"}],
            }
            r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return True, latency_ms, "messages_ok"
            try:
                err = r.json()
            except Exception:
                err = {"status_code": r.status_code, "text": r.text[:300]}
            return False, latency_ms, f"messages_fail:{stable_json_dumps(err)[:300]}"
        except requests.RequestException as e:
            latency_ms = int((time.time() - t0) * 1000)
            return False, latency_ms, f"request_error:{str(e)[:200]}"

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/v1/messages"
            payload = {
                "model": model_id,
                "max_tokens": 32,
                "messages": [{"role": "user", "content": prompt}],
            }
            r = requests.post(url, headers=self._headers(), data=json.dumps(payload), timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                data = r.json()
                content = data.get("content", [])
                text = content[0].get("text", "") if content else ""
                return True, text, latency_ms
            return False, "", latency_ms
        except requests.RequestException:
            return False, "", int((time.time() - t0) * 1000)


# =============================================================================
# Google (Gemini) Watcher
# =============================================================================

class GoogleWatcher(ProviderWatcher):
    provider_name = "google"

    def __init__(self, api_key: str, timeout_s: int = 30):
        self.api_key = api_key
        self.base_url = "https://generativelanguage.googleapis.com"
        self.timeout_s = timeout_s

    def poll_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/v1beta/models?key={self.api_key}"
        r = requests.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def list_model_ids_and_fingerprints(self, raw: Dict[str, Any]) -> Dict[str, str]:
        models = raw.get("models", [])
        out: Dict[str, str] = {}
        for m in models:
            # Google returns "models/gemini-2.0-flash" — normalize to just the model name
            mid = m.get("name", "")
            if mid.startswith("models/"):
                mid = mid[7:]
            if not mid:
                continue
            fp = sha256_bytes(stable_json_dumps(m).encode("utf-8"))
            out[mid] = fp
        return out

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/v1beta/models/{model_id}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": "Return exactly: OK"}]}],
                "generationConfig": {"maxOutputTokens": 16, "temperature": 0},
            }
            r = requests.post(url, json=payload, timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return True, latency_ms, "generate_ok"
            try:
                err = r.json()
            except Exception:
                err = {"status_code": r.status_code, "text": r.text[:300]}
            return False, latency_ms, f"generate_fail:{stable_json_dumps(err)[:300]}"
        except requests.RequestException as e:
            latency_ms = int((time.time() - t0) * 1000)
            return False, latency_ms, f"request_error:{str(e)[:200]}"

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/v1beta/models/{model_id}:generateContent?key={self.api_key}"
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": 32, "temperature": 0},
            }
            r = requests.post(url, json=payload, timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                data = r.json()
                candidates = data.get("candidates", [])
                if candidates:
                    parts = candidates[0].get("content", {}).get("parts", [])
                    text = parts[0].get("text", "") if parts else ""
                    return True, text, latency_ms
            return False, "", latency_ms
        except requests.RequestException:
            return False, "", int((time.time() - t0) * 1000)


# xAI, Mistral, DeepSeek, OpenRouter, Venice, Volink, Zhipu, Moonshot
# All use OpenAI-compatible API format — instantiated via OpenAICompatWatcher
# in PROVIDER_REGISTRY below.  No individual watcher classes needed.


# =============================================================================
# Ollama Watcher (local instances)
# =============================================================================
# No auth needed — just needs network access to the Ollama instance.

class OllamaWatcher(ProviderWatcher):
    provider_name = "ollama"

    def __init__(self, base_url: str = "http://localhost:11434", timeout_s: int = 15):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def poll_models(self) -> Dict[str, Any]:
        url = f"{self.base_url}/api/tags"
        r = requests.get(url, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def list_model_ids_and_fingerprints(self, raw: Dict[str, Any]) -> Dict[str, str]:
        models = raw.get("models", [])
        out: Dict[str, str] = {}
        for m in models:
            mid = m.get("name") or m.get("model")
            if not mid:
                continue
            # Include digest, size, modified_at in fingerprint for change detection
            fp = sha256_bytes(stable_json_dumps(m).encode("utf-8"))
            out[mid] = fp
        return out

    def canary_call(self, model_id: str) -> Tuple[bool, int, str]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model_id,
                "prompt": "Return exactly: OK",
                "stream": False,
                "options": {"num_predict": 16, "temperature": 0},
            }
            r = requests.post(url, json=payload, timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                return True, latency_ms, "generate_ok"
            try:
                err = r.json()
            except Exception:
                err = {"status_code": r.status_code, "text": r.text[:300]}
            return False, latency_ms, f"generate_fail:{stable_json_dumps(err)[:300]}"
        except requests.RequestException as e:
            latency_ms = int((time.time() - t0) * 1000)
            return False, latency_ms, f"request_error:{str(e)[:200]}"

    def benchmark_call(self, model_id: str, prompt: str) -> Tuple[bool, str, int]:
        t0 = time.time()
        try:
            url = f"{self.base_url}/api/generate"
            payload = {
                "model": model_id,
                "prompt": prompt,
                "stream": False,
                "options": {"num_predict": 32, "temperature": 0},
            }
            r = requests.post(url, json=payload, timeout=self.timeout_s)
            latency_ms = int((time.time() - t0) * 1000)
            if r.status_code == 200:
                text = r.json().get("response", "")
                return True, text, latency_ms
            return False, "", latency_ms
        except requests.RequestException:
            return False, "", int((time.time() - t0) * 1000)


# =============================================================================
# Capability Registry — per-model structured metadata
# =============================================================================

@dataclass
class ModelCapability:
    model_id: str
    provider: str
    max_context_tokens: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    latency_p50_ms: Optional[int] = None
    latency_p95_ms: Optional[int] = None
    reliability: Optional[float] = None  # success rate 0.0-1.0 over sliding window
    cost_class: Optional[str] = None     # "free" | "low" | "mid" | "high" | "premium"
    last_updated: Optional[str] = None
    canary_history: List[Dict[str, Any]] = field(default_factory=list)  # ring buffer of {ok, latency_ms}

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        return {k: v for k, v in d.items() if v is not None and v != []}


class CapabilityRegistry:
    """Persisted capability registry — builds over time from canary data + metadata."""

    WINDOW_SIZE = 20  # sliding window for reliability and latency

    def __init__(self, registry_path: Path):
        self.path = registry_path
        self.entries: Dict[str, ModelCapability] = {}
        self._load()

    def _load(self) -> None:
        data = read_json(self.path)
        if not data or not isinstance(data, dict):
            return
        for key, val in data.items():
            self.entries[key] = ModelCapability(**val)

    def save(self) -> None:
        out = {k: v.to_dict() for k, v in self.entries.items()}
        write_text_atomic(self.path, json.dumps(out, indent=2, sort_keys=True))

    def _key(self, provider: str, model_id: str) -> str:
        return f"{provider}/{model_id}"

    def update_from_canary(self, provider: str, model_id: str,
                           ok: bool, latency_ms: int) -> None:
        """Update latency and reliability from canary result using sliding window."""
        key = self._key(provider, model_id)
        cap = self.entries.get(key, ModelCapability(model_id=model_id, provider=provider))

        # Append to sliding window, trim to WINDOW_SIZE
        cap.canary_history.append({"ok": ok, "latency_ms": latency_ms})
        if len(cap.canary_history) > self.WINDOW_SIZE:
            cap.canary_history = cap.canary_history[-self.WINDOW_SIZE:]

        # Compute reliability from window
        successes = sum(1 for h in cap.canary_history if h.get("ok"))
        cap.reliability = round(successes / len(cap.canary_history), 4)

        # Compute latency percentiles from successful calls in window
        ok_latencies = sorted(h["latency_ms"] for h in cap.canary_history
                              if h.get("ok") and h.get("latency_ms") is not None)
        if ok_latencies:
            cap.latency_p50_ms = ok_latencies[len(ok_latencies) // 2]
            p95_idx = min(int((len(ok_latencies) - 1) * 0.95), len(ok_latencies) - 1)
            cap.latency_p95_ms = ok_latencies[p95_idx]

        cap.last_updated = utc_now_iso()
        self.entries[key] = cap

    def update_from_metadata(self, provider: str, model_id: str,
                             raw_model: Dict[str, Any]) -> None:
        """Extract capability metadata from provider model listing."""
        key = self._key(provider, model_id)
        cap = self.entries.get(key, ModelCapability(model_id=model_id, provider=provider))

        # OpenRouter is richest: context_length, pricing
        if "context_length" in raw_model:
            cap.max_context_tokens = raw_model["context_length"]
        # Google: inputTokenLimit
        if "inputTokenLimit" in raw_model:
            cap.max_context_tokens = raw_model["inputTokenLimit"]
        # OpenRouter pricing -> cost_class
        pricing = raw_model.get("pricing", {})
        if isinstance(pricing, dict) and "prompt" in pricing:
            try:
                cost_per_1k = float(pricing["prompt"]) * 1000
                if cost_per_1k == 0:
                    cap.cost_class = "free"
                elif cost_per_1k < 0.5:
                    cap.cost_class = "low"
                elif cost_per_1k < 3.0:
                    cap.cost_class = "mid"
                elif cost_per_1k < 15.0:
                    cap.cost_class = "high"
                else:
                    cap.cost_class = "premium"
            except (ValueError, TypeError):
                pass

        cap.last_updated = utc_now_iso()
        self.entries[key] = cap

    def query(self, min_context: Optional[int] = None,
              min_reliability: Optional[float] = None,
              max_cost_class: Optional[str] = None,
              provider: Optional[str] = None) -> List[ModelCapability]:
        """Query registry with filters. Returns matching capabilities."""
        cost_order = {"free": 0, "low": 1, "mid": 2, "high": 3, "premium": 4}
        max_cost_rank = cost_order.get(max_cost_class, 999) if max_cost_class else 999

        results = []
        for cap in self.entries.values():
            if provider and cap.provider != provider:
                continue
            if min_context and (cap.max_context_tokens is None or cap.max_context_tokens < min_context):
                continue
            if min_reliability and (cap.reliability is None or cap.reliability < min_reliability):
                continue
            if cap.cost_class and cost_order.get(cap.cost_class, 999) > max_cost_rank:
                continue
            results.append(cap)
        return results


# =============================================================================
# Experiment Invariants — safe drift detection
# =============================================================================

@dataclass
class ExperimentLock:
    experiment_id: str
    created_at: str
    provider_registry_hash: str
    model_ids_hash: str
    epoch_id: str = ""
    manifest_hash: str = ""
    benchmark_canonical_baselines: Optional[Dict[str, Dict[str, str]]] = None
    benchmark_variant_distributions: Optional[Dict[str, Dict[str, List[str]]]] = None
    benchmark_epoch: str = ""
    benchmark_coverage: int = 0
    benchmark_staleness_warning: bool = False
    notes: Optional[str] = None
    # Legacy fields for backward compatibility with v1 lockfiles
    canary_benchmark_hash: Optional[str] = None
    benchmark_response_hashes: Optional[Dict[str, Dict[str, str]]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = dataclasses.asdict(self)
        # Omit None/empty legacy fields in new lockfiles
        if not d.get("canary_benchmark_hash"):
            d.pop("canary_benchmark_hash", None)
        if not d.get("benchmark_response_hashes"):
            d.pop("benchmark_response_hashes", None)
        return d

    @property
    def is_v2(self) -> bool:
        """True if this lock was created under v2 semantics."""
        return bool(self.epoch_id) and self.benchmark_canonical_baselines is not None

    @property
    def behavioral_status(self) -> str:
        """Returns 'LOCKED' if v2 canonical baselines present, else 'UNLOCKED_BEHAVIORAL'."""
        if self.benchmark_canonical_baselines:
            return "LOCKED"
        return "UNLOCKED_BEHAVIORAL"


class ExperimentInvariantManager:
    """Manages experiment lockfiles for drift detection."""

    CANARY_PROMPT = "Return exactly: OK"

    def __init__(self, locks_dir: Path):
        self.locks_dir = locks_dir
        mkdirp(locks_dir)

    def _snapshot_hashes(self, state_dir: Path) -> Tuple[str, str]:
        """Compute current provider registry hash and model IDs hash.
        Uses semantic fingerprinting (Change 3) when epoch data is available,
        falls back to raw hashes for v1 state."""
        registry_parts = []
        model_ids_parts = []
        for snap_file in sorted(state_dir.glob("*_snapshot_latest.json")):
            snap = read_json(snap_file)
            if not snap:
                continue
            provider = snap.get("provider", "")
            # Use semantic registry hash if fingerprints available
            semantic_hash = snap.get("semantic_registry_hash")
            if semantic_hash:
                registry_parts.append(f"{provider}:{semantic_hash}")
            else:
                registry_parts.append(f"{provider}:{snap.get('raw_hash', '')}")
            models = snap.get("models", {})
            for mid in sorted(models.keys()):
                model_ids_parts.append(f"{provider}/{mid}")

        registry_hash = sha256_bytes("|".join(registry_parts).encode("utf-8"))
        model_ids_hash = sha256_bytes("|".join(model_ids_parts).encode("utf-8"))
        return registry_hash, model_ids_hash

    def _get_current_epoch(self, state_dir: Path) -> Tuple[str, str]:
        """Read current epoch_id and manifest_hash from epoch pointer."""
        pointer = read_json(state_dir / "current_epoch.json")
        if not pointer:
            return "", ""
        epoch_id = pointer.get("epoch_id", "")
        manifest_file = pointer.get("manifest_file", "")
        manifest_hash = ""
        if manifest_file:
            manifest = read_json(state_dir / manifest_file)
            if manifest:
                manifest_hash = manifest.get("manifest_hash", "")
        return epoch_id, manifest_hash

    def create_lock(self, experiment_id: str, state_dir: Path,
                    benchmark_store: Optional['BenchmarkStore'] = None,
                    notes: Optional[str] = None,
                    benchmark_before_lock: bool = False,
                    watchers: Optional[List] = None) -> ExperimentLock:
        """Create an experiment lockfile capturing current model ecosystem state.

        If benchmark_before_lock is True and watchers are provided, runs fresh
        benchmarks for all models before creating the lock.

        Lockfiles store canonical baselines (not binary hashes) per Change 5/GPT-5.2 Thinking Pass 2.
        """
        registry_hash, model_ids_hash = self._snapshot_hashes(state_dir)
        epoch_id, manifest_hash = self._get_current_epoch(state_dir)

        # Run fresh benchmarks if requested
        if benchmark_before_lock and watchers:
            bench_path = state_dir / "benchmark_hashes.json"
            if benchmark_store is None:
                benchmark_store = BenchmarkStore(bench_path)
            for w in watchers:
                snap_path = state_dir / f"{w.provider_name}_snapshot_latest.json"
                snap = read_json(snap_path)
                if not snap:
                    continue
                models = snap.get("models", {})
                for mid in models.keys():
                    for pi, prompt in enumerate(BENCHMARK_PROMPTS):
                        try:
                            ok, text, _lat = w.benchmark_call(mid, prompt)
                            if ok and text:
                                benchmark_store.check_and_update(
                                    w.provider_name, mid, pi, prompt, text)
                        except Exception:
                            pass
            benchmark_store.save()

        # Capture canonical baselines from BenchmarkStore
        canonical_baselines: Optional[Dict[str, Dict[str, str]]] = None
        variant_dists: Optional[Dict[str, Dict[str, List[str]]]] = None
        bench_coverage = 0
        bench_epoch = ""
        staleness_warning = False

        if benchmark_store and benchmark_store.baselines:
            canonical_baselines = {}
            variant_dists = {}
            for model_key, prompts in benchmark_store.baselines.items():
                canonical_baselines[model_key] = {}
                variant_dists[model_key] = {}
                for prompt_key, baseline in prompts.items():
                    if baseline.canonical_baseline is not None:
                        canonical_baselines[model_key][prompt_key] = baseline.canonical_baseline
                        bench_coverage += 1
                    if baseline.variant_hashes:
                        variant_dists[model_key][prompt_key] = list(baseline.variant_hashes)
            bench_epoch = epoch_id
            # Clean up empty dicts
            if not any(variant_dists.values()):
                variant_dists = None
        elif benchmark_store is None or not benchmark_store.baselines:
            staleness_warning = True

        lock = ExperimentLock(
            experiment_id=experiment_id,
            created_at=utc_now_iso(),
            provider_registry_hash=registry_hash,
            model_ids_hash=model_ids_hash,
            epoch_id=epoch_id,
            manifest_hash=manifest_hash,
            benchmark_canonical_baselines=canonical_baselines,
            benchmark_variant_distributions=variant_dists,
            benchmark_epoch=bench_epoch,
            benchmark_coverage=bench_coverage,
            benchmark_staleness_warning=staleness_warning,
            notes=notes,
        )
        lock_path = self.locks_dir / f"{experiment_id}.lock.json"
        write_text_atomic(lock_path, json.dumps(lock.to_dict(), indent=2))
        return lock

    def check_invariants(self, experiment_id: str, state_dir: Path,
                         benchmark_store: Optional['BenchmarkStore'] = None) -> Dict[str, Any]:
        """Check if current state matches experiment lock. Returns drift report.

        v2 lockfiles: Uses MATCH/VARIANT/DRIFT logic against canonical baselines.
        Variants do NOT invalidate the lock — only confirmed DRIFT does.

        v1 lockfiles: Marks behavioral component as UNLOCKED_BEHAVIORAL.
        Structural checks (registry, model list) still apply.
        """
        lock_path = self.locks_dir / f"{experiment_id}.lock.json"
        lock_data = read_json(lock_path)
        if not lock_data:
            return {"status": "NO_LOCK", "experiment_id": experiment_id,
                    "message": f"No lockfile found for experiment '{experiment_id}'"}

        # Handle both v1 and v2 lockfiles gracefully
        known_fields = {f.name for f in dataclasses.fields(ExperimentLock)}
        filtered = {k: v for k, v in lock_data.items() if k in known_fields}
        lock = ExperimentLock(**filtered)

        registry_hash, model_ids_hash = self._snapshot_hashes(state_dir)

        drifts = []
        variants = []
        warnings = []

        if registry_hash != lock.provider_registry_hash:
            drifts.append("PROVIDER_REGISTRY_DRIFT")
        if model_ids_hash != lock.model_ids_hash:
            drifts.append("MODEL_IDS_DRIFT")

        # Behavioral check — v2 canonical baselines (MATCH/VARIANT/DRIFT logic)
        if lock.benchmark_canonical_baselines and benchmark_store:
            drifted_models = []
            variant_models = []
            for model_key, locked_prompts in lock.benchmark_canonical_baselines.items():
                if model_key not in benchmark_store.baselines:
                    continue
                for prompt_key, locked_canonical in locked_prompts.items():
                    current_baselines = benchmark_store.baselines.get(model_key, {})
                    current_bl = current_baselines.get(prompt_key)
                    if current_bl is None:
                        continue
                    current_canonical = current_bl.canonical_baseline
                    if current_canonical is None:
                        continue
                    if current_canonical == locked_canonical:
                        pass  # MATCH — no action
                    elif current_bl.candidate_drift_streak >= CONFIRM_STREAK:
                        # Confirmed drift
                        drifted_models.append(model_key)
                        break
                    else:
                        # Check frequency threshold
                        window_count = current_bl.observation_window.count(current_canonical)
                        window_freq = (window_count / len(current_bl.observation_window)
                                       if current_bl.observation_window else 0)
                        if (window_freq >= CONFIRM_FREQ
                                and len(current_bl.observation_window) >= CONFIRM_STREAK):
                            drifted_models.append(model_key)
                            break
                        else:
                            variant_models.append(model_key)
            if drifted_models:
                drifts.append(f"BENCHMARK_DRIFT({','.join(sorted(set(drifted_models))[:5])})")
            if variant_models:
                variants.append(f"BENCHMARK_VARIANT({','.join(sorted(set(variant_models))[:5])})")

        # Legacy v1 lockfile — cannot check behavioral drift with canonical values
        elif lock.benchmark_response_hashes and not lock.benchmark_canonical_baselines:
            warnings.append("UNLOCKED_BEHAVIORAL: v1 lockfile uses binary hashes, "
                            "not canonical baselines. Re-lock under v2 semantics "
                            "with --benchmark-before-lock for behavioral drift detection.")

        if drifts:
            result: Dict[str, Any] = {
                "status": "DRIFT_DETECTED",
                "experiment_id": experiment_id,
                "drifts": drifts,
                "locked_at": lock.created_at,
                "checked_at": utc_now_iso(),
                "message": f"WARN: Experiment '{experiment_id}' invariants violated: {', '.join(drifts)}. "
                           f"Results collected after this point may mix pre/post-drift behavior.",
            }
            if variants:
                result["variants"] = variants
            if warnings:
                result["warnings"] = warnings
            return result

        result = {
            "status": "CLEAN",
            "experiment_id": experiment_id,
            "locked_at": lock.created_at,
            "checked_at": utc_now_iso(),
        }
        if variants:
            result["variants"] = variants
        if warnings:
            result["warnings"] = warnings
            result["behavioral_status"] = lock.behavioral_status
        return result

    def list_locks(self) -> List[str]:
        """List all active experiment IDs."""
        return [p.stem.replace(".lock", "")
                for p in self.locks_dir.glob("*.lock.json")]


# =============================================================================
# v1 -> v2 Migration
# =============================================================================

def migrate_v1_to_v2(state_dir: Path, events_file: Path) -> bool:
    """One-time migration from v1 state directory to v2 epoch-based state.

    Detection: v1 artifacts exist AND current_epoch.json does not.
    Idempotent: re-running after migration is a no-op.

    Returns True if migration was performed, False if not needed.
    """
    epoch_pointer = state_dir / "current_epoch.json"
    v1_snapshots = list(state_dir.glob("*_snapshot_latest.json"))

    # Already migrated or no v1 state
    if epoch_pointer.exists() or not v1_snapshots:
        return False

    ts = utc_now_iso()
    print(f"\n  [MIGRATION] v1 -> v2 state migration detected")
    print(f"  Found {len(v1_snapshots)} v1 snapshot(s)")

    # Step 1: Backup v1 state
    backup_name = f"v1_backup_{ts.replace(':', '-').replace('T', '_')}"
    backup_dir = state_dir / backup_name
    mkdirp(backup_dir)
    for f in state_dir.iterdir():
        if f.is_file() and f.name != ".observatory.lock":
            shutil.copy2(f, backup_dir / f.name)
    print(f"  Backed up v1 state to {backup_dir.name}/")

    # Step 2: Create epoch 0 from existing snapshots
    epoch_id = "v2-migration-0"
    model_count = 0
    providers_manifest: Dict[str, Any] = {}
    manifest_hash_parts = []

    for snap_file in sorted(v1_snapshots):
        snap = read_json(snap_file)
        if not snap:
            continue
        provider = snap.get("provider", snap_file.stem.replace("_snapshot_latest", ""))
        models = snap.get("models", {})
        model_count += len(models)

        # Write epoch-stamped snapshot
        epoch_snap_name = f"{provider}_snapshot_{epoch_id}.json"
        atomic_write_json(state_dir / epoch_snap_name, snap)

        snap_hash = sha256_bytes(stable_json_dumps(snap).encode("utf-8"))
        providers_manifest[provider] = {
            "snapshot_file": epoch_snap_name,
            "hash": snap_hash,
        }
        manifest_hash_parts.append(f"{provider}:{snap_hash}")

    manifest_hash = sha256_bytes("|".join(sorted(manifest_hash_parts)).encode("utf-8"))
    manifest = {
        "epoch": epoch_id,
        "providers": providers_manifest,
        "manifest_hash": manifest_hash,
        "migration": True,
    }
    manifest_file = f"epoch_manifest_{epoch_id}.json"
    atomic_write_json(state_dir / manifest_file, manifest)

    # Step 3: Update epoch pointer
    atomic_write_json(epoch_pointer, {
        "epoch_id": epoch_id,
        "manifest_file": manifest_file,
        "manifest_hash": manifest_hash,
        "created_at": ts,
    })

    # Step 4: Mark benchmark baselines as NEEDS_REBASELINE
    bench_path = state_dir / "benchmark_hashes.json"
    bench_data = read_json(bench_path)
    if bench_data and isinstance(bench_data, dict):
        bench_data["_needs_rebaseline"] = True
        bench_data["_migration_note"] = ("v1 binary hashes cannot be converted to "
                                         "canonical values. First v2 benchmark run "
                                         "will populate canonical baselines.")
        atomic_write_json(bench_path, bench_data)

    # Step 5: Emit single MIGRATION_REBASELINE event (not per-model flood)
    event = {
        "type": "MIGRATION_REBASELINE",
        "timestamp": ts,
        "epoch_id": epoch_id,
        "models_rebaselined": model_count,
        "providers_migrated": list(providers_manifest.keys()),
        "message": (f"v1->v2 migration complete. {model_count} models across "
                    f"{len(providers_manifest)} providers. Fingerprints rebaselined. "
                    f"Benchmark baselines marked NEEDS_REBASELINE."),
    }
    try:
        with open(events_file, "a") as f:
            f.write(json.dumps(event, sort_keys=True) + "\n")
    except OSError:
        pass

    print(f"  Migration complete: epoch 0 created with {model_count} models")
    print(f"  Benchmark baselines marked NEEDS_REBASELINE")
    print(f"  Single MIGRATION_REBASELINE event emitted\n")
    return True


# =============================================================================
# Webhook Notification System — tiered severity
# =============================================================================

class Severity:
    INFO = "INFO"
    WARN = "WARN"
    CRITICAL = "CRITICAL"


# Event type -> severity mapping
EVENT_SEVERITY = {
    "MODEL_ADDED": Severity.INFO,
    "MODEL_REMOVED": Severity.WARN,
    "METADATA_CHANGED": Severity.INFO,
    "CANARY_RESULT": Severity.INFO,
    "BENCHMARK_RESULT": Severity.INFO,
    "BENCHMARK_VARIANT": Severity.INFO,
    "BENCHMARK_MISSING": Severity.WARN,
    "BENCHMARK_FAILURE": Severity.INFO,
    "BEHAVIOR_DRIFT": Severity.CRITICAL,
    "MIGRATION_REBASELINE": Severity.INFO,
    "STATE_CORRUPT": Severity.WARN,
    "POLL_ERROR": Severity.WARN,
}

# Experiment drift is always critical
DRIFT_SEVERITY = Severity.CRITICAL


class WebhookNotifier:
    """Sends notifications to Discord, Slack, or generic HTTP endpoints."""

    def __init__(self, config_path: Path):
        self.config_path = config_path
        self.hooks: List[Dict[str, Any]] = []
        self._cooldowns: Dict[str, float] = {}  # "event_key" -> last_sent_timestamp
        self.default_cooldown_s = 300  # 5 minutes between duplicate notifications
        self._load()

    def _load(self) -> None:
        data = read_json(self.config_path)
        if isinstance(data, list):
            self.hooks = data
        elif isinstance(data, dict) and "webhooks" in data:
            self.hooks = data["webhooks"]
            self.default_cooldown_s = data.get("cooldown_seconds", 300)

    @property
    def enabled(self) -> bool:
        return len(self.hooks) > 0

    def _cooldown_key(self, event_type: str, provider: str, model_id: str) -> str:
        return f"{event_type}:{provider}:{model_id}"

    def _is_cooled_down(self, key: str) -> bool:
        last = self._cooldowns.get(key, 0)
        return (time.time() - last) >= self.default_cooldown_s

    def notify(self, event_type: str, provider: str, model_id: str,
               message: str, severity: Optional[str] = None) -> None:
        """Send notification to all configured webhooks that match severity threshold.
        Respects cooldown to prevent notification spam."""
        if not self.hooks:
            return

        # Cooldown check — CRITICAL always fires, others respect cooldown
        sev = severity or EVENT_SEVERITY.get(event_type, Severity.INFO)
        cooldown_key = self._cooldown_key(event_type, provider, model_id)
        if sev != Severity.CRITICAL and not self._is_cooled_down(cooldown_key):
            return
        self._cooldowns[cooldown_key] = time.time()

        sev_order = {Severity.INFO: 0, Severity.WARN: 1, Severity.CRITICAL: 2}
        sev_rank = sev_order.get(sev, 0)

        for hook in self.hooks:
            min_sev = hook.get("min_severity", Severity.INFO)
            if sev_order.get(min_sev, 0) > sev_rank:
                continue

            url = hook.get("url", "")
            if not url:
                continue

            hook_type = hook.get("type", "generic")
            try:
                if hook_type == "discord":
                    self._send_discord(url, sev, event_type, provider, model_id, message)
                elif hook_type == "slack":
                    self._send_slack(url, sev, event_type, provider, model_id, message)
                else:
                    self._send_generic(url, sev, event_type, provider, model_id, message)
            except Exception as e:
                print(f"  [WEBHOOK ERROR] {hook_type} -> {str(e)[:100]}")

    def notify_drift(self, experiment_id: str, drifts: List[str]) -> None:
        """Special notification for experiment invariant drift."""
        msg = f"Experiment '{experiment_id}' invariants violated: {', '.join(drifts)}"
        self.notify("EXPERIMENT_DRIFT", "system", experiment_id, msg, Severity.CRITICAL)

    def _severity_emoji(self, sev: str) -> str:
        return {"INFO": "INFO", "WARN": "WARN", "CRITICAL": "CRITICAL"}.get(sev, "EVENT")

    def _severity_color(self, sev: str) -> int:
        """Discord embed color."""
        return {Severity.INFO: 0x3498db, Severity.WARN: 0xf39c12, Severity.CRITICAL: 0xe74c3c}.get(sev, 0x95a5a6)

    def _send_discord(self, url: str, sev: str, event_type: str,
                      provider: str, model_id: str, message: str) -> None:
        payload = {
            "embeds": [{
                "title": f"[{sev}] {event_type}",
                "description": message,
                "color": self._severity_color(sev),
                "fields": [
                    {"name": "Provider", "value": provider, "inline": True},
                    {"name": "Model", "value": model_id, "inline": True},
                ],
                "timestamp": utc_now_iso(),
                "footer": {"text": "Model Observatory"},
            }]
        }
        requests.post(url, json=payload, timeout=10)

    def _send_slack(self, url: str, sev: str, event_type: str,
                    provider: str, model_id: str, message: str) -> None:
        payload = {
            "blocks": [
                {
                    "type": "header",
                    "text": {"type": "plain_text",
                             "text": f"[{sev}] {event_type}"}
                },
                {
                    "type": "section",
                    "text": {"type": "mrkdwn", "text": message},
                    "fields": [
                        {"type": "mrkdwn", "text": f"*Provider:* {provider}"},
                        {"type": "mrkdwn", "text": f"*Model:* {model_id}"},
                    ]
                },
            ]
        }
        requests.post(url, json=payload, timeout=10)

    def _send_generic(self, url: str, sev: str, event_type: str,
                      provider: str, model_id: str, message: str) -> None:
        payload = {
            "severity": sev,
            "event_type": event_type,
            "provider": provider,
            "model_id": model_id,
            "message": message,
            "timestamp": utc_now_iso(),
        }
        requests.post(url, json=payload, timeout=10)


# =============================================================================
# Diff logic
# =============================================================================

@dataclass
class Snapshot:
    timestamp: str
    provider: str
    raw_hash: str
    models: Dict[str, str]

def compute_diff(prev: Dict[str, str], curr: Dict[str, str]) -> Dict[str, List[str]]:
    prev_ids = set(prev.keys())
    curr_ids = set(curr.keys())
    added = sorted(curr_ids - prev_ids)
    removed = sorted(prev_ids - curr_ids)
    common = prev_ids & curr_ids
    changed = sorted([mid for mid in common if prev[mid] != curr[mid]])
    unchanged = sorted(list(common - set(changed)))
    return {"added": added, "removed": removed, "changed": changed, "unchanged": unchanged}


# =============================================================================
# Worker function — pure, no side effects (Change 2: single-writer)
# =============================================================================

def poll_provider(
    watcher: ProviderWatcher,
    prev_models: Dict[str, str],
    do_canary_on_added: bool = True,
    do_canary_on_changed: bool = False,
    do_benchmark: bool = False,
    benchmark_models: Optional[List[str]] = None,
) -> PollResult:
    """Poll a single provider. Pure function — no shared state mutation.
    Returns an immutable PollResult for the main thread to merge."""
    provider = watcher.provider_name
    events: List[ModelEvent] = []
    canary_results: List[Dict[str, Any]] = []
    benchmark_results: List[Dict[str, Any]] = []
    metadata_updates: List[Dict[str, Any]] = []

    # 1. Poll models
    raw = watcher.poll_models()
    raw_bytes = stable_json_dumps(raw).encode("utf-8")
    raw_hash = sha256_bytes(raw_bytes)
    curr_models = watcher.list_model_ids_and_fingerprints(raw)

    # 2. Compute diff against previous snapshot (read-only)
    diff = compute_diff(prev_models, curr_models)

    # 3. Generate events
    for mid in diff["added"]:
        events.append(ModelEvent(
            timestamp=utc_now_iso(), provider=provider, model_id=mid,
            event_type="MODEL_ADDED", raw_snapshot_hash=raw_hash,
            details={"fingerprint": curr_models.get(mid)},
        ))

    for mid in diff["removed"]:
        events.append(ModelEvent(
            timestamp=utc_now_iso(), provider=provider, model_id=mid,
            event_type="MODEL_REMOVED", raw_snapshot_hash=raw_hash,
            details={"prev_fingerprint": prev_models.get(mid)},
        ))

    for mid in diff["changed"]:
        events.append(ModelEvent(
            timestamp=utc_now_iso(), provider=provider, model_id=mid,
            event_type="METADATA_CHANGED", raw_snapshot_hash=raw_hash,
            details={"prev_fingerprint": prev_models.get(mid),
                     "new_fingerprint": curr_models.get(mid)},
        ))

    # 4. Collect metadata updates for registry
    raw_models = raw.get("data", raw.get("models", []))
    if isinstance(raw_models, list):
        for m in raw_models:
            if isinstance(m, dict):
                mid = m.get("id") or m.get("name", "")
                if mid.startswith("models/"):
                    mid = mid[7:]
                if mid:
                    metadata_updates.append({"model_id": mid, "raw_model": m})

    # 5. Run canary calls
    canary_targets: List[str] = []
    if do_canary_on_added:
        canary_targets.extend(diff["added"])
    if do_canary_on_changed:
        canary_targets.extend(diff["changed"])
    seen: Set[str] = set()
    canary_targets = [x for x in canary_targets if not (x in seen or seen.add(x))]

    for mid in canary_targets:
        ok, latency_ms, notes = watcher.canary_call(mid)
        canary_results.append({
            "model_id": mid, "ok": ok, "latency_ms": latency_ms, "notes": notes,
        })
        events.append(ModelEvent(
            timestamp=utc_now_iso(), provider=provider, model_id=mid,
            event_type="CANARY_RESULT", callable=ok,
            canary_latency_ms=latency_ms, notes=notes,
            raw_snapshot_hash=raw_hash,
        ))

    # 6. Run benchmark calls (collect raw responses — main thread canonicalizes/stores)
    if do_benchmark and benchmark_models is not None:
        targets = benchmark_models if benchmark_models else list(curr_models.keys())
        for mid in targets:
            if mid not in curr_models:
                continue
            for i, prompt in enumerate(BENCHMARK_PROMPTS):
                ok, response_text, latency_ms = watcher.benchmark_call(mid, prompt)
                benchmark_results.append({
                    "model_id": mid, "prompt_idx": i, "prompt": prompt,
                    "ok": ok, "response_text": response_text, "latency_ms": latency_ms,
                })

    return PollResult(
        provider=provider,
        snapshot=raw,
        raw_hash=raw_hash,
        model_fingerprints=curr_models,
        diff=diff,
        events=events,
        canary_results=canary_results,
        benchmark_results=benchmark_results,
        metadata_updates=metadata_updates,
    )


# =============================================================================
# Epoch Management (Change 1: global poll epoch)
# =============================================================================

EPOCH_RETENTION = 10  # keep this many recent epochs


def generate_epoch_id() -> str:
    """Generate a monotonic epoch identifier."""
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_epoch_snapshot(state_dir: Path, provider: str, epoch_id: str,
                         curr_models: Dict[str, str], raw_hash: str) -> Path:
    """Write an epoch-stamped snapshot file. Returns the path."""
    snapshot = Snapshot(
        timestamp=utc_now_iso(),
        provider=provider,
        raw_hash=raw_hash,
        models=curr_models,
    )
    # Epoch-stamped file
    epoch_path = state_dir / f"{provider}_snapshot_{epoch_id}.json"
    write_text_atomic(epoch_path, stable_json_dumps(dataclasses.asdict(snapshot)))
    # Also update _latest pointer for backward compatibility
    latest_path = state_dir / f"{provider}_snapshot_latest.json"
    write_text_atomic(latest_path, stable_json_dumps(dataclasses.asdict(snapshot)))
    return epoch_path


def write_epoch_manifest(state_dir: Path, epoch_id: str,
                         provider_hashes: Dict[str, str]) -> Dict[str, Any]:
    """Write epoch manifest and return it."""
    # Build manifest
    providers_section = {}
    for provider, raw_hash in sorted(provider_hashes.items()):
        providers_section[provider] = {
            "snapshot_file": f"{provider}_snapshot_{epoch_id}.json",
            "hash": raw_hash,
        }
    manifest_input = "|".join(f"{p}:{h}" for p, h in sorted(provider_hashes.items()))
    manifest_hash = sha256_bytes(manifest_input.encode("utf-8"))

    manifest = {
        "epoch": epoch_id,
        "timestamp": utc_now_iso(),
        "providers": providers_section,
        "manifest_hash": manifest_hash,
    }

    manifest_path = state_dir / f"epoch_manifest_{epoch_id}.json"
    write_text_atomic(manifest_path, json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def update_epoch_pointer(state_dir: Path, epoch_id: str) -> None:
    """Atomically update current_epoch.json to point to the latest epoch."""
    pointer = {"current_epoch": epoch_id, "updated_at": utc_now_iso()}
    pointer_path = state_dir / "current_epoch.json"
    write_text_atomic(pointer_path, json.dumps(pointer, indent=2))


def cleanup_old_epochs(state_dir: Path, keep: int = EPOCH_RETENTION) -> None:
    """Remove old epoch files beyond retention limit."""
    manifests = sorted(state_dir.glob("epoch_manifest_*.json"))
    if len(manifests) <= keep:
        return
    for old_manifest in manifests[:-keep]:
        # Read manifest to find associated snapshot files
        data = read_json(old_manifest)
        if data and "providers" in data:
            for prov_info in data["providers"].values():
                snap_file = state_dir / prov_info.get("snapshot_file", "")
                if snap_file.exists():
                    try:
                        snap_file.unlink()
                    except OSError:
                        pass
        try:
            old_manifest.unlink()
        except OSError:
            pass


# =============================================================================
# Main thread merge — processes PollResults sequentially
# =============================================================================

def merge_poll_results(
    results: List[PollResult],
    epoch_id: str,
    state_dir: Path,
    events_file: Path,
    cap_reg: CapabilityRegistry,
    bench_store: Optional[BenchmarkStore],
    notifier: Optional[WebhookNotifier],
) -> Tuple[int, int]:
    """Merge all PollResults into state. Main thread only.
    Returns (total_models, total_events).

    Epoch commit ordering (per spec):
    1. Write epoch-stamped snapshots
    2. Write epoch manifest
    3. Save CapabilityRegistry
    4. Save BenchmarkStore
    5. Append events to events.jsonl
    6. Update current_epoch.json pointer
    7. Fire webhooks (best-effort, after all state durable)
    """
    total_models = 0
    total_events = 0
    provider_hashes: Dict[str, str] = {}
    all_events: List[ModelEvent] = []
    webhook_queue: List[ModelEvent] = []

    # --- Step 1: Write snapshots + merge data ---
    for result in results:
        if result.error:
            print(f"  {result.provider:15s} | ERROR: {result.error[:100]}")
            all_events.append(ModelEvent(
                timestamp=utc_now_iso(), provider=result.provider,
                model_id="__POLL_ERROR__", event_type="POLL_ERROR",
                notes=result.error,
            ))
            continue

        # Write epoch-stamped snapshot
        write_epoch_snapshot(state_dir, result.provider, epoch_id,
                            result.model_fingerprints, result.raw_hash or "")
        provider_hashes[result.provider] = result.raw_hash or ""

        # Merge metadata into registry
        for update in result.metadata_updates:
            cap_reg.update_from_metadata(result.provider, update["model_id"], update["raw_model"])

        # Merge canary results into registry
        for canary in result.canary_results:
            cap_reg.update_from_canary(result.provider, canary["model_id"],
                                       canary["ok"], canary["latency_ms"])

        # Merge benchmark results into store (three-state classification)
        if bench_store:
            for bench in result.benchmark_results:
                if not bench["ok"]:
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BENCHMARK_FAILURE",
                        notes=f"prompt_{bench['prompt_idx']}: benchmark call failed",
                        details={"prompt_index": bench["prompt_idx"]},
                    ))
                    continue
                if not bench["response_text"]:
                    continue
                drift_status = bench_store.check_and_update(
                    result.provider, bench["model_id"],
                    bench["prompt_idx"], bench["prompt"],
                    bench["response_text"])
                if drift_status == "BEHAVIOR_DRIFT":
                    tier_sev = TIER_SEVERITY.get(bench["prompt_idx"], "CRITICAL")
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BEHAVIOR_DRIFT",
                        notes=f"prompt_{bench['prompt_idx']}: confirmed drift (severity={tier_sev})",
                        details={"prompt_index": bench["prompt_idx"],
                                 "tier_severity": tier_sev},
                    ))
                elif drift_status == "VARIANT":
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BENCHMARK_VARIANT",
                        notes=f"prompt_{bench['prompt_idx']}: nondeterministic variant",
                        details={"prompt_index": bench["prompt_idx"]},
                    ))
                elif drift_status == "NEW":
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BENCHMARK_RESULT", callable=True,
                        canary_latency_ms=bench["latency_ms"],
                        notes=f"prompt_{bench['prompt_idx']}: baseline established",
                    ))

        # Collect events from worker
        all_events.extend(result.events)
        webhook_queue.extend(result.events)

        # Print summary
        model_count = len(result.model_fingerprints)
        total_models += model_count
        if result.diff:
            total_events += len(result.diff["added"]) + len(result.diff["removed"]) + len(result.diff["changed"])
            print_summary(result.provider, result.diff, model_count)

    # --- Step 2: Write epoch manifest ---
    if provider_hashes:
        write_epoch_manifest(state_dir, epoch_id, provider_hashes)

    # --- Step 3: Save CapabilityRegistry ---
    cap_reg.save()

    # --- Step 4: Save BenchmarkStore ---
    if bench_store:
        bench_store.save()

    # --- Step 5: Append events ---
    for event in all_events:
        append_line(events_file, event.to_jsonl())

    # --- Step 6: Update epoch pointer ---
    if provider_hashes:
        update_epoch_pointer(state_dir, epoch_id)

    # --- Step 7: Fire webhooks (best-effort) ---
    if notifier and notifier.enabled:
        for event in webhook_queue:
            notifier.notify(
                event.event_type, event.provider, event.model_id,
                f"{event.event_type}: {event.provider}/{event.model_id}" +
                (f" -- {event.notes}" if event.notes else ""))

    return total_models, total_events


# =============================================================================
# Summary printer
# =============================================================================

def print_summary(provider: str, diff: Dict[str, List[str]], model_count: int) -> None:
    added = diff["added"]
    removed = diff["removed"]
    changed = diff["changed"]

    status = "no changes"
    if added or removed or changed:
        parts = []
        if added:
            parts.append(f"+{len(added)} added")
        if removed:
            parts.append(f"-{len(removed)} removed")
        if changed:
            parts.append(f"~{len(changed)} changed")
        status = ", ".join(parts)

    print(f"  {provider:15s} | {model_count:4d} models | {status}")

    if added:
        for mid in added[:10]:
            print(f"    + {mid}")
        if len(added) > 10:
            print(f"    ... and {len(added) - 10} more")
    if removed:
        for mid in removed[:10]:
            print(f"    - {mid}")
    if changed:
        for mid in changed[:5]:
            print(f"    ~ {mid}")
        if len(changed) > 5:
            print(f"    ... and {len(changed) - 5} more")


# =============================================================================
# PID lockfile — prevent overlapping instances
# =============================================================================

_lock_fd = None  # must stay open for process lifetime


def acquire_instance_lock(state_dir: Path) -> bool:
    """Acquire exclusive instance lock. Returns True if acquired, exits if another instance running."""
    global _lock_fd
    lock_path = state_dir / ".observatory.lock"
    mkdirp(state_dir)
    _lock_fd = open(lock_path, "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except (BlockingIOError, OSError):
        print("[SKIP] Another observatory instance is running.")
        sys.exit(0)


# =============================================================================
# Provider factory
# =============================================================================

PROVIDER_REGISTRY = {
    "openai":     ("OPENAI_API_KEY",     lambda k: OpenAIWatcher(api_key=k, base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com"))),
    "anthropic":  ("ANTHROPIC_API_KEY",   lambda k: AnthropicWatcher(api_key=k)),
    "google":     ("GOOGLE_API_KEY",      lambda k: GoogleWatcher(api_key=k)),
    "xai":        ("XAI_API_KEY",         lambda k: OpenAICompatWatcher("xai", k, "https://api.x.ai")),
    "mistral":    ("MISTRAL_API_KEY",     lambda k: OpenAICompatWatcher("mistral", k, "https://api.mistral.ai")),
    "deepseek":   ("DEEPSEEK_API_KEY",    lambda k: OpenAICompatWatcher("deepseek", k, "https://api.deepseek.com", models_path="/models", chat_path="/chat/completions")),
    "openrouter": ("OPENROUTER_API_KEY",  lambda k: OpenAICompatWatcher("openrouter", k, "https://openrouter.ai/api")),
    "venice":     ("VENICE_API_KEY",      lambda k: OpenAICompatWatcher("venice", k, "https://api.venice.ai", models_path="/api/v1/models", chat_path="/api/v1/chat/completions")),
    "volink":     ("VOLINK_API_KEY",      lambda k: OpenAICompatWatcher("volink", k, "https://api.volink.org/v1", models_path="/models", chat_path="/chat/completions")),
    "zhipu":      ("ZHIPU_API_KEY",       lambda k: OpenAICompatWatcher("zhipu", k, "https://open.bigmodel.cn/api/paas/v4", models_path="/models", chat_path="/chat/completions")),
    "moonshot":   ("MOONSHOT_API_KEY",    lambda k: OpenAICompatWatcher("moonshot", k, "https://api.moonshot.cn")),
    "ollama":     (None,                  lambda _: OllamaWatcher(base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"))),
}


def build_watchers(requested: Optional[List[str]] = None) -> List[ProviderWatcher]:
    """Build watcher instances for all providers that have keys configured."""
    watchers = []
    targets = requested or list(PROVIDER_REGISTRY.keys())

    for name in targets:
        if name not in PROVIDER_REGISTRY:
            print(f"  [WARN] Unknown provider: {name}, skipping")
            continue

        env_var, factory = PROVIDER_REGISTRY[name]

        if name == "ollama":
            try:
                w = factory(None)
                requests.get(f"{w.base_url}/api/tags", timeout=3)
                watchers.append(w)
            except Exception:
                print(f"  [SKIP] ollama: not reachable at {os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')}")
            continue

        key = os.environ.get(env_var, "").strip()
        if not key:
            if requested:
                print(f"  [SKIP] {name}: {env_var} not set")
            continue

        watchers.append(factory(key))

    return watchers


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-provider model availability watcher")
    parser.add_argument("--providers", type=str, default=None,
                        help="Comma-separated list of providers to poll (default: all configured)")
    parser.add_argument("--list-providers", action="store_true",
                        help="Show which providers have keys configured and exit")
    parser.add_argument("--canary-changed", action="store_true",
                        help="Also run canary calls on models with metadata changes")
    parser.add_argument("--benchmark", action="store_true",
                        help="Run deterministic benchmark hashing on all current models (detects semantic drift)")
    parser.add_argument("--benchmark-models", type=str, default=None,
                        help="Comma-separated model IDs to benchmark (default: only new models, or all with --benchmark)")

    # Experiment invariant commands
    parser.add_argument("--lock-experiment", type=str, metavar="EXP_ID",
                        help="Create an experiment lockfile capturing current model ecosystem state")
    parser.add_argument("--benchmark-before-lock", action="store_true",
                        help="Run fresh benchmarks for all models before creating lock (recommended)")
    parser.add_argument("--check-experiment", type=str, metavar="EXP_ID",
                        help="Check if current state matches an experiment's lockfile")
    parser.add_argument("--list-experiments", action="store_true",
                        help="List all active experiment locks")
    parser.add_argument("--lock-notes", type=str, default=None,
                        help="Optional notes to attach to experiment lock")

    # Capability registry commands
    parser.add_argument("--show-registry", action="store_true",
                        help="Show the capability registry")
    parser.add_argument("--query-registry", type=str, metavar="FILTER",
                        help="Query registry: 'context>200000', 'reliability>0.9', 'cost<=mid', 'provider=openrouter'")

    args = parser.parse_args()

    state_dir = Path(os.environ.get("WATCHER_STATE_DIR", "./state")).resolve()
    events_file = Path(os.environ.get("WATCHER_EVENTS_FILE", str(state_dir / "events.jsonl"))).resolve()
    registry_path = state_dir / "capability_registry.json"
    locks_dir = state_dir / "experiment_locks"

    # --- Experiment invariant commands (no lock needed) ---
    if args.list_experiments:
        mgr = ExperimentInvariantManager(locks_dir)
        locks = mgr.list_locks()
        if locks:
            print(f"Active experiment locks ({len(locks)}):")
            for eid in locks:
                lock_data = read_json(locks_dir / f"{eid}.lock.json")
                created = lock_data.get("created_at", "?") if lock_data else "?"
                notes = lock_data.get("notes", "") if lock_data else ""
                print(f"  {eid:30s} | locked {created}" + (f" | {notes}" if notes else ""))
        else:
            print("No active experiment locks.")
        return

    if args.lock_experiment:
        mgr = ExperimentInvariantManager(locks_dir)
        bench_path = state_dir / "benchmark_hashes.json"
        bench = BenchmarkStore(bench_path) if bench_path.exists() else None

        # Build watchers if --benchmark-before-lock is set
        lock_watchers = None
        if args.benchmark_before_lock:
            requested = [p.strip() for p in args.providers.split(",")] if args.providers else None
            lock_watchers = build_watchers(requested)
            if not lock_watchers:
                print("[ERROR] --benchmark-before-lock requires at least one configured provider")
                sys.exit(1)
            print(f"Running fresh benchmarks for {len(lock_watchers)} provider(s)...")

        lock = mgr.create_lock(args.lock_experiment, state_dir,
                               benchmark_store=bench, notes=args.lock_notes,
                               benchmark_before_lock=args.benchmark_before_lock,
                               watchers=lock_watchers)
        print(f"Experiment locked: {lock.experiment_id}")
        print(f"  Created: {lock.created_at}")
        print(f"  Epoch: {lock.epoch_id or '(no epoch)'}")
        print(f"  Registry hash: {lock.provider_registry_hash[:16]}...")
        print(f"  Model IDs hash: {lock.model_ids_hash[:16]}...")
        print(f"  Behavioral: {lock.behavioral_status}")
        if lock.benchmark_coverage:
            print(f"  Benchmark coverage: {lock.benchmark_coverage} (model x prompt baselines)")
        if lock.benchmark_staleness_warning:
            print(f"  WARNING: No benchmark baselines found. Use --benchmark-before-lock for behavioral lock.")
        if lock.notes:
            print(f"  Notes: {lock.notes}")
        return

    if args.check_experiment:
        mgr = ExperimentInvariantManager(locks_dir)
        bench_path = state_dir / "benchmark_hashes.json"
        bench = BenchmarkStore(bench_path) if bench_path.exists() else None
        report = mgr.check_invariants(args.check_experiment, state_dir, benchmark_store=bench)
        status = report["status"]
        if status == "CLEAN":
            print(f"CLEAN: Experiment '{args.check_experiment}' invariants hold.")
            print(f"  Locked: {report['locked_at']}")
            if report.get("behavioral_status") == "UNLOCKED_BEHAVIORAL":
                print(f"  Behavioral: UNLOCKED (v1 lockfile — re-lock with --benchmark-before-lock)")
        elif status == "DRIFT_DETECTED":
            print(f"DRIFT DETECTED: {report['message']}")
            for d in report.get("drifts", []):
                print(f"  - {d}")
        else:
            print(f"  {report.get('message', 'Unknown status')}")
        # Show variants (not drift — informational)
        for v in report.get("variants", []):
            print(f"  [VARIANT] {v}")
        # Show warnings (e.g., UNLOCKED_BEHAVIORAL)
        for w in report.get("warnings", []):
            print(f"  [WARN] {w}")
        return

    # --- Capability registry commands (no lock needed) ---
    if args.show_registry:
        cap_reg = CapabilityRegistry(registry_path)
        if not cap_reg.entries:
            print("Capability registry is empty. Run a poll first.")
            return
        print(f"Capability Registry ({len(cap_reg.entries)} models):\n")
        print(f"  {'Provider/Model':<50s} {'Context':>10s} {'Rel':>6s} {'P50ms':>7s} {'Cost':>8s}")
        print(f"  {'-'*50} {'-'*10} {'-'*6} {'-'*7} {'-'*8}")
        for key in sorted(cap_reg.entries.keys()):
            c = cap_reg.entries[key]
            ctx = str(c.max_context_tokens) if c.max_context_tokens else "-"
            rel = f"{c.reliability:.2f}" if c.reliability is not None else "-"
            p50 = str(c.latency_p50_ms) if c.latency_p50_ms else "-"
            cost = c.cost_class or "-"
            print(f"  {key:<50s} {ctx:>10s} {rel:>6s} {p50:>7s} {cost:>8s}")
        return

    if args.query_registry:
        cap_reg = CapabilityRegistry(registry_path)
        kwargs: Dict[str, Any] = {}
        for part in args.query_registry.split(","):
            part = part.strip()
            if part.startswith("context>"):
                kwargs["min_context"] = int(part.split(">")[1])
            elif part.startswith("reliability>"):
                kwargs["min_reliability"] = float(part.split(">")[1])
            elif part.startswith("cost<="):
                kwargs["max_cost_class"] = part.split("<=")[1]
            elif part.startswith("provider="):
                kwargs["provider"] = part.split("=")[1]
        results = cap_reg.query(**kwargs)
        if results:
            print(f"Query results ({len(results)} matches):")
            for c in results:
                ctx = str(c.max_context_tokens) if c.max_context_tokens else "?"
                print(f"  {c.provider}/{c.model_id} -- context:{ctx} rel:{c.reliability} cost:{c.cost_class}")
        else:
            print("No models match the query.")
        return

    # --- Provider listing (no lock needed) ---
    if args.list_providers:
        print("Provider configuration status:")
        for name, (env_var, _) in PROVIDER_REGISTRY.items():
            if name == "ollama":
                url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
                try:
                    requests.get(f"{url}/api/tags", timeout=3)
                    print(f"  {name:15s} | REACHABLE at {url}")
                except Exception:
                    print(f"  {name:15s} | NOT REACHABLE at {url}")
            else:
                key = os.environ.get(env_var, "").strip()
                prov_status = "CONFIGURED" if key else "NOT SET"
                masked = f"{key[:8]}...{key[-4:]}" if len(key) > 12 else ("***" if key else "--")
                print(f"  {name:15s} | {prov_status:14s} | {env_var}={masked}")
        return

    # =========================================================================
    # Main poll loop — single-writer pattern with epoch management
    # =========================================================================

    # Acquire PID lockfile (prevents overlapping cron instances)
    acquire_instance_lock(state_dir)

    requested = [p.strip() for p in args.providers.split(",")] if args.providers else None
    watchers = build_watchers(requested)

    if not watchers:
        print("[ERROR] No providers configured. Set API key env vars or check --list-providers")
        sys.exit(1)

    mkdirp(state_dir)
    mkdirp(events_file.parent)

    # v1 -> v2 migration (one-time, idempotent)
    migrate_v1_to_v2(state_dir, events_file)

    cap_reg = CapabilityRegistry(registry_path)
    bench_store_path = state_dir / "benchmark_hashes.json"
    bench_store = BenchmarkStore(bench_store_path) if (args.benchmark or args.benchmark_models) else None
    webhook_config_path = state_dir / "webhooks.json"
    notifier = WebhookNotifier(webhook_config_path)

    # Parse benchmark model targets
    do_benchmark = bool(args.benchmark or args.benchmark_models)
    bench_models_explicit = None
    if args.benchmark_models:
        bench_models_explicit = [m.strip() for m in args.benchmark_models.split(",")]

    # Generate epoch ID
    epoch_id = generate_epoch_id()

    ts = utc_now_iso()
    print(f"\n[{ts}] Model Observatory -- polling {len(watchers)} provider(s)")
    print(f"  State: {state_dir}")
    print(f"  Events: {events_file}")
    print(f"  Epoch: {epoch_id}")
    if notifier.enabled:
        print(f"  Webhooks: {len(notifier.hooks)} configured")
    if bench_store:
        print(f"  Benchmarks: {bench_store_path}")
    print()

    # --- Phase A: Load previous snapshots (read-only for workers) ---
    prev_snapshots: Dict[str, Dict[str, str]] = {}
    for w in watchers:
        snap_path = state_dir / f"{w.provider_name}_snapshot_latest.json"
        snap = read_json(snap_path)
        prev_snapshots[w.provider_name] = (snap.get("models") if isinstance(snap, dict) else {}) or {}

    # --- Phase B: Parallel poll — workers return PollResults ---
    poll_results: List[PollResult] = []

    def _poll_worker(watcher: ProviderWatcher) -> PollResult:
        """Worker function — pure, returns PollResult."""
        prev_models = prev_snapshots.get(watcher.provider_name, {})

        # Determine benchmark targets for this provider
        watcher_bench_models = bench_models_explicit
        if args.benchmark and not args.benchmark_models:
            # Benchmark ALL current models for this provider
            watcher_bench_models = []  # empty = all (poll_provider handles this)

        return poll_provider(
            watcher=watcher,
            prev_models=prev_models,
            do_canary_on_added=True,
            do_canary_on_changed=args.canary_changed,
            do_benchmark=do_benchmark,
            benchmark_models=watcher_bench_models,
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, len(watchers))) as executor:
        future_to_provider = {executor.submit(_poll_worker, w): w.provider_name for w in watchers}
        for future in concurrent.futures.as_completed(future_to_provider):
            provider_name = future_to_provider[future]
            try:
                result = future.result()
                poll_results.append(result)
            except Exception as e:
                # Create error PollResult — don't lose the epoch
                poll_results.append(PollResult(
                    provider=provider_name,
                    snapshot=None,
                    raw_hash=None,
                    model_fingerprints={},
                    diff=None,
                    events=[],
                    canary_results=[],
                    benchmark_results=[],
                    metadata_updates=[],
                    error=f"{type(e).__name__}: {str(e)[:300]}",
                ))

    # --- Phase C: Main thread merge (single-writer, epoch commit order) ---
    total_models, total_events = merge_poll_results(
        results=poll_results,
        epoch_id=epoch_id,
        state_dir=state_dir,
        events_file=events_file,
        cap_reg=cap_reg,
        bench_store=bench_store,
        notifier=notifier,
    )

    # --- Phase D: Cleanup old epochs ---
    cleanup_old_epochs(state_dir, keep=EPOCH_RETENTION)

    # --- Phase E: Check experiment invariants ---
    inv_mgr = ExperimentInvariantManager(locks_dir)
    active_locks = inv_mgr.list_locks()
    if active_locks:
        # Load benchmark store for invariant checks even if benchmarks weren't run this poll
        inv_bench = bench_store
        if inv_bench is None:
            inv_bench_path = state_dir / "benchmark_hashes.json"
            if inv_bench_path.exists():
                inv_bench = BenchmarkStore(inv_bench_path)
        print(f"\n  Experiment invariant checks ({len(active_locks)}):")
        for eid in active_locks:
            report = inv_mgr.check_invariants(eid, state_dir, benchmark_store=inv_bench)
            status_icon = "OK" if report["status"] == "CLEAN" else "DRIFT"
            extra = ""
            if report["status"] == "DRIFT_DETECTED":
                extra = f" -- {', '.join(report.get('drifts', []))}"
            elif report.get("behavioral_status") == "UNLOCKED_BEHAVIORAL":
                extra = " -- behavioral: UNLOCKED (v1 lock)"
            print(f"    [{status_icon}] {eid}{extra}")
            if report["status"] == "DRIFT_DETECTED" and notifier.enabled:
                notifier.notify_drift(eid, report.get("drifts", []))

    print(f"\n  Total: {total_models} models across {len(watchers)} providers, {total_events} change event(s)")
    print(f"  Registry: {len(cap_reg.entries)} models tracked")
    print(f"  Epoch: {epoch_id}")
    print(f"[{utc_now_iso()}] Done.\n")


if __name__ == "__main__":
    # Load .env from script directory if available
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
    else:
        load_dotenv()  # try default locations
    main()
```

---

# SECTION 4: TUI DASHBOARD (observatory_tui.py)

```python
#!/usr/bin/env python3
"""
Model Observatory — Terminal UI Dashboard
==========================================
Reads state from model_observatory.py's state directory and presents
a live overview of provider topology, drift events, capability
registry, and experiment locks.

Usage:
  python observatory_tui.py                    # default state dir
  python observatory_tui.py --state-dir ./state
  python observatory_tui.py --watch            # auto-refresh every 30s

Keys:
  q/ESC     Quit
  r         Refresh
  1-5       Switch panels
  j/k       Scroll down/up
  Tab       Next panel
"""

from __future__ import annotations

import argparse
import curses
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# State readers (read what model_observatory.py writes)
# =============================================================================

def read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def load_snapshots(state_dir: Path) -> Dict[str, Dict]:
    """Load all provider snapshots."""
    snaps = {}
    for f in sorted(state_dir.glob("*_snapshot_latest.json")):
        data = read_json(f)
        if data:
            provider = data.get("provider", f.stem.replace("_snapshot_latest", ""))
            snaps[provider] = data
    return snaps


def load_events(state_dir: Path, limit: int = 100) -> List[Dict]:
    """Load most recent events from JSONL."""
    events_file = state_dir / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().split("\n")
    events = []
    for line in lines[-limit:]:
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return list(reversed(events))


def load_registry(state_dir: Path) -> Dict[str, Dict]:
    """Load capability registry."""
    data = read_json(state_dir / "capability_registry.json")
    return data if isinstance(data, dict) else {}


def load_benchmarks(state_dir: Path) -> Dict[str, Dict]:
    """Load benchmark hashes."""
    data = read_json(state_dir / "benchmark_hashes.json")
    return data if isinstance(data, dict) else {}


def load_experiment_locks(state_dir: Path) -> List[Dict]:
    """Load all experiment locks."""
    locks_dir = state_dir / "experiment_locks"
    if not locks_dir.exists():
        return []
    locks = []
    for f in sorted(locks_dir.glob("*.lock.json")):
        data = read_json(f)
        if data:
            locks.append(data)
    return locks


# =============================================================================
# Color pairs
# =============================================================================

COLOR_TITLE = 1
COLOR_OK = 2
COLOR_WARN = 3
COLOR_CRIT = 4
COLOR_DIM = 5
COLOR_HIGHLIGHT = 6
COLOR_HEADER = 7
COLOR_ACCENT = 8


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(COLOR_TITLE, curses.COLOR_CYAN, -1)
    curses.init_pair(COLOR_OK, curses.COLOR_GREEN, -1)
    curses.init_pair(COLOR_WARN, curses.COLOR_YELLOW, -1)
    curses.init_pair(COLOR_CRIT, curses.COLOR_RED, -1)
    curses.init_pair(COLOR_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(COLOR_HIGHLIGHT, curses.COLOR_BLACK, curses.COLOR_CYAN)
    curses.init_pair(COLOR_HEADER, curses.COLOR_BLACK, curses.COLOR_WHITE)
    curses.init_pair(COLOR_ACCENT, curses.COLOR_MAGENTA, -1)


# =============================================================================
# Drawing helpers
# =============================================================================

def safe_addstr(win, y: int, x: int, text: str, attr=0):
    """Write string, clipping to window bounds."""
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    max_len = w - x - 1
    if max_len <= 0:
        return
    try:
        win.addnstr(y, x, text, max_len, attr)
    except curses.error:
        pass


def draw_box(win, y: int, x: int, h: int, w: int, title: str = "", attr=0):
    """Draw a bordered box with optional title."""
    max_h, max_w = win.getmaxyx()
    if y >= max_h or x >= max_w:
        return
    # Clamp
    h = min(h, max_h - y)
    w = min(w, max_w - x)
    if h < 2 or w < 2:
        return

    # Top border
    safe_addstr(win, y, x, "+" + "-" * (w - 2) + "+", attr)
    # Sides
    for row in range(y + 1, y + h - 1):
        safe_addstr(win, row, x, "|", attr)
        safe_addstr(win, row, x + w - 1, "|", attr)
    # Bottom
    safe_addstr(win, y + h - 1, x, "+" + "-" * (w - 2) + "+", attr)
    # Title
    if title:
        safe_addstr(win, y, x + 2, f" {title} ", curses.color_pair(COLOR_TITLE) | curses.A_BOLD)


# =============================================================================
# Panel renderers
# =============================================================================

def render_providers(win, y: int, snapshots: Dict, scroll: int = 0) -> int:
    """Render provider topology overview. Returns lines used."""
    h, w = win.getmaxyx()
    row = y

    safe_addstr(win, row, 2, f"{'Provider':<15s} {'Models':>7s} {'Last Poll':>22s} {'Hash':>18s}",
                curses.color_pair(COLOR_HEADER))
    row += 1

    providers = sorted(snapshots.keys())
    visible = providers[scroll:]

    for provider in visible:
        if row >= h - 2:
            break
        snap = snapshots[provider]
        model_count = len(snap.get("models", {}))
        ts = snap.get("timestamp", "?")[:19]
        raw_hash = snap.get("raw_hash", "?")[:16]

        color = COLOR_OK if model_count > 0 else COLOR_WARN
        safe_addstr(win, row, 2, f"{provider:<15s}", curses.color_pair(color) | curses.A_BOLD)
        safe_addstr(win, row, 18, f"{model_count:>7d}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 26, f"{ts:>22s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 49, f"{raw_hash:>18s}", curses.color_pair(COLOR_DIM))
        row += 1

    total = sum(len(s.get("models", {})) for s in snapshots.values())
    row += 1
    safe_addstr(win, row, 2, f"Total: {total} models across {len(snapshots)} providers",
                curses.color_pair(COLOR_ACCENT))
    return row - y + 1


def render_events(win, y: int, events: List[Dict], scroll: int = 0) -> int:
    """Render recent event log."""
    h, w = win.getmaxyx()
    row = y

    safe_addstr(win, row, 2, f"{'Time':>19s}  {'Type':<18s} {'Provider':<12s} {'Model':<30s}",
                curses.color_pair(COLOR_HEADER))
    row += 1

    visible = events[scroll:]
    for evt in visible:
        if row >= h - 2:
            break
        ts = evt.get("timestamp", "?")[:19]
        etype = evt.get("event_type", "?")
        provider = evt.get("provider", "?")
        model_id = evt.get("model_id", "?")

        # Color by severity
        if etype == "BEHAVIOR_DRIFT":
            color = COLOR_CRIT
        elif etype in ("MODEL_REMOVED", "POLL_ERROR"):
            color = COLOR_WARN
        elif etype == "MODEL_ADDED":
            color = COLOR_OK
        else:
            color = COLOR_DIM

        safe_addstr(win, row, 2, f"{ts:>19s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 23, f"{etype:<18s}", curses.color_pair(color) | curses.A_BOLD)
        safe_addstr(win, row, 42, f"{provider:<12s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 55, model_id[:w - 57] if len(model_id) > w - 57 else model_id,
                    curses.color_pair(COLOR_DIM))
        row += 1

    return row - y + 1


def render_registry(win, y: int, registry: Dict, scroll: int = 0) -> int:
    """Render capability registry."""
    h, w = win.getmaxyx()
    row = y

    safe_addstr(win, row, 2,
                f"{'Provider/Model':<45s} {'Context':>10s} {'Rel':>6s} {'P50':>7s} {'P95':>7s} {'Cost':>8s}",
                curses.color_pair(COLOR_HEADER))
    row += 1

    keys = sorted(registry.keys())
    visible = keys[scroll:]

    for key in visible:
        if row >= h - 2:
            break
        c = registry[key]
        ctx = str(c.get("max_context_tokens", "-")) if c.get("max_context_tokens") else "-"
        rel = f"{c['reliability']:.2f}" if c.get("reliability") is not None else "-"
        p50 = str(c.get("latency_p50_ms", "-")) if c.get("latency_p50_ms") else "-"
        p95 = str(c.get("latency_p95_ms", "-")) if c.get("latency_p95_ms") else "-"
        cost = c.get("cost_class", "-") or "-"

        # Color reliability
        rel_val = c.get("reliability")
        if rel_val is not None:
            if rel_val >= 0.95:
                rel_color = COLOR_OK
            elif rel_val >= 0.8:
                rel_color = COLOR_WARN
            else:
                rel_color = COLOR_CRIT
        else:
            rel_color = COLOR_DIM

        display_key = key[:43] if len(key) > 43 else key
        safe_addstr(win, row, 2, f"{display_key:<45s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 48, f"{ctx:>10s}", curses.color_pair(COLOR_ACCENT))
        safe_addstr(win, row, 59, f"{rel:>6s}", curses.color_pair(rel_color))
        safe_addstr(win, row, 66, f"{p50:>7s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 74, f"{p95:>7s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 82, f"{cost:>8s}", curses.color_pair(COLOR_DIM))
        row += 1

    row += 1
    safe_addstr(win, row, 2, f"{len(registry)} models in registry",
                curses.color_pair(COLOR_ACCENT))
    return row - y + 1


def render_benchmarks(win, y: int, benchmarks: Dict, scroll: int = 0) -> int:
    """Render benchmark hash status."""
    h, w = win.getmaxyx()
    row = y

    safe_addstr(win, row, 2, f"{'Provider/Model':<45s} {'Prompts Hashed':>15s}",
                curses.color_pair(COLOR_HEADER))
    row += 1

    keys = sorted(benchmarks.keys())
    visible = keys[scroll:]

    for key in visible:
        if row >= h - 2:
            break
        hashes = benchmarks[key]
        count = len(hashes) if isinstance(hashes, dict) else 0

        display_key = key[:43] if len(key) > 43 else key
        safe_addstr(win, row, 2, f"{display_key:<45s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 48, f"{count:>15d}", curses.color_pair(COLOR_OK if count > 0 else COLOR_WARN))
        row += 1

    row += 1
    safe_addstr(win, row, 2, f"{len(benchmarks)} models baselined",
                curses.color_pair(COLOR_ACCENT))
    return row - y + 1


def render_experiments(win, y: int, locks: List[Dict], scroll: int = 0) -> int:
    """Render experiment locks."""
    h, w = win.getmaxyx()
    row = y

    if not locks:
        safe_addstr(win, row, 2, "No active experiment locks.", curses.color_pair(COLOR_DIM))
        return 1

    safe_addstr(win, row, 2, f"{'Experiment':<30s} {'Locked At':>22s} {'Notes':<30s}",
                curses.color_pair(COLOR_HEADER))
    row += 1

    visible = locks[scroll:]
    for lock in visible:
        if row >= h - 2:
            break
        eid = lock.get("experiment_id", "?")
        created = lock.get("created_at", "?")[:19]
        notes = lock.get("notes", "") or ""

        safe_addstr(win, row, 2, f"{eid:<30s}", curses.color_pair(COLOR_ACCENT) | curses.A_BOLD)
        safe_addstr(win, row, 33, f"{created:>22s}", curses.color_pair(COLOR_DIM))
        safe_addstr(win, row, 56, notes[:w - 58] if len(notes) > w - 58 else notes,
                    curses.color_pair(COLOR_DIM))
        row += 1

    return row - y + 1


# =============================================================================
# Main TUI loop
# =============================================================================

PANELS = ["Providers", "Events", "Registry", "Benchmarks", "Experiments"]


def main_tui(stdscr, state_dir: Path, auto_refresh: bool = False):
    init_colors()
    curses.curs_set(0)
    stdscr.timeout(500 if auto_refresh else -1)

    current_panel = 0
    scroll = 0
    last_refresh = 0
    refresh_interval = 30  # seconds

    while True:
        now = time.time()
        if auto_refresh and (now - last_refresh) >= refresh_interval:
            last_refresh = now

        # Load state
        snapshots = load_snapshots(state_dir)
        events = load_events(state_dir, limit=200)
        registry = load_registry(state_dir)
        benchmarks = load_benchmarks(state_dir)
        locks = load_experiment_locks(state_dir)

        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # Title bar
        title = " MODEL OBSERVATORY "
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        safe_addstr(stdscr, 0, 0, " " * w, curses.color_pair(COLOR_HEADER))
        safe_addstr(stdscr, 0, (w - len(title)) // 2, title,
                    curses.color_pair(COLOR_HEADER) | curses.A_BOLD)
        safe_addstr(stdscr, 0, w - len(ts) - 2, ts, curses.color_pair(COLOR_HEADER))

        # Panel tabs
        tab_x = 1
        for i, name in enumerate(PANELS):
            label = f" {i + 1}:{name} "
            if i == current_panel:
                safe_addstr(stdscr, 1, tab_x, label,
                            curses.color_pair(COLOR_HIGHLIGHT) | curses.A_BOLD)
            else:
                safe_addstr(stdscr, 1, tab_x, label, curses.color_pair(COLOR_DIM))
            tab_x += len(label) + 1

        # Status indicators on tab bar
        drift_events = [e for e in events[:20] if e.get("event_type") == "BEHAVIOR_DRIFT"]
        poll_errors = [e for e in events[:20] if e.get("event_type") == "POLL_ERROR"]
        status_parts = []
        if drift_events:
            status_parts.append(f"DRIFT:{len(drift_events)}")
        if poll_errors:
            status_parts.append(f"ERR:{len(poll_errors)}")
        if status_parts:
            status_str = " | ".join(status_parts)
            safe_addstr(stdscr, 1, w - len(status_str) - 2, status_str,
                        curses.color_pair(COLOR_CRIT) | curses.A_BOLD)

        # Panel content
        content_y = 3
        if current_panel == 0:
            render_providers(stdscr, content_y, snapshots, scroll)
        elif current_panel == 1:
            render_events(stdscr, content_y, events, scroll)
        elif current_panel == 2:
            render_registry(stdscr, content_y, registry, scroll)
        elif current_panel == 3:
            render_benchmarks(stdscr, content_y, benchmarks, scroll)
        elif current_panel == 4:
            render_experiments(stdscr, content_y, locks, scroll)

        # Footer
        footer = " q:Quit  r:Refresh  1-5:Panel  j/k:Scroll  Tab:Next "
        if auto_refresh:
            footer += f" [auto-refresh {refresh_interval}s]"
        safe_addstr(stdscr, h - 1, 0, " " * w, curses.color_pair(COLOR_HEADER))
        safe_addstr(stdscr, h - 1, (w - len(footer)) // 2, footer, curses.color_pair(COLOR_HEADER))

        stdscr.refresh()

        # Input
        try:
            key = stdscr.getch()
        except curses.error:
            continue

        if key in (ord('q'), ord('Q'), 27):  # q or ESC
            break
        elif key == ord('r'):
            last_refresh = 0  # force refresh
        elif key == ord('\t'):
            current_panel = (current_panel + 1) % len(PANELS)
            scroll = 0
        elif key in (ord('j'), curses.KEY_DOWN):
            scroll += 1
        elif key in (ord('k'), curses.KEY_UP):
            scroll = max(0, scroll - 1)
        elif key in (ord('1'), ord('2'), ord('3'), ord('4'), ord('5')):
            current_panel = key - ord('1')
            scroll = 0


def main():
    parser = argparse.ArgumentParser(description="Model Observatory -- Terminal UI Dashboard")
    parser.add_argument("--state-dir", type=str, default="./state",
                        help="Path to model_observatory state directory")
    parser.add_argument("--watch", action="store_true",
                        help="Auto-refresh every 30 seconds")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).resolve()
    if not state_dir.exists():
        print(f"[ERROR] State directory not found: {state_dir}")
        print("  Run model_observatory.py first to populate state.")
        sys.exit(1)

    curses.wrapper(main_tui, state_dir, args.watch)


if __name__ == "__main__":
    main()
```

---

# SECTION 5: ROUTING ORACLE (routing_oracle.py)

```python
#!/usr/bin/env python3
"""
Routing Oracle — Policy-driven model selection from capability registry
=======================================================================
Importable module for any downstream consumer (application layer) to
query the observatory's capability registry and make routing decisions.

Usage (as library):
    from routing_oracle import RoutingOracle

    oracle = RoutingOracle("/path/to/model-observatory/state")

    # Find best model for a task
    model = oracle.route(
        min_context=200000,
        min_reliability=0.9,
        max_cost="mid",
        prefer="latency",   # or "reliability", "context", "cost"
    )

    # Get all candidates
    candidates = oracle.candidates(min_context=100000)

    # Check if a specific model is healthy
    healthy = oracle.is_healthy("openrouter/gpt-5.2")

    # Get full status for routing decisions
    status = oracle.model_status("openrouter/gpt-5.2")

Usage (CLI):
    python routing_oracle.py --route "context>200000,reliability>0.9,cost<=mid"
    python routing_oracle.py --status openrouter/gpt-5.2
    python routing_oracle.py --healthy openrouter/gpt-5.2
    python routing_oracle.py --list-by reliability
    python routing_oracle.py --policy default    # show routing recommendations
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(path: Path) -> Optional[Any]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


COST_ORDER = {"free": 0, "low": 1, "mid": 2, "high": 3, "premium": 4}


@dataclass
class ModelProfile:
    """Full model profile combining registry + benchmark + snapshot data."""
    key: str  # "provider/model_id"
    provider: str
    model_id: str
    max_context_tokens: Optional[int] = None
    supports_tools: Optional[bool] = None
    supports_streaming: Optional[bool] = None
    latency_p50_ms: Optional[int] = None
    latency_p95_ms: Optional[int] = None
    reliability: Optional[float] = None
    cost_class: Optional[str] = None
    benchmark_count: int = 0
    has_drift: bool = False
    last_updated: Optional[str] = None

    @property
    def cost_rank(self) -> int:
        return COST_ORDER.get(self.cost_class, 999) if self.cost_class else 999

    def matches(self, min_context: Optional[int] = None,
                min_reliability: Optional[float] = None,
                max_cost: Optional[str] = None,
                provider: Optional[str] = None,
                exclude_drift: bool = True) -> bool:
        """Check if model matches routing criteria."""
        if exclude_drift and self.has_drift:
            return False
        if provider and self.provider != provider:
            return False
        if min_context and (self.max_context_tokens is None or self.max_context_tokens < min_context):
            return False
        if min_reliability and (self.reliability is None or self.reliability < min_reliability):
            return False
        if max_cost:
            max_rank = COST_ORDER.get(max_cost, 999)
            if self.cost_rank > max_rank:
                return False
        return True


class RoutingOracle:
    """Policy-driven model routing from observatory state."""

    def __init__(self, state_dir: str | Path):
        self.state_dir = Path(state_dir)
        self._profiles: Dict[str, ModelProfile] = {}
        self._load()

    def _load(self) -> None:
        """Load registry, benchmarks, and recent events."""
        registry = _read_json(self.state_dir / "capability_registry.json") or {}
        benchmarks = _read_json(self.state_dir / "benchmark_hashes.json") or {}

        # Check recent events for drift
        drift_models = set()
        events_file = self.state_dir / "events.jsonl"
        if events_file.exists():
            lines = events_file.read_text(encoding="utf-8").strip().split("\n")
            for line in lines[-500:]:
                try:
                    evt = json.loads(line)
                    if evt.get("event_type") == "BEHAVIOR_DRIFT":
                        key = f"{evt.get('provider', '')}/{evt.get('model_id', '')}"
                        drift_models.add(key)
                except json.JSONDecodeError:
                    continue

        # Build profiles
        for key, data in registry.items():
            parts = key.split("/", 1)
            provider = parts[0] if len(parts) > 1 else "unknown"
            model_id = parts[1] if len(parts) > 1 else key

            bench_data = benchmarks.get(key, {})
            bench_count = len(bench_data) if isinstance(bench_data, dict) else 0

            self._profiles[key] = ModelProfile(
                key=key,
                provider=provider,
                model_id=model_id,
                max_context_tokens=data.get("max_context_tokens"),
                supports_tools=data.get("supports_tools"),
                supports_streaming=data.get("supports_streaming"),
                latency_p50_ms=data.get("latency_p50_ms"),
                latency_p95_ms=data.get("latency_p95_ms"),
                reliability=data.get("reliability"),
                cost_class=data.get("cost_class"),
                benchmark_count=bench_count,
                has_drift=key in drift_models,
                last_updated=data.get("last_updated"),
            )

    def refresh(self) -> None:
        """Reload state from disk."""
        self._profiles.clear()
        self._load()

    @property
    def profiles(self) -> Dict[str, ModelProfile]:
        return self._profiles

    def candidates(self, min_context: Optional[int] = None,
                   min_reliability: Optional[float] = None,
                   max_cost: Optional[str] = None,
                   provider: Optional[str] = None,
                   exclude_drift: bool = True) -> List[ModelProfile]:
        """Get all models matching criteria."""
        return [p for p in self._profiles.values()
                if p.matches(min_context, min_reliability, max_cost, provider, exclude_drift)]

    def route(self, min_context: Optional[int] = None,
              min_reliability: Optional[float] = None,
              max_cost: Optional[str] = None,
              provider: Optional[str] = None,
              prefer: str = "reliability",
              exclude_drift: bool = True) -> Optional[ModelProfile]:
        """Route to best model matching criteria.

        prefer: "reliability" | "latency" | "context" | "cost"
        """
        matches = self.candidates(min_context, min_reliability, max_cost, provider, exclude_drift)
        if not matches:
            return None

        if prefer == "reliability":
            matches.sort(key=lambda p: -(p.reliability or 0))
        elif prefer == "latency":
            matches.sort(key=lambda p: p.latency_p50_ms or 999999)
        elif prefer == "context":
            matches.sort(key=lambda p: -(p.max_context_tokens or 0))
        elif prefer == "cost":
            matches.sort(key=lambda p: p.cost_rank)

        return matches[0]

    def is_healthy(self, key: str) -> bool:
        """Check if a model is healthy (exists, reliable, no drift)."""
        profile = self._profiles.get(key)
        if not profile:
            return False
        if profile.has_drift:
            return False
        if profile.reliability is not None and profile.reliability < 0.5:
            return False
        return True

    def model_status(self, key: str) -> Dict[str, Any]:
        """Full status dict for a model."""
        profile = self._profiles.get(key)
        if not profile:
            return {"found": False, "key": key}
        return {
            "found": True,
            "key": key,
            "provider": profile.provider,
            "model_id": profile.model_id,
            "context": profile.max_context_tokens,
            "reliability": profile.reliability,
            "latency_p50": profile.latency_p50_ms,
            "latency_p95": profile.latency_p95_ms,
            "cost_class": profile.cost_class,
            "benchmarks": profile.benchmark_count,
            "has_drift": profile.has_drift,
            "healthy": self.is_healthy(key),
            "last_updated": profile.last_updated,
        }

    def default_routing_policy(self) -> Dict[str, Optional[str]]:
        """Generate default routing recommendations.

        Returns role -> best model key mapping based on common requirements.
        """
        policy = {}

        # Long-context synthesis (>200k, reliable)
        synth = self.route(min_context=200000, min_reliability=0.8, prefer="context")
        policy["long_context_synthesis"] = synth.key if synth else None

        # Math/precision (reliable, low latency)
        math = self.route(min_reliability=0.9, prefer="reliability")
        policy["precision_math"] = math.key if math else None

        # Adversarial/destruction testing (any cost, need reliability)
        adversarial = self.route(min_reliability=0.7, prefer="reliability")
        policy["adversarial_review"] = adversarial.key if adversarial else None

        # Budget-friendly routing (cost-first)
        budget = self.route(max_cost="low", prefer="cost")
        policy["budget_routing"] = budget.key if budget else None

        # Fast interactive (latency-first)
        fast = self.route(prefer="latency", min_reliability=0.7)
        policy["fast_interactive"] = fast.key if fast else None

        # Maximum context (context-first, any cost)
        max_ctx = self.route(prefer="context")
        policy["maximum_context"] = max_ctx.key if max_ctx else None

        return policy


# =============================================================================
# CLI interface
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Routing Oracle -- query model capability registry")
    parser.add_argument("--state-dir", type=str, default="./state",
                        help="Path to model_observatory state directory")
    parser.add_argument("--route", type=str, metavar="FILTER",
                        help="Route to best model: 'context>200000,reliability>0.9,cost<=mid,prefer=latency'")
    parser.add_argument("--status", type=str, metavar="KEY",
                        help="Full status for a model (e.g. 'openrouter/gpt-5.2')")
    parser.add_argument("--healthy", type=str, metavar="KEY",
                        help="Check if model is healthy")
    parser.add_argument("--list-by", type=str, choices=["reliability", "latency", "context", "cost"],
                        help="List all models sorted by attribute")
    parser.add_argument("--policy", type=str, choices=["default"],
                        help="Show routing policy recommendations")
    args = parser.parse_args()

    state_dir = Path(args.state_dir).resolve()
    oracle = RoutingOracle(state_dir)

    if args.route:
        kwargs: Dict[str, Any] = {}
        for part in args.route.split(","):
            part = part.strip()
            if part.startswith("context>"):
                kwargs["min_context"] = int(part.split(">")[1])
            elif part.startswith("reliability>"):
                kwargs["min_reliability"] = float(part.split(">")[1])
            elif part.startswith("cost<="):
                kwargs["max_cost"] = part.split("<=")[1]
            elif part.startswith("provider="):
                kwargs["provider"] = part.split("=")[1]
            elif part.startswith("prefer="):
                kwargs["prefer"] = part.split("=")[1]

        result = oracle.route(**kwargs)
        if result:
            print(f"Routed: {result.key}")
            print(f"  Context: {result.max_context_tokens}")
            print(f"  Reliability: {result.reliability}")
            print(f"  Latency P50: {result.latency_p50_ms}ms")
            print(f"  Cost: {result.cost_class}")
            print(f"  Drift: {'YES' if result.has_drift else 'clean'}")
        else:
            print("No model matches the criteria.")
        return

    if args.status:
        status = oracle.model_status(args.status)
        print(json.dumps(status, indent=2))
        return

    if args.healthy:
        healthy = oracle.is_healthy(args.healthy)
        print(f"{args.healthy}: {'HEALTHY' if healthy else 'UNHEALTHY'}")
        sys.exit(0 if healthy else 1)

    if args.list_by:
        profiles = list(oracle.profiles.values())
        if args.list_by == "reliability":
            profiles.sort(key=lambda p: -(p.reliability or 0))
        elif args.list_by == "latency":
            profiles.sort(key=lambda p: p.latency_p50_ms or 999999)
        elif args.list_by == "context":
            profiles.sort(key=lambda p: -(p.max_context_tokens or 0))
        elif args.list_by == "cost":
            profiles.sort(key=lambda p: p.cost_rank)

        print(f"{'Model':<50s} {'Context':>10s} {'Rel':>6s} {'P50ms':>7s} {'Cost':>8s} {'Drift':>6s}")
        print("-" * 90)
        for p in profiles:
            ctx = str(p.max_context_tokens) if p.max_context_tokens else "-"
            rel = f"{p.reliability:.2f}" if p.reliability is not None else "-"
            lat = str(p.latency_p50_ms) if p.latency_p50_ms else "-"
            cost = p.cost_class or "-"
            drift = "DRIFT" if p.has_drift else "ok"
            drift_color = drift
            print(f"{p.key:<50s} {ctx:>10s} {rel:>6s} {lat:>7s} {cost:>8s} {drift:>6s}")
        return

    if args.policy == "default":
        policy = oracle.default_routing_policy()
        print("Routing Policy Recommendations:")
        print("=" * 50)
        for role, model_key in policy.items():
            display_role = role.replace("_", " ").title()
            if model_key:
                status = oracle.model_status(model_key)
                ctx = status.get("context", "?")
                rel = status.get("reliability", "?")
                print(f"\n  {display_role}:")
                print(f"    Model: {model_key}")
                print(f"    Context: {ctx}  Reliability: {rel}")
            else:
                print(f"\n  {display_role}:")
                print(f"    No candidate available")
        return

    # Default: show summary
    print(f"Routing Oracle -- {len(oracle.profiles)} models loaded from {state_dir}")
    healthy = sum(1 for p in oracle.profiles.values() if oracle.is_healthy(p.key))
    drifted = sum(1 for p in oracle.profiles.values() if p.has_drift)
    print(f"  Healthy: {healthy}  Drifted: {drifted}  Total: {len(oracle.profiles)}")
    print("\nUse --route, --status, --list-by, or --policy for details.")


if __name__ == "__main__":
    main()
```
