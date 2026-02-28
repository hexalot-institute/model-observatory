# Model Observatory v2.0 — Architectural Repair Spec (Amended)

**Date:** 2026-02-27
**Status:** SPEC LOCKED
**Amendments:** GPT-5.2 Thinking Pass 1 (7 amendments) + GPT-5.2 Thinking Pass 2 (5 amendments) + GPT-5.2 Thinking Pass 3 (6 amendments) merged per Shadow Directive.
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
| 13 | GPT-5.2 Thinking Pass 3 | Skeptic implementation check | 6 amendments — epoch pointer schema, partial epoch carry-forward, R=3 worker binding, canonicalization failure handling, severity vocabulary, baseline epoch tracking |

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
     The manifest MUST represent every configured provider even on partial success.
     Each provider entry includes "status": "fresh" | "carried_forward".
     Carried-forward providers include "from_epoch": "<source_epoch_id>".
     The manifest_hash commits to the tuple (provider, snapshot_file, hash, status)
     for every provider — not just those that succeeded this epoch.
  6. Atomically replace current_epoch.json with an explicit pointer object:
     {
       "epoch_id": "<epoch_id>",
       "manifest_file": "epoch_manifest_<epoch_id>.json",
       "manifest_hash": "<sha256(manifest canonical form)>",
       "updated_at": "<utc iso8601>"
     }
     Readers MUST treat the epoch as committed only if current_epoch.json parses and the referenced manifest file parses. Writers MUST NOT use alternative keys (e.g., current_epoch) — epoch_id/manifest_file are the stable contract.
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
- **Tier 3 — JSON stability (prompt index 4):** Parse JSON even if wrapped in code fences or followed by commentary. Canonicalize via `json.loads` + `json.dumps(sort_keys=True, separators=(',',':'))`. If JSON parse fails, the system MUST NOT drop the observation. It MUST emit `BENCHMARK_FAILURE` with `reason=parse_failed`, store `"__UNPARSEABLE__"` as the canonical surrogate in the observation window, and ensure that missing observations do not increment drift-by-absence. Empty `response_text` MUST emit `BENCHMARK_MISSING` (WARN).
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

**Implementation constraint (R resampling worker binding):** The R confirmation samples MUST be collected inside the worker function while it remains pure (no state writes have occurred). Workers return all R benchmark samples in `benchmark_results`. The main thread computes `v_mode` from all R samples before applying any thresholds or writing state. The `CONFIRM_SAMPLES` constant (default `R=3`) MUST be exercised by control flow — i.e., the code path that triggers resampling must be reachable and tested, not dead code behind an always-false guard.

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
    2: "INFO",       # Tier 1 sequence
    3: "WARN",       # Tier 2 reasoning
    4: "CRITICAL",   # Tier 3 JSON schema
    5: "CRITICAL",   # Tier 4 tool-call format
}
# Note: severity vocabulary is strictly INFO/WARN/CRITICAL — no LOW/MEDIUM/HIGH.
# If more granularity is desired, carry as a separate "tier_level" field
# without overloading severity.
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

**BenchmarkStore baseline persistence:** `BenchmarkStore` MUST persist per baseline:
- `baseline_set_at`: UTC ISO8601 timestamp of when the canonical baseline was established
- `baseline_epoch_id`: the epoch_id during which the baseline was measured

Lockfiles MUST store `baseline_set_at` and `baseline_epoch_id` for each baseline entry so that Acceptance Criteria #3 ("every behavioral baseline in a lock was measured during the lock's epoch or explicitly acknowledged as stale") can be verified mechanically — not by convention or manual inspection.

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
