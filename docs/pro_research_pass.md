# Observatory v2 Spec — Pro Research Technical Findings

**Purpose:** Deep technical validation of all five architectural changes before Tywin stress test.  
**Date:** 2026-02-27  
**Method:** Targeted research across concurrency, file I/O, hashing, and LLM non-determinism domains.  
**Verdict:** Spec is architecturally sound. Six implementation landmines found. Zero require design changes. All addressable during build.

---

## Summary: What This Research Tested

Each of the five architectural changes was probed for implementation-level failure modes that the Council review cycle (focused on architecture) may not have surfaced. The research targeted:

1. **Change 1 (Global Poll Epoch):** Atomic file semantics, manifest write ordering, epoch pointer race windows
2. **Change 2 (Single-Writer Pattern):** `concurrent.futures` return semantics, `ThreadPoolExecutor` thread safety boundaries, PID lockfile reliability
3. **Change 3 (Semantic Registry Hash):** JSON serialization determinism edge cases, `sort_keys` behavior on nested structures, set/list ordering traps
4. **Change 4 (Tier-Sensitive Drift + Response Hash Canary):** LLM non-determinism at temperature=0, canary hash stability, birthday-attack probability on truncated hashes
5. **Change 5 (Benchmark-Before-Lock):** Staleness window definition, enforcement gap between benchmark run and lock creation

---

## Finding 1: Atomic Epoch Pointer Update Requires fsync + Directory Sync

**Affects:** Change 1 (Global Poll Epoch), step 6 — `current_epoch.json` pointer update

**The Problem:**  
The spec says "atomically update pointer: `current_epoch.json` → `epoch_manifest_{epoch_id}.json`." On POSIX, `os.replace()` (formerly `os.rename()`) is atomic for the directory entry — but the *data* in the new file may not be durable yet. If the process crashes between write and the OS flushing to disk, you can get a zero-length or corrupt `current_epoch.json`.

**The Pattern (from `atomicwrites` library and POSIX best practice):**

```python
import os, tempfile

def atomic_write_json(path, data):
    dir_path = os.path.dirname(path) or "."
    # Write to temp file in same directory (same filesystem = atomic rename)
    fd, tmp = tempfile.mkstemp(dir=dir_path, suffix=".tmp")
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())        # flush file data to disk
        os.replace(tmp, path)            # atomic rename
        # Sync directory metadata so the new entry is durable
        dir_fd = os.open(dir_path, os.O_RDONLY)
        try:
            os.fsync(dir_fd)
        finally:
            os.close(dir_fd)
    except:
        os.unlink(tmp)                   # cleanup on failure
        raise
```

**Why it matters:**  
Every file the spec writes during Phase A (snapshots, manifest, registry, benchmark store) and the epoch pointer itself should use this pattern. Without the directory fsync, a power loss can leave a dangling or empty pointer — and the next run would see no valid epoch. This is the difference between "works in testing" and "survives production crashes."

**Recommendation:** Extract `atomic_write_json()` as a utility. Use it for every state file write. The spec's existing `write_text_atomic` concept is correct in intent but needs the directory fsync addition.

---

## Finding 2: `concurrent.futures.ThreadPoolExecutor` — Return Values Are Safe, But Exception Handling Has a Trap

**Affects:** Change 2 (Single-Writer Pattern)

**The Good News:**  
`concurrent.futures` is the correct tool for the spec's design. Workers submit via `executor.submit()`, return `Future` objects, and the main thread collects results via `future.result()`. The `PollResult` dataclass crosses the thread boundary cleanly — no shared mutable state, no locks needed.

**The Trap:**  
If a worker raises an exception, `future.result()` re-raises it in the main thread. But if you're iterating futures in a loop and one raises, the remaining futures are abandoned. Those abandoned futures may still be running (or queued) and their results are silently discarded.

For the observatory, this means: if OpenAI polling throws a network error, the main thread might skip Anthropic and Google results that completed successfully.

**The Fix:**

```python
results = []
for future in concurrent.futures.as_completed(future_to_provider):
    provider = future_to_provider[future]
    try:
        result = future.result()
        results.append(result)
    except Exception as e:
        # Create an error PollResult instead of losing the epoch
        results.append(PollResult(
            provider=provider,
            snapshot=None,
            error=str(e),
            # ... empty fields ...
        ))
```

**Why it matters:** The spec already contemplates partial epochs (manifests with error fields for failed providers). But the implementation must ensure failed workers produce error PollResults, not exceptions that abort the main thread's collection loop.

**Also confirmed:** `requests.Session` is NOT thread-safe. Each worker thread should create its own session (or use thread-local storage). The spec's single-writer pattern makes this natural — each `_poll_one` worker is independent — but it's worth noting explicitly during implementation.

---

## Finding 3: JSON Determinism Has More Edge Cases Than `sort_keys=True` Covers

**Affects:** Change 3 (Semantic Registry Hash)

**The spec's `compute_model_fingerprint()` is well-designed** — it extracts stable fields and uses `stable_json_dumps()` with `sort_keys=True`. But there are three edge cases that can produce false-positive drift even with sorted keys:

### 3a. Float Representation Instability

Python's `json.dumps` renders floats using `repr()`, which can produce different string representations for the same logical value across Python versions or platforms. Example: `0.1` might serialize as `"0.1"` on one system and `"0.10000000000000001"` on another (rare in modern Python, but real on 32-bit builds).

**Fix:** If pricing or numeric fields are included in fingerprints, round to a fixed precision before hashing, or convert to string with explicit formatting.

### 3b. Unicode Normalization

Provider APIs might return model names with different Unicode normalization forms (NFC vs NFD). `"café"` in NFC and NFD are visually identical but have different byte representations. `json.dumps` preserves the input form, so two "identical" model names could hash differently.

**Fix:** Apply `unicodedata.normalize('NFC', s)` to all string values before fingerprinting.

### 3c. `None` vs. Missing Key Ambiguity

The spec already handles this correctly with the "only include fields that ARE present" comment. But verify that `stable_json_dumps` doesn't emit `null` for `None` values that were explicitly included. The spec's `if key in m and m[key] is not None` guard is correct — just ensure it's applied consistently across all hashing paths.

---

## Finding 4: Canary Response Hashing Will Show "Drift" From LLM Non-Determinism — This Is Expected and the Spec Must Account For It

**Affects:** Change 4 (Tier-Sensitive Drift + Response Hash Canary)

**This is the most important finding.**

The spec replaces prompt-hash canaries (Angry Tyrion's "dead canary" critique) with response-hash canaries. The intent is correct: hash the actual model output to detect behavioral changes. But research confirms a critical reality:

**Even at temperature=0, LLM APIs are NOT deterministic.**

Sources of non-determinism (confirmed across OpenAI, Anthropic, and Google documentation):

- **GPU floating-point non-associativity:** Parallel matrix operations accumulate rounding errors differently depending on thread scheduling. Same weights, same input, slightly different logit scores.
- **Batch contention (MoE models):** GPT-4 and similar Mixture-of-Experts models route tokens based on what *other requests* are in the same batch. Your prompt gets different expert paths depending on server load.
- **Infrastructure differences:** Different GPU hardware (A100 vs H100), different CUDA versions, different quantization states across the serving fleet produce subtly different outputs.
- **Silent model updates:** Providers update model weights without changing model IDs. This is *exactly what the observatory should detect*, but it's indistinguishable from random non-determinism via hash comparison alone.

**What this means for the spec:**

A canary that sends `"What is 2+2?"` and hashes the response will get different hashes across runs — not because the model changed, but because `"The answer is 4."` vs `"4"` vs `"2+2 = 4"` are all valid temperature=0 outputs depending on batch context.

**The spec needs a "canary tolerance" model, not a binary match:**

```python
@dataclass
class CanaryBaseline:
    prompt: str
    expected_semantic_content: str  # "4" — the semantic answer
    baseline_responses: List[str]   # last N responses (ring buffer)
    baseline_hashes: Set[str]       # known-good response hashes
    
    def check(self, new_response: str) -> CanaryVerdict:
        new_hash = sha256(new_response)
        if new_hash in self.baseline_hashes:
            return CanaryVerdict.MATCH
        # Semantic check: does the response still contain "4"?
        if self.expected_semantic_content in new_response:
            # New wording, same answer — update baseline
            self.baseline_hashes.add(new_hash)
            return CanaryVerdict.VARIANT
        else:
            return CanaryVerdict.DRIFT  # Real behavioral change
```

**Three-state canary instead of binary:**
- **MATCH:** Exact hash match with known baseline. No action.
- **VARIANT:** Different wording, same semantic content. Log, update baseline, no alert.
- **DRIFT:** Semantic content changed. Alert. This is a real model behavioral change.

**Why this matters:** Without the three-state model, the canary will fire on almost every poll run, training operators to ignore it. That's worse than no canary at all — it's a false-positive factory. The spec's canary design is architecturally correct (response hash, not prompt hash), but the implementation must account for the fundamental non-determinism of the signal source.

---

## Finding 5: Hash Truncation Collision Risk Is Negligible But Should Be Documented

**Affects:** Change 3 (Semantic Registry Hash), Change 1 (Manifest Hash)

Oberyn Kill #5 flagged hash truncation collision risk. The birthday attack math:

- **Full SHA-256 (256 bits):** Collision at ~2^128 operations. Not a concern for any system on Earth.
- **Truncated to 16 hex chars (64 bits):** Collision at ~2^32 ≈ 4.3 billion operations. For a system hashing a few hundred model fingerprints, the probability of collision is approximately `n²/2^65` where n is the number of distinct fingerprints. At n=1000, that's ~1 in 36 quadrillion. Not a practical concern.
- **Truncated to 8 hex chars (32 bits):** Collision at ~2^16 ≈ 65,536. Now we're in dangerous territory.

**Recommendation:** Use at minimum 16 hex characters (64 bits) for all display/comparison hashes. Use full SHA-256 internally for lockfile commitments and manifest hashes where collision resistance is load-bearing. Document the truncation policy so future maintainers don't shorten hashes for aesthetics.

---

## Finding 6: PID Lockfile (fcntl.flock) Has Platform and Failure-Mode Nuances

**Affects:** Change 2 bonus fix — inter-process cron overlap prevention

The spec's `fcntl.flock()` pattern is correct for Linux, but has nuances:

- **flock is advisory, not mandatory.** Any process that doesn't check the lock can still write to locked files. This is fine for the observatory (cooperative single-process design) but shouldn't be relied upon as a security boundary.
- **flock does NOT work on NFS.** If the state directory is on a network filesystem, flock may silently succeed without actually locking. For SSH-only tower access, this is unlikely but worth documenting.
- **Process death releases the lock automatically.** This is actually a feature — if the observatory crashes, the next cron invocation will acquire the lock cleanly. No stale lockfile cleanup needed.
- **The file descriptor must stay open.** The lock is held as long as the fd is open. Closing the fd (or garbage-collecting the file object) releases the lock. Store the fd in a global or pass it through the run lifetime.

**The spec's pattern is correct as written.** Just ensure:

```python
# Store globally — do NOT let this get garbage-collected
_lock_fd = None

def acquire_instance_lock(state_dir):
    global _lock_fd
    _lock_fd = open(state_dir / ".observatory.lock", "w")
    try:
        fcntl.flock(_lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("[SKIP] Another observatory instance is running.")
        sys.exit(0)
```

---

## Structural Verdict

| Spec Change | Research Verdict | Implementation Notes |
|---|---|---|
| 1. Global Poll Epoch | **Sound.** | Add directory fsync to atomic writes. Pattern matches Apache Iceberg's immutable snapshot + manifest design. |
| 2. Single-Writer Pattern | **Sound.** | Use `as_completed()` with error PollResults, not bare exception propagation. One Session per worker thread. |
| 3. Semantic Registry Hash | **Sound.** | Add float rounding, Unicode NFC normalization. Edge cases, not design flaws. |
| 4. Tier-Sensitive Drift + Canary | **Sound, needs tolerance model.** | Three-state canary (MATCH/VARIANT/DRIFT) instead of binary hash comparison. LLM non-determinism is fundamental, not a bug. |
| 5. Benchmark-Before-Lock | **Sound.** | No novel findings. Spec's design is clean. |

**Overall:** The spec's five changes are architecturally correct. The research found no design-level flaws — only implementation-level landmines that are standard engineering concerns. The most important finding is #4 (canary tolerance), which changes the implementation approach for one component but not the architectural direction.

The epoch/manifest pattern independently converges with Apache Iceberg's metadata architecture (immutable snapshots, manifest lists, atomic pointer updates) — which is a strong validation signal. Different domain, same structural solution to the same class of problem (temporal coherence under concurrent mutation).

---

## Recommendation

Hand this alongside the spec to Tywin. Let him decide whether Finding #4 requires a spec amendment or is an implementation detail. Everything else is "notes for the builder."

No council. No drama. Just implementation.
