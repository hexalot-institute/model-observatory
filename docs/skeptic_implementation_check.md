# Skeptic Implementation Check (2026-02-27)

According to a document from **2026-02-27**, the v2 spec and the included v2 implementation do **not** currently survive a skeptic implementation check in the two highest‑risk places: (1) the epoch pointer/lock epoch chain, and (2) Change 4’s “tier‑sensitive drift” claim that it tolerates LLM non‑determinism. This is not a “new bug hunt” — it’s a verification failure against the spec’s own acceptance criteria.

---

## A) Spec integrity check (spec vs implementation)

### A1) Change 1: Epoch pointer contract is underspecified in the spec and internally inconsistent in implementation

**Spec requirement (Change 1, step 6):** “Atomically update pointer: `current_epoch.json -> epoch_manifest_{epoch_id}.json`.”

**Implementation reality (two incompatible schemas):**

* `update_epoch_pointer()` writes `{"current_epoch": epoch_id, "updated_at": ...}` (no manifest reference, no manifest hash).
* `ExperimentInvariantManager._get_current_epoch()` reads `epoch_id` and `manifest_file` (and derives `manifest_hash` by loading the manifest file). This will return empty strings under the `current_epoch` schema.

**Direct consequence:** locks cannot reliably “reference epoch_id and manifest_hash” (Acceptance Criteria #2/#3) because the pointer schema the lock-reader expects is not the pointer schema the writer produces.

**Sequential conflict:** the migration routine writes the *other* schema (`epoch_id`, `manifest_file`, `manifest_hash`) into `current_epoch.json`, but normal runtime will overwrite it with the `current_epoch` schema on the next poll, breaking `_get_current_epoch()` again.

---

### A2) Change 3: “Semantic registry hashing” is defined, but not actually used where it matters

The spec’s acceptance criteria require registry drift computed from semantic primitives and stable under list ordering.

The document includes an implementation of `compute_model_fingerprint()` that normalizes fields (NFC, float rounding, etc.).

But watchers still compute “fingerprints” as `sha256(stable_json_dumps(raw_model_dict))`, which reintroduces sensitivity to irrelevant payload fields and list ordering inside those dicts:

* Ollama watcher: hashes the entire model entry `m` as JSON.
* OpenAI compat watcher: same pattern.

Also, epoch snapshots written by `write_epoch_snapshot()` store only `raw_hash` and `models` (model_id → fingerprint), with **no `semantic_registry_hash` field**.

So the “semantic” path exists in text, but is not the actual invariant used by state or locks.

---

### A3) Change 4: The spec’s non‑determinism tolerance claim does not survive implementation

The spec claims:

* No `BEHAVIOR_DRIFT` from a single observation.
* **R=3 confirmation samples on mismatch** (immediate resampling).
* **K=3** consecutive polls or **T=0.6** frequency threshold in W=20.

**Implementation check: resampling (R) is not implemented.**

* Worker benchmark loop calls `benchmark_call()` **exactly once per prompt**; it never performs “R=3 confirmation samples on mismatch.”
* `CONFIRM_SAMPLES = 3` exists but is unused in the control flow.

That alone falsifies the spec’s central “handles non-determinism” claim: MoE routing variance / floating‑point non‑associativity can yield two or more plausible outputs at temperature=0; without within‑poll resampling, a random streak can satisfy K (or early window frequency) and produce a false “confirmed drift” over time.

**Implementation check: JSON normalization failure does not “hash raw and continue”; it silently drops the observation.**

* Spec explicitly says: if JSON parse fails, “hash raw text as-is and go through same confirmation flow.”
* Code path: `canonicalize_benchmark()` returns `ok=False` on JSON parse failure.
* `BenchmarkStore.check_and_update()` returns `None` when `ok` is false — meaning no baseline update, no window update, no variant/drift classification.

That creates a **blind spot**: a model can start producing “almost JSON” or prefixed explanations and the drift system will simply stop observing that benchmark rather than classifying it as variant/drift. That is explicitly contrary to the spec’s intended behavior.

**Implementation check: “missing benchmark is not mismatch” is asserted, but missing results are silently ignored without `BENCHMARK_MISSING`.**

* Merge loop: if `bench["response_text"]` is empty, it just `continue`s (no event, no accounting).
* Spec requires a WARN‑level `STATE_CORRUPT` event on corruption and calls out missing benchmark tolerance; it also implies missing benchmark should be visible, not silent.

---

### A4) Severity vocabulary conflict: spec defines tier severity levels the notifier cannot represent

The implementation defines:

```python
TIER_SEVERITY = {0:"INFO", 1:"INFO", 2:"LOW", 3:"MEDIUM", 4:"HIGH", 5:"CRITICAL"}
```

But the notifier only has emojis and severity semantics for `INFO`, `WARN`, `CRITICAL`.

So the spec’s “tier-sensitive severity” is underspecified (what are LOW/MEDIUM/HIGH?) and the implementation can’t faithfully convey it as severity (it ends up only as a note).

---

## B) Research integration: Finding #4 (three-state MATCH/VARIANT/DRIFT) — spec change or implementation detail?

It is a **spec change**, not merely an implementation detail.

Reason: it changes the observable contract of the system and the acceptance criteria. A binary response hash is fundamentally incompatible with non‑deterministic LLM APIs; therefore, the spec must define:

* what constitutes MATCH vs VARIANT vs DRIFT,
* what thresholds apply,
* and how lockfiles interpret these states.

The document already reflects this (three-state language appears in Change 4 and in lock semantics).

However: the *implementation* does not yet implement the critical part of that change (R resampling + non-silent handling of parse failures), so the spec is correct to treat it as a contract-level change, not as a “local detail.”

---

## C) Deep research pass: what the research file missed (or what the implementation still violates)

### C1) Error recovery paths: required events are not emitted

Spec requires: on JSON corruption, rename file and emit WARN `STATE_CORRUPT`.

Current code’s `read_json()` renames/prints but does not route this into the event system; it is not connected to an emitted `STATE_CORRUPT` event in the main merge.

### C2) Migration (v1 → v2): pointer schema and lockfile compatibility are not enforced end-to-end

Spec says v1 lockfiles with hash-only behavioral invariants must be marked `UNLOCKED_BEHAVIORAL` and require re-lock.

Migration code shown only:
* writes epoch 0,
* sets `_needs_rebaseline` in benchmark file,
* emits `MIGRATION_REBASELINE`.

There is no corresponding mandatory sweep that rewrites or flags existing v1 lockfiles in the locks directory in the migration path shown.

### C3) Ordering assumptions: Change 1 pointer schema must be fixed before Change 5 can be true

Spec asserts dependency chain: Change 2 → Change 1 → (Change 4 + Change 5).

But in practice: **Change 5 cannot function** (locks referencing epoch/manifest) until the pointer schema is unambiguous and consistent across writer/reader.

---

## D) Final verdict: spec amendments required

The spec does need amendments. Not because the architecture is wrong — but because several contracts are ambiguous or internally inconsistent, and those ambiguities have already produced contradictory implementations (pointer schema, severity vocabulary, missing/parse-fail benchmark handling, baseline freshness enforceability).

Below are **six amendments** with exact replacement text.

### Amendment 1 — Change 1: make `current_epoch.json` schema explicit

Replace (Change 1 design step 6):
> `6. Atomically update pointer: current_epoch.json -> epoch_manifest_{epoch_id}.json`

With:
> `6. Atomically replace current_epoch.json with an explicit pointer object:`
>
> ```json
> {
>   "epoch_id": "<epoch_id>",
>   "manifest_file": "epoch_manifest_<epoch_id>.json",
>   "manifest_hash": "<sha256(manifest canonical form)>",
>   "updated_at": "<utc iso8601>"
> }
> ```
>
> `Readers MUST treat the epoch as committed only if current_epoch.json parses and the referenced manifest file parses. Writers MUST NOT use alternative keys (e.g., current_epoch) — epoch_id/manifest_file are the stable contract.`

### Amendment 2 — Change 1: define partial-epoch carry-forward in the manifest schema

Replace the manifest example in Change 1 design step 5:
> `"providers": { "openai": {"snapshot_file": "...", "hash": "..."}, ... }`

With:
> `epoch_manifest_<epoch_id>.json MUST represent every configured provider, even on partial success:`
>
> ```json
> {
>   "epoch": "<epoch_id>",
>   "timestamp": "<utc iso8601>",
>   "providers": {
>     "<provider>": {
>       "status": "fresh" | "carried_forward",
>       "snapshot_file": "<provider>_snapshot_<epoch_or_prev_epoch>.json",
>       "hash": "<provider_registry_hash>",
>       "from_epoch": "<prev_epoch_id>"
>     }
>   },
>   "manifest_hash": "<sha256(sorted provider:hash:status:from_epoch))>"
> }
> ```
>
> `A provider poll failure MUST NOT remove the provider from the manifest; it MUST be carried_forward from the last committed epoch.`

### Amendment 3 — Change 4: explicitly bind “R confirmation samples” to worker behavior and data shape

Insert after:
> `R=3 confirmation samples on mismatch (immediate re-sampling)`

Insert:
> `Implementation constraint:`
> `R resampling REQUIRES additional provider API calls and therefore MUST occur in the worker (poll_provider) while the worker is still pure (no state writes).`
> `Workers MAY return multiple benchmark samples for the same (provider, model_id, prompt_idx). The main thread MUST group samples by (provider, model_id, prompt_idx), canonicalize each sample, and compute v_mode (most frequent canonical value) before applying MATCH/VARIANT/DRIFT thresholds.`
> `A defined constant (e.g., CONFIRM_SAMPLES) MUST be exercised by control flow; unused constants do not satisfy this requirement.`

### Amendment 4 — Change 4: canonicalization failure and missing benchmark observations must never be silent

Replace:
> `If JSON parse fails, hash raw text as-is and go through same confirmation flow.`

With:
> `If canonicalization fails, the system MUST NOT drop the observation. It MUST:`
> `1) emit BENCHMARK_FAILURE with reason=parse_failed,`
> `2) store a canonical value of "__UNPARSEABLE__" (or raw-hash surrogate) into the observation window,`
> `3) ensure missing/failed observations do not increment drift-by-absence.`
> `If benchmarking was attempted but response_text is empty, emit BENCHMARK_MISSING (WARN) and record a missing observation (no data).`

### Amendment 5 — Event semantics: remove undefined severity levels

Replace severities outside `{INFO, WARN, CRITICAL}` with:

```python
TIER_SEVERITY = {
  1: "INFO",
  2: "INFO",
  3: "WARN",
  4: "CRITICAL",
}
```

Carry extra granularity in a separate field (tier_level), not severity.

### Amendment 6 — Change 5: make “baseline measured during lock epoch” enforceable

Insert:
> `BenchmarkStore MUST persist baseline_set_at and baseline_epoch_id per (provider, model_id, prompt_idx). Lockfiles MUST store baseline_epoch_id values so “baseline measured during lock epoch” can be verified mechanically.`

---

## Bottom line

The spec is directionally right (three-state drift is mandatory).  
The implementation does not currently satisfy the spec’s acceptance criteria (pointer schema inconsistency; R resampling absent; parse/missing silently dropped; severity mismatch).

