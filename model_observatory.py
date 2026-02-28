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
from collections import Counter
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

# Track corrupt files for STATE_CORRUPT event emission (C1)
_corrupt_files: List[str] = []


def read_json(path: Path) -> Optional[Any]:
    """Read a JSON file with corruption handling.
    On parse failure: renames corrupt file, prints warning, returns None.
    Tracks corrupt files for STATE_CORRUPT event emission."""
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
        _corrupt_files.append(f"{path.name}: {e}")
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
    2: "INFO",       # Tier 1 sequence
    3: "WARN",       # Tier 2 reasoning
    4: "CRITICAL",   # Tier 3 JSON schema
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
        # Extract all number tokens and check if "4" appears
        nums = re.findall(r'\b\d+\b', t)
        if "4" in nums:
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
    baseline_set_at: Optional[str] = None
    baseline_epoch_id: Optional[str] = None
    observation_window: List[str] = field(default_factory=list)
    variant_hashes: Set[str] = field(default_factory=set)
    candidate_drift_value: Optional[str] = None
    candidate_drift_streak: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "canonical_baseline": self.canonical_baseline,
            "baseline_set_at": self.baseline_set_at,
            "baseline_epoch_id": self.baseline_epoch_id,
            "observation_window": self.observation_window,
            "variant_hashes": list(self.variant_hashes),
            "candidate_drift_value": self.candidate_drift_value,
            "candidate_drift_streak": self.candidate_drift_streak,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'BenchmarkBaseline':
        return cls(
            canonical_baseline=d.get("canonical_baseline"),
            baseline_set_at=d.get("baseline_set_at"),
            baseline_epoch_id=d.get("baseline_epoch_id"),
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
        self.current_epoch_id: Optional[str] = None  # set by caller for baseline tracking
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
            baseline.baseline_set_at = utc_now_iso()
            baseline.baseline_epoch_id = self.current_epoch_id
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
            baseline.baseline_set_at = utc_now_iso()
            baseline.baseline_epoch_id = self.current_epoch_id
            baseline.variant_hashes = {raw_hash}
            baseline.candidate_drift_value = None
            baseline.candidate_drift_streak = 0
            return "DRIFT"

        # Check drift confirmation: T frequency in window
        window_count = baseline.observation_window.count(canonical_value)
        window_freq = window_count / len(baseline.observation_window) if baseline.observation_window else 0
        if window_freq >= CONFIRM_FREQ and len(baseline.observation_window) >= CONFIRM_STREAK:
            baseline.canonical_baseline = canonical_value
            baseline.baseline_set_at = utc_now_iso()
            baseline.baseline_epoch_id = self.current_epoch_id
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
        Returns drift type or None if stable.
        Canonicalization failure stores __UNPARSEABLE__ surrogate (Amendment 4)."""
        ok, canonical = canonicalize_benchmark(prompt_index, response_text)
        raw_hash = sha256_bytes(response_text.encode("utf-8"))

        if not ok:
            # Canonicalization failed — store surrogate, do NOT silently drop
            canonical = "__UNPARSEABLE__"
            result = self.classify_observation(provider, model_id, prompt_index, canonical, raw_hash)
            return "BENCHMARK_FAILURE"

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
            fp = compute_model_fingerprint(m) if isinstance(m, dict) else sha256_bytes(str(m).encode("utf-8"))
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
            fp = compute_model_fingerprint(m)
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
            fp = compute_model_fingerprint(m)
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
            fp = compute_model_fingerprint(m)
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
        Reads from epoch manifest (not *_snapshot_latest.json) to prevent
        Franken-snapshot locks. Falls back to latest snapshots only if
        no epoch exists (pre-migration state)."""
        registry_parts = []
        model_ids_parts = []

        # Try to read from current epoch manifest first
        pointer = read_json(state_dir / "current_epoch.json")
        if pointer and pointer.get("manifest_file"):
            manifest = read_json(state_dir / pointer["manifest_file"])
            if manifest and "providers" in manifest:
                for provider in sorted(manifest["providers"].keys()):
                    prov_info = manifest["providers"][provider]
                    snap_file = prov_info.get("snapshot_file", "")
                    snap = read_json(state_dir / snap_file) if snap_file else None
                    if not snap:
                        continue
                    # Use the manifest's committed hash for registry
                    registry_parts.append(f"{provider}:{prov_info.get('hash', '')}")
                    models = snap.get("models", {})
                    for mid in sorted(models.keys()):
                        model_ids_parts.append(f"{provider}/{mid}")

                registry_hash = sha256_bytes("|".join(registry_parts).encode("utf-8"))
                model_ids_hash = sha256_bytes("|".join(model_ids_parts).encode("utf-8"))
                return registry_hash, model_ids_hash

        # Fallback: no epoch yet — read from latest snapshots (v1 compat)
        for snap_file in sorted(state_dir.glob("*_snapshot_latest.json")):
            snap = read_json(snap_file)
            if not snap:
                continue
            provider = snap.get("provider", "")
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

        Lockfiles store canonical baselines (not binary hashes) per Change 5/Tywin Pass 2.
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

    # Step 6: Flag existing v1 lockfiles as needing re-lock (C2)
    locks_dir = state_dir / "experiment_locks"
    if locks_dir.exists():
        v1_locks = list(locks_dir.glob("*.lock.json"))
        for lock_path in v1_locks:
            lock_data = read_json(lock_path)
            if not lock_data:
                continue
            # If it has benchmark_response_hashes but no benchmark_canonical_baselines,
            # it's a v1 behavioral lock
            if lock_data.get("benchmark_response_hashes") and not lock_data.get("benchmark_canonical_baselines"):
                lock_data["_v1_migration_note"] = (
                    "UNLOCKED_BEHAVIORAL: v1 lockfile uses binary hashes. "
                    "Re-lock with --benchmark-before-lock for v2 canonical baselines.")
                atomic_write_json(lock_path, lock_data)
                print(f"  Flagged v1 lockfile: {lock_path.name}")

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

    # 6. Run benchmark calls with R=3 confirmation sampling on mismatch
    if do_benchmark and benchmark_models is not None:
        targets = benchmark_models if benchmark_models else list(curr_models.keys())
        for mid in targets:
            if mid not in curr_models:
                continue
            for i, prompt in enumerate(BENCHMARK_PROMPTS):
                ok, response_text, latency_ms = watcher.benchmark_call(mid, prompt)
                samples = [{"ok": ok, "response_text": response_text, "latency_ms": latency_ms}]
                # R=3 confirmation: if first call succeeded, canonicalize and check
                # if it's a potential mismatch, resample R-1 more times
                if ok and response_text:
                    canon_ok, canon_val = canonicalize_benchmark(i, response_text)
                    if canon_ok:
                        # Collect R-1 additional samples for mode determination
                        for _ in range(CONFIRM_SAMPLES - 1):
                            try:
                                r_ok, r_text, r_lat = watcher.benchmark_call(mid, prompt)
                                samples.append({"ok": r_ok, "response_text": r_text, "latency_ms": r_lat})
                            except Exception:
                                pass
                benchmark_results.append({
                    "model_id": mid, "prompt_idx": i, "prompt": prompt,
                    "ok": ok, "response_text": response_text, "latency_ms": latency_ms,
                    "confirmation_samples": samples,
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
                         provider_hashes: Dict[str, str],
                         all_providers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Write epoch manifest and return it.
    Includes carry-forward for providers that failed to poll (Amendment 2)."""
    # Build manifest — every configured provider must appear
    providers_section: Dict[str, Any] = {}

    # Fresh providers from this epoch
    for provider, raw_hash in sorted(provider_hashes.items()):
        providers_section[provider] = {
            "status": "fresh",
            "snapshot_file": f"{provider}_snapshot_{epoch_id}.json",
            "hash": raw_hash,
        }

    # Carry forward missing providers from previous epoch
    if all_providers:
        pointer = read_json(state_dir / "current_epoch.json")
        prev_manifest = None
        if pointer and pointer.get("manifest_file"):
            prev_manifest = read_json(state_dir / pointer["manifest_file"])
        for provider in all_providers:
            if provider not in providers_section:
                # Not polled this epoch — carry forward
                carried = {"status": "carried_forward", "snapshot_file": "", "hash": ""}
                if prev_manifest and "providers" in prev_manifest:
                    prev_prov = prev_manifest["providers"].get(provider, {})
                    carried["snapshot_file"] = prev_prov.get("snapshot_file", "")
                    carried["hash"] = prev_prov.get("hash", "")
                    carried["from_epoch"] = prev_manifest.get("epoch", "")
                providers_section[provider] = carried

    # Manifest hash commits to provider:snapshot_file:hash:status
    manifest_parts = []
    for p in sorted(providers_section.keys()):
        info = providers_section[p]
        manifest_parts.append(
            f"{p}:{info.get('snapshot_file', '')}:{info.get('hash', '')}:{info.get('status', '')}"
        )
    manifest_hash = sha256_bytes("|".join(manifest_parts).encode("utf-8"))

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
    """Atomically update current_epoch.json to point to the latest epoch.
    Schema matches migration and _get_current_epoch() expectations."""
    manifest_file = f"epoch_manifest_{epoch_id}.json"
    # Read manifest hash from the manifest we just wrote
    manifest_hash = ""
    manifest_data = read_json(state_dir / manifest_file)
    if manifest_data:
        manifest_hash = manifest_data.get("manifest_hash", "")
    pointer = {
        "epoch_id": epoch_id,
        "manifest_file": manifest_file,
        "manifest_hash": manifest_hash,
        "created_at": utc_now_iso(),
    }
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
    all_provider_names: Optional[List[str]] = None,
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

        # Merge benchmark results into store (three-state classification with R=3 mode)
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
                    # Empty response — emit BENCHMARK_MISSING (Amendment 4)
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BENCHMARK_MISSING",
                        notes=f"prompt_{bench['prompt_idx']}: empty response",
                        details={"prompt_index": bench["prompt_idx"]},
                    ))
                    continue
                # R=3 confirmation: use mode of all samples' canonical values
                samples = bench.get("confirmation_samples", [])
                canonical_values = []
                for s in samples:
                    if s.get("ok") and s.get("response_text"):
                        c_ok, c_val = canonicalize_benchmark(bench["prompt_idx"], s["response_text"])
                        if c_ok:
                            canonical_values.append(c_val)
                # Use mode canonical value for classification (R=3 consensus)
                if canonical_values:
                    mode_val = Counter(canonical_values).most_common(1)[0][0]
                    # Find a sample with this canonical value for the raw hash
                    use_response = bench["response_text"]
                    for s in samples:
                        if s.get("ok") and s.get("response_text"):
                            c_ok, c_val = canonicalize_benchmark(bench["prompt_idx"], s["response_text"])
                            if c_ok and c_val == mode_val:
                                use_response = s["response_text"]
                                break
                    drift_status = bench_store.check_and_update(
                        result.provider, bench["model_id"],
                        bench["prompt_idx"], bench["prompt"],
                        use_response)
                else:
                    drift_status = bench_store.check_and_update(
                        result.provider, bench["model_id"],
                        bench["prompt_idx"], bench["prompt"],
                        bench["response_text"])
                if drift_status == "BENCHMARK_FAILURE":
                    all_events.append(ModelEvent(
                        timestamp=utc_now_iso(), provider=result.provider,
                        model_id=bench["model_id"],
                        event_type="BENCHMARK_FAILURE",
                        notes=f"prompt_{bench['prompt_idx']}: canonicalization failed (reason=parse_failed)",
                        details={"prompt_index": bench["prompt_idx"],
                                 "reason": "parse_failed"},
                    ))
                elif drift_status == "BEHAVIOR_DRIFT":
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

    # --- Step 2: Write epoch manifest (with carry-forward for failed providers) ---
    if provider_hashes:
        write_epoch_manifest(state_dir, epoch_id, provider_hashes,
                             all_providers=all_provider_names)

    # --- Step 3: Save CapabilityRegistry ---
    cap_reg.save()

    # --- Step 4: Save BenchmarkStore ---
    if bench_store:
        bench_store.save()

    # --- Step 5: Append events (including STATE_CORRUPT from read_json) ---
    if _corrupt_files:
        for corrupt_info in _corrupt_files:
            all_events.append(ModelEvent(
                timestamp=utc_now_iso(), provider="system", model_id="",
                event_type="STATE_CORRUPT",
                notes=f"Corrupt state file: {corrupt_info}",
            ))
        _corrupt_files.clear()
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

    # Set epoch on bench_store for baseline tracking (Amendment 6)
    if bench_store:
        bench_store.current_epoch_id = epoch_id

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
        all_provider_names=[w.provider_name for w in watchers],
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
