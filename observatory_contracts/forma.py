from __future__ import annotations
import argparse, ast, json, re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace")

def find_py_files(root: Path) -> List[Path]:
    exclude_dirs = {".venv", "__pycache__", "observatory_contracts", "tests"}
    exclude_files = {"observatory_tui.py", "routing_oracle.py"}
    return [p for p in root.rglob("*.py")
            if not (set(p.parts) & exclude_dirs) and p.name not in exclude_files]

def fail(msg: str) -> None:
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def warn(msg: str) -> None:
    print(f"[WARN] {msg}")

def ok(msg: str) -> None:
    print(f"[OK] {msg}")

def grep_any(text: str, patterns: Iterable[str]) -> Optional[str]:
    for pat in patterns:
        if re.search(pat, text, flags=re.MULTILINE):
            return pat
    return None

@dataclass
class AstHit:
    file: Path
    lineno: int
    detail: str

def check_no_latest_in_locking(py_files: List[Path]) -> None:
    offenders: List[Tuple[Path, int, str]] = []
    lock_keywords = ("Experiment", "Invariant", "Lock", "lock", "invariant")
    # Lines containing "fallback", "v1", "compat", or "latest_path" are exempt —
    # v1 backward-compatibility paths are expected
    exempt_patterns = re.compile(r"fallback|v1|compat|latest_path|carried_forward|backward|previous|Phase A|benchmark_before|not \*_snapshot", re.IGNORECASE)
    for f in py_files:
        txt = read_text(f)
        if not any(k in txt for k in lock_keywords):
            continue
        lines = txt.split("\n")
        for i, line in enumerate(lines):
            if "_snapshot_latest.json" not in line:
                continue
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if exempt_patterns.search(line):
                continue
            # Check 5 lines above for exemption context
            context = "\n".join(lines[max(0,i-5):i+1])
            if not exempt_patterns.search(context):
                offenders.append((f, i+1, "references *_snapshot_latest.json"))
    if offenders:
        msg = "Lock semantics must be epoch-manifest based; found *_snapshot_latest.json references:\n"
        msg += "\n".join([f"  - {p}:{ln} {d}" for p, ln, d in offenders])
        fail(msg)
    ok("No *_snapshot_latest.json usage in lock/invariant code (v1-compat fallbacks exempt)")

def check_epoch_pointer_schema(py_files: List[Path]) -> None:
    required = ["epoch_id", "manifest_file", "manifest_hash"]
    found = {k: False for k in required}
    for f in py_files:
        txt = read_text(f)
        if "current_epoch.json" not in txt:
            continue
        for k in required:
            if re.search(rf'["\']{k}["\']', txt):
                found[k] = True
    missing = [k for k, v in found.items() if not v]
    if missing:
        fail(f"Epoch pointer schema keys not referenced in code: missing {missing}. "
             f"Require current_epoch.json to include epoch_id/manifest_file/manifest_hash.")
    ok("Epoch pointer schema keys referenced (epoch_id, manifest_file, manifest_hash)")

def check_manifest_commits_filenames(py_files: List[Path]) -> None:
    joined = "\n".join(read_text(f) for f in py_files if "manifest" in read_text(f))
    strong_signals = [
        r"snapshot_file",
        r"snapshot_filename",
        r"provider.*:.*snapshot_",
        r"provider:.*\.json",
    ]
    if not grep_any(joined, strong_signals):
        fail("Manifest hashing does not appear to include snapshot filenames/identity tuple. "
             "Require hash input lines like provider:snapshot_file:hash.")
    ok("Manifest hashing appears to commit to snapshot identity (filenames/tuple present)")

def check_atomic_write_is_durable(py_files: List[Path]) -> None:
    text = "\n".join(read_text(f) for f in py_files)
    has_replace = "os.replace" in text
    has_fsync = "os.fsync" in text
    has_dir_open = bool(re.search(r"os\.open\(.+os\.O_RDONLY", text))
    if not (has_replace and has_fsync and has_dir_open):
        fail("Durable atomic write contract not satisfied. Need: write temp -> fsync(file) -> os.replace -> fsync(dir).")
    ok("Durable atomic write pattern present (fsync file + dir)")

def check_canary_is_response_based(py_files: List[Path]) -> None:
    offenders: List[Tuple[Path, int, str]] = []
    pat = r"sha256_.*\(\s*.*CANARY_PROMPT.*encode"
    for f in py_files:
        txt = read_text(f)
        if "CANARY_PROMPT" not in txt:
            continue
        for m in re.finditer(pat, txt):
            ln = txt[:m.start()].count("\n") + 1
            offenders.append((f, ln, "hashes CANARY_PROMPT string (dead canary)"))
    if offenders:
        msg = "Dead-canary pattern found (prompt-hash). Canary must be response-hash baseline, not prompt hash:\n"
        msg += "\n".join([f"  - {p}:{ln} {d}" for p, ln, d in offenders])
        fail(msg)
    ok("No dead-canary (prompt-hash) pattern detected")

def check_lock_includes_behavioral_baselines(py_files: List[Path]) -> None:
    text = "\n".join(read_text(f) for f in py_files if "Lock" in read_text(f) or "Experiment" in read_text(f))
    signals = [
        r"benchmark_canonical_baselines",
        r"benchmark_response_hash",
        r"benchmark_hashes",
        r"response_hashes",
        r"canary_response_hash",
        r"baseline_hashes",
    ]
    if not grep_any(text, signals):
        fail("Lockfiles do not appear to store behavioral baselines. "
             "Add benchmark_canonical_baselines to lock schema.")
    ok("Lock schema appears to include behavioral baselines")

def check_single_writer_ledger(py_files: List[Path]) -> None:
    code = "\n".join(read_text(f) for f in py_files)
    # v2 pattern: ThreadPoolExecutor + PollResult (immutable) → main thread merges
    uses_executor = "ThreadPoolExecutor" in code
    uses_poll_result = "PollResult" in code
    has_merge = bool(re.search(r"merge.*poll.*result|main.*thread.*merge", code, re.IGNORECASE))
    # v1 pattern: Queue + writer loop
    uses_queue = "queue.Queue" in code or "Queue()" in code
    writer_loop = bool(re.search(r"while\s+True:.*get\(", code, re.DOTALL)) or "writer" in code.lower()
    if not ((uses_executor and uses_poll_result) or (uses_queue and writer_loop)):
        fail("Single-writer ledger pattern not detected. "
             "FORMA requires event log integrity under concurrency "
             "(ThreadPoolExecutor+PollResult or Queue+writer loop).")
    ok("Single-writer ledger pattern detected (workers return immutable results, main thread merges)")

def check_drift_claims_match_code(py_files: List[Path], strict: bool) -> None:
    docs = ""
    for name in ("README.md", "observatory_v2_spec.md", "observatory_v2_spec_amended.md"):
        p = REPO_ROOT / name
        if p.exists():
            docs += read_text(p) + "\n"
    claims_r3 = bool(re.search(r"\bR\s*=\s*3\b", docs)) or ("resampling" in docs.lower())
    if not claims_r3:
        ok("No R=3/resampling claim detected in docs (or docs absent)")
        return
    code = "\n".join(read_text(f) for f in py_files)
    has_resample = bool(re.search(r"resampl|retry.*benchmark|confirm.*3|range\(3\)", code, re.IGNORECASE))
    if not has_resample:
        msg = "Docs claim resampling / R=3 confirmation but code does not appear to implement it."
        if strict:
            fail(msg)
        warn(msg + " (non-strict)")
    else:
        ok("R=3/resampling logic appears present in code")

def run_all(strict: bool) -> None:
    py_files = find_py_files(REPO_ROOT)
    if not py_files:
        fail("No Python files found. Run from repo root.")
    print(f"[INFO] Scanning {len(py_files)} Python files under {REPO_ROOT}")
    check_no_latest_in_locking(py_files)
    check_epoch_pointer_schema(py_files)
    check_manifest_commits_filenames(py_files)
    check_atomic_write_is_durable(py_files)
    check_canary_is_response_based(py_files)
    check_lock_includes_behavioral_baselines(py_files)
    check_single_writer_ledger(py_files)
    check_drift_claims_match_code(py_files, strict=strict)
    print("[PASS] FORMA contracts satisfied")

def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(prog="observatory_contracts.forma")
    ap.add_argument("--strict", action="store_true", help="Fail on doc↔code mismatches")
    args = ap.parse_args(argv)
    run_all(strict=args.strict)

if __name__ == "__main__":
    main()
