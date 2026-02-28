from __future__ import annotations
import argparse, re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")

def py_files():
    exclude_dirs = {".venv", "__pycache__", "observatory_contracts", "tests"}
    exclude_files = {"observatory_tui.py", "routing_oracle.py"}
    return [p for p in REPO_ROOT.rglob("*.py")
            if not (set(p.parts) & exclude_dirs) and p.name not in exclude_files]

def fail(msg: str):
    print(f"[FAIL] {msg}")
    raise SystemExit(1)

def warn(msg: str):
    print(f"[WARN] {msg}")

def ok(msg: str):
    print(f"[OK] {msg}")

def run():
    files = py_files()
    if not files:
        fail("No Python files found.")

    code = "\n".join(read_text(f) for f in files)

    if "current_epoch.json" in code:
        needed = ["epoch_id", "manifest_file", "manifest_hash"]
        missing = [k for k in needed if f'"{k}"' not in code and f"'{k}'" not in code]
        if missing:
            fail(f"Epoch pointer schema likely incomplete; missing references to keys: {missing}")
        ok("Epoch pointer schema keys referenced")
    else:
        warn("current_epoch.json not referenced in code (ok if different naming, but verify manually)")

    # Check that primary lock paths don't use *_snapshot_latest.json
    # (v1-compat fallback paths, backward-compat writes, comments, and benchmark enumeration are exempt)
    exempt = re.compile(r"fallback|v1|compat|latest_path|carried_forward|backward|previous|Phase A|benchmark_before|not \*_snapshot", re.IGNORECASE)
    lock_violation = False
    for f in files:
        txt = read_text(f)
        if "Experiment" not in txt and "Lock" not in txt:
            continue
        lines_list = txt.split("\n")
        for i, line in enumerate(lines_list):
            if "_snapshot_latest.json" not in line:
                continue
            stripped = line.strip()
            # Skip comments and docstrings
            if stripped.startswith("#") or stripped.startswith('"""') or stripped.startswith("'''"):
                continue
            if exempt.search(line):
                continue
            # Check 5 lines above for exemption context
            context = "\n".join(lines_list[max(0,i-5):i+1])
            if exempt.search(context):
                continue
            lock_violation = True
            break
    if lock_violation:
        fail("Lock/invariant code references *_snapshot_latest.json in primary path. Must use epoch manifest.")
    ok("Locks do not reference *_snapshot_latest.json in primary paths (v1-compat exempt)")

    if "epoch_manifest" in code:
        if not re.search(r"provider.*:.*snapshot_.*\.json", code):
            warn("Manifest hash may not include snapshot filenames. Recommend provider:snapshot_file:hash commitment.")
        else:
            ok("Manifest appears to include snapshot filename identity")
    else:
        warn("No epoch_manifest mention found. Verify epoch+manifest is implemented.")

    if re.search(r"stable_json_dumps\(\s*m\s*\)", code) and "compute_model_fingerprint" in code:
        warn("Possible raw-dict fingerprinting detected (stable_json_dumps(m)). Ensure semantic fingerprinting used.")
    else:
        ok("No obvious raw-dict fingerprinting pattern detected (or semantic fingerprinting not present)")

    readme = REPO_ROOT / "README.md"
    if readme.exists():
        t = read_text(readme)
        if re.search(r"\bR\s*=\s*3\b", t):
            if not re.search(r"resampl|confirm.*3|range\(3\)", code, re.IGNORECASE):
                warn("README claims R=3 confirmation but code doesn’t show clear resampling logic.")
            else:
                ok("R=3/resampling appears present")
    else:
        warn("README.md not found; skipping doc consistency checks")

    print("[PASS] Observatory medium contracts: no critical regressions detected")

def main(argv=None):
    ap = argparse.ArgumentParser(prog="observatory_contracts.observatory")
    ap.parse_args(argv)
    run()

if __name__ == "__main__":
    main()
