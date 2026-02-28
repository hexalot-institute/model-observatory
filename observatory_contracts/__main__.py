from __future__ import annotations
import argparse
from . import forma as forma_mod
from . import observatory as obs_mod

def main(argv=None):
    ap = argparse.ArgumentParser(prog="observatory_contracts", description="Contract verifiers for Model Observatory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("forma", help="FORMA-grade verifier")
    p1.add_argument("--strict", action="store_true", help="Fail on doc↔code mismatches")
    p1.add_argument("--repo-root", default=".", help="Repo root (default: .)")

    p2 = sub.add_parser("observatory", help="Medium verifier")
    p2.add_argument("--repo-root", default=".", help="Repo root (default: .)")

    args = ap.parse_args(argv)
    if args.cmd == "forma":
        forma_args = ["--strict"] if args.strict else []
        forma_mod.main(forma_args)
    else:
        obs_mod.main([])

if __name__ == "__main__":
    main()
