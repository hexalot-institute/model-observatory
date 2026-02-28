from __future__ import annotations
import argparse
from . import forma as forma_mod
from . import observatory as obs_mod

def main(argv=None):
    ap = argparse.ArgumentParser(prog="observatory_contracts", description="Contract verifiers for Model Observatory.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("forma", help="FORMA-grade verifier")
    p1.add_argument("--strict", action="store_true", help="Fail on doc↔code mismatches")

    sub.add_parser("observatory", help="Medium verifier")

    args = ap.parse_args(argv)
    if args.cmd == "forma":
        forma_mod.main(["--strict"] if args.strict else [])
    else:
        obs_mod.main([])

if __name__ == "__main__":
    main()
