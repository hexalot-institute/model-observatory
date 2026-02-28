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
