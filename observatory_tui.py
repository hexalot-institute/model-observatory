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
