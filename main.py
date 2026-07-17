#!/usr/bin/env python3
"""CLI entry for the FxPro MT5 dual-OLS + Maker/Checker autonomous loop."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.loop_engine import LoopEngine


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError("config root must be a mapping")
    return cfg


def setup_logging(log_dir: str | Path) -> None:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "loop.log"
    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    handlers: list[logging.Handler] = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_file, encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)


def _pick_trader(engine: LoopEngine, symbol: str | None):
    if symbol is None:
        return engine.traders[0]
    for t in engine.traders:
        if t.symbol == symbol or t.cfg.state_key.upper() == symbol.lstrip("#").upper():
            return t
    raise SystemExit(f"Symbol not found in config: {symbol}")


def cmd_once(engine: LoopEngine, args: argparse.Namespace) -> int:
    if not engine.start_connection():
        logging.error("MT5 connection failed")
        return 1
    try:
        trader = _pick_trader(engine, args.symbol)
        decision = trader.evaluate()
        if decision is None:
            print(json.dumps({"ok": False, "error": "no decision"}, indent=2))
            return 2
        out = trader.maybe_trade(decision)
        print(json.dumps(out, indent=2, default=str))
        return 0
    finally:
        engine.stop()


def cmd_backtest(engine: LoopEngine, args: argparse.Namespace) -> int:
    if not engine.start_connection():
        logging.error("MT5 connection failed")
        return 1
    try:
        trader = _pick_trader(engine, args.symbol)
        months = args.months or int(engine.config.get("validator", {}).get("lookback_months", 6))
        out = trader.backtest_and_validate(months=months)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 2
    finally:
        engine.stop()


def cmd_optimize(engine: LoopEngine, args: argparse.Namespace) -> int:
    if not engine.start_connection():
        logging.error("MT5 connection failed")
        return 1
    try:
        trader = _pick_trader(engine, args.symbol)
        months = args.months or int(engine.config.get("validator", {}).get("lookback_months", 6))
        out = trader.optimize(months=months)
        print(json.dumps(out, indent=2, default=str))
        return 0 if out.get("ok") else 2
    finally:
        engine.stop()


def cmd_loop(engine: LoopEngine, args: argparse.Namespace) -> int:
    engine.run_forever()
    return 0


def cmd_review(engine: LoopEngine, args: argparse.Namespace) -> int:
    if not engine.start_connection():
        logging.error("MT5 connection failed")
        return 1
    try:
        out = engine.review_subloop()
        print(json.dumps(out, indent=2, default=str))
        return 0
    finally:
        engine.stop()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="FxPro MT5 dual OLS + Anthropic Maker/Checker autonomous loop"
    )
    p.add_argument("--config", default=str(ROOT / "config.yaml"), help="Path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    once = sub.add_parser("once", help="Evaluate signal and optionally place limit order")
    once.add_argument("--symbol", default=None)
    once.set_defaults(func=cmd_once)

    bt = sub.add_parser("backtest", help="Backtest + validator gates")
    bt.add_argument("--symbol", default=None)
    bt.add_argument("--months", type=int, default=None)
    bt.set_defaults(func=cmd_backtest)

    opt = sub.add_parser(
        "optimize",
        help="Maker→Checker→Validator search (grid fallback if needed)",
    )
    opt.add_argument("--symbol", default=None)
    opt.add_argument("--months", type=int, default=None)
    opt.set_defaults(func=cmd_optimize)

    loop = sub.add_parser("loop", help="Resident M30 + weekend review loop")
    loop.set_defaults(func=cmd_loop)

    rev = sub.add_parser("review", help="Force weekend review sub-loop once")
    rev.set_defaults(func=cmd_review)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    log_dir = cfg.get("loop", {}).get("log_dir", "logs")
    setup_logging(ROOT / log_dir if not Path(log_dir).is_absolute() else log_dir)
    engine = LoopEngine(cfg)
    return int(args.func(engine, args))


if __name__ == "__main__":
    raise SystemExit(main())
