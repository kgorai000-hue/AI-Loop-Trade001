# AI-Loop-Trade001 — Windows VPS Dual-OLS + Maker/Checker

Windows-VPS CLI system that connects to FxPro MetaTrader 5, trades `#US30`
on M30 with a dual-window OLS slope strategy, sizes with half-Kelly, and
self-improves via **Anthropic Maker → Checker → mathematical Validator**
(with nested walk-forward / grid-search fallback).

**Target OS: Windows x86-64.** The official `MetaTrader5` package on PyPI ships
Windows wheels only (no source distribution), so this project is not supported
on Ubuntu/Linux for live MT5 access.

## Architecture

1. **Maker** (`claude-sonnet-4-5` by default) proposes dual-OLS parameter JSON.
2. **Checker** (`claude-opus-4-8` by default) adversarially approves/rejects.
3. **Validator** enforces hard gates (DD, Sharpe band, HAC/bootstrap p-value,
   DSR, PBO, trade-count floors, OOS / nested holdout).
4. Accepted params persist in `state/US30/STATE.md`; failures append to `SKILL.md`.
5. **KillSwitchMonitor** thread flattens and locks if account DD ≥ 10%.

## Setup (Windows VPS)

Requirements:

- Windows Server / Windows 10+ (x86-64)
- FxPro MetaTrader 5 terminal installed and logged in (demo recommended)
- Python 3.10+ (64-bit)

```powershell
cd C:\path\to\AI-Loop-Trade001
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements-windows.txt
```

`requirements.txt` pins core scientific / Anthropic / pytest deps (used by CI on
Linux). `requirements-windows.txt` adds the Windows-only `MetaTrader5` wheel.

1. Edit `config.yaml`: `mt5.login`, `mt5.password`, `mt5.server`, and
   `mt5.path` (full path to `terminal64.exe` if auto-detect fails).
2. Set `ANTHROPIC_API_KEY` for Maker/Checker (optional; without it, optimize
   falls back to grid search).
3. Keep `EXECUTE: false` until backtests pass. Set `EXECUTE: true` only for demo
   (`account_type: demo`). Live requires `allow_live: true`.

### Environment variable (PowerShell)

```powershell
$env:ANTHROPIC_API_KEY = "sk-ant-..."
cd C:\path\to\AI-Loop-Trade001
.\.venv\Scripts\Activate.ps1
python main.py loop
```

To unlock after a kill-switch lock, set `locked: false` in `state/US30/STATE.md`
manually after reviewing the cause (no automatic unlock).

### Optional: Task Scheduler

Run `python main.py loop` at logon via Task Scheduler (highest privileges if MT5
needs them). Prefer a logged-on session where the MT5 terminal stays running.

## CI

GitHub Actions (`.github/workflows/ci.yml`) runs `pytest` on Ubuntu for Python
3.11 and 3.12 using `requirements.txt`. The real `MetaTrader5` package is not
installed in CI; `tests/conftest.py` provides a stub.

Locally:

```powershell
pip install -r requirements.txt
pytest -q
```

## CLI

```powershell
# One-shot signal evaluation (no resident loop)
python main.py once --symbol "#US30"

# Backtest last 6 months against validator gates
python main.py backtest --symbol "#US30"

# Maker→Checker→Validator (nested / grid fallback if needed)
python main.py optimize --symbol "#US30"

# Resident loop: M30 bars + kill-switch + weekend review
python main.py loop

# Force weekend review once
python main.py review
```

## Validator constitution (defaults)

| Gate | Rule |
|------|------|
| Max drawdown | < 10% |
| Sharpe | 1.5 ≤ Sharpe ≤ 3.0 |
| Significance | HAC / block-bootstrap p-value (Bonferroni vs candidate count) |
| DSR / PBO | Deflated Sharpe ≥ 0.95; search PBO ≤ 0.50 |
| Sample size | ≥ 40 full-sample trades; ≥ 15 OOS; ≥ 10 per regime |
| OOS | IS→OOS Sharpe degradation ≤ 30% |
| Costs | Spread once + commission/slippage each way; ≥ 10 bps **round-trip** floor |

## Layout

| Path | Role |
|------|------|
| `src/connection.py` | MT5 worker thread / initialize / reconnect |
| `src/data.py` | M30 OHLC feed (closed bars only) |
| `src/strategy.py` | Dual OLS slope signals |
| `src/metrics.py` | Sharpe, DD, IC, OOS degradation |
| `src/inference.py` | HAC, bootstrap, DSR, PBO |
| `src/backtest.py` | Limit fills + account sizing backtester |
| `src/validator.py` | Rejection gates |
| `src/search.py` | Ranking rows + PBO search gate |
| `src/risk.py` | Half-Kelly + cost model |
| `src/execution.py` | Limit orders + kill flatten |
| `src/optimizer.py` | Grid search fallback |
| `src/splits.py` | Holdout / walk-forward splits |
| `src/anthropic_client.py` | Backoff + prompt cache |
| `src/maker.py` / `src/checker.py` | LLM intelligence layer |
| `src/intelligence.py` | Maker→Checker→Validator orchestrator |
| `src/kill_switch.py` | DD monitor thread |
| `src/persistence.py` | `STATE.md` / `SKILL.md` |
| `src/symbol_trader.py` | Per-symbol orchestration |
| `src/loop_engine.py` | Poll + weekend review |
| `state/US30/` | Per-symbol persistence |

## Strategy (summary)

- OLS slope on close: `b_long` (default 240 bars), `b_short` (48 bars).
- Same sign → trend follow `sign(b_long)`.
- Opposite sign → mean-reversion against `b_short`.
- Price–line distance is ignored; slope relationship only.
- Maker never emits strategy code — only `long_window` / `short_window` / `max_hold_bars`.
