# AI-Loop-Trade001 — Windows VPS Dual-OLS + Maker/Checker

Windows-VPS CLI system that connects to FxPro MetaTrader 5, trades `#US30`
on M30 with a dual-window OLS slope strategy, sizes with capped risk (cold-start
fixed fraction until enough trades for Kelly), and
self-improves via **Anthropic Maker → Checker → mathematical Validator**
(with nested walk-forward / grid-search fallback).

**Target OS: Windows x86-64.** The official `MetaTrader5` package on PyPI ships
Windows wheels only (no source distribution), so this project is not supported
on Ubuntu/Linux for live MT5 access.

## Architecture

1. **Maker** (`claude-sonnet-4-5` by default) proposes dual-OLS parameter JSON.
2. **Checker** (`claude-opus-4-8` by default) adversarially approves/rejects.
3. **Validator** enforces hard gates (DD, Sharpe band, HAC/bootstrap p-value,
   DSR, PBO, trade-count floors, OOS / nested rolling OOS gate).
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

1. Edit `config.yaml`: `mt5.server` and `mt5.path` (full path to
   `terminal64.exe` if auto-detect fails). Do **not** put login/password in
   `config.yaml` (tracked by Git).
2. Set MT5 credentials via environment / `.env` / gitignored `secrets.yaml`
   (see `.env.example` and `secrets.example.yaml`). Leave them unset to attach
   to an already-logged-in FxPro terminal session.
3. Set `ANTHROPIC_API_KEY` for Maker/Checker (optional; without it, optimize
   falls back to grid search).
4. Keep `EXECUTE: false` until backtests pass. Set `EXECUTE: true` only for demo
   (`account_type: demo`). Live requires `allow_live: true`.

### Secrets (PowerShell)

```powershell
# Option A: environment / .env
copy .env.example .env
# edit .env → ANTHROPIC_API_KEY, MT5_LOGIN, MT5_PASSWORD

# Option B: gitignored YAML next to config
copy secrets.example.yaml secrets.yaml
# edit secrets.yaml → mt5.login / mt5.password

$env:ANTHROPIC_API_KEY = "sk-ant-..."   # if not using .env
cd C:\path\to\AI-Loop-Trade001
.\.venv\Scripts\Activate.ps1
python main.py loop
```

To unlock after a kill-switch lock, set `locked: false` in `state/US30/STATE.md`
manually after reviewing the cause (no automatic unlock). The kill-switch monitor
detects the unlock on the next poll and resumes drawdown watching (it does not
stay stopped in `LOCKED_AND_FLAT`).

### Optional: Task Scheduler

Run `python main.py loop` at logon via Task Scheduler (highest privileges if MT5
needs them). Prefer a logged-on session where the MT5 terminal stays running.
Relative `paths.state_dir` / `loop.log_dir` (and non-empty `mt5.path`) are
resolved against the config file's directory at load time, so an empty Task
Scheduler "Start in" folder cannot create a second `state/` tree.

Only one `main.py` process may run per install: startup takes an OS-level lock
(Windows named mutex + `state/.ai_loop_trade.lock`). A second start exits
immediately so Task Scheduler and a manual launch cannot double-trade.

## CI

GitHub Actions (`.github/workflows/ci.yml`):

| Job | Runner | What it covers |
|-----|--------|----------------|
| Ubuntu pytest | `ubuntu-latest` · Py 3.11/3.12 | Full logic suite; `MetaTrader5` stubbed |
| Windows MT5 wheel compat | `windows-latest` · Py 3.12 | Real `MetaTrader5` wheel import, official constants, `OrderSendResult` / `TradeDeal` / `SymbolInfo` field layouts, `order_send` without a terminal, filling-mode mapping, Windows `os.replace` STATE writes |

Live FxPro terminal connection and broker-specific `filling_mode` discovery are **not** exercised in CI (no terminal / credentials on runners).

Locally (Linux or Windows without the wheel):

```powershell
pip install -r requirements.txt
pytest -q
```

On a Windows VPS with the wheel:

```powershell
pip install -r requirements-windows.txt
pytest -q tests/test_windows_mt5_compat.py
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
| Significance | HAC p-value (unadjusted); DSR/PBO handle selection bias |
| DSR / PBO | Deflated Sharpe ≥ 0.95 (SR* uses **cross-trial** Sharpe dispersion); search PBO ≤ 0.50 |
| Sample size | ≥ 40 full-sample trades; ≥ 15 OOS; ≥ 10 per regime |
| OOS | IS→OOS Sharpe degradation ≤ 30% |
| Costs | Spread once + commission/slippage each way; ≥ 10 bps **round-trip** floor |

Bootstrap p-values are optional diagnostics (`pvalue_method: block_bootstrap|max`).
The block bootstrap uses **fixed-length circular blocks** by default (wrap-around;
no short tail blocks). Do not pair them with Bonferroni at low
`block_bootstrap_reps`: the floor
`1/(n_boot+1)` can sit above `alpha/m`, making every grid candidate impossible.
If you enable a bootstrap gate, use `multiple_testing: none` or raise reps so
`1/(n_boot+1) < alpha/m` (e.g. ≥12 000 for 120 tests at α=0.05). `fdr_bh` now
runs a true Benjamini–Hochberg pass over the search family (not α/m).

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
| `src/risk.py` | Cold-start risk + Kelly (after min trades) + cost model |
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
