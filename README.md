# AI-Loop-Trade001 — VPS Dual-OLS + Maker/Checker

Ubuntu-VPS friendly CLI system that connects to FxPro MetaTrader 5, trades `#US30`
on M30 with a dual-window OLS slope strategy, sizes with half-Kelly, and
self-improves via **Anthropic Maker → Checker → mathematical Validator**
(with grid-search fallback).

## Architecture

1. **Maker** (`claude-sonnet-4-5` by default) proposes dual-OLS parameter JSON.
2. **Checker** (`claude-opus-4` by default) adversarially approves/rejects.
3. **Validator** enforces hard gates (DD, Sharpe band, p-value, OOS, costs).
4. Accepted params persist in `state/US30/STATE.md`; failures append to `SKILL.md`.
5. **KillSwitchMonitor** thread flattens and locks if account DD ≥ 10%.

## Setup (Ubuntu VPS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

1. Install FxPro MT5 terminal and log in (demo recommended).
2. Edit `config.yaml`: `mt5.login`, `mt5.password`, `mt5.server`, optional `mt5.path`.
3. Export `ANTHROPIC_API_KEY` for Maker/Checker (optional; without it, optimize falls back to grid search).
4. Keep `EXECUTE: false` until backtests pass. Set `EXECUTE: true` only for demo
   (`account_type: demo`). Live requires `allow_live: true`.

### Headless / systemd sketch

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cd /path/to/AI-Loop-Trade001
source .venv/bin/activate
python main.py loop
```

To unlock after a kill-switch lock, set `locked: false` in `state/US30/STATE.md`
manually after reviewing the cause (no automatic unlock).

## CLI

```bash
# One-shot signal evaluation (no resident loop)
python main.py once --symbol "#US30"

# Backtest last 6 months against validator gates
python main.py backtest --symbol "#US30"

# Maker→Checker→Validator (grid fallback if needed)
python main.py optimize --symbol "#US30"

# Resident loop: M30 bars + kill-switch + weekend review
python main.py loop

# Force weekend review once
python main.py review
```

## Validator constitution

| Gate | Rule |
|------|------|
| Max drawdown | < 10% |
| Sharpe | 1.5 ≤ Sharpe ≤ 3.0 |
| Significance | p-value < 0.05 |
| OOS | IS→OOS Sharpe degradation ≤ 30% |
| Costs | Spread/commission or ≥ 10 bps floor |

## Layout

| Path | Role |
|------|------|
| `src/connection.py` | MT5 initialize / reconnect |
| `src/data.py` | M30 OHLC feed |
| `src/strategy.py` | Dual OLS slope signals |
| `src/metrics.py` | Sharpe, DD, p-value, IC, OOS |
| `src/backtest.py` | Event-driven backtester + costs |
| `src/validator.py` | Rejection gates |
| `src/risk.py` | Half-Kelly + cost model |
| `src/execution.py` | Limit orders + kill flatten |
| `src/optimizer.py` | Grid search fallback |
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
