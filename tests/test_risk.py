from __future__ import annotations

from src.risk import RiskManager


def test_max_loss_lots_use_tick_value_not_notional():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=0.6,
        default_reward_risk=2.0,
        max_fraction=0.10,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
    )
    # full kelly = 0.6 - 0.4/2 = 0.4 → capped to 0.10
    # equity 10_000 → risk_capital 1000
    # stop = 1% of 40000 = 400; tick 1.0 value 1.0 → 400 loss/lot
    # lots = 1000/400 = 2.5 → floor to 2.5 with step 0.01
    decision = risk.position_lots(
        equity=10_000,
        price=40_000,
        contract_size=1.0,
        volume_step=0.01,
        tick_size=1.0,
        tick_value=1.0,
        point=1.0,
        side_long=True,
        digits=1,
    )
    assert decision.lots == 2.5
    assert decision.stop_loss == 39_600.0
    assert decision.risk_per_lot == 400.0
    assert "max_loss" in decision.message


def test_open_risk_reduces_budget():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=0.20,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=0.20,
    )
    # kelly budget = 0.2 * 10000 = 2000; open_risk 1500 → remaining 500
    # stop dist 100 on price 10000; tick 1/1 → 100/lot → lots=5
    decision = risk.position_lots(
        equity=10_000,
        price=10_000,
        contract_size=1.0,
        tick_size=1.0,
        tick_value=1.0,
        point=1.0,
        open_risk=1_500,
    )
    assert decision.risk_capital == 500.0
    assert decision.lots == 5.0


def test_record_trade_updates_empirical_wr():
    risk = RiskManager(lookback_trades=10)
    for pnl in [10, 10, 10, -5, -5]:
        risk.record_trade(pnl)
    wr, rr = risk.estimate_wr_rr()
    assert wr == 0.6
    assert abs(rr - 2.0) < 1e-9


def test_margin_cap_limits_lots():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=1.0,
        stop_pct=0.001,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        max_margin_fraction=0.5,
        max_lots=100,
    )
    decision = risk.position_lots(
        equity=100_000,
        price=100.0,
        contract_size=1.0,
        tick_size=0.01,
        tick_value=0.01,
        point=0.01,
        free_margin=1_000,
        margin_per_lot=100.0,
        volume_step=0.01,
    )
    # margin allows 1000*0.5/100 = 5 lots
    assert decision.margin_capped is True
    assert decision.lots == 5.0
