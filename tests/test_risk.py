from __future__ import annotations

from types import SimpleNamespace

from src.risk import CostModel, RiskManager


def test_round_trip_floor_is_not_doubled():
    """10 bps config is a round-trip floor, not 10bps each way."""
    cost = CostModel(round_trip_floor=0.001)
    assert abs(cost.round_trip_fraction() - 0.001) < 1e-12
    assert abs(cost.one_way_fraction() - 0.0005) < 1e-12


def test_spread_counted_once_per_round_trip():
    cost = CostModel(
        spread_fraction=0.0004,  # 4 bps full spread
        commission_one_way=0.0001,  # 1 bps each side
        slippage_one_way=0.0,
        round_trip_floor=0.0,
    )
    # RT = 4bps + 2*1bps = 6bps; not 2*(4+1)=10bps
    assert abs(cost.round_trip_fraction() - 0.0006) < 1e-12
    assert abs(cost.one_way_fraction() - 0.0003) < 1e-12


def test_from_symbol_info_does_not_double_spread():
    info = SimpleNamespace(
        spread=4.0,
        point=1.0,
        trade_contract_size=1.0,
        bid=40_000.0,
        ask=40_004.0,
    )
    tick = SimpleNamespace(bid=40_000.0, ask=40_004.0)
    cost = CostModel.from_symbol_info(
        info,
        tick=tick,
        risk_cfg={"round_trip_floor_bps": 0, "commission_one_way_bps": 0},
    )
    # mid=40002; spread frac = 4/40002
    expected_spread = 4.0 / 40_002.0
    assert abs(cost.spread_fraction - expected_spread) < 1e-12
    assert abs(cost.round_trip_fraction() - expected_spread) < 1e-12


def test_from_risk_config_accepts_min_cost_bps_alias():
    cost = CostModel.from_risk_config({"min_cost_bps": 10})
    assert abs(cost.round_trip_floor - 0.001) < 1e-12


def test_max_loss_lots_use_tick_value_not_notional():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=0.6,
        default_reward_risk=2.0,
        max_fraction=0.10,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        kelly_min_trades=0,
    )
    # full kelly = 0.6 - 0.4/2 = 0.4 -> capped to 0.10
    # equity 10_000 -> risk_capital 1000
    # stop = 1% of 40000 = 400; tick 1.0 value 1.0 -> 400 loss/lot
    # lots = 1000/400 = 2.5 -> floor to 2.5 with step 0.01
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
        kelly_min_trades=0,
    )
    # kelly budget = 0.2 * 10000 = 2000; open_risk 1500 -> remaining 500
    # stop dist 100 on price 10000; tick 1/1 -> 100/lot -> lots=5
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
        kelly_min_trades=0,
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


def test_live_path_blocks_notional_fallback_without_ticks():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=0.10,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        kelly_min_trades=0,
    )
    decision = risk.position_lots(
        equity=10_000,
        price=40_000,
        contract_size=1.0,
        tick_size=0.0,
        tick_value=0.0,
        allow_notional_fallback=False,
    )
    assert decision.lots == 0.0
    assert "notional fallback disabled" in decision.message


def test_backtest_may_use_notional_fallback():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=1.0,
        default_reward_risk=1.0,
        max_fraction=0.10,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
        max_lots=100,
        kelly_min_trades=0,
    )
    decision = risk.position_lots(
        equity=10_000,
        price=40_000,
        contract_size=1.0,
        tick_size=None,
        tick_value=None,
        volume_step=0.01,
        allow_notional_fallback=True,
        win_rate=1.0,
        reward_risk=1.0,
    )
    assert decision.lots > 0
    assert "notional_fallback" in decision.message


def test_cold_start_uses_fixed_fraction_not_default_kelly():
    """Unlearned defaults must not size at ~10% half-Kelly."""
    risk = RiskManager(
        half_kelly=True,
        default_win_rate=0.52,
        default_reward_risk=1.5,
        max_fraction=0.005,
        cold_start_fraction=0.005,
        kelly_min_trades=30,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=0.01,
    )
    frac, mode = risk.sizing_fraction()
    assert mode == "cold_start"
    assert abs(frac - 0.005) < 1e-12
    # Half-Kelly on defaults would be ~0.10 before cap -- must not be used yet.
    assert risk.kelly_fraction() == 0.005  # capped
    assert not risk.kelly_ready()

    decision = risk.position_lots(
        equity=10_000,
        price=40_000,
        contract_size=1.0,
        tick_size=1.0,
        tick_value=1.0,
        point=1.0,
        volume_step=0.01,
    )
    assert "cold_start" in decision.message
    # risk_capital <= 0.5% of equity (and open-risk cap 1%)
    assert decision.risk_capital <= 50.0 + 1e-9


def test_kelly_activates_after_min_trades():
    risk = RiskManager(
        half_kelly=False,
        default_win_rate=0.6,
        default_reward_risk=2.0,
        max_fraction=0.10,
        cold_start_fraction=0.005,
        kelly_min_trades=5,
        lookback_trades=50,
        stop_pct=0.01,
        gap_buffer_mult=1.0,
        max_open_risk_fraction=1.0,
    )
    for pnl in [10, 10, 10, -5, -5]:
        risk.record_trade(pnl)
    assert risk.kelly_ready()
    frac, mode = risk.sizing_fraction()
    assert mode == "kelly"
    # Empirical W=0.6 R=2 -> full Kelly = 0.4 capped to 0.10
    assert abs(frac - 0.10) < 1e-12


def test_buffered_stop_distance_applies_gap_mult():
    risk = RiskManager(stop_pct=0.01, gap_buffer_mult=1.25)
    base = risk.stop_distance_price(40_000.0, point=1.0)
    assert abs(base - 400.0) < 1e-9
    assert abs(risk.buffered_stop_distance(40_000.0, point=1.0) - 500.0) < 1e-9
