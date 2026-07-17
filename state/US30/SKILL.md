# SKILL — US30

Lessons from Checker rejections, Validator failures, and kill-switch events.
Maker and the grid optimizer read this before the next search.

## Constitution (Checker reference)

- Reject max drawdown >= 10%.
- Require 1.5 <= Sharpe <= 3.0 (Sharpe > 3.0 → overfitting risk).
- Require return mean p-value < 0.05.
- Require IS/OOS Sharpe degradation <= 30%.
- Always apply round-trip costs with a floor of 10 bps (0.1%) optimism tax.
- Strategy is fixed dual-OLS on M30; Maker may only change long_window / short_window / max_hold_bars.

## Lessons

- (none yet)
