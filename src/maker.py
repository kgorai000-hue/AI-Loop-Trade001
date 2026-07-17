"""Maker: Anthropic-driven dual-OLS parameter proposal (no strategy code)."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .anthropic_client import AnthropicClient, AnthropicClientError
from .strategy import StrategyParams

logger = logging.getLogger(__name__)

MAKER_SYSTEM = """You are Maker, a quant parameter explorer for a FIXED dual-window OLS strategy on M30 #US30.

Strategy rules (immutable -- do NOT invent new indicators or code):
- OLS slope on close: b_long over long_window bars, b_short over short_window bars.
- Same sign -> trend-follow sign(b_long). Opposite sign -> mean-revert against b_short.
- Price distance to the regression line is IGNORED.

Your job: propose JSON parameter candidates that explore the search space while
respecting past SKILL lessons (avoid known failure modes).

Output ONLY valid JSON (no markdown prose) with this schema:
{
  "candidates": [
    {
      "long_window": <int>,
      "short_window": <int>,
      "max_hold_bars": <int>,
      "rationale": "<short reason>"
    }
  ]
}

Hard constraints:
- short_window < long_window
- long_window within allowed long range
- short_window within allowed short range
- max_hold_bars within allowed hold range
- Do not emit Python code or new strategy logic.
"""


@dataclass
class MakerCandidate:
    params: StrategyParams
    rationale: str = ""


class StrategyMaker:
    def __init__(
        self,
        client: AnthropicClient,
        model: str = "claude-sonnet-4-5",
        n_candidates: int = 8,
        long_range: Optional[tuple[int, int]] = None,
        short_range: Optional[tuple[int, int]] = None,
        hold_range: Optional[tuple[int, int]] = None,
    ) -> None:
        self.client = client
        self.model = model
        self.n_candidates = max(1, int(n_candidates))
        self.long_range = long_range or (180, 280)
        self.short_range = short_range or (24, 72)
        self.hold_range = hold_range or (8, 24)

    def propose(
        self,
        *,
        current_params: StrategyParams,
        last_metrics: dict[str, Any],
        skills_text: str,
    ) -> list[MakerCandidate]:
        cached = (
            f"SKILL LESSONS:\n{skills_text}\n\n"
            f"ALLOWED RANGES:\n"
            f"- long_window: [{self.long_range[0]}, {self.long_range[1]}]\n"
            f"- short_window: [{self.short_range[0]}, {self.short_range[1]}]\n"
            f"- max_hold_bars: [{self.hold_range[0]}, {self.hold_range[1]}]\n"
        )
        user = (
            f"Propose exactly {self.n_candidates} distinct dual-OLS parameter candidates.\n"
            f"Current params: {current_params.as_dict()}\n"
            f"Last metrics: {last_metrics}\n"
            "Favor diversity across the allowed ranges while avoiding SKILL failure modes."
        )
        try:
            raw = self.client.messages_create(
                model=self.model,
                system=MAKER_SYSTEM,
                user=user,
                cached_context=cached,
                max_tokens=2048,
                temperature=0.4,
            )
            data = self.client.extract_json(raw)
        except (AnthropicClientError, Exception) as exc:
            logger.error("Maker propose failed: %s", exc)
            return []

        return self._parse_candidates(data)

    def _parse_candidates(self, data: Any) -> list[MakerCandidate]:
        if not isinstance(data, dict):
            return []
        items = data.get("candidates")
        if not isinstance(items, list):
            return []

        out: list[MakerCandidate] = []
        seen: set[tuple[int, int, int]] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            try:
                lw = int(item["long_window"])
                sw = int(item["short_window"])
                mh = int(item["max_hold_bars"])
            except (KeyError, TypeError, ValueError):
                continue
            if not self._in_range(lw, sw, mh):
                logger.info("Maker candidate out of range: %s", item)
                continue
            key = (lw, sw, mh)
            if key in seen:
                continue
            seen.add(key)
            out.append(
                MakerCandidate(
                    params=StrategyParams(long_window=lw, short_window=sw, max_hold_bars=mh),
                    rationale=str(item.get("rationale") or ""),
                )
            )
        return out

    def _in_range(self, lw: int, sw: int, mh: int) -> bool:
        if sw >= lw:
            return False
        if not (self.long_range[0] <= lw <= self.long_range[1]):
            return False
        if not (self.short_range[0] <= sw <= self.short_range[1]):
            return False
        if not (self.hold_range[0] <= mh <= self.hold_range[1]):
            return False
        return True
