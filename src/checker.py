"""Checker: adversarial Anthropic review of Maker parameter candidates."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from .anthropic_client import AnthropicClient, AnthropicClientError
from .maker import MakerCandidate
from .strategy import StrategyParams

logger = logging.getLogger(__name__)

CHECKER_SYSTEM = """You are Checker, an adversarial quant auditor. Your only job is to REJECT
suspicious dual-OLS parameter candidates before they waste a backtest.

You do NOT invent new strategies. You review parameters for:
- Unrealistic windows (too short for M30 noise, or long/short nearly equal)
- Likely data-snooping / overfit seeking (extreme max_hold, odd round-number hunting)
- Violations of SKILL lessons (explicit avoid_* constraints)
- Thin economic rationale

Dual-OLS strategy is FIXED (slope sign agreement → trend; disagreement → mean-reversion).
Price-line distance is ignored by design — do not reject for that.

Output ONLY valid JSON:
{
  "reviews": [
    {
      "long_window": <int>,
      "short_window": <int>,
      "max_hold_bars": <int>,
      "decision": "approve" | "reject",
      "reason": "<concise>"
    }
  ]
}
"""


@dataclass
class CheckerReview:
    params: StrategyParams
    approved: bool
    reason: str = ""


class StrategyChecker:
    def __init__(
        self,
        client: AnthropicClient,
        model: str = "claude-opus-4-8",
    ) -> None:
        self.client = client
        self.model = model

    def review(
        self,
        candidates: list[MakerCandidate],
        *,
        skills_text: str,
    ) -> list[CheckerReview]:
        if not candidates:
            return []

        cached = f"SKILL LESSONS:\n{skills_text}\n"
        payload = [
            {
                "long_window": c.params.long_window,
                "short_window": c.params.short_window,
                "max_hold_bars": c.params.max_hold_bars,
                "rationale": c.rationale,
            }
            for c in candidates
        ]
        user = (
            "Adversarially review each candidate. Prefer reject when uncertain.\n"
            f"CANDIDATES:\n{payload}"
        )
        try:
            raw = self.client.messages_create(
                model=self.model,
                system=CHECKER_SYSTEM,
                user=user,
                cached_context=cached,
                max_tokens=2048,
                temperature=0.0,
            )
            data = self.client.extract_json(raw)
        except (AnthropicClientError, Exception) as exc:
            logger.error("Checker review failed: %s", exc)
            # Fail closed: reject all on API/parse failure
            return [
                CheckerReview(params=c.params, approved=False, reason=f"checker_error: {exc}")
                for c in candidates
            ]

        return self._parse_reviews(data, candidates)

    def _parse_reviews(
        self,
        data: Any,
        candidates: list[MakerCandidate],
    ) -> list[CheckerReview]:
        by_key: dict[tuple[int, int, int], MakerCandidate] = {
            (c.params.long_window, c.params.short_window, c.params.max_hold_bars): c
            for c in candidates
        }
        reviews: list[CheckerReview] = []
        items = []
        if isinstance(data, dict):
            items = data.get("reviews") or []
        if not isinstance(items, list):
            items = []

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
            key = (lw, sw, mh)
            if key not in by_key:
                continue
            decision = str(item.get("decision", "reject")).strip().lower()
            approved = decision in ("approve", "pass", "accepted", "accept")
            reviews.append(
                CheckerReview(
                    params=StrategyParams(long_window=lw, short_window=sw, max_hold_bars=mh),
                    approved=approved,
                    reason=str(item.get("reason") or ""),
                )
            )
            seen.add(key)

        # Any candidate missing from Checker response → reject
        for key, cand in by_key.items():
            if key not in seen:
                reviews.append(
                    CheckerReview(
                        params=cand.params,
                        approved=False,
                        reason="missing from checker response",
                    )
                )
        return reviews
