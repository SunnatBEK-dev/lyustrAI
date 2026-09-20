from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from ai_sdk.agents.coordination import CoordinationError


class AICapability(str, Enum):
    CONTEXT = "context"
    REASONING = "reasoning"
    SYNTHESIS = "synthesis"


class MultiModelRoute(str, Enum):
    FAST = "fast"
    CONTEXT = "context"
    REASONING = "reasoning"
    FULL = "full"


class RoutingSignal(str, Enum):
    USER_OVERRIDE = "user_override"
    RETRIEVAL_CONTEXT = "retrieval_context"
    CONTEXT_KEYWORD = "context_keyword"
    REASONING_KEYWORD = "reasoning_keyword"
    MULTI_PART = "multi_part"
    LONG_REQUEST = "long_request"


_ROUTE_CAPABILITIES = {
    MultiModelRoute.FAST: (AICapability.SYNTHESIS,),
    MultiModelRoute.CONTEXT: (
        AICapability.CONTEXT,
        AICapability.SYNTHESIS,
    ),
    MultiModelRoute.REASONING: (
        AICapability.REASONING,
        AICapability.SYNTHESIS,
    ),
    MultiModelRoute.FULL: (
        AICapability.CONTEXT,
        AICapability.REASONING,
        AICapability.SYNTHESIS,
    ),
}

_ROUTE_MODEL_REQUESTS = {
    MultiModelRoute.FAST: 1,
    MultiModelRoute.CONTEXT: 2,
    MultiModelRoute.REASONING: 2,
    MultiModelRoute.FULL: 3,
}


@dataclass(frozen=True, init=False)
class RoutingDecision:
    route: MultiModelRoute
    capabilities: tuple[AICapability, ...]
    signals: tuple[RoutingSignal, ...]

    def __init__(
        self,
        route: MultiModelRoute,
        signals: tuple[RoutingSignal, ...] = (),
    ) -> None:
        if not isinstance(route, MultiModelRoute):
            raise TypeError("Adaptive Multi-Model route is invalid.")
        normalized_signals = tuple(signals)
        if any(not isinstance(signal, RoutingSignal) for signal in normalized_signals):
            raise TypeError("Routing signals are invalid.")
        if len(normalized_signals) != len(set(normalized_signals)):
            raise CoordinationError("Routing signals must be unique.")
        object.__setattr__(self, "route", route)
        object.__setattr__(
            self,
            "capabilities",
            _ROUTE_CAPABILITIES[route],
        )
        object.__setattr__(
            self,
            "signals",
            normalized_signals,
        )

    @property
    def estimated_model_requests(self) -> int:
        return _ROUTE_MODEL_REQUESTS[self.route]


class CapabilityRouter:
    """Choose a bounded Adaptive Multi-Model workflow without another model call."""

    _CONTEXT_MARKERS = (
        "retrieved context:",
        "manba konteksti:",
    )
    _CONTEXT_TERMS = frozenset(
        {
            "citation",
            "citations",
            "context",
            "dalil",
            "document",
            "documents",
            "evidence",
            "fact",
            "facts",
            "fakt",
            "faktlar",
            "hujjat",
            "hujjatlar",
            "iqtibos",
            "kontekst",
            "manba",
            "manbalar",
            "research",
            "source",
            "sources",
            "tadqiqot",
            "tekshir",
            "verification",
            "verify",
        }
    )
    _CONTEXT_PREFIXES = (
        "dalil",
        "fakt",
        "hujjat",
        "iqtibos",
        "kontekst",
        "manba",
        "tadqiq",
        "tekshir",
    )
    _REASONING_TERMS = frozenset(
        {
            "analysis",
            "analyze",
            "architecture",
            "arxitektura",
            "compare",
            "comparison",
            "chuqur",
            "design",
            "isbot",
            "loyiha",
            "nega",
            "plan",
            "proof",
            "reason",
            "reasoning",
            "reja",
            "sabab",
            "solution",
            "solve",
            "taqqosla",
            "taqqoslash",
            "tahlil",
            "tradeoff",
            "yechim",
        }
    )
    _REASONING_PREFIXES = (
        "reja",
        "tahlil",
        "taqqos",
        "yech",
    )
    _NUMBERED_ITEM = re.compile(r"(?m)^\s*\d+[.)]\s+")
    _WORD = re.compile(r"\w+", re.UNICODE)

    def __init__(
        self,
        *,
        long_request_chars: int = 800,
        max_analyzed_chars: int = 20_000,
        forced_route: MultiModelRoute | None = None,
    ) -> None:
        for value, label in (
            (long_request_chars, "Long request threshold"),
            (max_analyzed_chars, "Maximum analyzed characters"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{label} must be positive.")
        if max_analyzed_chars < long_request_chars:
            raise ValueError(
                "Maximum analyzed characters cannot be below "
                "the long request threshold."
            )
        if forced_route is not None and not isinstance(
            forced_route,
            MultiModelRoute,
        ):
            raise TypeError("Forced Adaptive Multi-Model route is invalid.")
        self.long_request_chars = long_request_chars
        self.max_analyzed_chars = max_analyzed_chars
        self.forced_route = forced_route

    def route(self, request: str) -> RoutingDecision:
        if not isinstance(request, str) or not request.strip():
            raise ValueError("Routing request cannot be empty.")
        if self.forced_route is not None:
            return RoutingDecision(
                self.forced_route,
                (RoutingSignal.USER_OVERRIDE,),
            )

        normalized = request.strip().casefold()
        analyzed = normalized[: self.max_analyzed_chars]
        words = set(self._WORD.findall(analyzed))
        signals: list[RoutingSignal] = []

        has_retrieval_context = any(
            marker in analyzed for marker in self._CONTEXT_MARKERS
        )
        has_context_keyword = bool(
            words.intersection(self._CONTEXT_TERMS)
            or self._has_prefix(
                words,
                self._CONTEXT_PREFIXES,
            )
        )
        has_reasoning_keyword = bool(
            words.intersection(self._REASONING_TERMS)
            or self._has_prefix(
                words,
                self._REASONING_PREFIXES,
            )
        )
        is_multi_part = (
            analyzed.count("?") >= 2 or len(self._NUMBERED_ITEM.findall(analyzed)) >= 2
        )
        is_long = len(normalized) >= self.long_request_chars

        if has_retrieval_context:
            signals.append(RoutingSignal.RETRIEVAL_CONTEXT)
        if has_context_keyword:
            signals.append(RoutingSignal.CONTEXT_KEYWORD)
        if has_reasoning_keyword:
            signals.append(RoutingSignal.REASONING_KEYWORD)
        if is_multi_part:
            signals.append(RoutingSignal.MULTI_PART)
        if is_long:
            signals.append(RoutingSignal.LONG_REQUEST)

        needs_context = has_retrieval_context or has_context_keyword or is_long
        needs_reasoning = has_reasoning_keyword or is_multi_part or is_long
        if needs_context and needs_reasoning:
            selected = MultiModelRoute.FULL
        elif needs_context:
            selected = MultiModelRoute.CONTEXT
        elif needs_reasoning:
            selected = MultiModelRoute.REASONING
        else:
            selected = MultiModelRoute.FAST

        return RoutingDecision(selected, tuple(signals))

    @staticmethod
    def _has_prefix(
        words: set[str],
        prefixes: tuple[str, ...],
    ) -> bool:
        return any(word.startswith(prefix) for word in words for prefix in prefixes)
