"""Per-browser-session conversation state. Plain data, no logic."""
from __future__ import annotations

from dataclasses import dataclass, field

import config
from tix.schemas import Product


@dataclass
class Session:
    messages: list[dict] = field(default_factory=list)   # {"role","content"[, "products": [card, ...]]}: UI + LLM history
    seen: dict = field(default_factory=dict)             # id -> Product returned by any tool this session (the only ids that may be shown)
    shown: list[Product] = field(default_factory=list)   # products on screen after the last answer
    last_search: dict = field(default_factory=dict)      # arguments of the last product search (for "cheaper"/"more")
    media: dict = field(default_factory=dict)            # id -> {"images": [...], "videos": [...]} learned from product details
    allowed_prices: set = field(default_factory=set)     # every real price seen (+ budgets the model searched with)
    pending_cards: list[dict] = field(default_factory=list)  # cards the model asked to display in the current turn
    found: tuple | None = None                           # (products, unambiguous) of the latest search in the current turn
    user_text: str = ""                                  # the message being answered

    def history(self) -> list[dict]:
        """Recent text-only turns for the LLM (tool traffic of earlier turns is dropped on purpose)."""
        return [{"role": m["role"], "content": m["content"]} for m in self.messages[-config.HISTORY_MESSAGES:]]

    def remember(self, products: list[Product]) -> None:
        for p in products:
            key = p.id if p.id is not None else p.name
            self.seen[key] = p
            for v in (p.price, p.price_before):
                if v is not None:
                    self.allowed_prices.add(float(v))
        while len(self.seen) > 400:  # bounded memory
            self.seen.pop(next(iter(self.seen)))

    def reset(self) -> None:
        self.__init__()
