"""
Source registry. One place that knows every adapter, so the retriever can build
the active set from names without importing each module by hand.
"""

from .base import Source
from .alpaca_news import AlpacaNews
from .rss_news import RSSNews
from .edgar_filings import EdgarFilings
from .yfinance_analyst import YFinanceAnalyst
from .gdelt import Gdelt

# name -> factory (zero-arg). Order here is the default merge order.
REGISTRY = {
    AlpacaNews.name: AlpacaNews,
    RSSNews.name: RSSNews,
    EdgarFilings.name: EdgarFilings,
    YFinanceAnalyst.name: YFinanceAnalyst,
    Gdelt.name: Gdelt,
}


def all_names() -> list[str]:
    return list(REGISTRY)


def default_names() -> list[str]:
    return [n for n, cls in REGISTRY.items() if cls.default_on]


def build(names=None) -> list[Source]:
    """Instantiate sources by name (defaults to every default_on source)."""
    if names is None:
        names = default_names()
    out = []
    for n in names:
        if n not in REGISTRY:
            raise ValueError(f"Unknown source '{n}'. Known: {', '.join(REGISTRY)}")
        out.append(REGISTRY[n]())
    return out
