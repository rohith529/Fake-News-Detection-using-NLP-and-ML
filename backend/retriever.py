"""Retriever: fetch live articles from trusted sources via Google News RSS."""
import functools, json, logging, re
from pathlib import Path
from urllib.parse import urlparse

import feedparser, requests
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

SOURCES = json.loads((Path(__file__).parent / "sources.json").read_text())
_log = logging.getLogger(__name__)


def _domain_of(url: str) -> str:
    """Extract the bare domain from a URL, stripping the 'www.' prefix."""
    if not url:
        return ""
    host = urlparse(url if "//" in url else "//" + url).netloc.lower()
    return host[4:] if host.startswith("www.") else host


def _cache_key(claim_text: str) -> tuple:
    """Return the first 10 non-stopword tokens from *claim_text* as a hashable tuple."""
    tokens = re.findall(r"[a-z]+", claim_text.lower())
    key_tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS][:10]
    return tuple(key_tokens)


@functools.lru_cache(maxsize=256)
def _fetch_uncached(key: tuple) -> tuple:
    """Fetch and filter trusted articles for *key*. Returns a tuple of dicts (hashable container)."""
    try:
        query = " ".join(key)
        r = requests.get(
            "https://news.google.com/rss/search",
            timeout=5,
            params=dict(q=query, hl="en-IN", gl="IN", ceid="IN:en"),
        )
        feed = feedparser.parse(r.content)

        results = []
        for e in feed.entries:
            title = e.title.rsplit(" - ", 1)[0].strip()
            link = e.link
            source_url = getattr(e.get("source", {}), "href", "") or link
            domain = _domain_of(source_url)

            reliability = SOURCES.get(domain)
            if reliability is None or reliability < 0.80:
                continue

            results.append({
                "title": title,
                "link": link,
                "source_domain": domain,
                "reliability": reliability,
            })

        return tuple(results[:8])

    except Exception:
        _log.exception("retriever: fetch failed")
        return ()


def fetch(claim_text: str) -> list[dict]:
    """Return up to 8 trusted articles for the claim.

    Each dict: {title: str, link: str, source_domain: str, reliability: float}
    Returns [] on timeout or any exception. Never raises.
    """
    return list(_fetch_uncached(_cache_key(claim_text)))
