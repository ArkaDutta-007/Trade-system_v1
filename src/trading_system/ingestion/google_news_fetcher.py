"""Google News RSS + article-content fetcher.

This module ports the useful parts of the older standalone news fetcher into
the project ingestion layer:

- Google News RSS search with date filters (free, no API key)
- async URL discovery across many tickers
- redirect decoding for news.google.com article links
- full article-body extraction using BeautifulSoup heuristics

The output is plain Python dict rows so `news_events.fetch_news()` can map them
into the project EVENT_SCHEMA.
"""
from __future__ import annotations

import asyncio
import base64
import json
import re
import ssl
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Iterable
from urllib.parse import quote, quote_plus, urlparse
import xml.etree.ElementTree as ET

import aiohttp
from bs4 import BeautifulSoup
import certifi

from ..utils import get_logger

logger = get_logger(__name__)

_GOOGLE_RSS = "https://news.google.com/rss/search"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
_URL_SEMAPHORE = asyncio.Semaphore(20)
_ARTICLE_SEMAPHORE = asyncio.Semaphore(10)


def _normalize_ws(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def _decode_google_news_url(link: str) -> str:
    """Decode `news.google.com/rss/articles/...` redirect links when possible."""
    if "news.google.com" not in link:
        return link

    try:
        path = urlparse(link).path
        token = path.rstrip("/").split("/")[-1]
        if not token:
            return link

        padding = (4 - len(token) % 4) % 4
        decoded = base64.urlsafe_b64decode(token + ("=" * padding)).decode("latin-1")
        decoded = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]", "", decoded)
        http_idx = decoded.find("http")
        if http_idx == -1:
            return link
        decoded = decoded[http_idx:]
        # If the string contains duplicated schema prefixes, keep the second one.
        tag = decoded.split(":", 1)[0]
        repeats = [m.start() for m in re.finditer(tag, decoded)]
        if len(repeats) > 1:
            decoded = decoded[repeats[1]:]
        return decoded or link
    except Exception:
        return link


def _google_news_rss_url(
    ticker: str,
    after_date: str,
    before_date: str,
    language: str = "en",
    country: str = "US",
) -> str:
    query = f'"{ticker}" after:{after_date} before:{before_date}'
    encoded = quote_plus(query)
    return (
        f"{_GOOGLE_RSS}?q={encoded}"
        f"&ceid={country}:{language}&hl={language}-{country}&gl={country}"
    )


def _parse_rss_entries(xml_text: str, ticker: str, max_urls: int) -> list[dict]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    rows: list[dict] = []
    for item in root.findall("./channel/item")[:max_urls]:
        title = _normalize_ws(item.findtext("title"))
        link = _normalize_ws(item.findtext("link"))
        pub = _normalize_ws(item.findtext("pubDate"))
        # We don't try to mock-decode here anymore. Use the original link for downstream decoding.
        source_url = link
        source_node = item.find("source")
        publisher_name = _normalize_ws(source_node.text if source_node is not None else "")
        fetch_url = link
        
        try:
            published_at = parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
        except Exception:
            published_at = None
        rows.append(
            {
                "ticker": ticker.upper(),
                "title": title,
                "source_url": source_url,
                "published_at": published_at,
                "publisher_name": publisher_name,
                "fetch_url": fetch_url,
            }
        )
    return rows


async def _fetch_rss_for_ticker(
    session: aiohttp.ClientSession,
    ticker: str,
    days: int,
    max_urls: int,
) -> list[dict]:
    now = datetime.now(timezone.utc)
    after_date = (now - timedelta(days=days)).date().isoformat()
    before_date = (now + timedelta(days=1)).date().isoformat()
    url = _google_news_rss_url(ticker, after_date, before_date)

    async with _URL_SEMAPHORE:
        async with session.get(url, timeout=20) as resp:
            if resp.status != 200:
                logger.debug(f"Google RSS failed for {ticker}: status={resp.status}")
                return []
            text = await resp.text()
    return _parse_rss_entries(text, ticker, max_urls=max_urls)


def _extract_article_text(html: str, fallback_title: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside", "form"]):
        tag.decompose()

    meta_title = soup.find("meta", attrs={"property": "og:title"})
    if meta_title and meta_title.get("content"):
        title = _normalize_ws(meta_title["content"])
    elif soup.title and soup.title.string:
        title = _normalize_ws(soup.title.string)
    else:
        title = fallback_title

    selector_order = [
        "article",
        "main",
        "[role='main']",
        ".article-body",
        ".story-body",
        ".entry-content",
        ".post-content",
    ]

    best_text = ""
    for selector in selector_order:
        try:
            nodes = soup.select(selector)
        except Exception:
            nodes = []
        for node in nodes:
            paras = [_normalize_ws(p.get_text(" ", strip=True)) for p in node.find_all("p")]
            paras = [p for p in paras if len(p) >= 40]
            text = _normalize_ws(" ".join(paras))
            if len(text) > len(best_text):
                best_text = text

    if not best_text:
        paras = [_normalize_ws(p.get_text(" ", strip=True)) for p in soup.find_all("p")]
        paras = [p for p in paras if len(p) >= 40]
        best_text = _normalize_ws(" ".join(paras))

    if not best_text:
        meta_desc = soup.find("meta", attrs={"name": "description"})
        best_text = _normalize_ws(meta_desc.get("content") if meta_desc else "")

    return title[:500], best_text[:8000]


async def _resolve_google_news_url_async(session: aiohttp.ClientSession, row: dict) -> dict:
    out = dict(row)
    url = out.get("source_url")
    if not url or "news.google.com" not in url:
        out["fetch_url"] = url
        return out

    try:
        path = urlparse(url).path
        token = path.rstrip("/").split("/")[-1]

        async with _ARTICLE_SEMAPHORE:
            async with session.get(url, allow_redirects=True, timeout=10) as r:
                html = await r.text(errors="ignore")

        sg_match = re.search(r'data-n-a-sg="([^"]+)"', html)
        ts_match = re.search(r'data-n-a-ts="([^"]+)"', html)
        if not (sg_match and ts_match):
            out["fetch_url"] = url
            return out

        sg, ts = sg_match.group(1), ts_match.group(1)

        req_url = "https://news.google.com/_/DotsSplashUi/data/batchexecute"
        payload = [
            "Fbv4je",
            f'["garturlreq",[["X","X",["X","X"],null,null,1,1,"US:en",null,1,null,null,null,null,null,0,1],"X","X",1,[1,1,1],1,1,null,0,0,null,0],"{token}",{ts},"{sg}"]',
        ]
        data = f"f.req={quote(json.dumps([[payload]]))}"

        headers = {
            "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8",
            "User-Agent": _HEADERS["User-Agent"],
        }

        async with _ARTICLE_SEMAPHORE:
            async with session.post(req_url, data=data, headers=headers, timeout=10) as r:
                resp_text = await r.text()

        parsed = json.loads(resp_text.split("\n\n")[1])[:-2]
        decoded = json.loads(parsed[0][2])[1]

        if decoded:
            out["fetch_url"] = decoded
            out["source_url"] = decoded  # Use the true resolved publisher URL!
        else:
            out["fetch_url"] = url
            
    except Exception as exc:
        logger.debug(f"Google URL resolve failed for {url}: {exc}")
        out["fetch_url"] = url

    return out

async def _fetch_article_body(session: aiohttp.ClientSession, row: dict) -> dict:
    out = dict(row)
    url = row["fetch_url"] if "fetch_url" in row else row.get("source_url")
    if not url:
        out["content"] = ""
        out.pop("fetch_url", None)
        return out

    if "news.google.com" in url:
        out["content"] = ""
        out.pop("fetch_url", None)
        return out

    try:
        async with _ARTICLE_SEMAPHORE:
            async with session.get(url, timeout=20, allow_redirects=True) as resp:
                if resp.status != 200:
                    out["content"] = ""
                    return out
                html = await resp.text(errors="ignore")
        title, content = _extract_article_text(html, fallback_title=row.get("title", ""))
        out["title"] = title
        out["content"] = content
        out.pop("fetch_url", None)
        return out
    except Exception as exc:
        logger.debug(f"Article fetch failed for {url}: {exc}")
        out["content"] = ""
        out.pop("fetch_url", None)
        return out


async def _collect_google_news_articles_async(
    tickers: Iterable[str],
    days: int = 7,
    max_urls_per_ticker: int = 10,
) -> list[dict]:
    timeout = aiohttp.ClientTimeout(total=30)
    ssl_ctx = ssl.create_default_context(cafile=certifi.where())
    connector = aiohttp.TCPConnector(ssl=ssl_ctx)
    async with aiohttp.ClientSession(headers=_HEADERS, timeout=timeout, connector=connector) as session:
        rss_tasks = [
            _fetch_rss_for_ticker(session, ticker, days=days, max_urls=max_urls_per_ticker)
            for ticker in tickers
        ]
        rss_rows_nested = await asyncio.gather(*rss_tasks, return_exceptions=False)

        deduped: list[dict] = []
        seen: set[tuple[str, str]] = set()
        for rows in rss_rows_nested:
            for row in rows:
                key = (row["ticker"], row.get("fetch_url") or row["source_url"], row["title"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(row)

        resolve_tasks = [_resolve_google_news_url_async(session, row) for row in deduped]
        resolved_rows = await asyncio.gather(*resolve_tasks, return_exceptions=False)

        deduped_again: list[dict] = []
        seen_again: set[tuple[str, str]] = set()
        for row in resolved_rows:
            key = (row["ticker"], row.get("source_url"), row["title"])
            if key in seen_again:
                continue
            seen_again.add(key)
            deduped_again.append(row)

        article_tasks = [_fetch_article_body(session, row) for row in deduped_again]
        return await asyncio.gather(*article_tasks, return_exceptions=False)


def collect_google_news_articles(
    tickers: Iterable[str],
    days: int = 7,
    max_urls_per_ticker: int = 10,
) -> list[dict]:
    """Collect recent article URLs and full text for the given tickers."""
    return asyncio.run(
        _collect_google_news_articles_async(
            tickers=tickers,
            days=days,
            max_urls_per_ticker=max_urls_per_ticker,
        )
    )