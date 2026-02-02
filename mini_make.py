# mini_make.py
from __future__ import annotations

import os
import re
import json
import time
import random
import hashlib
import logging
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Any, Dict, List, Optional, Tuple

import requests

# Optional: playwright for dynamic pages (kept for parity with original)
from playwright.sync_api import sync_playwright


# ----------------------------
# Logging
# ----------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("mini_make")


# ----------------------------
# Config
# ----------------------------
KST = ZoneInfo("Asia/Seoul")

NAVER_RANKING_URL = os.getenv(
    "NAVER_RANKING_URL", "https://m.entertain.naver.com/ranking"
)

USE_OPENAI = os.getenv("USE_OPENAI", "0").strip() in ("1", "true", "True", "YES", "yes")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

# Wikimedia/Commons settings
COMMONS_API = "https://commons.wikimedia.org/w/api.php"
WIKIDATA_API = "https://www.wikidata.org/w/api.php"

COMMONS_THUMB_WIDTH = int(os.getenv("COMMONS_THUMB_WIDTH", "1080"))
COMMONS_SEARCH_LIMIT = int(os.getenv("COMMONS_SEARCH_LIMIT", "12"))
COMMONS_P18_LIMIT = int(os.getenv("COMMONS_P18_LIMIT", "10"))

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")
MAX_ARTICLES = int(os.getenv("MAX_ARTICLES", "30"))
RANDOM_SEED = os.getenv("RANDOM_SEED", "").strip()

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "20"))

# Beat structure for 60s short
BEATS = [
    {"id": 1, "t": "0-2s", "goal": "강한 훅"},
    {"id": 2, "t": "2-10s", "goal": "상황/주인공 제시"},
    {"id": 3, "t": "10-25s", "goal": "핵심 전개/포인트"},
    {"id": 4, "t": "25-40s", "goal": "추가 정보/맥락"},
    {"id": 5, "t": "40-55s", "goal": "정리/여운"},
    {"id": 6, "t": "55-60s", "goal": "콜투액션"},
]


# ----------------------------
# Utilities
# ----------------------------
def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def now_kst_iso() -> str:
    return datetime.now(tz=KST).isoformat(timespec="seconds")


def safe_filename(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE).strip("_")
    if not s:
        s = "item"
    return s[:max_len]


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def http_get(url: str, headers: Optional[dict] = None, params: Optional[dict] = None) -> requests.Response:
    return requests.get(url, headers=headers, params=params, timeout=REQUEST_TIMEOUT)


def http_stream_download(url: str, out_path: str, headers: Optional[dict] = None) -> None:
    with requests.get(url, headers=headers, stream=True, timeout=REQUEST_TIMEOUT) as r:
        r.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 256):
                if chunk:
                    f.write(chunk)


def json_dump(path: str, data: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def text_dump(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


# ----------------------------
# Article scraping (Naver)
# ----------------------------
@dataclass
class Article:
    url: str
    title: str = ""
    description: str = ""
    published: str = ""
    content: str = ""


def collect_naver_ranking_links(max_items: int = 30) -> List[str]:
    """
    Use Playwright to render mobile Naver ranking list and collect links.
    """
    links: List[str] = []
    logger.info(f"[NAVER] goto {NAVER_RANKING_URL}")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.goto(NAVER_RANKING_URL, wait_until="networkidle", timeout=60_000)
        page.wait_for_timeout(1500)

        # Collect anchors under ranking list
        anchors = page.query_selector_all("a[href]")
        for a in anchors:
            href = a.get_attribute("href") or ""
            if not href:
                continue
            # target article links
            if "/read" in href or "/article" in href:
                if href.startswith("/"):
                    href = "https://m.entertain.naver.com" + href
                if href.startswith("https://"):
                    links.append(href)

        browser.close()

    # Dedup while preserving order
    seen = set()
    uniq = []
    for u in links:
        if u in seen:
            continue
        seen.add(u)
        uniq.append(u)

    logger.info(f"[NAVER] items={len(uniq)}")
    return uniq[:max_items]


def fetch_article(url: str) -> Article:
    """
    Fetch article HTML and parse OG meta and a rough text body (best-effort).
    For robustness we use requests (no JS); Naver mobile pages usually have OG tags.
    """
    logger.info(f"[ARTICLE] fetch {url}")
    r = http_get(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; mini_make/1.0; +https://example.com)"
        },
    )
    r.raise_for_status()
    html = r.text

    def og(prop: str) -> str:
        # property="og:title" content="..."
        m = re.search(
            rf'property="{re.escape(prop)}"\s+content="([^"]*)"', html, flags=re.I
        )
        return (m.group(1).strip() if m else "")

    title = og("og:title")
    desc = og("og:description")
    published = ""

    # Try to find published time (varies)
    m = re.search(r'data-date-time="([^"]+)"', html)
    if m:
        published = m.group(1).strip()
    else:
        m = re.search(r'"publishedTime"\s*:\s*"([^"]+)"', html)
        if m:
            published = m.group(1).strip()

    # crude body extraction: remove scripts/styles, then pick larger text blocks
    body = re.sub(r"(?is)<(script|style).*?>.*?</\1>", " ", html)
    body = re.sub(r"(?is)<br\s*/?>", "\n", body)
    body = re.sub(r"(?is)<[^>]+>", " ", body)
    body = re.sub(r"&nbsp;", " ", body)
    body = re.sub(r"\s+", " ", body).strip()

    # limit
    content = body[:2000]

    return Article(url=url, title=title, description=desc, published=published, content=content)


# ----------------------------
# OpenAI (optional)
# ----------------------------
def openai_chat_json(system: str, user: str) -> Optional[dict]:
    """
    Minimal OpenAI Chat Completions call (best-effort).
    Uses the Responses API shape? The original likely used chat.completions; keep simple.
    """
    if not (USE_OPENAI and OPENAI_API_KEY):
        return None

    try:
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": OPENAI_MODEL,
            "temperature": 0.7,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
        }
        r = requests.post(url, headers=headers, json=payload, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        return json.loads(content)
    except Exception as e:
        logger.warning(f"[OPENAI] failed: {e}")
        return None


# ----------------------------
# Script + Person extraction
# ----------------------------
def build_script(article: Article) -> dict:
    """
    Return a dict containing:
      - short_title
      - script: list of beats with {t, caption, narration, keywords}
      - people: list of person name strings
    """
    system = (
        "You are a Korean short-form news scriptwriter. "
        "Return compact JSON only. No markdown."
    )
    user = {
        "task": "Generate a 60-second YouTube Shorts script with 6 beats.",
        "format": {
            "short_title": "string",
            "people": ["string"],
            "beats": [
                {
                    "id": "1..6",
                    "time": "e.g. 0-2s",
                    "caption": "On-screen caption (Korean, short)",
                    "narration": "Voiceover (Korean, natural)",
                    "keywords": ["visual keywords", "for image search"],
                }
            ],
        },
        "article": {
            "title": article.title,
            "description": article.description,
            "published": article.published,
            "content": article.content,
            "url": article.url,
        },
        "constraints": [
            "No defamation. Use cautious wording like '보도에 따르면', '추정된다' when uncertain.",
            "Avoid quoting long text from the article.",
            "Keep each beat narration 1-2 sentences max.",
            "keywords should be nouns, 2-6 items per beat.",
            "people should include likely related person names if any, else empty list.",
        ],
    }

    j = openai_chat_json(system, json.dumps(user, ensure_ascii=False))
    if j and isinstance(j, dict) and "beats" in j:
        return {
            "short_title": j.get("short_title") or article.title[:40],
            "people": [p for p in (j.get("people") or []) if isinstance(p, str) and p.strip()],
            "beats": j.get("beats"),
        }

    # Fallback template
    fallback_beats = []
    for b in BEATS:
        fallback_beats.append(
            {
                "id": b["id"],
                "time": b["t"],
                "caption": article.title[:18],
                "narration": f"{article.title}. {article.description}"[:120],
                "keywords": ["entertainment", "celebrity", "news", "Korea"],
            }
        )

    people = heuristic_people_from_text(article.title + " " + article.description)
    return {
        "short_title": article.title[:40] if article.title else "쇼츠 뉴스",
        "people": people,
        "beats": fallback_beats,
    }


def heuristic_people_from_text(text: str) -> List[str]:
    """
    Very rough heuristic for Korean person names:
    - Finds 2~4 Korean syllables sequences, avoids common words.
    - Dedup and return top few.
    """
    if not text:
        return []
    cands = re.findall(r"[가-힣]{2,4}", text)
    stop = {
        "오늘", "연예", "뉴스", "공개", "논란", "근황", "소식", "사진", "영상",
        "공식", "입장", "드라마", "영화", "가수", "배우", "아이돌", "결혼",
        "이별", "사과", "해명", "발표", "방송", "출연", "컴백",
    }
    out = []
    for c in cands:
        if c in stop:
            continue
        if len(c) == 2 and c.endswith("들"):
            continue
        if c not in out:
            out.append(c)
    # Keep only a few
    return out[:4]


# ----------------------------
# Wikimedia Commons / Wikidata image fetcher (copyright-safe sources)
# ----------------------------
@dataclass
class CommonsImage:
    title: str  # "File:xxx.jpg"
    page_url: str
    thumb_url: str
    orig_url: Optional[str]
    license_short: Optional[str]
    license_url: Optional[str]
    author: Optional[str]
    credit: Optional[str]


def wikidata_search_entity(query: str, limit: int = 5) -> List[dict]:
    params = {
        "action": "wbsearchentities",
        "search": query,
        "language": "ko",
        "format": "json",
        "limit": limit,
    }
    r = http_get(WIKIDATA_API, params=params)
    r.raise_for_status()
    return (r.json().get("search") or [])


def wikidata_get_claim_p18(entity_id: str, limit: int = 10) -> List[str]:
    """
    P18: image filename on Commons (without 'File:' prefix sometimes)
    """
    params = {
        "action": "wbgetentities",
        "ids": entity_id,
        "props": "claims",
        "format": "json",
    }
    r = http_get(WIKIDATA_API, params=params)
    r.raise_for_status()
    entities = r.json().get("entities") or {}
    ent = entities.get(entity_id) or {}
    claims = ent.get("claims") or {}
    p18 = claims.get("P18") or []
    files = []
    for c in p18[:limit]:
        mainsnak = c.get("mainsnak") or {}
        datavalue = mainsnak.get("datavalue") or {}
        value = datavalue.get("value")
        if isinstance(value, str) and value.strip():
            files.append(value.strip())
    return files


def commons_get_imageinfo(file_titles: List[str]) -> List[CommonsImage]:
    """
    For each file title, fetch imageinfo including URLs and extmetadata.
    """
    if not file_titles:
        return []

    # Normalize titles to "File:..."
    norm = []
    for t in file_titles:
        t = t.strip()
        if not t:
            continue
        if not t.lower().startswith("file:"):
            t = "File:" + t
        norm.append(t)

    params = {
        "action": "query",
        "titles": "|".join(norm),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata",
        "iiurlwidth": str(COMMONS_THUMB_WIDTH),
        "format": "json",
    }
    r = http_get(COMMONS_API, params=params)
    r.raise_for_status()
    data = r.json()
    pages = (data.get("query") or {}).get("pages") or {}
    out: List[CommonsImage] = []

    for _, page in pages.items():
        title = page.get("title") or ""
        pageid = page.get("pageid")
        page_url = f"https://commons.wikimedia.org/?curid={pageid}" if pageid else ""
        infos = page.get("imageinfo") or []
        if not infos:
            continue
        ii = infos[0]
        thumb_url = ii.get("thumburl") or ""
        orig_url = ii.get("url")
        ext = (ii.get("extmetadata") or {})

        # Read license metadata if present
        license_short = _ext_value(ext, "LicenseShortName")
        license_url = _ext_value(ext, "LicenseUrl")
        author = _ext_value(ext, "Artist")
        credit = _ext_value(ext, "Credit")

        out.append(
            CommonsImage(
                title=title,
                page_url=page_url,
                thumb_url=thumb_url,
                orig_url=orig_url,
                license_short=license_short,
                license_url=license_url,
                author=author,
                credit=credit,
            )
        )
    return out


def _ext_value(ext: Dict[str, Any], key: str) -> Optional[str]:
    v = ext.get(key) or {}
    if isinstance(v, dict):
        val = v.get("value")
        if isinstance(val, str):
            # remove html tags
            val = re.sub(r"(?is)<[^>]+>", "", val).strip()
            return val or None
    return None


def commons_search_files(query: str, limit: int = 12) -> List[str]:
    """
    Search commons files by text query. Returns file titles.
    """
    params = {
        "action": "query",
        "list": "search",
        "srsearch": f'filetype:bitmap {query}',
        "srnamespace": "6",  # File:
        "srlimit": str(limit),
        "format": "json",
    }
    r = http_get(COMMONS_API, params=params)
    r.raise_for_status()
    items = (r.json().get("query") or {}).get("search") or []
    titles = []
    for it in items:
        t = it.get("title")
        if isinstance(t, str) and t.lower().startswith("file:"):
            titles.append(t)
    return titles


def is_license_allowed(img: CommonsImage) -> bool:
    """
    Conservative filter:
    - Allow common CC and public domain-like licenses.
    - Reject if license is missing.
    """
    ls = (img.license_short or "").lower()
    if not ls:
        return False

    allow_tokens = [
        "cc by", "cc-by", "cc by-sa", "cc-by-sa",
        "public domain", "pd", "cc0",
        "attribution", "share alike",
    ]
    deny_tokens = [
        "fair use", "non-free", "all rights reserved", "arr",
        "copyrighted", "restricted",
    ]
    for d in deny_tokens:
        if d in ls:
            return False
    for a in allow_tokens:
        if a in ls:
            return True

    # Some commons entries have "Creative Commons Attribution 4.0" etc
    if "creative commons" in ls:
        return True

    return False


def gather_images_for_people(people: List[str], need: int = 6) -> Tuple[List[CommonsImage], List[dict]]:
    """
    Collect Commons images using (1) Wikidata P18 then (2) Commons search.
    Returns (images, debug_log).
    """
    images: List[CommonsImage] = []
    debug: List[dict] = []

    for name in people:
        if len(images) >= need:
            break
        name = name.strip()
        if not name:
            continue

        step = {"person": name, "wikidata_entities": [], "p18_files": [], "commons_search_files": [], "picked": []}

        # 1) Wikidata entity search
        entities = wikidata_search_entity(name, limit=5)
        step["wikidata_entities"] = [{"id": e.get("id"), "label": e.get("label"), "desc": e.get("description")} for e in entities]

        # 2) For each entity, try P18
        p18_files: List[str] = []
        for ent in entities:
            eid = ent.get("id")
            if not eid:
                continue
            p18 = wikidata_get_claim_p18(eid, limit=COMMONS_P18_LIMIT)
            for f in p18:
                if f not in p18_files:
                    p18_files.append(f)
        step["p18_files"] = p18_files

        # 3) Resolve metadata and filter allowed licenses
        imgs = commons_get_imageinfo(p18_files)
        for img in imgs:
            if len(images) >= need:
                break
            if is_license_allowed(img):
                if not any(x.title == img.title for x in images):
                    images.append(img)
                    step["picked"].append(img.title)

        if len(images) < need:
            # 4) Commons search as fallback
            # Add extra context tokens to bias toward portraits/photos
            q = f'{name} portrait'
            sfiles = commons_search_files(q, limit=COMMONS_SEARCH_LIMIT)
            step["commons_search_files"] = sfiles

            imgs2 = commons_get_imageinfo(sfiles)
            for img in imgs2:
                if len(images) >= need:
                    break
                if is_license_allowed(img):
                    if not any(x.title == img.title for x in images):
                        images.append(img)
                        step["picked"].append(img.title)

        debug.append(step)

    return images[:need], debug


def gather_generic_news_images(article: Article, need: int = 6) -> Tuple[List[CommonsImage], List[str]]:
    """
    If no people found, search commons with generic queries built from article title/desc keywords.
    """
    text = f"{article.title} {article.description}".strip()
    # crude keyword extraction: keep korean/english tokens longer than 2
    tokens = re.findall(r"[A-Za-z]{3,}|[가-힣]{2,}", text)
    tokens = [t for t in tokens if len(t) >= 2]
    # reduce and dedup
    seen = set()
    kws = []
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        kws.append(t)
    if not kws:
        kws = ["Korea", "entertainment", "celebrity"]

    queries = [
        " ".join(kws[:3]) + " photo",
        " ".join(kws[:2]) + " portrait",
        "Korean celebrity portrait",
        "red carpet photo",
    ]

    images: List[CommonsImage] = []
    used_titles = set()
    used_queries: List[str] = []

    for q in queries:
        if len(images) >= need:
            break
        used_queries.append(q)
        files = commons_search_files(q, limit=COMMONS_SEARCH_LIMIT)
        imgs = commons_get_imageinfo(files)
        for img in imgs:
            if len(images) >= need:
                break
            if is_license_allowed(img) and img.title not in used_titles:
                images.append(img)
                used_titles.add(img.title)

    return images[:need], used_queries


def download_commons_images(images: List[CommonsImage], out_dir: str) -> List[dict]:
    """
    Download thumb images (or original if thumb missing) to out_dir.
    Returns manifest items with license/source.
    """
    ensure_dir(out_dir)
    manifest: List[dict] = []

    for idx, img in enumerate(images, start=1):
        url = img.thumb_url or img.orig_url
        if not url:
            continue

        ext = ".jpg"
        m = re.search(r"\.(jpg|jpeg|png|webp)(?:\?|$)", url, flags=re.I)
        if m:
            ext = "." + m.group(1).lower()

        fname = f"image_{idx:02d}{ext}"
        path = os.path.join(out_dir, fname)

        logger.info(f"[COMMONS] download {img.title} -> {fname}")
        http_stream_download(url, path)

        manifest.append(
            {
                "file": fname,
                "source_page": img.page_url,
                "commons_title": img.title,
                "download_url": url,
                "license_short": img.license_short,
                "license_url": img.license_url,
                "author": img.author,
                "credit": img.credit,
            }
        )

    return manifest


# ----------------------------
# Main pipeline
# ----------------------------
def run_pipeline() -> dict:
    if RANDOM_SEED:
        random.seed(RANDOM_SEED)

    ensure_dir(OUTPUT_DIR)

    # 1) collect ranking links
    links = collect_naver_ranking_links(max_items=min(MAX_ARTICLES, 100))
    if not links:
        raise RuntimeError("No ranking links collected.")

    # 2) pick random article
    pick = random.choice(links)
    article = fetch_article(pick)

    # 3) build script (OpenAI optional)
    script_obj = build_script(article)
    short_title = script_obj.get("short_title") or (article.title[:40] if article.title else "쇼츠 뉴스")
    people = script_obj.get("people") or []

    # 4) build run folder
    run_id = sha1(article.url + now_kst_iso())[:10]
    run_dir = os.path.join(OUTPUT_DIR, f"{safe_filename(short_title)}_{run_id}")
    ensure_dir(run_dir)

    # Save article + script
    json_dump(os.path.join(run_dir, "article.json"), article.__dict__)
    json_dump(os.path.join(run_dir, "script.json"), script_obj)

    # Human-readable script
    lines = []
    lines.append(f"TITLE: {short_title}")
    lines.append(f"URL: {article.url}")
    if article.published:
        lines.append(f"PUBLISHED: {article.published}")
    lines.append("")
    for b in script_obj.get("beats") or []:
        lines.append(f"[{b.get('time')}] {b.get('caption')}")
        lines.append(f"VO: {b.get('narration')}")
        kws = b.get("keywords") or []
        if kws:
            lines.append(f"KW: {', '.join(kws)}")
        lines.append("")
    text_dump(os.path.join(run_dir, "script.txt"), "\n".join(lines).strip() + "\n")

    # 5) gather images from Wikimedia Commons (copyright-safe)
    images: List[CommonsImage] = []
    debug_log: Any = None

    if people:
        logger.info(f"[IMAGES] people={people}")
        images, debug_log = gather_images_for_people(people, need=6)

    if len(images) < 6:
        logger.info("[IMAGES] fallback to generic commons search")
        more, used_queries = gather_generic_news_images(article, need=6 - len(images))
        images.extend(more)
        debug_log = {"people_debug": debug_log, "fallback_queries": used_queries}

    # 6) download
    img_dir = os.path.join(run_dir, "images")
    manifest = download_commons_images(images[:6], img_dir)
    json_dump(os.path.join(run_dir, "images_manifest.json"), manifest)
    json_dump(os.path.join(run_dir, "images_debug.json"), debug_log)

    # 7) return summary
    summary = {
        "run_dir": run_dir,
        "short_title": short_title,
        "article_url": article.url,
        "people": people,
        "images_count": len(manifest),
    }
    json_dump(os.path.join(run_dir, "run_summary.json"), summary)
    logger.info(f"[DONE] {summary}")
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    args = parser.parse_args()

    _ = run_pipeline()

    if not args.once:
        # compatibility: keep process alive if needed (default exit)
        pass


if __name__ == "__main__":
    main()

