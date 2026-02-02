# mini_make.py
from __future__ import annotations

import os
import json
import random
import time
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Callable
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus

from playwright.sync_api import sync_playwright


# ===========================
# 설정
# ===========================
OUT_DIR = Path("outputs")
IMG_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))
USE_OPENAI = os.getenv("USE_OPENAI", "1") == "1"

# Wikidata / Commons (프로필/스틸컷 보충용)
WIKIDATA_SEARCH = (
    "https://www.wikidata.org/w/api.php?action=wbsearchentities&format=json&language=ko&limit=1&search="
)
WIKIDATA_ENTITY = "https://www.wikidata.org/w/api.php?action=wbgetentities&format=json&props=claims&ids="


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def _clean_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    return " ".join(s.replace("\u00a0", " ").split()).strip()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return {"_error": True, "_status": int(getattr(e, "code", 0) or 0), "_url": url, "_body": body[:2000]}
    except URLError as e:
        return {"_error": True, "_status": 0, "_url": url, "_body": str(e)}


def _download(url: str, out_path: Path, timeout: int = 60, referer: str | None = None) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {
        "User-Agent": UA,
        "Accept": "image/avif,image/webp,image/*,*/*;q=0.8",
    }
    if referer:
        headers["Referer"] = referer

    req = Request(url, headers=headers)
    try:
        with urlopen(req, timeout=timeout) as resp:
            data = resp.read()
        out_path.write_bytes(data)
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        raise RuntimeError(f"download_failed status={getattr(e,'code',0)} url={url} body={body[:200]}") from e
    except URLError as e:
        raise RuntimeError(f"download_failed url={url} err={e}") from e


def retry(fn: Callable[[], Any], tries: int = 3, base_sleep: float = 1.2) -> Any:
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base_sleep * (2**i) + random.random() * 0.2)
    if last:
        raise last
    raise RuntimeError("retry_failed")


# ===========================
# mini make (flow)
# ===========================
@dataclass
class Node:
    name: str
    fn: Callable[[dict[str, Any]], Any]


class Flow:
    def __init__(self):
        self.nodes: dict[str, Node] = {}
        self.links: dict[str, str] = {}

    def add(self, node: Node):
        self.nodes[node.name] = node

    def connect(self, a: str, b: str):
        self.links[a] = b

    def run(self, start: str):
        ctx: dict[str, Any] = {}
        cur = start
        while cur is not None:
            log(f"[실행] {cur}")
            nxt = self.nodes[cur].fn(ctx)
            cur = nxt if isinstance(nxt, str) else self.links.get(cur)
        return ctx


# ===========================
# 네이버 랭킹에서 기사 랜덤 선택
# ===========================
def _walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _is_article_url(url: str) -> bool:
    if "entertain.naver.com" not in url:
        return False
    return ("/article/" in url) or ("/home/article/" in url)


def _extract_items_from_json(obj) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for node in _walk_json(obj):
        if not isinstance(node, dict):
            continue

        title = None
        for k in ("title", "headline", "articleTitle", "subject", "newsTitle", "name"):
            v = node.get(k)
            if isinstance(v, str):
                s = v.strip()
                if 6 <= len(s) <= 120:
                    title = s
                    break

        url = None
        for k in ("url", "link", "href", "mobileUrl", "pcUrl"):
            v = node.get(k)
            if isinstance(v, str) and v.strip().startswith("http"):
                url = v.strip()
                break

        if not url:
            oid = node.get("oid") or node.get("officeId")
            aid = node.get("aid") or node.get("articleId")
            if oid and aid:
                url = f"https://m.entertain.naver.com/home/article/{oid}/{aid}"

        if title and url and _is_article_url(url):
            out.append({"title": title, "url": url})

    uniq = {}
    for it in out:
        uniq[it["title"] + "|" + it["url"]] = it
    return list(uniq.values())


def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    json_items: list[dict[str, str]] = []
    json_urls: list[str] = []

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 1280, "height": 720},
            user_agent=UA,
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = context.new_page()

        MAX_JSON = 60

        def on_response(resp):
            try:
                if len(json_urls) >= MAX_JSON:
                    return
                ct = (resp.headers.get("content-type") or "").lower()
                if "application/json" not in ct:
                    return
                u = resp.url
                if not any(k in u for k in ("ranking", "graphql", "api")):
                    return

                json_urls.append(u)
                try:
                    data = resp.json()
                    json_items.extend(_extract_items_from_json(data))
                except Exception:
                    pass
            except Exception:
                pass

        page.on("response", on_response)

        try:
            log(f"[네이버] goto {RANKING_URL}")
            retry(lambda: page.goto(RANKING_URL, wait_until="domcontentloaded", timeout=60000), tries=3, base_sleep=1.5)
            page.wait_for_timeout(2000)

            for _ in range(3):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1200)

            dom_items = page.evaluate(
                """() => {
                    const out = [];
                    const links = Array.from(document.querySelectorAll('a'));
                    for (const a of links) {
                      const rawHref = a.getAttribute('href') || '';
                      const href = rawHref ? new URL(rawHref, location.origin).href : '';
                      const text = ((a.innerText || a.textContent || '')).trim().replace(/\\s+/g,' ');
                      if (!href || !text) continue;

                      const okDomain = href.includes('entertain.naver.com');
                      const okPath = href.includes('/article/') || href.includes('/home/article/');
                      if (okDomain && okPath && text.length >= 6 && text.length <= 120) {
                        out.push({title: text, url: href});
                      }
                    }
                    const uniq = new Map();
                    for (const x of out) uniq.set(x.title + '|' + x.url, x);
                    return Array.from(uniq.values());
                }"""
            ) or []

            next_data_text = page.evaluate(
                """() => {
                    const el = document.querySelector('#__NEXT_DATA__');
                    return el ? el.textContent : '';
                }"""
            )
            next_items: list[dict[str, str]] = []
            if next_data_text and isinstance(next_data_text, str) and len(next_data_text) > 50:
                try:
                    next_json = json.loads(next_data_text)
                    next_items = _extract_items_from_json(next_json)
                except Exception:
                    next_items = []

            all_map = {}
            for it in (json_items or []):
                all_map[it["title"] + "|" + it["url"]] = it
            for it in (next_items or []):
                all_map[it["title"] + "|" + it["url"]] = it
            for it in (dom_items or []):
                all_map[it["title"] + "|" + it["url"]] = it

            all_items = list(all_map.values())
            log(f"[네이버] items={len(all_items)}")

            _write_json(OUT_DIR / "naver_items_sample.json", all_items[:200])

            if not all_items:
                _write_text(OUT_DIR / "naver_debug.html", page.content())
                page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                _write_text(OUT_DIR / "naver_json_urls.txt", "\n".join(json_urls[:500]))
                raise RuntimeError("기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")

            chosen = random.choice(all_items)
            return chosen["title"], chosen["url"], RANKING_URL

        finally:
            context.close()
            browser.close()


# ===========================
# 기사 컨텍스트(OG + 본문 일부)
# ===========================
def fetch_article_context(article_url: str) -> dict[str, Any]:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 1280, "height": 720},
            user_agent=UA,
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = context.new_page()

        try:
            retry(lambda: page.goto(article_url, wait_until="domcontentloaded", timeout=60000), tries=3, base_sleep=1.5)
            page.wait_for_timeout(1500)

            data = page.evaluate(
                """() => {
                    function metaBy(sel) {
                      const el = document.querySelector(sel);
                      return el ? (el.getAttribute('content') || '') : '';
                    }
                    function pickBody() {
                      const selectors = ['#articleBodyContents','div#dic_area','div._article_content','div.article_body','article','main'];
                      let best = '';
                      for (const sel of selectors) {
                        const el = document.querySelector(sel);
                        if (!el) continue;
                        const t = (el.innerText || '').trim();
                        if (t.length > best.length) best = t;
                      }
                      if (!best) best = (document.body && document.body.innerText) ? document.body.innerText : '';
                      return best;
                    }

                    const og_title = metaBy('meta[property="og:title"]');
                    const og_description = metaBy('meta[property="og:description"]');
                    const published = metaBy('meta[property="article:published_time"]') || metaBy('meta[name="article:published_time"]');
                    const body = pickBody();
                    return { og_title, og_description, published_time: published, body_text: body };
                }"""
            )

            return {
                "final_url": page.url,
                "og_title": _clean_text(data.get("og_title", ""))[:200],
                "og_description": _clean_text(data.get("og_description", ""))[:300],
                "published_time": _clean_text(data.get("published_time", ""))[:60],
                "body_snippet": _clean_text(data.get("body_text", ""))[:1200],
            }
        finally:
            context.close()
            browser.close()


# ===========================
# 대본 생성: 템플릿(폴백)
# ===========================
def build_60s_shorts_script_template(inputs: dict[str, Any]) -> dict[str, Any]:
    topic = inputs["topic"]
    beats = [
        {"t": "0-2s", "voice": f"오늘 연예 랭킹 한 줄 요약! {topic}", "onscreen": "오늘 랭킹", "broll": "ranking scroll"},
        {"t": "2-10s", "voice": "왜 뜨는지 제목/요약 기준으로 핵심만 볼게요.", "onscreen": "왜 뜸?", "broll": "news keywords"},
        {"t": "10-25s", "voice": "포인트 1. 사람들이 멈춰보는 키워드가 있어요.", "onscreen": "포인트1", "broll": "headline highlight"},
        {"t": "25-40s", "voice": "포인트 2. 댓글·공유가 생길 포인트가 보여요.", "onscreen": "포인트2", "broll": "social comments"},
        {"t": "40-55s", "voice": "포인트 3. 다음 이슈로 이어질 흐름이 보입니다.", "onscreen": "포인트3", "broll": "news collage"},
        {"t": "55-60s", "voice": "내일 랭킹도 자동으로 정리할게요. 구독!", "onscreen": "구독!", "broll": "subscribe"},
    ]
    return {
        "_generator": "template",
        "topic": topic,
        "title_short": topic[:28],
        "description": "네이버 연예 랭킹 기반 요약(제목/요약 기준)",
        "beats": beats,
        "hashtags": ["#연예", "#네이버", "#랭킹", "#쇼츠", "#자동화"],
        "notes": "정보가 부족한 부분은 '제목/요약 기준'으로 표현",
    }


# ===========================
# 대본 생성: OpenAI(구조화 JSON) + 폴백
# ===========================
def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "title_short": {"type": "string"},
            "description": {"type": "string"},
            "beats": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "t": {"type": "string"},
                        "voice": {"type": "string"},
                        "onscreen": {"type": "string"},
                        "broll": {"type": "string"},
                    },
                    "required": ["t", "voice", "onscreen", "broll"],
                },
            },
            "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 10},
            "notes": {"type": "string"},
        },
        "required": ["topic", "title_short", "description", "beats", "hashtags", "notes"],
    }

    system = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다. "
        "입력(제목/OG설명/본문발췌/링크) 밖의 사실을 단정하지 말고, 불확실하면 '제목/요약 기준'이라고 표현해라. "
        "과장/루머/비방 금지. "
        "정확히 6구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s). "
        "onscreen은 12자 이내, voice는 짧고 말로 읽기 좋게."
    )

    user_payload = {
        "topic": inputs["topic"],
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "article_context": inputs.get("article_context", {}),
    }

    _write_json(OUT_DIR / "openai_script_payload.json", user_payload)

    resp = retry(
        lambda: client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "name": "shorts_script", "strict": True, "schema": schema}},
            max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        ),
        tries=3,
        base_sleep=1.5,
    )

    data = json.loads(resp.output_text)
    data["_generator"] = f"openai:{OPENAI_MODEL}"

    for b in data.get("beats", []):
        b["t"] = _clean_text(b.get("t", ""))[:20]
        b["voice"] = _clean_text(b.get("voice", ""))[:220]
        b["onscreen"] = _clean_text(b.get("onscreen", ""))[:12]
        b["broll"] = _clean_text(b.get("broll", ""))[:80]
    return data


# ===========================
# 이미지 수집: 기사(OG/본문) + Wikidata(Commons)
# ===========================
def _http_json_simple(url: str, timeout: int = 30) -> Any:
    return _http_json(url, headers={"User-Agent": UA, "Accept": "application/json"}, timeout=timeout)


def extract_image_urls_from_article(article_url: str) -> list[str]:
    """기사 페이지에서 og:image + 본문 img url 추출"""
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 1280, "height": 720},
            user_agent=UA,
            extra_http_headers={"Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8"},
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = context.new_page()
        try:
            retry(lambda: page.goto(article_url, wait_until="domcontentloaded", timeout=60000), tries=3, base_sleep=1.5)
            page.wait_for_timeout(1200)

            urls = page.evaluate(
                """() => {
                  const out = [];
                  const push = (u) => {
                    if (!u) return;
                    if (u.startsWith('data:')) return;
                    try { u = new URL(u, location.href).href; } catch(e) { return; }
                    if (u.includes('sprite') || u.includes('sp_')) return;
                    out.push(u);
                  };

                  // OG / Twitter
                  const metas = [
                    'meta[property="og:image"]',
                    'meta[name="twitter:image"]',
                    'meta[property="twitter:image"]',
                  ];
                  for (const sel of metas) {
                    const el = document.querySelector(sel);
                    if (el) push(el.getAttribute('content') || '');
                  }

                  // 본문 후보
                  const areas = ['#articleBodyContents','div#dic_area','div._article_content','div.article_body','article','main'];
                  for (const sel of areas) {
                    const root = document.querySelector(sel);
                    if (!root) continue;
                    const imgs = Array.from(root.querySelectorAll('img'));
                    for (const img of imgs) {
                      push(img.getAttribute('src') || '');
                      push(img.getAttribute('data-src') || '');
                      push(img.getAttribute('data-lazy-src') || '');
                      push(img.getAttribute('data-original') || '');
                    }
                  }

                  return Array.from(new Set(out));
                }"""
            ) or []

            cleaned: list[str] = []
            for u in urls:
                u = (u or "").strip()
                if not u:
                    continue
                low = u.lower()
                if any(x in low for x in (".svg", ".ico", "pixel", "tracker")):
                    continue
                cleaned.append(u)

            seen = set()
            final = []
            for u in cleaned:
                if u in seen:
                    continue
                seen.add(u)
                final.append(u)
            return final
        finally:
            context.close()
            browser.close()


def wikidata_commons_image_urls(name: str) -> list[str]:
    """Wikidata 검색 → P18(이미지) → Commons FilePath URL 생성"""
    name = _clean_text(name)
    if not name:
        return []

    s = _http_json_simple(WIKIDATA_SEARCH + quote_plus(name), timeout=25)
    items = (s or {}).get("search", []) if isinstance(s, dict) else []
    if not items:
        return []

    qid = (items[0] or {}).get("id")
    if not qid:
        return []

    e = _http_json_simple(WIKIDATA_ENTITY + quote_plus(qid), timeout=25)
    ent = (((e or {}).get("entities") or {}).get(qid) or {}) if isinstance(e, dict) else {}
    claims = ent.get("claims") or {}
    p18 = claims.get("P18") or []

    urls: list[str] = []
    for c in p18[:3]:
        try:
            filename = c["mainsnak"]["datavalue"]["value"]
            if not isinstance(filename, str):
                continue
            fp = "https://commons.wikimedia.org/wiki/Special:FilePath/" + quote_plus(filename.replace(" ", "_"))
            urls.append(fp)
        except Exception:
            continue

    seen = set()
    out = []
    for u in urls:
        if u in seen:
            continue
        seen.add(u)
        out.append(u)
    return out


def extract_names_fallback(topic: str) -> list[str]:
    """OpenAI 없을 때: 제목에서 고유명사 추정(완벽하지 않음)"""
    t = _clean_text(topic)
    if not t:
        return []
    t = t[:60]
    parts = [p.strip() for p in t.replace("“", " ").replace("”", " ").replace("'", " ").split() if p.strip()]
    cands = []
    if len(parts) >= 2:
        cands.append(" ".join(parts[:2]))
    if len(parts) >= 3:
        cands.append(" ".join(parts[:3]))
    if parts:
        cands.append(parts[0])

    seen = set()
    out = []
    for c in cands:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out[:3]


def extract_names_openai(topic: str, article_context: dict[str, Any]) -> list[str]:
    """가능하면 OpenAI로 제목/요약에서 '당사자 이름' 1~3개만 추출"""
    from openai import OpenAI

    client = OpenAI()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {"names": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}},
        "required": ["names"],
    }

    system = (
        "너는 기사 제목/요약에서 '당사자(인물/그룹) 이름'만 뽑는다. "
        "추측 금지. 불명확하면 제목에서 가장 가능성 높은 고유명사 1~2개만 반환. "
        "반드시 1~3개."
    )
    payload = {
        "topic": topic,
        "og_title": (article_context or {}).get("og_title", ""),
        "og_description": (article_context or {}).get("og_description", ""),
    }

    resp = retry(
        lambda: client.responses.create(
            model=OPENAI_MODEL,
            input=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            text={"format": {"type": "json_schema", "name": "names", "strict": True, "schema": schema}},
            max_output_tokens=200,
        ),
        tries=3,
        base_sleep=1.5,
    )

    data = json.loads(resp.output_text)
    names = [_clean_text(x)[:50] for x in (data.get("names") or []) if _clean_text(x)]

    seen = set()
    out = []
    for n in names:
        if n in seen:
            continue
        seen.add(n)
        out.append(n)
    return out[:3] if out else []


def collect_images_from_web(ctx: dict[str, Any]) -> dict[str, Any]:
    topic = ctx["inputs"]["topic"]
    source_url = ctx["inputs"]["source_url"]
    article_context = ctx["inputs"].get("article_context") or {}

    candidates: list[dict[str, Any]] = []

    # 1) 기사에서 이미지 URL 최대 확보
    try:
        urls = extract_image_urls_from_article(source_url)
        _write_json(OUT_DIR / "article_image_urls.json", {"source_url": source_url, "urls": urls[:200]})
        for u in urls:
            candidates.append({"provider": "naver_article", "url": u})
    except Exception:
        _write_text(OUT_DIR / "article_images_error.txt", traceback.format_exc())

    # 2) 부족하면 Wikidata/Commons에서 당사자 이미지 보충
    if len(candidates) < 6:
        names: list[str] = []
        if USE_OPENAI and os.getenv("OPENAI_API_KEY"):
            try:
                names = extract_names_openai(topic, article_context)
                _write_json(OUT_DIR / "wikidata_names.json", {"source": "openai", "names": names})
            except Exception:
                _write_text(OUT_DIR / "wikidata_names_error.txt", traceback.format_exc())
                names = []

        if not names:
            names = extract_names_fallback(topic)
            _write_json(OUT_DIR / "wikidata_names.json", {"source": "fallback", "names": names})

        commons_urls: list[str] = []
        for n in names:
            commons_urls.extend(wikidata_commons_image_urls(n))

        _write_json(OUT_DIR / "commons_image_urls.json", {"names": names, "urls": commons_urls})

        for u in commons_urls:
            candidates.append({"provider": "wikidata_commons", "url": u})

    # 3) 유니크 URL
    seen = set()
    uniq: list[dict[str, Any]] = []
    for c in candidates:
        u = c.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        uniq.append(c)

    if not uniq:
        raise RuntimeError(
            "이미지 후보를 찾지 못했습니다. outputs/article_image_urls.json / outputs/commons_image_urls.json 확인"
        )

    # 4) 6장 다운로드 (부족하면 반복 채움)
    results: list[dict[str, Any]] = []
    files: list[str] = []

    run_id = os.getenv("GITHUB_RUN_ID", "local")
    stamp = now_kst().strftime("%Y%m%d_%H%M%S")

    for i in range(6):
        pick = uniq[i] if i < len(uniq) else uniq[i % len(uniq)]
        url = pick["url"]
        provider = pick["provider"]

        out_path = IMG_DIR / f"{stamp}_{run_id}_beat{i+1:02d}_{provider}.jpg"
        log(f"[WEBIMG] beat{i+1} provider={provider} url={url}")

        try:
            ref = source_url if provider == "naver_article" else None
            _download(url, out_path, timeout=60, referer=ref)
            rec = {"beat": i + 1, "ok": True, "provider": provider, "file": str(out_path), "url": url}
        except Exception as e:
            rec = {
                "beat": i + 1,
                "ok": False,
                "provider": provider,
                "file": str(out_path),
                "url": url,
                "error": str(e),
            }

        results.append(rec)
        if rec["ok"]:
            files.append(str(out_path))

    manifest = {"topic": topic, "provider": "web", "images": results, "files": files}
    _write_json(OUT_DIR / "images_manifest.json", manifest)
    return manifest


# ===========================
# 노드들
# ===========================
def node_load_inputs(ctx: dict[str, Any]):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    ctx["inputs"] = {"topic": title, "source_url": article_url, "ranking_page": ranking_url}
    try:
        ctx["inputs"]["article_context"] = fetch_article_context(article_url)
    except Exception as e:
        ctx["inputs"]["article_context"] = {"error": str(e)}
    return "MAKE_SCRIPT"


def node_make_script(ctx: dict[str, Any]):
    inputs = ctx["inputs"]

    if USE_OPENAI and os.getenv("OPENAI_API_KEY"):
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inputs)
            return "COLLECT_IMAGES"
        except Exception:
            _write_text(OUT_DIR / "openai_error.txt", traceback.format_exc())

    ctx["shorts"] = build_60s_shorts_script_template(inputs)
    return "COLLECT_IMAGES"


def node_collect_images(ctx: dict[str, Any]):
    try:
        ctx["images"] = collect_images_from_web(ctx)
    except Exception:
        _write_text(OUT_DIR / "images_error.txt", traceback.format_exc())
        ctx["images"] = {"provider": "web", "ok": False, "error": "collect_images_failed"}
    return "SAVE_FILES"


def node_save_files(ctx: dict[str, Any]):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"shorts_{stamp}_{run_id}.json"

    src = ctx["inputs"]
    shorts = ctx["shorts"]

    lines: list[str] = []
    lines.append(f"DATE(KST): {kst.isoformat()}")
    lines.append(f"GENERATOR: {shorts.get('_generator')}")
    lines.append(f"TOPIC: {shorts.get('topic')}")
    lines.append(f"SOURCE_URL: {src.get('source_url')}")
    lines.append(f"RANKING_PAGE: {src.get('ranking_page')}")

    ac = src.get("article_context") or {}
    lines.append(f"ARTICLE_OG_TITLE: {ac.get('og_title','')}")
    lines.append(f"ARTICLE_OG_DESC: {ac.get('og_description','')}")
    lines.append("")

    for b in shorts.get("beats", []):
        lines.append(f'[{b.get("t")}] {b.get("voice")}')
        lines.append(f'  - ONSCREEN: {b.get("onscreen")}')
        lines.append(f'  - BROLL: {b.get("broll")}')
    lines.append("")
    lines.append("HASHTAGS: " + " ".join(shorts.get("hashtags", [])))

    if "images" in ctx and isinstance(ctx["images"], dict):
        ok = sum(1 for x in (ctx["images"].get("images") or []) if x.get("ok"))
        lines.append("")
        lines.append(f"IMAGES: {ok}/6 -> outputs/images/")
        lines.append("IMAGES_MANIFEST: outputs/images_manifest.json")

    _write_text(script_path, "\n".join(lines))

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": src,
        "shorts": shorts,
        "images": ctx.get("images", {}),
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    _write_json(meta_path, meta)

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}
    return "PRINT"


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("GENERATOR: " + str(ctx["shorts"].get("_generator")))
    if "images" in ctx and isinstance(ctx["images"], dict):
        ok = sum(1 for x in (ctx["images"].get("images") or []) if x.get("ok"))
        log(f"IMAGES: {ok}/6 -> outputs/images/")
    log("SCRIPT FILE: " + ctx["outputs"]["script_path"])
    log("META FILE: " + ctx["outputs"]["meta_path"])
    return None


# ===========================
# 연결/실행
# ===========================
flow = Flow()
flow.add(Node("LOAD_INPUTS", node_load_inputs))
flow.add(Node("MAKE_SCRIPT", node_make_script))
flow.add(Node("COLLECT_IMAGES", node_collect_images))
flow.add(Node("SAVE_FILES", node_save_files))
flow.add(Node("PRINT", node_print))

flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
flow.connect("MAKE_SCRIPT", "COLLECT_IMAGES")
flow.connect("COLLECT_IMAGES", "SAVE_FILES")
flow.connect("SAVE_FILES", "PRINT")

if __name__ == "__main__":
    try:
        flow.run("LOAD_INPUTS")
        log("\n[끝] OK")
    except Exception:
        _write_text(OUT_DIR / "error.txt", traceback.format_exc())
        raise
