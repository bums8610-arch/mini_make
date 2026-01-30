# mini_make.py
from __future__ import annotations

import os
import json
import random
import re
import traceback
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus
from urllib.request import Request, urlopen

from playwright.sync_api import sync_playwright

# ===========================
# 기본 설정
# ===========================
OUT_DIR = Path("outputs")
IMG_DIR = OUT_DIR / "images"
OUT_DIR.mkdir(parents=True, exist_ok=True)
IMG_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"

# 대본 분량(문자수) 보정 범위(현재는 편집 단계 미사용, 추후 확장 가능)
VOICE_CHARS_MIN = int(os.getenv("VOICE_CHARS_MIN", "360"))
VOICE_CHARS_MAX = int(os.getenv("VOICE_CHARS_MAX", "620"))

# OpenAI 텍스트 모델
OPENAI_TEXT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

# OpenAI TTS
OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    "한국어 뉴스 쇼츠 톤. 또박또박, 과장 없이, 빠르지 않게."
)

# 이미지(Pexels)
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
IMAGES_PER_BEAT = int(os.getenv("IMAGES_PER_BEAT", "1"))  # 기본 1장/구간
PEXELS_PER_PAGE = int(os.getenv("PEXELS_PER_PAGE", "20"))

# 비디오(쇼츠) 해상도
W, H = 1080, 1920
FPS = 30

# ===========================
# 유틸
# ===========================
def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _write_text(path: Path, s: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(s, encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _http_json(url: str, headers: dict[str, str] | None = None, timeout: int = 30) -> Any:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def _http_download(url: str, out_path: Path, headers: dict[str, str] | None = None, timeout: int = 60) -> None:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.read())


def parse_time_range(t: str) -> tuple[float, float]:
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*-\s*(\d+(?:\.\d+)?)\s*s\s*$", t)
    if not m:
        raise ValueError(f"Bad time range: {t}")
    return float(m.group(1)), float(m.group(2))


def srt_ts(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    ms = int(round(seconds * 1000))
    hh = ms // 3600000
    ms %= 3600000
    mm = ms // 60000
    ms %= 60000
    ss = ms // 1000
    ms %= 1000
    return f"{hh:02d}:{mm:02d}:{ss:02d},{ms:03d}"


# ===========================
# mini make flow
# ===========================
@dataclass
class Node:
    name: str
    fn: callable


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
# 네이버 랭킹: 랜덤 기사 선택
# ===========================
def _walk_json(obj):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


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

        if title and url and "entertain.naver.com" in url:
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
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        )
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = context.new_page()

        def on_response(resp):
            try:
                ct = (resp.headers.get("content-type") or "").lower()
                if "application/json" in ct:
                    json_urls.append(resp.url)
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
            page.goto(RANKING_URL, wait_until="domcontentloaded", timeout=60000)
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
                      const okPath = href.includes('/article/') || href.includes('/home/article/') || href.includes('/ranking/read');
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
            if isinstance(next_data_text, str) and len(next_data_text) > 50:
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
        context = browser.new_context(locale="ko-KR", timezone_id="Asia/Seoul", viewport={"width": 1280, "height": 720})
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
        page = context.new_page()

        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
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

            og_title = _clean_text(data.get("og_title", ""))
            og_description = _clean_text(data.get("og_description", ""))
            published_time = _clean_text(data.get("published_time", ""))

            body_text = (data.get("body_text", "") or "").replace("\u00a0", " ")
            body_text = re.sub(r"\s+\n", "\n", body_text)
            body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()
            body_excerpt = _clean_text(body_text)[:1800]

            if not og_title and not og_description and len(body_excerpt) < 40:
                _write_text(OUT_DIR / "article_debug.html", page.content())
                page.screenshot(path=str(OUT_DIR / "article_debug.png"), full_page=True)

            return {
                "og_title": og_title,
                "og_description": og_description,
                "published_time": published_time,
                "body_excerpt": body_excerpt,
            }

        finally:
            context.close()
            browser.close()


# ===========================
# 대본 생성(템플릿 백업)
# ===========================
def build_60s_shorts_script_template(topic: str, article_ctx: dict[str, Any]) -> dict[str, Any]:
    hint = (article_ctx.get("og_description") or "").strip()[:60]
    beats = [
        {"t": "0-2s",   "voice": f"오늘 연예 랭킹, {topic}.", "onscreen": "오늘의 랭킹", "broll": "entertainment news ranking"},
        {"t": "2-10s",  "voice": "핵심만 1분으로 정리합니다.", "onscreen": "1분 요약", "broll": "news keywords"},
        {"t": "10-25s", "voice": (hint and f"기사 요약 힌트: {hint}." or "제목에서 포인트를 먼저 잡아볼게요."), "onscreen": "포인트", "broll": "headline concept"},
        {"t": "25-40s", "voice": "사람들이 멈춰보는 지점이 있어요.", "onscreen": "반응 포인트", "broll": "social reaction"},
        {"t": "40-55s", "voice": "자세한 내용은 기사 확인이 가장 안전합니다.", "onscreen": "기사 확인", "broll": "reading news on phone"},
        {"t": "55-60s", "voice": "내일 랭킹도 자동으로 요약. 구독!", "onscreen": "구독", "broll": "subscribe button"},
    ]
    return {
        "topic": topic,
        "title_short": topic[:28],
        "description": "네이버 연예 랭킹 기반 60초 요약(템플릿)",
        "beats": beats,
        "hashtags": ["#연예", "#네이버", "#랭킹", "#쇼츠", "#자동화"],
        "notes": "OpenAI 실패/미설정 시 템플릿으로 생성",
        "_generator": "template",
        "_style": "template",
        "_voice_chars": sum(len(b["voice"]) for b in beats),
    }


# ===========================
# OpenAI 대본 생성(2단계: facts -> script)
# ===========================
def _pick_style() -> str:
    env_style = os.getenv("SHORTS_STYLE", "").strip().lower()
    styles = ["timeline", "qna", "mythfact", "3points"]
    return env_style if env_style in styles else random.choice(styles)


def _voice_total_chars(beats: list[dict[str, Any]]) -> int:
    return sum(len((b.get("voice") or "").strip()) for b in beats)


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    client = OpenAI()
    style = _pick_style()
    article_ctx = inputs.get("article_ctx", {}) or {}

    facts_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "style": {"type": "string"},
            "facts": {"type": "array", "minItems": 3, "maxItems": 8, "items": {"type": "string"}},
            "what_is_unknown": {"type": "array", "minItems": 1, "maxItems": 5, "items": {"type": "string"}},
            "angles": {"type": "array", "minItems": 2, "maxItems": 5, "items": {"type": "string"}},
            "safe_wording_rules": {"type": "array", "minItems": 3, "maxItems": 8, "items": {"type": "string"}},
        },
        "required": ["topic", "style", "facts", "what_is_unknown", "angles", "safe_wording_rules"],
    }

    sys_facts = (
        "너는 한국어 쇼츠 작가의 리서처다. "
        "입력(제목/OG요약/본문 일부)에서 확인 가능한 내용만 facts로 뽑아라. "
        "추측/루머/비방 금지. 불확실한 건 what_is_unknown에 넣어라."
    )

    user_facts_payload = {
        "topic": inputs["topic"],
        "style": style,
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "og_title": article_ctx.get("og_title", ""),
        "og_description": article_ctx.get("og_description", ""),
        "published_time": article_ctx.get("published_time", ""),
        "body_excerpt": article_ctx.get("body_excerpt", ""),
    }

    facts_resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": sys_facts},
            {"role": "user", "content": json.dumps(user_facts_payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "facts_pack", "strict": True, "schema": facts_schema}},
        max_output_tokens=650,
    )
    facts_pack = json.loads(facts_resp.output_text)

    script_schema = {
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
            "hashtags": {"type": "array", "minItems": 4, "maxItems": 10, "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["topic", "title_short", "description", "beats", "hashtags", "notes"],
    }

    sys_script = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다. "
        "facts_pack.facts에 있는 내용만 '사실처럼' 말해라. "
        "facts 밖의 내용은 단정 금지(필요 시 '기사 요약 기준'으로 표현). "
        "과장/루머/비방 금지. "
        "정확히 6구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s)으로 구성. "
        "onscreen은 12자 이내. voice는 말로 읽기 좋게."
        f"스타일은 {style}."
    )

    user_script_payload = {
        "topic": inputs["topic"],
        "style": style,
        "facts_pack": facts_pack,
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
    }

    script_resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": sys_script},
            {"role": "user", "content": json.dumps(user_script_payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "shorts_script", "strict": True, "schema": script_schema}},
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )
    draft = json.loads(script_resp.output_text)
    draft["_generator"] = "openai_v2"
    draft["_style"] = style
    draft["_voice_chars"] = _voice_total_chars(draft.get("beats") or [])
    return draft


# ===========================
# (1번) OpenAI로 "스톡사진에 잘 맞는" 이미지 검색어 6개 만들기
# ===========================
def generate_image_queries_openai(inputs: dict[str, Any], shorts: dict[str, Any]) -> dict[str, Any]:
    """
    beats(6개)마다:
      - search_query: Pexels에 바로 넣을 영어 검색어(짧고 구체적)
      - fallback_query: 결과 없을 때 대체 검색어(더 일반적)
      - avoid: 피해야 할 단어(텍스트/로고/스크린샷 등)
    """
    from openai import OpenAI

    client = OpenAI()

    beats = shorts.get("beats") or []
    if not (isinstance(beats, list) and len(beats) == 6):
        return {"ok": False, "reason": "beats not found"}

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "topic": {"type": "string"},
            "items": {
                "type": "array",
                "minItems": 6,
                "maxItems": 6,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "beat": {"type": "integer", "minimum": 1, "maximum": 6},
                        "search_query": {"type": "string"},
                        "fallback_query": {"type": "string"},
                        "keywords": {"type": "array", "minItems": 3, "maxItems": 10, "items": {"type": "string"}},
                        "avoid": {"type": "array", "minItems": 2, "maxItems": 8, "items": {"type": "string"}},
                    },
                    "required": ["beat", "search_query", "fallback_query", "keywords", "avoid"],
                },
            },
            "notes": {"type": "string"},
        },
        "required": ["topic", "items", "notes"],
    }

    article_ctx = inputs.get("article_ctx", {}) or {}

    sys = (
        "너는 쇼츠 영상 편집자의 '스톡 이미지 검색어' 전문가다.\n"
        "목표: Pexels 같은 스톡 사이트에서 결과가 잘 뜨는 영어 검색어를 6개 만든다.\n"
        "규칙:\n"
        "- 인물 실명/특정 연예인 이름은 되도록 넣지 말고, 일반화(celebrity, pop star, actor 등)로 표현.\n"
        "- 'screenshot, ranking, naver' 같은 화면 캡처 느낌 단어는 피한다.\n"
        "- search_query는 4~8 단어로, 장면(장소+인물+행동+감정)을 포함.\n"
        "- fallback_query는 더 일반적인 2~5 단어.\n"
        "- avoid에는 text, logo, watermark, screenshot 같은 피해야 할 단어를 넣는다.\n"
        "- beats의 broll/onscreen/voice에서 '장면'을 상상해 만든다.\n"
    )

    user_payload = {
        "topic": inputs.get("topic", ""),
        "og_title": article_ctx.get("og_title", ""),
        "og_description": article_ctx.get("og_description", ""),
        "beats": [
            {
                "beat": i + 1,
                "t": b.get("t"),
                "voice": (b.get("voice") or "")[:220],
                "onscreen": b.get("onscreen"),
                "broll": b.get("broll"),
            }
            for i, b in enumerate(beats)
        ],
    }

    resp = client.responses.create(
        model=OPENAI_TEXT_MODEL,
        input=[
            {"role": "system", "content": sys},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "image_query_pack", "strict": True, "schema": schema}},
        max_output_tokens=700,
    )

    pack = json.loads(resp.output_text)

    # 최소 안전 정리
    for it in pack.get("items", []):
        it["search_query"] = _clean_text(it.get("search_query", ""))[:90]
        it["fallback_query"] = _clean_text(it.get("fallback_query", ""))[:60]

    return {"ok": True, "pack": pack}


# ===========================
# 이미지 수집: Pexels
# ===========================
def pexels_search(query: str, per_page: int = 20) -> list[dict[str, Any]]:
    if not PEXELS_API_KEY:
        return []
    url = f"https://api.pexels.com/v1/search?query={quote_plus(query)}&per_page={per_page}&orientation=portrait"
    headers = {"Authorization": PEXELS_API_KEY}
    data = _http_json(url, headers=headers, timeout=30)
    return data.get("photos", []) or []


def score_pexels_photo(p: dict[str, Any]) -> float:
    """
    세로(9:16) 비율 + 해상도 높을수록 점수↑
    """
    w = float(p.get("width") or 0)
    h = float(p.get("height") or 0)
    if w <= 0 or h <= 0:
        return 0.0
    ratio = w / h  # portrait면 0.5625(=9/16) 근처가 좋음
    target = 9.0 / 16.0
    ratio_score = max(0.0, 1.0 - abs(ratio - target) * 2.0)  # 대충 근접도
    res_score = (w * h) / (3000 * 4000)  # 12MP를 1.0 근처로
    return ratio_score * 0.7 + min(1.5, res_score) * 0.3


def collect_images_from_pexels_with_openai(topic: str, beats: list[dict[str, Any]], query_pack: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI가 만든 query_pack.items(6개)를 따라 이미지 수집.
    - search_query → 없으면 fallback_query → 그래도 없으면 topic 기반 일반 쿼리
    - 각 beat마다 상위 점수 사진을 선택(랜덤 아님, 품질 우선)
    """
    items = (query_pack.get("items") or [])
    if not (isinstance(items, list) and len(items) == 6):
        raise RuntimeError("image_query_pack.items가 올바르지 않습니다.")

    results: list[dict[str, Any]] = []
    files: list[Path] = []

    for idx in range(1, 7):
        it = items[idx - 1]
        q1 = it.get("search_query", "")
        q2 = it.get("fallback_query", "")
        avoid = it.get("avoid", []) or []

        # Pexels는 negative filter가 없어서, avoid는 참고용으로만 기록
        log(f"[PEXELS] beat{idx} q1={q1} / q2={q2}")

        photos = pexels_search(q1, per_page=PEXELS_PER_PAGE) if q1 else []
        used_query = q1

        if not photos and q2:
            photos = pexels_search(q2, per_page=PEXELS_PER_PAGE)
            used_query = q2

        if not photos:
            fallback = _clean_text(f"{topic} portrait")[:80]
            photos = pexels_search(fallback, per_page=PEXELS_PER_PAGE)
            used_query = fallback

        if not photos:
            results.append({"beat": idx, "query": used_query, "ok": False, "reason": "no results"})
            continue

        # 품질 우선: 점수 높은 순으로 정렬 후 상위 3개 중 랜덤(완전 고정 방지)
        ranked = sorted(photos, key=score_pexels_photo, reverse=True)
        pick = random.choice(ranked[: min(3, len(ranked))])

        src = pick.get("src") or {}
        dl = src.get("large2x") or src.get("large") or src.get("original")
        if not dl:
            results.append({"beat": idx, "query": used_query, "ok": False, "reason": "no download url"})
            continue

        out_path = IMG_DIR / f"beat{idx:02d}_pexels_{pick.get('id')}.jpg"
        _http_download(dl, out_path, timeout=60)

        results.append(
            {
                "beat": idx,
                "query": used_query,
                "avoid": avoid,
                "ok": True,
                "provider": "pexels",
                "file": str(out_path),
                "id": pick.get("id"),
                "page_url": pick.get("url"),
                "photographer": pick.get("photographer"),
                "photographer_url": pick.get("photographer_url"),
                "download_url": dl,
                "w": pick.get("width"),
                "h": pick.get("height"),
                "score": score_pexels_photo(pick),
            }
        )
        files.append(out_path)

    _write_json(OUT_DIR / "images_manifest.json", {"provider": "pexels", "topic": topic, "images": results})
    return {"provider": "pexels", "images": results, "files": [str(p) for p in files]}


def build_image_queries_fallback(topic: str, beats: list[dict[str, Any]]) -> list[str]:
    queries = []
    for b in beats:
        q = " ".join([topic, b.get("broll", ""), b.get("onscreen", "")]).strip()
        q = _clean_text(q)[:80]
        queries.append(q)
    return queries


def collect_images_from_pexels_basic(topic: str, beats: list[dict[str, Any]]) -> dict[str, Any]:
    queries = build_image_queries_fallback(topic, beats)
    results: list[dict[str, Any]] = []
    files: list[Path] = []

    for idx, q in enumerate(queries, start=1):
        log(f"[PEXELS-basic] beat{idx} query={q}")
        photos = pexels_search(q, per_page=PEXELS_PER_PAGE)
        if not photos:
            results.append({"beat": idx, "query": q, "ok": False, "reason": "no results"})
            continue

        ranked = sorted(photos, key=score_pexels_photo, reverse=True)
        pick = random.choice(ranked[: min(3, len(ranked))])

        src = pick.get("src") or {}
        dl = src.get("large2x") or src.get("large") or src.get("original")
        if not dl:
            results.append({"beat": idx, "query": q, "ok": False, "reason": "no download url"})
            continue

        out_path = IMG_DIR / f"beat{idx:02d}_pexels_{pick.get('id')}.jpg"
        _http_download(dl, out_path, timeout=60)
        results.append(
            {
                "beat": idx,
                "query": q,
                "ok": True,
                "provider": "pexels",
                "file": str(out_path),
                "id": pick.get("id"),
                "page_url": pick.get("url"),
                "photographer": pick.get("photographer"),
                "photographer_url": pick.get("photographer_url"),
                "download_url": dl,
                "w": pick.get("width"),
                "h": pick.get("height"),
                "score": score_pexels_photo(pick),
            }
        )
        files.append(out_path)

    _write_json(OUT_DIR / "images_manifest.json", {"provider": "pexels", "topic": topic, "images": results})
    return {"provider": "pexels", "images": results, "files": [str(p) for p in files]}


# ===========================
# 음성 생성: OpenAI TTS
# ===========================
def make_voice_openai(narration: str, out_mp3: Path) -> None:
    from openai import OpenAI

    narration = narration.strip()
    if len(narration) > 3900:
        narration = narration[:3900]

    client = OpenAI()
    out_mp3.parent.mkdir(parents=True, exist_ok=True)

    with client.audio.speech.with_streaming_response.create(
        model=OPENAI_TTS_MODEL,
        voice=OPENAI_TTS_VOICE,
        input=narration,
        instructions=OPENAI_TTS_INSTRUCTIONS,
    ) as response:
        response.stream_to_file(out_mp3)


# ===========================
# 자막(SRT)
# ===========================
def make_captions_srt(beats: list[dict[str, Any]], out_srt: Path) -> None:
    lines = []
    for i, b in enumerate(beats, start=1):
        start, end = parse_time_range(b["t"])
        text = (b.get("voice") or "").strip().replace("\n", " ")
        lines.append(str(i))
        lines.append(f"{srt_ts(start)} --> {srt_ts(end)}")
        lines.append(text)
        lines.append("")
    _write_text(out_srt, "\n".join(lines))


# ===========================
# 비디오 합성: ffmpeg
# ===========================
def build_video_ffmpeg(image_paths: list[str], beats: list[dict[str, Any]], audio_mp3: Path, captions_srt: Path, out_mp4: Path) -> None:
    durations = []
    for b in beats:
        s, e = parse_time_range(b["t"])
        durations.append(max(0.5, e - s))

    if not image_paths:
        raise RuntimeError("No images to build video.")

    while len(image_paths) < len(durations):
        image_paths.append(image_paths[-1])

    cmd = ["ffmpeg", "-y"]
    for img, d in zip(image_paths[: len(durations)], durations):
        cmd += ["-loop", "1", "-t", f"{d}", "-i", img]

    cmd += ["-i", str(audio_mp3)]

    filter_parts = []
    for i in range(len(durations)):
        filter_parts.append(
            f"[{i}:v]scale={W}:{H}:force_original_aspect_ratio=increase,"
            f"crop={W}:{H},fps={FPS},format=yuv420p,setsar=1[v{i}]"
        )
    v_in = "".join([f"[v{i}]" for i in range(len(durations))])
    filter_parts.append(f"{v_in}concat=n={len(durations)}:v=1:a=0[vcat]")
    filter_parts.append(f"[vcat]subtitles={captions_srt.as_posix()}[vout]")
    filter_complex = ";".join(filter_parts)

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", f"{len(durations)}:a",
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-r", str(FPS),
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        str(out_mp4),
    ]

    log("[FFMPEG] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ===========================
# 노드들
# ===========================
def node_load_inputs(ctx: dict[str, Any]):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    article_ctx = fetch_article_context(article_url)
    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
        "article_ctx": article_ctx,
    }


def node_make_script(ctx: dict[str, Any]):
    inp = ctx["inputs"]
    if os.getenv("OPENAI_API_KEY"):
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inp)
            return
        except Exception as e:
            _write_text(OUT_DIR / "openai_error.txt", str(e))

    ctx["shorts"] = build_60s_shorts_script_template(inp["topic"], inp.get("article_ctx", {}))


def node_collect_images(ctx: dict[str, Any]):
    topic = ctx["inputs"]["topic"]
    beats = ctx["shorts"]["beats"]

    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY가 없습니다. GitHub Secrets에 PEXELS_API_KEY를 추가하세요.")

    # (1번) OpenAI로 검색어 생성 → Pexels 검색 품질 상승
    if os.getenv("OPENAI_API_KEY"):
        try:
            qres = generate_image_queries_openai(ctx["inputs"], ctx["shorts"])
            if qres.get("ok"):
                pack = qres["pack"]
                _write_json(OUT_DIR / "image_queries.json", pack)
                ctx["image_queries"] = pack
                ctx["images"] = collect_images_from_pexels_with_openai(topic, beats, pack)
                return
        except Exception as e:
            _write_text(OUT_DIR / "image_query_error.txt", str(e))

    # OpenAI 실패 시 basic
    ctx["images"] = collect_images_from_pexels_basic(topic, beats)


def node_make_voice(ctx: dict[str, Any]):
    beats = ctx["shorts"]["beats"]
    narration = "\n".join([b["voice"] for b in beats])

    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")
    audio_path = OUT_DIR / f"voice_{stamp}_{run_id}.mp3"

    make_voice_openai(narration, audio_path)
    ctx["audio"] = {"path": str(audio_path), "text_len": len(narration)}


def node_make_video(ctx: dict[str, Any]):
    beats = ctx["shorts"]["beats"]

    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    captions_srt = OUT_DIR / f"captions_{stamp}_{run_id}.srt"
    make_captions_srt(beats, captions_srt)

    audio_mp3 = Path(ctx["audio"]["path"])
    image_files = ctx["images"]["files"]

    out_mp4 = OUT_DIR / f"shorts_{stamp}_{run_id}.mp4"
    build_video_ffmpeg(image_files, beats, audio_mp3, captions_srt, out_mp4)

    ctx["video"] = {"path": str(out_mp4), "captions": str(captions_srt)}


def node_save_meta(ctx: dict[str, Any]):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    meta_path = OUT_DIR / f"meta_{stamp}_{run_id}.json"
    _write_json(meta_path, {"date_kst": kst.isoformat(), "run_id": run_id, **ctx})
    ctx["meta"] = {"path": str(meta_path)}


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("SCRIPT_GENERATOR: " + str(ctx["shorts"].get("_generator")))
    if "image_queries" in ctx:
        log("IMAGE_QUERIES: outputs/image_queries.json")
    log("AUDIO: " + str(ctx.get("audio", {}).get("path")))
    log("VIDEO: " + str(ctx.get("video", {}).get("path")))
    log("META: " + str(ctx.get("meta", {}).get("path")))


# ===========================
# 실행 연결
# ===========================
flow = Flow()
flow.add(Node("LOAD_INPUTS", node_load_inputs))
flow.add(Node("MAKE_SCRIPT", node_make_script))
flow.add(Node("COLLECT_IMAGES", node_collect_images))
flow.add(Node("MAKE_VOICE", node_make_voice))
flow.add(Node("MAKE_VIDEO", node_make_video))
flow.add(Node("SAVE_META", node_save_meta))
flow.add(Node("PRINT", node_print))

flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
flow.connect("MAKE_SCRIPT", "COLLECT_IMAGES")
flow.connect("COLLECT_IMAGES", "MAKE_VOICE")
flow.connect("MAKE_VOICE", "MAKE_VIDEO")
flow.connect("MAKE_VIDEO", "SAVE_META")
flow.connect("SAVE_META", "PRINT")

if __name__ == "__main__":
    try:
        ctx = flow.run("LOAD_INPUTS")
        log("\n[끝] OK")
    except Exception:
        _write_text(OUT_DIR / "error.txt", traceback.format_exc())
        raise

