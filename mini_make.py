# mini_make.py
from __future__ import annotations

import os
import json
import random
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen
from urllib.parse import quote_plus


# ===========================
# 공통 유틸
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

PEXELS_API_KEY = os.getenv("PEXELS_API_KEY", "").strip()
PEXELS_PER_PAGE = int(os.getenv("PEXELS_PER_PAGE", "25"))  # 검색 결과 개수


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def _clean_text(s: str) -> str:
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
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8"))


def _download(url: str, out_path: Path, headers: dict[str, str] | None = None, timeout: int = 60) -> None:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(resp.read())


# ===========================
# mini make (flow)
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
# 네이버 랭킹: 랜덤 기사 선택(Playwright)
# ===========================
from playwright.sync_api import sync_playwright


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
            user_agent=UA,
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
                    next_json = JSON.parse(next_data_text)  # intentionally wrong? no, below is python json
                except Exception:
                    next_items = []
            # 올바른 파싱(파이썬 json)
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
# 기사 컨텍스트(간단 OG + 본문 일부)
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
            page.wait_for_timeout(1200)
            data = page.evaluate(
                """() => {
                    function metaBy(sel) {
                      const el = document.querySelector(sel);
                      return el ? (el.getAttribute('content') || '') : '';
                    }
                    function pickBody() {
                      const selectors = [
                        '#articleBodyContents','div#dic_area','div._article_content',
                        'div.article_body','article','main'
                      ];
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
# 대본: OpenAI(구조화 JSON) + 폴백
# ===========================
def build_60s_shorts_script_template(inputs: dict[str, Any]) -> dict[str, Any]:
    topic = inputs["topic"]
    beats = [
        {"t": "0-2s", "voice": f"오늘 연예 랭킹 한 줄 요약! {topic}", "onscreen": "오늘 랭킹", "broll": "ranking scroll"},
        {"t": "2-10s", "voice": "왜 뜨는지 제목과 요약 기준으로만 핵심을 볼게요.", "onscreen": "왜 뜸?", "broll": "news keywords"},
        {"t": "10-25s", "voice": "포인트 1. 사람들이 멈춰보는 키워드가 들어가 있어요.", "onscreen": "포인트1", "broll": "headline highlight"},
        {"t": "25-40s", "voice": "포인트 2. 댓글과 공유가 생기는 지점이 있어요.", "onscreen": "포인트2", "broll": "social comments"},
        {"t": "40-55s", "voice": "포인트 3. 다음 이슈로 이어질 흐름이 보입니다.", "onscreen": "포인트3", "broll": "news collage"},
        {"t": "55-60s", "voice": "내일 랭킹도 자동으로 정리할게요. 구독!", "onscreen": "구독!", "broll": "subscribe button"},
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


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))
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
        "입력으로 받은 제목/OG설명/본문 발췌/링크 밖의 사실을 단정하지 말고, "
        "불확실하면 '제목/요약 기준'이라고 표현해라. "
        "과장/루머/비방 금지. "
        "정확히 6구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s). "
        "onscreen은 12자 이내. voice는 읽기 쉽게 짧게."
    )

    user_payload = {
        "topic": inputs["topic"],
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "article_context": inputs.get("article_context", {}),
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "shorts_script", "strict": True, "schema": schema}},
        max_output_tokens=max_out,
    )

    data = json.loads(resp.output_text)

    # 최소 정리
    data["_generator"] = f"openai:{model}"
    if isinstance(data.get("beats"), list):
        for b in data["beats"]:
            b["t"] = _clean_text(b.get("t", ""))[:20]
            b["voice"] = _clean_text(b.get("voice", ""))[:220]
            b["onscreen"] = _clean_text(b.get("onscreen", ""))[:12]
            b["broll"] = _clean_text(b.get("broll", ""))[:80]
    return data


# ===========================
# 이미지 수집(Pexels)
# ===========================
def pexels_search(query: str, per_page: int = 25) -> list[dict[str, Any]]:
    if not PEXELS_API_KEY:
        return []
    url = f"https://api.pexels.com/v1/search?query={quote_plus(query)}&per_page={per_page}&orientation=portrait"
    headers = {"Authorization": PEXELS_API_KEY}
    data = _http_json(url, headers=headers, timeout=30)
    return data.get("photos", []) or []


def score_photo(p: dict[str, Any]) -> float:
    """세로 비율(9:16 근처) + 해상도 기준 점수"""
    w = float(p.get("width") or 0)
    h = float(p.get("height") or 0)
    if w <= 0 or h <= 0:
        return 0.0
    ratio = w / h
    target = 9.0 / 16.0
    ratio_score = max(0.0, 1.0 - abs(ratio - target) * 2.0)
    res_score = min(1.0, (w * h) / (3000 * 4000))
    return ratio_score * 0.7 + res_score * 0.3


def build_image_queries_fallback(beats: list[dict[str, Any]]) -> list[str]:
    """
    OpenAI가 없어도 무조건 결과가 뜨도록,
    구간별로 '스톡에서 잘 나오는' 영어 쿼리를 고정 제공.
    """
    base = [
        "K-pop celebrity news on smartphone",
        "reporter writing entertainment news",
        "people reading news on phone indoors",
        "social media comments scrolling close up",
        "entertainment news collage montage",
        "subscribe button on screen close up",
    ]
    if len(beats) == 6:
        return base
    return base[: max(1, len(beats))]


def generate_image_queries_openai(shorts: dict[str, Any]) -> list[str]:
    """
    OpenAI가 있으면 beats(broll/voice/onscreen)를 보고
    Pexels용 영어 검색어 6개를 만듦. 실패하면 예외.
    """
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    client = OpenAI()

    schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "queries": {"type": "array", "minItems": 6, "maxItems": 6, "items": {"type": "string"}}
        },
        "required": ["queries"],
    }

    system = (
        "너는 영상 편집자다. Pexels에서 잘 검색되는 영어 검색어 6개를 만들어라. "
        "각 검색어는 4~8단어, 장면(인물/장소/행동)이 떠오르게. "
        "실명/로고/스크린샷/워터마크 유도 단어는 피한다."
    )

    beats = shorts.get("beats") or []
    payload = {
        "topic": shorts.get("topic", ""),
        "beats": [
            {
                "t": b.get("t", ""),
                "voice": (b.get("voice", "") or "")[:120],
                "onscreen": b.get("onscreen", ""),
                "broll": b.get("broll", ""),
            }
            for b in beats
        ],
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "image_queries", "strict": True, "schema": schema}},
        max_output_tokens=300,
    )
    data = json.loads(resp.output_text)
    queries = [ _clean_text(q)[:90] for q in data["queries"] ]
    if len(queries) != 6:
        raise RuntimeError("OpenAI image queries length != 6")
    return queries


def collect_images_from_pexels(ctx: dict[str, Any]) -> dict[str, Any]:
    topic = ctx["inputs"]["topic"]
    beats = ctx["shorts"]["beats"]

    if not PEXELS_API_KEY:
        raise RuntimeError("PEXELS_API_KEY가 없습니다. GitHub Secrets에 PEXELS_API_KEY를 추가하세요.")

    # 1) OpenAI로 쿼리 생성 시도(있으면 품질↑)
    queries = None
    if os.getenv("OPENAI_API_KEY"):
        try:
            queries = generate_image_queries_openai(ctx["shorts"])
            _write_json(OUT_DIR / "image_queries.json", {"source": "openai", "queries": queries})
        except Exception as e:
            _write_text(OUT_DIR / "image_query_error.txt", str(e))
            queries = None

    # 2) 실패/키 없음이면 고정 쿼리로 폴백
    if not queries:
        queries = build_image_queries_fallback(beats)
        _write_json(OUT_DIR / "image_queries.json", {"source": "fallback", "queries": queries})

    results: list[dict[str, Any]] = []
    files: list[str] = []

    for i in range(6):
        q = queries[i]
        log(f"[PEXELS] beat{i+1} query={q}")

        photos = pexels_search(q, per_page=PEXELS_PER_PAGE)
        if not photos:
            results.append({"beat": i + 1, "query": q, "ok": False, "reason": "no_results"})
            continue

        ranked = sorted(photos, key=score_photo, reverse=True)
        pick = random.choice(ranked[: min(3, len(ranked))])  # 상위 3개 중 랜덤

        src = pick.get("src") or {}
        dl = src.get("large2x") or src.get("large") or src.get("original")
        if not dl:
            results.append({"beat": i + 1, "query": q, "ok": False, "reason": "no_download_url"})
            continue

        out_path = IMG_DIR / f"beat{i+1:02d}_pexels_{pick.get('id')}.jpg"
        _download(dl, out_path, timeout=60)

        rec = {
            "beat": i + 1,
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
            "score": score_photo(pick),
        }
        results.append(rec)
        files.append(str(out_path))

    manifest = {
        "topic": topic,
        "provider": "pexels",
        "images": results,
        "files": files,
    }
    _write_json(OUT_DIR / "images_manifest.json", manifest)
    return manifest


# ===========================
# 노드들
# ===========================
def node_load_inputs(ctx: dict[str, Any]):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
    }
    try:
        ctx["inputs"]["article_context"] = fetch_article_context(article_url)
    except Exception as e:
        ctx["inputs"]["article_context"] = {"error": str(e)}
    return "MAKE_SCRIPT"


def node_make_script(ctx: dict[str, Any]):
    inputs = ctx["inputs"]
    use_openai = os.getenv("USE_OPENAI", "1") == "1"
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if use_openai and has_key:
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inputs)
            return "COLLECT_IMAGES"
        except Exception:
            _write_text(OUT_DIR / "openai_error.txt", traceback.format_exc())

    ctx["shorts"] = build_60s_shorts_script_template(inputs)
    return "COLLECT_IMAGES"


def node_collect_images(ctx: dict[str, Any]):
    ctx["images"] = collect_images_from_pexels(ctx)
    return "SAVE_FILES"


def node_save_files(ctx: dict[str, Any]):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"shorts_{stamp}_{run_id}.json"

    src = ctx["inputs"]
    shorts = ctx["shorts"]

    txt = []
    txt.append(f"DATE(KST): {kst.isoformat()}")
    txt.append(f"GENERATOR: {shorts.get('_generator')}")
    txt.append(f"TOPIC: {shorts.get('topic')}")
    txt.append(f"SOURCE_URL: {src.get('source_url')}")
    txt.append(f"RANKING_PAGE: {src.get('ranking_page')}")
    ac = src.get("article_context") or {}
    txt.append(f"ARTICLE_OG_TITLE: {ac.get('og_title','')}")
    txt.append(f"ARTICLE_OG_DESC: {ac.get('og_description','')}")
    txt.append("")

    for b in shorts.get("beats", []):
        txt.append(f'[{b.get("t")}] {b.get("voice")}')
        txt.append(f'  - ONSCREEN: {b.get("onscreen")}')
        txt.append(f'  - BROLL: {b.get("broll")}')
    txt.append("")
    txt.append("TITLE_SHORT: " + (shorts.get("title_short") or ""))
    txt.append("DESCRIPTION: " + (shorts.get("description") or ""))
    txt.append("NOTES: " + (shorts.get("notes") or ""))
    txt.append("")
    txt.append("HASHTAGS: " + " ".join(shorts.get("hashtags", [])))
    txt.append("")
    if "images" in ctx:
        ok = sum(1 for x in (ctx["images"].get("images") or []) if x.get("ok"))
        txt.append(f"IMAGES: {ok}/6 saved to outputs/images/")
        txt.append("IMAGES_MANIFEST: outputs/images_manifest.json")

    _write_text(script_path, "\n".join(txt))

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": src,
        "shorts": shorts,
        "images": ctx.get("images", {}),
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}
    return "PRINT"


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("GENERATOR: " + str(ctx["shorts"].get("_generator")))
    if "images" in ctx:
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

if __name__ == "__main__":
    try:
        ctx = flow.run("LOAD_INPUTS")
        log("\n[끝] OK")
    except Exception:
        (OUT_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
