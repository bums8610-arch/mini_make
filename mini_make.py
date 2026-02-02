from __future__ import annotations

import json
import os
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote_plus
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError

# =========================
# 기본 설정
# =========================
OUT_DIR = Path("outputs")
IMG_DIR = OUT_DIR / "images"
PEOPLE_DIR = IMG_DIR / "people"
ARTICLE_DIR = IMG_DIR / "article"
FILL_DIR = IMG_DIR / "fill"
FINAL_DIR = IMG_DIR / "final"
for d in (OUT_DIR, IMG_DIR, PEOPLE_DIR, ARTICLE_DIR, FILL_DIR, FINAL_DIR):
    d.mkdir(parents=True, exist_ok=True)

RANKING_URL = os.getenv("RANKING_URL", "https://m.entertain.naver.com/ranking").strip()

UA = os.getenv(
    "USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36",
)

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

# 이미지 수집 옵션
DOWNLOAD_PERSON_IMAGES = os.getenv("DOWNLOAD_PERSON_IMAGES", "1").strip() == "1"
DOWNLOAD_ARTICLE_IMAGES = os.getenv("DOWNLOAD_ARTICLE_IMAGES", "0").strip() == "1"
MAX_PEOPLE = int(os.getenv("MAX_PEOPLE", "5"))
MAX_ARTICLE_IMAGES = int(os.getenv("MAX_ARTICLE_IMAGES", "12"))

# ✅ 최종 이미지 고정 개수
FIXED_IMAGE_COUNT = int(os.getenv("FIXED_IMAGE_COUNT", "8"))  # 요청: 8장 고정


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def _clean_text(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _safe_filename(name: str, max_len: int = 80) -> str:
    name = _clean_text(name)
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = re.sub(r"\s+", "_", name).strip("_")
    return (name[:max_len] or "file").strip("_")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _http_get_bytes(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 60) -> bytes:
    req = Request(url, headers=headers or {})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _download(url: str, dst: Path, timeout: int = 60) -> bool:
    try:
        data = _http_get_bytes(url, headers={"User-Agent": UA, "Accept": "*/*"}, timeout=timeout)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(data)
        return True
    except Exception:
        return False


def _http_json(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 30) -> Any:
    try:
        raw = _http_get_bytes(url, headers=headers, timeout=timeout)
        return json.loads(raw.decode("utf-8"))
    except HTTPError as e:
        body = ""
        try:
            body = e.read().decode("utf-8", "replace")
        except Exception:
            body = ""
        return {"_error": True, "_status": int(getattr(e, "code", 0) or 0), "_url": url, "_body": body[:3000]}
    except URLError as e:
        return {"_error": True, "_status": 0, "_url": url, "_body": str(e)}
    except Exception as e:
        return {"_error": True, "_status": 0, "_url": url, "_body": str(e)}


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

    def add(self, node: Node) -> None:
        self.nodes[node.name] = node

    def connect(self, a: str, b: str) -> None:
        self.links[a] = b

    def run(self, start: str) -> dict[str, Any]:
        ctx: dict[str, Any] = {}
        cur: str | None = start
        while cur is not None:
            log(f"[실행] {cur}")
            nxt = self.nodes[cur].fn(ctx)
            cur = nxt if isinstance(nxt, str) else self.links.get(cur)
        return ctx


# ===========================
# 1) 네이버 연예 랭킹: 랜덤 기사 선택
# ===========================
def _walk_json(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _extract_items_from_json(obj: Any) -> list[dict[str, str]]:
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

    uniq: dict[str, dict[str, str]] = {}
    for it in out:
        uniq[it["title"] + "|" + it["url"]] = it
    return list(uniq.values())


def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(user_agent=UA, locale="ko-KR", viewport={"width": 420, "height": 900})
        page = context.new_page()

        json_items: list[dict[str, str]] = []
        json_urls: list[str] = []

        def on_response(resp):
            try:
                ct = resp.headers.get("content-type", "")
                if "json" not in ct:
                    return
                url = resp.url
                if "ranking" not in url and "entertain" not in url:
                    return
                data = resp.json()
                items = _extract_items_from_json(data)
                if items:
                    json_items.extend(items)
                    json_urls.append(url)
            except Exception:
                return

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
                      if (!rawHref) continue;
                      const href = new URL(rawHref, location.origin).href;
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

            next_items: list[dict[str, str]] = []
            next_data_text = page.evaluate(
                """() => {
                    const el = document.querySelector('#__NEXT_DATA__');
                    return el ? el.textContent : '';
                }"""
            )
            if isinstance(next_data_text, str) and len(next_data_text) > 50:
                try:
                    next_json = json.loads(next_data_text)
                    next_items = _extract_items_from_json(next_json)
                except Exception:
                    next_items = []

            all_map: dict[str, dict[str, str]] = {}
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
                _write_text(OUT_DIR / "naver_json_urls.txt", "\n".join(json_urls[:200]))
                raise RuntimeError("기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")

            chosen = random.choice(all_items)
            return chosen["title"], chosen["url"], RANKING_URL

        finally:
            context.close()
            browser.close()


# ===========================
# 2) 기사 컨텍스트(OG + 본문 일부 + og:image) — 타임아웃 방지 버전
# ===========================
def fetch_article_context(article_url: str) -> dict[str, Any]:
    def meta_content(page, selector: str) -> str:
        # locator로 기다리지 않고, 없으면 즉시 ""
        try:
            el = page.query_selector(selector)
            if not el:
                return ""
            return el.get_attribute("content") or ""
        except Exception:
            return ""

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(user_agent=UA, locale="ko-KR", viewport={"width": 420, "height": 900})
        page = context.new_page()
        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(800)

            og_title = meta_content(page, 'meta[property="og:title"]')
            og_description = meta_content(page, 'meta[property="og:description"]')
            og_image = meta_content(page, 'meta[property="og:image"]')
            published_time = meta_content(page, 'meta[property="article:published_time"]')

            body_text = ""
            for sel in ("article", "main", "div#content", "div.article_body", "div#newsct_article", "div#dic_area"):
                try:
                    loc = page.locator(sel)
                    if loc.count() > 0:
                        body_text = loc.first.inner_text() or ""
                        if len(body_text.strip()) > 200:
                            break
                except PlaywrightTimeoutError:
                    continue
                except Exception:
                    continue

            body_excerpt = _clean_text(body_text)[:1800]

            return {
                "og_title": _clean_text(og_title),
                "og_description": _clean_text(og_description),
                "og_image": _clean_text(og_image),
                "published_time": _clean_text(published_time),
                "body_excerpt": body_excerpt,
                "final_url": page.url,
            }
        finally:
            context.close()
            browser.close()


# ===========================
# 3) OpenAI로 대본 생성(항상 OpenAI만)
# ===========================
def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 없습니다. GitHub Secrets에 OPENAI_API_KEY를 추가하세요.")


def _pick_style() -> str:
    styles = [
        "차분하고 사실 중심",
        "속도감 있게 핵심만",
        "댓글 반응 포인트 중심",
        "팩트-추정 구분을 강조",
        "연예뉴스 아나운서 톤",
    ]
    return random.choice(styles)


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    _require_openai_key()
    from openai import OpenAI

    client = OpenAI()
    style = _pick_style()

    schema: dict[str, Any] = {
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
                    },
                    "required": ["t", "voice", "onscreen"],
                },
            },
            "people": {
                "type": "array",
                "minItems": 0,
                "maxItems": MAX_PEOPLE,
                "items": {"type": "string"},
            },
            "hashtags": {"type": "array", "minItems": 4, "maxItems": 10, "items": {"type": "string"}},
            "notes": {"type": "string"},
        },
        "required": ["topic", "title_short", "description", "beats", "people", "hashtags", "notes"],
    }

    time_slots = ["0-2s", "2-10s", "10-25s", "25-40s", "40-55s", "55-60s"]

    system = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다.\n"
        "입력(JSON)에는 기사 제목/OG설명/본문 발췌가 있을 수 있다.\n"
        "- 입력에 없는 사실을 단정하지 말고, 불확실하면 '기사 제목/발췌 기준'이라고 표현.\n"
        "- 과장/루머/비방/명예훼손/개인정보 추정 금지.\n"
        "- 정확히 6구간으로 작성: 0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s.\n"
        "- onscreen: 12자 이내(한국어), voice: 한 구간당 1~2문장.\n"
        f"- 전체 톤/스타일: {style}\n"
        "- people: 기사 제목/발췌에 등장하는 실명 인물만 배열로 출력. 없으면 [].\n"
        "출력은 반드시 JSON(schema strict)만."
    )

    payload = {
        "topic": inputs["topic"],
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "article_context": inputs.get("article_context", {}),
    }

    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
        text={"format": {"type": "json_schema", "name": "shorts_script", "strict": True, "schema": schema}},
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
    )

    data = json.loads(resp.output_text)

    beats = data.get("beats") if isinstance(data.get("beats"), list) else []
    if len(beats) != 6:
        raise RuntimeError("OpenAI 응답 beats 형식이 올바르지 않습니다(6개 필수).")

    for i, b in enumerate(beats):
        b["t"] = time_slots[i]
        b["voice"] = _clean_text(b.get("voice", ""))[:240]
        b["onscreen"] = _clean_text(b.get("onscreen", ""))[:12]

    ppl = data.get("people")
    if not isinstance(ppl, list):
        ppl = []
    norm = []
    seen = set()
    for x in ppl:
        name = _clean_text(x)
        if not name or len(name) > 40:
            continue
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        norm.append(name)
    data["people"] = norm[:MAX_PEOPLE]

    data["_generator"] = f"openai:{OPENAI_MODEL}"
    data["_style"] = style
    return data


# ===========================
# 4) 인물 이미지 다운로드 (Wikipedia ko/en 대표 이미지)
# ===========================
def _wiki_page_image(person_name: str, lang: str = "ko") -> dict[str, Any]:
    title = quote_plus(person_name)
    url = (
        f"https://{lang}.wikipedia.org/w/api.php"
        f"?action=query&format=json&prop=pageimages&piprop=original&redirects=1&titles={title}"
    )
    data = _http_json(url, headers={"User-Agent": UA}, timeout=20)
    if isinstance(data, dict) and data.get("_error"):
        return {"ok": False, "lang": lang, "error": data}

    pages = (((data or {}).get("query") or {}).get("pages") or {})
    if not isinstance(pages, dict) or not pages:
        return {"ok": False, "lang": lang, "error": "no_pages"}

    for _, p in pages.items():
        if not isinstance(p, dict):
            continue
        original = p.get("original") or {}
        src = original.get("source")
        if isinstance(src, str) and src.startswith("http"):
            return {"ok": True, "lang": lang, "page_title": p.get("title"), "image_url": src}
    return {"ok": False, "lang": lang, "error": "no_image"}


def download_people_images(ctx: dict[str, Any]) -> dict[str, Any]:
    people = (ctx.get("shorts") or {}).get("people") or []
    if not DOWNLOAD_PERSON_IMAGES:
        return {"ok": False, "skipped": True, "reason": "DOWNLOAD_PERSON_IMAGES=0"}

    if not isinstance(people, list) or not people:
        return {"ok": True, "people": [], "files": []}

    results = []
    files = []

    for name in people:
        safe = _safe_filename(name)
        info = _wiki_page_image(name, "ko")
        if not info.get("ok"):
            info = _wiki_page_image(name, "en")

        rec = {"name": name, "source": "wikipedia", **info}
        if info.get("ok") and isinstance(info.get("image_url"), str):
            ext = ".jpg"
            url = info["image_url"]
            tail = url.split("?")[0].split("/")[-1]
            if "." in tail:
                eg = "." + tail.split(".")[-1].lower()
                if 2 <= len(eg) <= 5:
                    ext = eg
            path = PEOPLE_DIR / f"{safe}{ext}"
            ok = _download(url, path, timeout=60)
            rec["saved_path"] = str(path) if ok else ""
            rec["download_ok"] = ok
            if ok:
                files.append(str(path))
        else:
            rec["download_ok"] = False

        results.append(rec)

    manifest = {"ok": True, "people": results, "files": files}
    _write_json(OUT_DIR / "people_images_manifest.json", manifest)
    return manifest


# ===========================
# 5) (옵션) 기사 이미지 다운로드: og:image + img 태그
# ===========================
def download_article_images(ctx: dict[str, Any]) -> dict[str, Any]:
    if not DOWNLOAD_ARTICLE_IMAGES:
        return {"ok": False, "skipped": True, "reason": "DOWNLOAD_ARTICLE_IMAGES=0"}

    inputs = ctx.get("inputs") or {}
    url = inputs.get("source_url")
    if not isinstance(url, str) or not url:
        return {"ok": False, "error": "no_source_url"}

    images: list[str] = []
    ac = (inputs.get("article_context") or {})
    og = ac.get("og_image")
    if isinstance(og, str) and og.startswith("http"):
        images.append(og)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(user_agent=UA, locale="ko-KR", viewport={"width": 420, "height": 900})
        page = context.new_page()
        try:
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)
            img_urls = page.evaluate(
                """() => {
                    const out = [];
                    const imgs = Array.from(document.querySelectorAll('img'));
                    for (const im of imgs) {
                      const src = im.currentSrc || im.src || '';
                      if (!src) continue;
                      if (!src.startsWith('http')) continue;
                      const w = im.naturalWidth || im.width || 0;
                      const h = im.naturalHeight || im.height || 0;
                      if (w && h && (w < 200 || h < 200)) continue;
                      out.push(src);
                    }
                    return Array.from(new Set(out));
                }"""
            ) or []
            if isinstance(img_urls, list):
                for u in img_urls:
                    if isinstance(u, str) and u.startswith("http"):
                        images.append(u)
        finally:
            context.close()
            browser.close()

    uniq = []
    seen = set()
    for u in images:
        key = u.split("#")[0]
        if key in seen:
            continue
        seen.add(key)
        uniq.append(u)

    uniq = uniq[:MAX_ARTICLE_IMAGES]

    results = []
    files = []
    for i, u in enumerate(uniq, start=1):
        ext = ".jpg"
        fname = u.split("?")[0].split("/")[-1]
        if "." in fname:
            eg = "." + fname.split(".")[-1].lower()
            if 2 <= len(eg) <= 5:
                ext = eg
        path = ARTICLE_DIR / f"article_{i:02d}{ext}"
        ok = _download(u, path, timeout=60)
        results.append({"index": i, "url": u, "saved_path": str(path) if ok else "", "download_ok": ok})
        if ok:
            files.append(str(path))

    manifest = {"ok": True, "source_url": url, "items": results, "files": files}
    _write_json(OUT_DIR / "article_images_manifest.json", manifest)
    return manifest


# ===========================
# 6) 부족하면 자동 생성 카드(PNG)로 채워서 최종 8장 고정
# ===========================
def _render_card_png(path: Path, title: str, subtitle: str) -> bool:
    # Playwright로 간단한 HTML 카드 렌더 -> screenshot PNG
    html = f"""
    <html><head><meta charset="utf-8"/>
    <style>
      body {{
        margin:0; width:1080px; height:1920px;
        background: linear-gradient(160deg, #0b0f1a, #1a1030);
        font-family: Arial, "Apple SD Gothic Neo", "Malgun Gothic", sans-serif;
        color: #ffffff;
        display:flex; align-items:center; justify-content:center;
      }}
      .card {{
        width: 920px; height: 1640px;
        border-radius: 48px;
        background: rgba(255,255,255,0.06);
        box-shadow: 0 30px 80px rgba(0,0,0,0.45);
        border: 1px solid rgba(255,255,255,0.10);
        padding: 96px 72px;
        box-sizing:border-box;
        display:flex; flex-direction:column; justify-content:space-between;
      }}
      .kicker {{
        font-size: 44px; opacity: 0.85; letter-spacing: 0.5px;
      }}
      .title {{
        margin-top: 24px;
        font-size: 86px; font-weight: 800; line-height: 1.06;
      }}
      .subtitle {{
        margin-top: 36px;
        font-size: 48px; opacity: 0.9; line-height: 1.3;
        white-space: pre-wrap;
      }}
      .footer {{
        font-size: 34px; opacity: 0.65;
        display:flex; justify-content:space-between;
      }}
      .pill {{
        padding: 16px 26px;
        border-radius: 999px;
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.10);
      }}
    </style></head>
    <body>
      <div class="card">
        <div>
          <div class="kicker">NAVER ENTERTAINMENT RANKING</div>
          <div class="title">{title}</div>
          <div class="subtitle">{subtitle}</div>
        </div>
        <div class="footer">
          <div class="pill">AUTO SHORTS</div>
          <div class="pill">{now_kst().strftime("%Y-%m-%d %H:%M KST")}</div>
        </div>
      </div>
    </body></html>
    """.strip()

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        try:
            page.set_content(html, wait_until="domcontentloaded")
            page.wait_for_timeout(300)
            path.parent.mkdir(parents=True, exist_ok=True)
            page.screenshot(path=str(path), full_page=True)
            return True
        finally:
            browser.close()


def ensure_fixed_8_images(ctx: dict[str, Any]) -> dict[str, Any]:
    # 1) 후보 파일 모으기(사람 -> 기사)
    cand: list[dict[str, Any]] = []
    downloads = ctx.get("downloads") or {}

    ppl_files = ((downloads.get("people") or {}).get("files") or [])
    for f in ppl_files:
        cand.append({"kind": "people", "path": f})

    art_files = ((downloads.get("article") or {}).get("files") or [])
    for f in art_files:
        cand.append({"kind": "article", "path": f})

    # dedupe
    seen = set()
    uniq: list[dict[str, Any]] = []
    for it in cand:
        p = str(it.get("path") or "")
        if not p or p in seen:
            continue
        seen.add(p)
        uniq.append(it)

    # 2) 8장 넘으면 8장만
    chosen = uniq[:FIXED_IMAGE_COUNT]

    # 3) 부족하면 fill 카드 생성으로 채우기
    need = FIXED_IMAGE_COUNT - len(chosen)
    if need > 0:
        topic = _clean_text((ctx.get("shorts") or {}).get("topic") or (ctx.get("inputs") or {}).get("topic") or "")
        beats = (ctx.get("shorts") or {}).get("beats") or []
        ons = []
        if isinstance(beats, list):
            for b in beats:
                t = _clean_text((b or {}).get("onscreen") or "")
                if t:
                    ons.append(t)

        for i in range(need):
            subtitle = ons[i % len(ons)] if ons else "핵심만 빠르게 정리"
            path = FILL_DIR / f"fill_{i+1:02d}.png"
            ok = _render_card_png(path, title=(topic or "오늘의 연예 랭킹"), subtitle=subtitle)
            if ok:
                chosen.append({"kind": "fill", "path": str(path)})
            else:
                # 최후: 빈 파일 방지용(이 경우 런 실패)
                raise RuntimeError("fill 이미지 생성 실패")

    # 4) 최종 8장을 final 폴더에 01~08로 고정 복사(확장자 png로 통일)
    final_items = []
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    # final 폴더 초기화(이전 파일이 섞이지 않게)
    for old in FINAL_DIR.glob("*"):
        try:
            old.unlink()
        except Exception:
            pass

    for idx, it in enumerate(chosen, start=1):
        src = Path(str(it["path"]))
        dst = FINAL_DIR / f"{idx:02d}.png"
        if src.suffix.lower() == ".png":
            dst.write_bytes(src.read_bytes())
        else:
            # jpg/webp 등은 playwright로 다시 png로 변환
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
                page = browser.new_page(viewport={"width": 1080, "height": 1920})
                try:
                    # file:// 로 로드해서 스크린샷으로 png 통일
                    page.goto(f"file://{src.resolve()}")
                    page.wait_for_timeout(200)
                    page.screenshot(path=str(dst), full_page=True)
                finally:
                    browser.close()

        final_items.append({"index": idx, "kind": it["kind"], "path": str(dst)})

    manifest = {"ok": True, "count": FIXED_IMAGE_COUNT, "items": final_items, "final_dir": str(FINAL_DIR)}
    _write_json(OUT_DIR / "images_fixed_manifest.json", manifest)
    return manifest


# ===========================
# Nodes
# ===========================
def node_load_inputs(ctx: dict[str, Any]) -> None:
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()

    try:
        article_ctx = fetch_article_context(article_url)
    except Exception as e:
        _write_text(OUT_DIR / "article_context_error.txt", traceback.format_exc())
        article_ctx = {"error": str(e), "final_url": article_url}

    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
        "article_context": article_ctx,
    }


def node_make_script(ctx: dict[str, Any]) -> None:
    ctx["shorts"] = build_60s_shorts_script_openai(ctx["inputs"])
    _write_json(OUT_DIR / "shorts.json", ctx["shorts"])


def node_download_images(ctx: dict[str, Any]) -> None:
    ctx["downloads"] = {}

    # people
    try:
        ctx["downloads"]["people"] = download_people_images(ctx)
    except Exception:
        _write_text(OUT_DIR / "people_images_error.txt", traceback.format_exc())
        ctx["downloads"]["people"] = {"ok": False, "error": "people_images_failed"}

    # article (optional)
    try:
        ctx["downloads"]["article"] = download_article_images(ctx)
    except Exception:
        _write_text(OUT_DIR / "article_images_error.txt", traceback.format_exc())
        ctx["downloads"]["article"] = {"ok": False, "error": "article_images_failed"}

    # ✅ 최종 8장 고정 생성
    ctx["images_fixed"] = ensure_fixed_8_images(ctx)


def node_save_files(ctx: dict[str, Any]) -> None:
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    src = ctx["inputs"]
    shorts = ctx["shorts"]
    downloads = ctx.get("downloads", {})
    fixed = ctx.get("images_fixed", {})

    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"meta_{stamp}_{run_id}.json"

    txt: list[str] = []
    txt.append(f"DATE(KST): {kst.isoformat()}")
    txt.append(f"GENERATOR: {shorts.get('_generator')}")
    txt.append(f"STYLE: {shorts.get('_style')}")
    txt.append(f"TOPIC: {shorts.get('topic')}")
    txt.append(f"SOURCE_URL: {src['source_url']}")
    txt.append(f"RANKING_PAGE: {src['ranking_page']}")
    txt.append("")
    ac = src.get("article_context", {}) or {}
    txt.append(f"OG_TITLE: {ac.get('og_title','')}")
    txt.append(f"OG_DESC: {ac.get('og_description','')}")
    txt.append(f"OG_IMAGE: {ac.get('og_image','')}")
    txt.append("")

    for b in shorts.get("beats", []):
        txt.append(f"[{b.get('t')}] {b.get('voice')}")
        txt.append(f"  - ONSCREEN: {b.get('onscreen')}")
    txt.append("")
    txt.append("PEOPLE: " + ", ".join(shorts.get("people", [])))
    txt.append("HASHTAGS: " + " ".join(shorts.get("hashtags", [])))

    txt.append("")
    txt.append(f"FIXED_IMAGES: {fixed.get('count')} -> outputs/images/final/01.png ~ 08.png")
    txt.append(f"DOWNLOAD_ARTICLE_IMAGES: {int(DOWNLOAD_ARTICLE_IMAGES)} (기사 이미지도 필요하면 1로)")
    txt.append(f"DOWNLOAD_PERSON_IMAGES: {int(DOWNLOAD_PERSON_IMAGES)}")

    script_path.write_text("\n".join(txt), encoding="utf-8")

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": src,
        "shorts": shorts,
        "downloads": downloads,
        "images_fixed": fixed,
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}


def node_print(ctx: dict[str, Any]) -> None:
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("PEOPLE: " + ", ".join(ctx["shorts"].get("people", [])))
    log("FINAL IMAGES: outputs/images/final/01.png ~ 08.png")
    log("SCRIPT FILE: " + ctx["outputs"]["script_path"])
    log("META FILE: " + ctx["outputs"]["meta_path"])


def build_flow() -> Flow:
    flow = Flow()
    flow.add(Node("LOAD_INPUTS", node_load_inputs))
    flow.add(Node("MAKE_SCRIPT", node_make_script))
    flow.add(Node("DOWNLOAD_IMAGES", node_download_images))
    flow.add(Node("SAVE_FILES", node_save_files))
    flow.add(Node("PRINT", node_print))

    flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
    flow.connect("MAKE_SCRIPT", "DOWNLOAD_IMAGES")
    flow.connect("DOWNLOAD_IMAGES", "SAVE_FILES")
    flow.connect("SAVE_FILES", "PRINT")
    return flow


if __name__ == "__main__":
    try:
        build_flow().run("LOAD_INPUTS")
        log("\n[끝] OK")
    except Exception:
        _write_text(OUT_DIR / "error.txt", traceback.format_exc())
        raise
