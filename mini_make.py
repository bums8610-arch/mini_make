# mini_make.py
from __future__ import annotations

import json
import os
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright


# =========================
# 기본 설정
# =========================
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

FINAL_DIR = OUT_DIR / "images" / "final"
RAW_DIR = OUT_DIR / "images" / "raw"
FINAL_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

RANKING_CANDIDATES = [
    "https://m.entertain.naver.com/ranking",
    "https://entertain.naver.com/ranking",
    "https://n.news.naver.com/entertain/ranking",
]

FIXED_IMAGE_COUNT = 8
MAX_PEOPLE = 8

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

OPENAI_TTS_MODEL = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
OPENAI_TTS_VOICE = os.getenv("OPENAI_TTS_VOICE", "marin")
OPENAI_TTS_INSTRUCTIONS = os.getenv(
    "OPENAI_TTS_INSTRUCTIONS",
    "또박또박, 과장 없이, 뉴스 앵커 톤으로 자연스럽게 읽어줘.",
)


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def _clean(s: Any) -> str:
    if not isinstance(s, str):
        return ""
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text or "", encoding="utf-8")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _http_json(url: str, headers: Optional[dict[str, str]] = None, timeout: int = 30) -> dict[str, Any]:
    req = Request(url, headers=headers or {"User-Agent": UA, "Accept": "application/json"})
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read()
    return json.loads(raw.decode("utf-8", errors="replace"))


def _download(url: str, out_path: Path, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as resp:
        out_path.write_bytes(resp.read())


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 없습니다. GitHub Secrets에 OPENAI_API_KEY를 추가하세요.")


# =========================
# Flow
# =========================
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


# =========================
# 네이버 랭킹: JSON 탐색 유틸
# =========================
def _walk_json(obj: Any):
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _extract_items_from_json(obj: Any) -> list[dict[str, str]]:
    """
    ✅ 도메인 허용:
      - entertain.naver.com
      - n.news.naver.com/entertain
    """
    out: list[dict[str, str]] = []
    for node in _walk_json(obj):
        if not isinstance(node, dict):
            continue

        title = ""
        for k in ("title", "headline", "articleTitle", "subject", "newsTitle", "name"):
            v = node.get(k)
            if isinstance(v, str) and 6 <= len(v.strip()) <= 120:
                title = v.strip()
                break

        url = ""
        for k in ("url", "link", "href", "mobileUrl", "pcUrl"):
            v = node.get(k)
            if isinstance(v, str) and v.strip().startswith("http"):
                url = v.strip()
                break

        if not (title and url):
            continue

        ok = ("entertain.naver.com" in url) or ("n.news.naver.com/entertain" in url)
        if ok:
            out.append({"title": title, "url": url})

    uniq: dict[str, dict[str, str]] = {}
    for it in out:
        uniq[it["title"] + "|" + it["url"]] = it
    return list(uniq.values())


def _dedup(items: list[dict[str, str]]) -> list[dict[str, str]]:
    m: dict[str, dict[str, str]] = {}
    for it in items:
        t = _clean(it.get("title") or it.get("text") or "")
        u = _clean(it.get("url") or it.get("href") or "")
        if t and u:
            m[t + "|" + u] = {"title": t, "url": u}
    return list(m.values())


# =========================
# 1) 랭킹에서 랜덤 기사 선택 (DOM + __NEXT_DATA__ + JSON 응답)
# =========================
def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    json_urls: list[str] = []

    for attempt in range(1, 4):
        for start_url in RANKING_CANDIDATES:
            with sync_playwright() as p:
                browser = p.chromium.launch(
                    headless=True,
                    args=[
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-blink-features=AutomationControlled",
                    ],
                )
                context = browser.new_context(
                    locale="ko-KR",
                    timezone_id="Asia/Seoul",
                    user_agent=UA,
                    viewport={"width": 1280, "height": 720},
                    extra_http_headers={
                        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
                        "Referer": "https://m.entertain.naver.com/",
                    },
                )
                context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")
                page = context.new_page()

                json_items: list[dict[str, str]] = []

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
                    log(f"[네이버] try={attempt}/3 goto {start_url}")
                    resp = page.goto(start_url, wait_until="domcontentloaded", timeout=60000)
                    status = resp.status if resp else -1

                    page.wait_for_timeout(1500)

                    # 스크롤로 더 로드
                    for _ in range(4):
                        page.mouse.wheel(0, 1600)
                        page.wait_for_timeout(600)

                    # DOM 링크 추출(✅ n.news 도 허용)
                    dom_items = page.evaluate(
                        """() => {
                          const out = [];
                          const links = Array.from(document.querySelectorAll('a'));
                          for (const a of links) {
                            const rawHref = a.getAttribute('href') || '';
                            const href = rawHref ? new URL(rawHref, location.origin).href : (a.href || '');
                            if (!href) continue;

                            const t =
                              (a.innerText || a.textContent || a.getAttribute('aria-label') || a.getAttribute('title') || '')
                              .trim()
                              .replace(/\\s+/g,' ');

                            if (!t) continue;

                            const okDomain =
                              href.includes('entertain.naver.com') ||
                              href.includes('n.news.naver.com/entertain');

                            const okPath =
                              href.includes('/article/') ||
                              href.includes('/home/article/') ||
                              href.includes('/ranking/read');

                            if (okDomain && okPath && t.length >= 6 && t.length <= 120) {
                              out.push({title: t, url: href});
                            }
                          }
                          const uniq = new Map();
                          for (const x of out) uniq.set(x.title + '|' + x.url, x);
                          return Array.from(uniq.values());
                        }"""
                    ) or []

                    # __NEXT_DATA__ 파싱
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

                    all_items = _dedup(json_items + next_items + dom_items)
                    log(f"[네이버] status={status} items={len(all_items)} final={page.url}")

                    if all_items:
                        chosen = random.choice(all_items)
                        return chosen["title"], chosen["url"], start_url

                    # 마지막 시도에서만 디버그 저장
                    if attempt == 3 and start_url == RANKING_CANDIDATES[-1]:
                        _write_text(OUT_DIR / "naver_debug.html", page.content())
                        page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                        _write_text(OUT_DIR / "naver_json_urls.txt", "\n".join(json_urls[:400]))

                finally:
                    try:
                        context.close()
                    except Exception:
                        pass
                    try:
                        browser.close()
                    except Exception:
                        pass

    raise RuntimeError("렌더링 후에도 기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")


# =========================
# 2) 기사 컨텍스트 (OG + 발췌)
# =========================
def fetch_article_context(article_url: str) -> dict[str, Any]:
    def meta_content(page, selector: str) -> str:
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
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        context = browser.new_context(
            user_agent=UA,
            locale="ko-KR",
            timezone_id="Asia/Seoul",
            viewport={"width": 420, "height": 900},
        )
        page = context.new_page()
        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(600)

            og_title = meta_content(page, 'meta[property="og:title"]')
            og_description = meta_content(page, 'meta[property="og:description"]')
            og_image = meta_content(page, 'meta[property="og:image"]')
            published_time = meta_content(page, 'meta[property="article:published_time"]')

            body_text = ""
            for sel in ("article", "main", "div#newsct_article", "div#dic_area", "div#content"):
                try:
                    el = page.query_selector(sel)
                    if el:
                        t = el.inner_text() or ""
                        if len(t.strip()) > 200:
                            body_text = t
                            break
                except Exception:
                    continue

            return {
                "final_url": page.url,
                "og_title": _clean(og_title),
                "og_description": _clean(og_description),
                "og_image": _clean(og_image),
                "published_time": _clean(published_time),
                "body_excerpt": _clean(body_text)[:1800],
            }
        finally:
            context.close()
            browser.close()


# =========================
# 3) OpenAI로 대본 생성(항상 OpenAI)
# =========================
def _pick_style() -> str:
    return random.choice([
        "차분하고 사실 중심",
        "속도감 있게 핵심만",
        "댓글 반응 포인트 중심",
        "팩트-추정 구분을 강조",
        "연예뉴스 아나운서 톤",
    ])


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    _require_openai_key()
    from openai import OpenAI

    client = OpenAI()
    style = _pick_style()

    time_slots = ["0-2s", "2-10s", "10-25s", "25-40s", "40-55s", "55-60s"]

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
            "people": {"type": "array", "items": {"type": "string"}, "maxItems": MAX_PEOPLE},
            "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 10},
            "notes": {"type": "string"},
        },
        "required": ["topic", "title_short", "description", "beats", "people", "hashtags", "notes"],
    }

    system = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다.\n"
        f"스타일: {style}\n"
        "- 입력에 없는 사실을 단정하지 말고, 불확실하면 '기사 제목/발췌 기준'이라고 표현.\n"
        "- 루머/비방/명예훼손/개인정보 추정 금지.\n"
        f"- beats는 정확히 6개. t는 다음 중 하나: {', '.join(time_slots)}\n"
        "- onscreen은 12자 이내(짧게), voice는 말로 읽기 좋게.\n"
        "- people은 기사/제목/발췌에서 식별 가능한 인물만(없으면 빈 배열).\n"
    )

    payload = {
        "topic_from_ranking": inputs.get("topic", ""),
        "source_url": inputs.get("source_url", ""),
        "ranking_page": inputs.get("ranking_page", ""),
        "article_ctx": inputs.get("article_ctx", {}),
        "time_slots": time_slots,
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

    # 보정
    beats = data.get("beats") or []
    if len(beats) != 6:
        raise RuntimeError(f"beats length != 6 (got {len(beats)})")
    for i, b in enumerate(beats):
        b["t"] = b.get("t") or time_slots[i]
        b["voice"] = _clean(b.get("voice"))[:260]
        b["onscreen"] = _clean(b.get("onscreen"))[:12]

    data["topic"] = _clean(data.get("topic")) or _clean(inputs.get("topic")) or "오늘 연예 랭킹"
    data["title_short"] = _clean(data.get("title_short"))[:40]
    data["description"] = _clean(data.get("description"))[:200]
    data["people"] = [p for p in (_clean(x) for x in (data.get("people") or [])) if p][:MAX_PEOPLE]
    data["hashtags"] = [h for h in (_clean(x) for x in (data.get("hashtags") or [])) if h][:10]
    data["notes"] = _clean(data.get("notes"))
    data["_generator"] = "openai"

    return data


# =========================
# 4) 인물 이미지(위키 썸네일) + OG 이미지 + 카드로 8장 채우기
# =========================
def wiki_thumbnail_url(name: str) -> str:
    name = _clean(name)
    if not name:
        return ""
    headers = {"User-Agent": UA, "Accept": "application/json"}
    for lang in ("ko", "en"):
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(name)}"
        try:
            data = _http_json(url, headers=headers, timeout=20)
        except Exception:
            continue
        thumb = (data.get("thumbnail") or {}).get("source") or ""
        if thumb:
            return thumb
    return ""


def render_to_1080x1920_png(image_path: Path, out_png: Path) -> None:
    html = f"""
    <html>
      <body style="margin:0;width:1080px;height:1920px;overflow:hidden;background:#000;">
        <img src="{image_path.as_uri()}" style="width:1080px;height:1920px;object-fit:cover;display:block;"/>
      </body>
    </html>
    """.strip()

    tmp = OUT_DIR / "tmp_wrap.html"
    tmp.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        try:
            page.goto(tmp.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_timeout(150)
            page.screenshot(path=str(out_png), full_page=True)
        finally:
            browser.close()

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def render_card_png(lines: list[str], out_png: Path) -> None:
    safe = [(_clean(x)[:22] if x else "") for x in lines][:6]
    while len(safe) < 6:
        safe.append("")
    html_lines = "".join([f"<div class='line'>{l}</div>" for l in safe if l])

    html = f"""
    <html>
      <head>
        <meta charset="utf-8"/>
        <style>
          body {{
            margin:0;width:1080px;height:1920px;background:#0b0b0b;color:#fff;
            display:flex;align-items:center;justify-content:center;font-family:Arial,sans-serif;
          }}
          .box {{
            width:900px;padding:70px;border-radius:28px;
            background:rgba(255,255,255,0.06);
            border:1px solid rgba(255,255,255,0.08);
          }}
          .line {{
            font-size:58px;line-height:1.15;font-weight:800;margin:16px 0;word-break:keep-all;
          }}
          .small {{ font-size:40px;opacity:0.9;font-weight:600; }}
        </style>
      </head>
      <body>
        <div class="box">
          {html_lines if html_lines else "<div class='line'>오늘 연예 랭킹</div><div class='line small'>자동 생성</div>"}
        </div>
      </body>
    </html>
    """.strip()

    tmp = OUT_DIR / "tmp_card.html"
    tmp.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        try:
            page.goto(tmp.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_timeout(150)
            page.screenshot(path=str(out_png), full_page=True)
        finally:
            browser.close()

    try:
        tmp.unlink(missing_ok=True)
    except Exception:
        pass


def build_final_images_8(inputs: dict[str, Any], shorts: dict[str, Any]) -> dict[str, Any]:
    # 폴더 청소
    for p in RAW_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass
    for p in FINAL_DIR.glob("*"):
        try:
            p.unlink()
        except Exception:
            pass

    topic = _clean(shorts.get("topic") or inputs.get("topic") or "오늘 연예 랭킹")
    people = [p for p in (_clean(x) for x in (shorts.get("people") or [])) if p][:MAX_PEOPLE]
    og_image = _clean((inputs.get("article_ctx") or {}).get("og_image"))

    manifest: dict[str, Any] = {"topic": topic, "people": people, "og_image": og_image, "final": []}

    # 1) slot1: og_image 시도
    slot = 1
    if og_image:
        try:
            raw = RAW_DIR / "og_image.img"
            _download(og_image, raw, timeout=60)
            out = FINAL_DIR / f"{slot:02d}.png"
            render_to_1080x1920_png(raw, out)
            manifest["final"].append({"slot": slot, "type": "og_image", "file": str(out), "src": og_image})
            slot += 1
        except Exception:
            pass

    # 2) people thumbnails
    for person in people:
        if slot > FIXED_IMAGE_COUNT:
            break
        url = wiki_thumbnail_url(person)
        if not url:
            continue
        try:
            raw = RAW_DIR / f"person_{slot:02d}.img"
            _download(url, raw, timeout=60)
            out = FINAL_DIR / f"{slot:02d}.png"
            render_to_1080x1920_png(raw, out)
            manifest["final"].append({"slot": slot, "type": "person", "person": person, "file": str(out), "src": url})
            slot += 1
        except Exception:
            continue

    # 3) 부족분 카드로 채우기
    while slot <= FIXED_IMAGE_COUNT:
        out = FINAL_DIR / f"{slot:02d}.png"
        line1 = topic[:18]
        line2 = people[slot - 2] if 0 <= (slot - 2) < len(people) else ""
        line3 = "오늘 연예 랭킹"
        render_card_png([line1, line2, line3], out)
        manifest["final"].append({"slot": slot, "type": "card", "file": str(out)})
        slot += 1

    _write_json(OUT_DIR / "images_manifest.json", manifest)
    return manifest


# =========================
# 5) 음성(OpenAI TTS) => outputs/voice.mp3 고정
# =========================
def make_voice_openai(narration: str, out_mp3: Path) -> None:
    _require_openai_key()
    from openai import OpenAI

    narration = _clean(narration)
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


# =========================
# Nodes
# =========================
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
    shorts = build_60s_shorts_script_openai(ctx["inputs"])
    ctx["shorts"] = shorts
    _write_json(OUT_DIR / "shorts.json", shorts)


def node_download_images(ctx: dict[str, Any]):
    ctx["images"] = build_final_images_8(ctx["inputs"], ctx["shorts"])


def node_make_voice(ctx: dict[str, Any]):
    beats = ctx["shorts"]["beats"]
    narration = "\n".join([_clean(b.get("voice")) for b in beats if _clean(b.get("voice"))])
    voice_path = OUT_DIR / "voice.mp3"
    make_voice_openai(narration, voice_path)
    ctx["voice"] = {"path": str(voice_path), "chars": len(narration)}


def node_save_outputs(ctx: dict[str, Any]):
    kst = now_kst()
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    inp = ctx["inputs"]
    shorts = ctx["shorts"]

    # shorts.txt
    lines: list[str] = []
    lines.append(f"DATE(KST): {kst.isoformat()}")
    lines.append(f"RUN_ID: {run_id}")
    lines.append(f"TOPIC: {shorts.get('topic')}")
    lines.append(f"TITLE_SHORT: {shorts.get('title_short')}")
    lines.append(f"SOURCE_URL: {inp.get('source_url')}")
    lines.append("")
    for b in shorts.get("beats", []):
        lines.append(f"[{b.get('t')}] {b.get('voice')}")
        lines.append(f"  - ONSCREEN: {b.get('onscreen')}")
    lines.append("")
    lines.append("PEOPLE: " + ", ".join(shorts.get("people") or []))
    lines.append("HASHTAGS: " + " ".join(shorts.get("hashtags") or []))
    lines.append("NOTES: " + (shorts.get("notes") or ""))
    _write_text(OUT_DIR / "shorts.txt", "\n".join(lines))

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": inp,
        "shorts": shorts,
        "voice": ctx.get("voice", {}),
        "images_manifest": str(OUT_DIR / "images_manifest.json"),
    }
    _write_json(OUT_DIR / "meta.json", meta)


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + _clean(ctx["inputs"]["topic"]))
    log("SOURCE_URL: " + _clean(ctx["inputs"]["source_url"]))
    log("SHORTS_JSON: outputs/shorts.json")
    log("SHORTS_TXT: outputs/shorts.txt")
    log("IMAGES_FINAL: outputs/images/final/01.png ~ 08.png")
    log("VOICE: outputs/voice.mp3")
    log("META: outputs/meta.json")


def build_flow() -> Flow:
    f = Flow()
    f.add(Node("LOAD_INPUTS", node_load_inputs))
    f.add(Node("MAKE_SCRIPT", node_make_script))
    f.add(Node("DOWNLOAD_IMAGES", node_download_images))
    f.add(Node("MAKE_VOICE", node_make_voice))
    f.add(Node("SAVE_OUTPUTS", node_save_outputs))
    f.add(Node("PRINT", node_print))

    f.connect("LOAD_INPUTS", "MAKE_SCRIPT")
    f.connect("MAKE_SCRIPT", "DOWNLOAD_IMAGES")
    f.connect("DOWNLOAD_IMAGES", "MAKE_VOICE")
    f.connect("MAKE_VOICE", "SAVE_OUTPUTS")
    f.connect("SAVE_OUTPUTS", "PRINT")
    return f


if __name__ == "__main__":
    try:
        build_flow().run("LOAD_INPUTS")
        log("END")
    except Exception:
        _write_text(OUT_DIR / "fatal_error.txt", traceback.format_exc())
        raise
