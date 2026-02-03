# mini_make.py
from __future__ import annotations

import json
import os
import random
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from playwright.sync_api import sync_playwright


# =========================
# 설정
# =========================
OUT_DIR = Path("outputs")
RAW_DIR = OUT_DIR / "images" / "raw"
FINAL_DIR = OUT_DIR / "images" / "final"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)
FINAL_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

# 고정 요구사항
FIXED_IMAGE_COUNT = 8
MAX_PEOPLE = 8

# OpenAI
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.2")
OPENAI_MAX_OUTPUT_TOKENS = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

# OpenAI TTS
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


def _clean_text(s: str) -> str:
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
    req = Request(url, headers=headers or {})
    try:
        with urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
        return json.loads(raw.decode("utf-8"))
    except (HTTPError, URLError) as e:
        return {"_error": True, "_url": url, "_err": str(e)}
    except Exception as e:
        return {"_error": True, "_url": url, "_err": repr(e)}


def _download(url: str, out_path: Path, timeout: int = 60) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
    with urlopen(req, timeout=timeout) as resp:
        data = resp.read()
    out_path.write_bytes(data)


def _require_openai_key() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY가 없습니다. GitHub Secrets에 OPENAI_API_KEY를 추가하세요.")


# =========================
# mini make (flow)
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
# 1) 네이버 연예 랭킹에서 랜덤 기사 선택 (Playwright)
# =========================
def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    """
    return: (title, article_url, ranking_page_url)
    실패하면 예외 -> Actions 실패(디스코드 실패 알림)
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"],
        )
        page = browser.new_page(user_agent=UA, locale="ko-KR", viewport={"width": 420, "height": 900})

        try:
            log(f"[네이버] goto {RANKING_URL}")
            page.goto(RANKING_URL, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)

            # 스크롤 조금 내려서 링크 더 로드(가끔 첫 화면만 잡히는 경우 방지)
            for _ in range(3):
                page.mouse.wheel(0, 1200)
                page.wait_for_timeout(500)

            items = page.evaluate(
                """() => {
                  const links = Array.from(document.querySelectorAll('a'));
                  const out = [];
                  for (const a of links) {
                    const text = (a.innerText || '').trim().replace(/\\s+/g,' ');
                    const href = a.href || '';
                    if (!text || !href) continue;

                    const looksLikeArticle =
                      href.includes('/home/article/') ||
                      href.includes('/ranking/read') ||
                      href.includes('n.news.naver.com/entertain/ranking/article/');

                    if (looksLikeArticle && text.length >= 6 && text.length <= 120) {
                      out.push({text, href});
                    }
                  }
                  const uniq = new Map();
                  for (const x of out) uniq.set(x.text + '|' + x.href, x);
                  return Array.from(uniq.values());
                }"""
            )

            log(f"[네이버] items={len(items)}")

            if not items:
                _write_text(OUT_DIR / "naver_debug.html", page.content())
                page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                raise RuntimeError("렌더링 후에도 기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")

            chosen = random.choice(items)
            return chosen["text"], chosen["href"], RANKING_URL

        finally:
            browser.close()


# =========================
# 2) 기사 컨텍스트(OG/본문 발췌) - 타임아웃 방지형
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
            args=["--no-sandbox", "--disable-dev-shm-usage", "--disable-blink-features=AutomationControlled"],
        )
        context = browser.new_context(user_agent=UA, locale="ko-KR", viewport={"width": 420, "height": 900})
        page = context.new_page()
        try:
            page.goto(article_url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(700)

            og_title = meta_content(page, 'meta[property="og:title"]')
            og_description = meta_content(page, 'meta[property="og:description"]')
            og_image = meta_content(page, 'meta[property="og:image"]')
            published_time = meta_content(page, 'meta[property="article:published_time"]')

            body_text = ""
            for sel in ("article", "main", "div#content", "div.article_body", "div#newsct_article", "div#dic_area"):
                try:
                    el = page.query_selector(sel)
                    if el:
                        t = el.inner_text() or ""
                        if len(t.strip()) > 200:
                            body_text = t
                            break
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


# =========================
# 3) OpenAI로 대본 생성 (항상 OpenAI)
# =========================
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
        f"스타일: {style}\n"
        "입력(JSON)에는 기사 제목/OG설명/본문 발췌가 있을 수 있다.\n"
        "- 입력에 없는 사실을 단정하지 말고, 불확실하면 '기사 제목/발췌 기준'이라고 표현.\n"
        "- 루머/비방/명예훼손/개인정보 추정 금지.\n"
        "- beats는 정확히 6개, t는 반드시 다음 슬롯 중 하나로: "
        + ", ".join(time_slots)
        + "\n"
        "- onscreen은 12자 이내(짧게), voice는 자연스럽게 말로 읽기 좋게.\n"
        "- people: 기사/제목/발췌에서 식별 가능한 인물 이름만(없으면 빈 배열).\n"
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

    # 안전 보정
    beats = data.get("beats") or []
    if len(beats) != 6:
        raise RuntimeError(f"beats length != 6 (got {len(beats)})")
    for i, b in enumerate(beats):
        b["voice"] = _clean_text(b.get("voice", ""))[:240]
        b["onscreen"] = _clean_text(b.get("onscreen", ""))[:12]
        if not b.get("t"):
            b["t"] = time_slots[i]

    data["_generator"] = "openai"
    return data


# =========================
# 4) 인물 이미지 수집 (Wikipedia 썸네일)
# =========================
def wiki_thumbnail_url(name: str) -> str:
    """
    ko/en 위키 요약 API에서 thumbnail.source를 찾음.
    """
    name = _clean_text(name)
    if not name:
        return ""

    headers = {"User-Agent": UA, "Accept": "application/json"}
    for lang in ("ko", "en"):
        url = f"https://{lang}.wikipedia.org/api/rest_v1/page/summary/{quote(name)}"
        data = _http_json(url, headers=headers, timeout=20)
        if data.get("_error"):
            continue
        thumb = (data.get("thumbnail") or {}).get("source") or ""
        if thumb:
            return thumb
    return ""


def render_to_1080x1920_png(image_path: Path, out_png: Path) -> None:
    """
    로컬 이미지를 object-fit: cover로 1080x1920에 꽉 채워 PNG로 스냅샷.
    (Pillow 없이도 크롭/리사이즈 품질 안정화)
    """
    out_png.parent.mkdir(parents=True, exist_ok=True)
    html = f"""
    <html>
      <head><meta charset="utf-8"></head>
      <body style="margin:0; width:1080px; height:1920px; overflow:hidden; background:#000;">
        <img src="{image_path.as_uri()}"
             style="width:1080px; height:1920px; object-fit:cover; display:block;" />
      </body>
    </html>
    """.strip()

    tmp_html = OUT_DIR / "tmp_image_wrap.html"
    tmp_html.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        try:
            page.goto(tmp_html.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_timeout(200)
            page.screenshot(path=str(out_png), full_page=True)
        finally:
            browser.close()

    try:
        tmp_html.unlink(missing_ok=True)
    except Exception:
        pass


def render_card_png(lines: list[str], out_png: Path) -> None:
    """
    이미지가 부족할 때 쓰는 대체 카드(1080x1920 PNG)
    """
    safe_lines = [(_clean_text(x)[:22] if x else "") for x in lines][:6]
    while len(safe_lines) < 6:
        safe_lines.append("")

    html_lines = "".join([f"<div class='line'>{l}</div>" for l in safe_lines if l])

    html = f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{
            margin:0; width:1080px; height:1920px;
            background: linear-gradient(135deg, #111 0%, #1b1b1b 40%, #0f0f0f 100%);
            color: #fff; font-family: Arial, sans-serif;
            display:flex; align-items:center; justify-content:center;
          }}
          .box {{
            width: 900px;
            padding: 60px 60px;
            border-radius: 28px;
            background: rgba(0,0,0,0.35);
            border: 1px solid rgba(255,255,255,0.08);
          }}
          .line {{
            font-size: 56px;
            line-height: 1.15;
            font-weight: 700;
            margin: 14px 0;
            word-break: keep-all;
          }}
          .small {{
            font-size: 38px; opacity: 0.9; font-weight: 500;
          }}
        </style>
      </head>
      <body>
        <div class="box">
          {html_lines if html_lines else "<div class='line'>오늘 연예 랭킹</div><div class='line small'>자동 생성</div>"}
        </div>
      </body>
    </html>
    """.strip()

    tmp_html = OUT_DIR / "tmp_card.html"
    tmp_html.write_text(html, encoding="utf-8")

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = browser.new_page(viewport={"width": 1080, "height": 1920})
        try:
            page.goto(tmp_html.as_uri(), wait_until="load", timeout=60000)
            page.wait_for_timeout(200)
            page.screenshot(path=str(out_png), full_page=True)
        finally:
            browser.close()

    try:
        tmp_html.unlink(missing_ok=True)
    except Exception:
        pass


def build_final_images_8(inputs: dict[str, Any], shorts: dict[str, Any]) -> dict[str, Any]:
    """
    - shorts.people 기반으로 위키 썸네일 다운로드
    - 8장 고정: 부족하면 카드로 채움
    - outputs/images/final/01.png~08.png 생성
    """
    # clean dirs
    if RAW_DIR.exists():
        for p in RAW_DIR.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass
    if FINAL_DIR.exists():
        for p in FINAL_DIR.glob("*"):
            try:
                p.unlink()
            except Exception:
                pass

    topic = _clean_text(shorts.get("topic") or inputs.get("topic") or "오늘 연예 랭킹")
    people: list[str] = shorts.get("people") or []
    people = [p for p in (_clean_text(x) for x in people) if p][:MAX_PEOPLE]

    manifest: dict[str, Any] = {"topic": topic, "people": people, "raw": [], "final": []}

    # 1) raw 다운로드
    for i, person in enumerate(people, start=1):
        url = wiki_thumbnail_url(person)
        if not url:
            manifest["raw"].append({"person": person, "ok": False, "reason": "no_wiki_thumb"})
            continue
        raw_path = RAW_DIR / f"person_{i:02d}.img"
        try:
            _download(url, raw_path, timeout=60)
            manifest["raw"].append({"person": person, "ok": True, "url": url, "file": str(raw_path)})
        except Exception as e:
            manifest["raw"].append({"person": person, "ok": False, "url": url, "reason": repr(e)})

    # 2) final 8장 만들기
    raw_files = [Path(x["file"]) for x in manifest["raw"] if x.get("ok") and x.get("file")]
    idx = 1

    for rf in raw_files[:FIXED_IMAGE_COUNT]:
        out_png = FINAL_DIR / f"{idx:02d}.png"
        render_to_1080x1920_png(rf, out_png)
        manifest["final"].append({"slot": idx, "type": "person", "raw": str(rf), "file": str(out_png)})
        idx += 1

    # 부족분은 카드로 채움
    while idx <= FIXED_IMAGE_COUNT:
        out_png = FINAL_DIR / f"{idx:02d}.png"
        line1 = topic[:18]
        line2 = (people[idx - 2] if 0 <= (idx - 2) < len(people) else "")
        line3 = "오늘 연예 랭킹"
        render_card_png([line1, line2, line3], out_png)
        manifest["final"].append({"slot": idx, "type": "card", "file": str(out_png)})
        idx += 1

    _write_json(OUT_DIR / "images_manifest.json", manifest)
    return manifest


# =========================
# 5) 음성 생성 (OpenAI TTS) - 항상 outputs/voice.mp3로 고정 저장
# =========================
def make_voice_openai(narration: str, out_mp3: Path) -> None:
    _require_openai_key()
    from openai import OpenAI

    narration = _clean_text(narration)
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
# 노드들
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
    ctx["shorts"] = build_60s_shorts_script_openai(ctx["inputs"])
    _write_json(OUT_DIR / "shorts.json", ctx["shorts"])


def node_download_images(ctx: dict[str, Any]):
    ctx["images"] = build_final_images_8(ctx["inputs"], ctx["shorts"])


def node_make_voice(ctx: dict[str, Any]):
    beats = ctx["shorts"]["beats"]
    narration = "\n".join([(b.get("voice") or "").strip() for b in beats if (b.get("voice") or "").strip()])
    voice_path = OUT_DIR / "voice.mp3"  # ✅ 고정 파일명
    make_voice_openai(narration, voice_path)
    ctx["voice"] = {"path": str(voice_path), "chars": len(narration)}


def node_save_outputs(ctx: dict[str, Any]):
    kst = now_kst()
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    # 사람이 보기 쉬운 txt
    shorts = ctx["shorts"]
    inp = ctx["inputs"]
    lines = []
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

    # meta
    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": ctx.get("inputs", {}),
        "shorts": ctx.get("shorts", {}),
        "images_manifest": str(OUT_DIR / "images_manifest.json"),
        "voice": ctx.get("voice", {}),
    }
    _write_json(OUT_DIR / "meta.json", meta)


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("SHORTS_JSON: outputs/shorts.json")
    log("SHORTS_TXT: outputs/shorts.txt")
    log("IMAGES_FINAL: outputs/images/final/01.png ~ 08.png")
    log("VOICE: outputs/voice.mp3")
    log("META: outputs/meta.json")


def build_flow() -> Flow:
    flow = Flow()
    flow.add(Node("LOAD_INPUTS", node_load_inputs))
    flow.add(Node("MAKE_SCRIPT", node_make_script))
    flow.add(Node("DOWNLOAD_IMAGES", node_download_images))
    flow.add(Node("MAKE_VOICE", node_make_voice))
    flow.add(Node("SAVE_OUTPUTS", node_save_outputs))
    flow.add(Node("PRINT", node_print))

    flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
    flow.connect("MAKE_SCRIPT", "DOWNLOAD_IMAGES")
    flow.connect("DOWNLOAD_IMAGES", "MAKE_VOICE")
    flow.connect("MAKE_VOICE", "SAVE_OUTPUTS")
    flow.connect("SAVE_OUTPUTS", "PRINT")
    return flow


if __name__ == "__main__":
    try:
        build_flow().run("LOAD_INPUTS")
        log("END")
    except Exception:
        # GitHub Actions에서 원인 확인 쉽게
        err = traceback.format_exc()
        _write_text(OUT_DIR / "fatal_error.txt", err)
        raise
