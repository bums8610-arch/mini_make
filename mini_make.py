# mini_make.py
from __future__ import annotations

import os
import json
import random
import re
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from typing import Any, Optional

from playwright.sync_api import sync_playwright

# ===========================
# 설정
# ===========================
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"

# 대본 길이(대략) 품질 게이트: voice(6구간 합) 글자수 기준
VOICE_CHARS_MIN = int(os.getenv("VOICE_CHARS_MIN", "360"))
VOICE_CHARS_MAX = int(os.getenv("VOICE_CHARS_MAX", "620"))

# 스타일: 환경변수로 고정 가능(없으면 랜덤)
# 예) SHORTS_STYLE=timeline
SHORTS_STYLE = os.getenv("SHORTS_STYLE", "").strip().lower()


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


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
# 네이버: 랭킹에서 랜덤 토픽 1개 뽑기
# (DOM + NEXT_DATA + 네트워크 JSON 폴백)
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
    """
    1) DOM에서 링크 수집(스크롤 포함)
    2) 실패하면 __NEXT_DATA__/네트워크 JSON 응답에서 제목/링크 추출
    3) 그래도 실패하면 outputs에 디버그 저장
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_urls: list[str] = []
    json_items: list[dict[str, str]] = []

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
            log(f"[네이버] dom_items={len(dom_items)}")

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

            json_items_dedup = {it["title"] + "|" + it["url"]: it for it in (json_items or [])}
            next_items_dedup = {it["title"] + "|" + it["url"]: it for it in (next_items or [])}
            dom_items_dedup = {it["title"] + "|" + it["url"]: it for it in (dom_items or [])}

            all_map = {}
            all_map.update(json_items_dedup)
            all_map.update(next_items_dedup)
            all_map.update(dom_items_dedup)
            all_items = list(all_map.values())

            log(f"[네이버] items(dom/json/next)={len(dom_items)}/{len(json_items_dedup)}/{len(next_items_dedup)} -> total={len(all_items)}")

            if not all_items:
                (OUT_DIR / "naver_debug.html").write_text(page.content(), encoding="utf-8")
                page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                (OUT_DIR / "naver_json_urls.txt").write_text("\n".join(json_urls[:500]), encoding="utf-8")
                raise RuntimeError("기사 링크를 찾지 못했습니다. outputs/naver_debug.* 및 naver_json_urls.txt 확인 필요")

            chosen = random.choice(all_items)
            return chosen["title"], chosen["url"], RANKING_URL

        finally:
            context.close()
            browser.close()


# ===========================
# 기사 컨텍스트(OG + 본문 일부) 추출
# ===========================
def _clean_text(s: str) -> str:
    s = s.replace("\u00a0", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def fetch_article_context(article_url: str) -> dict[str, Any]:
    """
    Playwright로 기사 페이지를 열고:
    - og:title / og:description
    - published time(가능하면)
    - 본문 텍스트 일부(가능하면) 추출
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

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
            viewport={"width": 1280, "height": 720},
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
        )
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
                      const selectors = [
                        '#articleBodyContents',
                        'div#dic_area',
                        'div._article_content',
                        'div.article_body',
                        'div#content',
                        'article',
                        'main'
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
                    return {
                      og_title,
                      og_description,
                      published_time: published,
                      body_text: body
                    };
                }"""
            )

            og_title = _clean_text(data.get("og_title", ""))
            og_description = _clean_text(data.get("og_description", ""))
            published_time = _clean_text(data.get("published_time", ""))

            body_text = data.get("body_text", "") or ""
            body_text = body_text.replace("\u00a0", " ")
            body_text = re.sub(r"\s+\n", "\n", body_text)
            body_text = re.sub(r"\n{3,}", "\n\n", body_text).strip()

            # 너무 길면 앞부분만 (OpenAI 입력 재료)
            body_excerpt = _clean_text(body_text)[:1800]

            if not og_title and not og_description and len(body_excerpt) < 40:
                # 디버그 저장
                (OUT_DIR / "article_debug.html").write_text(page.content(), encoding="utf-8")
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
# 대본 생성: 템플릿(백업)
# ===========================
def build_60s_shorts_script_template(topic: str, ctx: dict[str, Any]) -> dict[str, Any]:
    # 가능한 경우 OG/본문 일부를 한 줄이라도 활용(빈약함 완화)
    og = (ctx.get("og_title") or "").strip()
    desc = (ctx.get("og_description") or "").strip()
    hint = desc[:60] if desc else ""
    if og and og != topic:
        topic_line = f"{topic} (기사: {og[:40]})"
    else:
        topic_line = topic

    beats = [
        {"t": "0-2s", "voice": f"오늘 연예 랭킹, {topic}.", "onscreen": "오늘의 랭킹", "broll": "랭킹 화면"},
        {"t": "2-10s", "voice": "지금 사람들이 왜 관심 갖는지, 핵심만 짚을게요.", "onscreen": "핵심만", "broll": "키워드 카드"},
        {"t": "10-25s", "voice": hint and f"요약 힌트는 이거예요: {hint}." or "제목에서 가장 큰 포인트가 드러납니다.", "onscreen": "포인트 1", "broll": "제목 클로즈업"},
        {"t": "25-40s", "voice": "이 이슈는 ‘상황’보다 ‘반응’이 빨리 퍼지는 타입입니다.", "onscreen": "반응 폭발", "broll": "댓글/공유 애니"},
        {"t": "40-55s", "voice": "정확한 내용은 기사 확인이 안전하고, 여기선 흐름만 정리했어요.", "onscreen": "흐름 정리", "broll": "타임라인 그래픽"},
        {"t": "55-60s", "voice": "내일 랭킹도 자동으로 1분 요약. 구독!", "onscreen": "구독", "broll": "구독 버튼"},
    ]

    return {
        "topic": topic_line,
        "title_short": topic[:28],
        "description": "네이버 연예 랭킹 기반 60초 요약(템플릿)",
        "beats": beats,
        "hashtags": ["#연예", "#네이버", "#랭킹", "#쇼츠", "#자동화"],
        "notes": "OpenAI 키가 없거나 실패 시 템플릿으로 생성됨",
        "_generator": "template",
    }


# ===========================
# OpenAI 협업: 2단계 생성(팩트/포인트 -> 대본) + 품질게이트
# ===========================
def _pick_style() -> str:
    styles = [
        "timeline",      # 타임라인 3포인트
        "qna",           # 질문-답 구조
        "mythfact",      # 오해 vs 사실(확인된 것만)
        "3points",       # 핵심 3줄 + 의미 2줄
    ]
    if SHORTS_STYLE in styles:
        return SHORTS_STYLE
    return random.choice(styles)


def _voice_total_chars(beats: list[dict[str, Any]]) -> int:
    total = 0
    for b in beats:
        total += len((b.get("voice") or "").strip())
    return total


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    inputs:
      - topic, source_url, ranking_page
      - article_ctx: {og_title, og_description, published_time, body_excerpt}
    """
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

    style = _pick_style()
    article_ctx = inputs.get("article_ctx", {}) or {}

    # (1) 팩트/포인트 추출 스키마
    facts_schema = {
        "type": "object",
        "properties": {
            "topic": {"type": "string"},
            "style": {"type": "string"},
            "facts": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string"},
            },
            "what_is_unknown": {
                "type": "array",
                "minItems": 1,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "angles": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {"type": "string"},
            },
            "safe_wording_rules": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {"type": "string"},
            },
        },
        "required": ["topic", "style", "facts", "what_is_unknown", "angles", "safe_wording_rules"],
    }

    sys_facts = (
        "너는 한국어 쇼츠 작가의 리서처다. "
        "아래 입력(제목/OG요약/본문 일부)에서 '확인 가능한 내용'만 뽑아라. "
        "추측, 루머, 비방은 금지. 불확실하면 what_is_unknown에 넣어라."
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
        model=model,
        input=[
            {"role": "system", "content": sys_facts},
            {"role": "user", "content": json.dumps(user_facts_payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "facts_pack",
                "strict": True,
                "schema": facts_schema,
            }
        },
        max_output_tokens=600,
    )
    facts_pack = json.loads(facts_resp.output_text)

    # (2) 대본 생성 스키마(6구간 고정)
    script_schema = {
        "type": "object",
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

    sys_script = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다. "
        "facts_pack의 facts에 있는 내용만 '사실처럼' 말해라. "
        "facts 밖의 내용은 단정 금지(필요하면 '기사 요약 기준' 또는 '가능성'으로 표현). "
        "과장, 루머, 비방 금지. "
        "정확히 6구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s)으로 구성해라. "
        "각 구간 onscreen은 12자 이내. voice는 말로 읽기 좋게 짧게. "
        f"스타일은 {style}로 구성해라."
    )

    user_script_payload = {
        "topic": inputs["topic"],
        "style": style,
        "facts_pack": facts_pack,
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
    }

    script_resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": sys_script},
            {"role": "user", "content": json.dumps(user_script_payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "shorts_script",
                "strict": True,
                "schema": script_schema,
            }
        },
        max_output_tokens=max_out,
    )
    draft = json.loads(script_resp.output_text)
    draft["_generator"] = "openai_v2"
    draft["_style"] = style

    # (3) 길이 품질 게이트: 너무 길거나 짧으면 한 번 더 다듬기
    beats = draft.get("beats") or []
    if isinstance(beats, list) and len(beats) == 6:
        total_chars = _voice_total_chars(beats)
        draft["_voice_chars"] = total_chars

        need_fix = total_chars < VOICE_CHARS_MIN or total_chars > VOICE_CHARS_MAX
        if need_fix:
            fix_mode = "shorten" if total_chars > VOICE_CHARS_MAX else "expand"
            sys_fix = (
                "너는 쇼츠 대본 편집자다. "
                "원본 구조(6구간)와 사실 제약은 유지하고, "
                f"voice 전체 분량을 {'줄여' if fix_mode=='shorten' else '늘려'}라. "
                "과장/루머/비방 금지. onscreen 12자 이내 유지."
            )
            user_fix_payload = {
                "fix_mode": fix_mode,
                "target_voice_chars_min": VOICE_CHARS_MIN,
                "target_voice_chars_max": VOICE_CHARS_MAX,
                "draft": draft,
                "facts_pack": facts_pack,
            }

            fix_resp = client.responses.create(
                model=model,
                input=[
                    {"role": "system", "content": sys_fix},
                    {"role": "user", "content": json.dumps(user_fix_payload, ensure_ascii=False)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "shorts_script_fixed",
                        "strict": True,
                        "schema": script_schema,
                    }
                },
                max_output_tokens=max_out,
            )
            fixed = json.loads(fix_resp.output_text)
            fixed["_generator"] = "openai_v2_edit"
            fixed["_style"] = style
            fixed["_voice_chars"] = _voice_total_chars(fixed.get("beats") or [])
            return fixed

    return draft


# ===========================
# 노드
# ===========================
def node_load_inputs(ctx: dict[str, Any]):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    article_ctx = fetch_article_context(article_url)

    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
        "article_ctx": article_ctx,  # OG + 본문 일부
    }


def node_make_script(ctx: dict[str, Any]):
    inp = ctx["inputs"]

    # OpenAI 키가 있으면 OpenAI로 생성, 실패하면 템플릿
    if os.getenv("OPENAI_API_KEY"):
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inp)
            return
        except Exception as e:
            (OUT_DIR / "openai_error.txt").write_text(str(e), encoding="utf-8")

    ctx["shorts"] = build_60s_shorts_script_template(inp["topic"], inp.get("article_ctx", {}))


def node_save_files(ctx: dict[str, Any]):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"shorts_{stamp}_{run_id}.json"

    src = ctx["inputs"]
    shorts = ctx["shorts"]

    txt: list[str] = []
    txt.append(f"DATE(KST): {kst.isoformat()}")
    txt.append(f"RUN_ID: {run_id}")
    txt.append(f"GENERATOR: {shorts.get('_generator')}")
    txt.append(f"STYLE: {shorts.get('_style', '')}")
    txt.append(f"VOICE_CHARS: {shorts.get('_voice_chars', '')}")
    txt.append(f"TOPIC: {shorts.get('topic')}")
    txt.append(f"SOURCE_URL: {src['source_url']}")
    txt.append(f"RANKING_PAGE: {src['ranking_page']}")
    a = src.get("article_ctx", {}) or {}
    txt.append(f"ARTICLE_OG_TITLE: {a.get('og_title','')}")
    txt.append(f"ARTICLE_OG_DESC: {a.get('og_description','')}")
    txt.append(f"ARTICLE_PUBLISHED: {a.get('published_time','')}")
    txt.append("")
    txt.append("=== SCRIPT ===")

    if isinstance(shorts.get("beats"), list) and len(shorts["beats"]) == 6:
        for b in shorts["beats"]:
            txt.append(f'[{b["t"]}] {b["voice"]}')
            txt.append(f'  - ONSCREEN: {b["onscreen"]}')
            txt.append(f'  - BROLL: {b["broll"]}')
    else:
        txt.append(shorts.get("notes", ""))

    txt.append("")
    txt.append("TITLE_SHORT: " + (shorts.get("title_short") or ""))
    txt.append("DESCRIPTION: " + (shorts.get("description") or ""))
    txt.append("NOTES: " + (shorts.get("notes") or ""))
    txt.append("")
    txt.append("HASHTAGS: " + " ".join(shorts.get("hashtags", [])))

    script_path.write_text("\n".join(txt), encoding="utf-8")

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": src,
        "shorts": shorts,
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("GENERATOR: " + str(ctx["shorts"].get("_generator")))
    log("SCRIPT FILE: " + ctx["outputs"]["script_path"])
    log("META FILE: " + ctx["outputs"]["meta_path"])


# ===========================
# 연결/실행
# ===========================
flow = Flow()
flow.add(Node("LOAD_INPUTS", node_load_inputs))
flow.add(Node("MAKE_SCRIPT", node_make_script))
flow.add(Node("SAVE_FILES", node_save_files))
flow.add(Node("PRINT", node_print))

flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
flow.connect("MAKE_SCRIPT", "SAVE_FILES")
flow.connect("SAVE_FILES", "PRINT")

if __name__ == "__main__":
    try:
        ctx = flow.run("LOAD_INPUTS")
        log("\n[끝] inputs = " + json.dumps(ctx.get("inputs"), ensure_ascii=False))
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
