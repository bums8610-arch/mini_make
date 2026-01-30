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

from playwright.sync_api import sync_playwright


# ===========================
# 공통 유틸
# ===========================
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"


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

            # 스크롤로 추가 로딩 유도
            for _ in range(3):
                page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                page.wait_for_timeout(1200)

            # DOM 링크 수집
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

            # __NEXT_DATA__도 시도
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

            og_title = _clean_text(data.get("og_title", ""))
            og_description = _clean_text(data.get("og_description", ""))
            published_time = _clean_text(data.get("published_time", ""))
            body_text = _clean_text(data.get("body_text", ""))

            return {
                "final_url": page.url,
                "og_title": og_title[:200],
                "og_description": og_description[:300],
                "published_time": published_time[:60],
                "body_snippet": body_text[:1200],
            }
        finally:
            context.close()
            browser.close()


# ===========================
# 대본: OpenAI(구조화 JSON) + 안전 폴백
# ===========================
def build_60s_shorts_script_template(inputs: dict[str, Any]) -> dict[str, Any]:
    topic = inputs["topic"]
    beats = [
        {"t": "0-2s", "voice": f"오늘 연예 랭킹 한 줄 요약! {topic}", "onscreen": "오늘 랭킹", "broll": "랭킹 페이지 스크롤"},
        {"t": "2-10s", "voice": "왜 갑자기 주목받는지, 제목 기준으로 핵심만 짚어볼게요.", "onscreen": "왜 뜸?", "broll": "기사 타이틀 클로즈업"},
        {"t": "10-25s", "voice": "포인트 1: 사람들이 멈춰 서는 키워드가 들어가 있어요.", "onscreen": "포인트1", "broll": "키워드 하이라이트"},
        {"t": "25-40s", "voice": "포인트 2: 댓글/공유를 부르는 한 문장이 있어요.", "onscreen": "포인트2", "broll": "댓글 창/공유 아이콘"},
        {"t": "40-55s", "voice": "포인트 3: 다음 이슈로 이어질 만한 흐름이 보여요.", "onscreen": "포인트3", "broll": "관련 기사 썸네일"},
        {"t": "55-60s", "voice": "내일 랭킹도 자동으로 뽑아올게요. 구독하고 같이 보죠!", "onscreen": "구독!", "broll": "구독 버튼 애니"},
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
                        "t": {"type": "string"},         # 0-2s ...
                        "voice": {"type": "string"},     # 내레이션
                        "onscreen": {"type": "string"},  # 화면 자막(짧게)
                        "broll": {"type": "string"},     # 비롤 아이디어(짧게)
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

    # ✅ 핵심 수정: gpt-4.1-mini에서는 reasoning 파라미터를 보내지 않는다.
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

    # 최소 안전 정리/검증
    if not isinstance(data.get("beats"), list) or len(data["beats"]) != 6:
        raise RuntimeError("OpenAI 결과 beats 형식이 깨졌습니다.")
    for b in data["beats"]:
        for k in ("t", "voice", "onscreen", "broll"):
            b[k] = _clean_text(b.get(k, ""))[:220]
        b["onscreen"] = b["onscreen"][:12]

    data["_generator"] = f"openai:{model}"
    data["topic"] = _clean_text(data.get("topic", inputs["topic"]))[:120]
    data["title_short"] = _clean_text(data.get("title_short", data["topic"]))[:40]
    data["description"] = _clean_text(data.get("description", ""))[:200]
    data["notes"] = _clean_text(data.get("notes", ""))[:400]
    if isinstance(data.get("hashtags"), list):
        data["hashtags"] = [str(x)[:24] for x in data["hashtags"]][:10]

    return data


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
    # 기사 컨텍스트는 여기서 미리 확보(실패해도 대본은 만들 수 있게 try)
    try:
        ctx["inputs"]["article_context"] = fetch_article_context(article_url)
    except Exception as e:
        ctx["inputs"]["article_context"] = {"error": str(e)}
    # next
    return "MAKE_SCRIPT"


def node_make_script(ctx: dict[str, Any]):
    inputs = ctx["inputs"]
    use_openai = os.getenv("USE_OPENAI", "1") == "1"
    has_key = bool(os.getenv("OPENAI_API_KEY"))

    if use_openai and has_key:
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inputs)
            return "SAVE_FILES"
        except Exception:
            # OpenAI 실패 기록 후 템플릿 폴백
            _write_text(OUT_DIR / "openai_error.txt", traceback.format_exc())

    ctx["shorts"] = build_60s_shorts_script_template(inputs)
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

    _write_text(script_path, "\n".join(txt))

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "inputs": src,
        "shorts": shorts,
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}
    return "PRINT"


def node_print(ctx: dict[str, Any]):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("GENERATOR: " + str(ctx["shorts"].get("_generator")))
    log("SCRIPT FILE: " + ctx["outputs"]["script_path"])
    log("META FILE: " + ctx["outputs"]["meta_path"])
    return None


# ===========================
# 연결/실행
# ===========================
flow = Flow()
flow.add(Node("LOAD_INPUTS", node_load_inputs))
flow.add(Node("MAKE_SCRIPT", node_make_script))
flow.add(Node("SAVE_FILES", node_save_files))
flow.add(Node("PRINT", node_print))

if __name__ == "__main__":
    try:
        ctx = flow.run("LOAD_INPUTS")
        log("\n[끝] inputs = " + json.dumps(ctx.get("inputs"), ensure_ascii=False))
    except Exception:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
