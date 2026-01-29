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


# ---------------------------
# 공통 유틸
# ---------------------------
OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"


def log(msg: str) -> None:
    print(msg, flush=True)


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


# ---------------------------
# mini make (flow)
# ---------------------------
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


# ---------------------------
# 네이버: 랜덤 토픽 1개 뽑기
# ---------------------------
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
    2) 실패하면 __NEXT_DATA__/JSON 응답에서 제목/링크 추출
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
            if next_data_text and isinstance(next_data_text, str) and len(next_data_text) > 50:
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

            log(
                f"[네이버] items(dom/json/next)={len(dom_items)}/{len(json_items_dedup)}/{len(next_items_dedup)} -> total={len(all_items)}"
            )

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


# ---------------------------
# 기사 메타(brief) 추출 (NameError 방지용으로 반드시 존재)
# ---------------------------
def fetch_article_brief(url: str) -> dict[str, str]:
    """
    기사 페이지에서 og:title / og:description 정도만 뽑아 '대본 재료'로 사용.
    실패해도 빈 값 반환(파이프라인은 계속 진행)
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True,
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            context = browser.new_context(locale="ko-KR", timezone_id="Asia/Seoul")
            page = context.new_page()

            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)

            def meta(selector: str) -> str:
                el = page.query_selector(selector)
                return (el.get_attribute("content") or "").strip() if el else ""

            og_title = meta("meta[property='og:title']")
            og_desc = meta("meta[property='og:description']") or meta("meta[name='description']")

            context.close()
            browser.close()

            return {"og_title": og_title, "og_description": og_desc}
    except Exception:
        return {"og_title": "", "og_description": ""}


# ---------------------------
# 대본 생성: 템플릿(백업)
# ---------------------------
def build_60s_shorts_script_template(topic: str) -> dict[str, Any]:
    narration = "\n".join(
        [
            f"[훅 0-2s] 오늘 연예 랭킹: {topic}",
            "[문제 2-10s] 왜 이게 급상승했는지 핵심만.",
            "[핵심1 10-25s] 제목에서 사람들이 멈춤.",
            "[핵심2 25-40s] 댓글/공유 포인트가 명확함.",
            "[핵심3 40-55s] 다음 이슈로 이어지는 흐름.",
            "[CTA 55-60s] 내일 랭킹도 자동으로 뽑아줄게. 구독.",
        ]
    )
    return {
        "topic": topic,
        "title_short": topic[:28],
        "description": "네이버 연예 랭킹 기준 60초 요약",
        "beats": [
            {"t": "0-2s", "voice": f"오늘 연예 랭킹, {topic}.", "onscreen": "오늘의 랭킹", "broll": "랭킹 화면"},
            {"t": "2-10s", "voice": "왜 뜨는지 핵심만 볼게요.", "onscreen": "왜 뜸?", "broll": "키워드 카드"},
            {"t": "10-25s", "voice": "제목에서 사람들이 멈추는 포인트가 있어요.", "onscreen": "포인트 1", "broll": "제목 클로즈업"},
            {"t": "25-40s", "voice": "댓글/공유가 생기는 지점이 명확합니다.", "onscreen": "포인트 2", "broll": "댓글 스크롤"},
            {"t": "40-55s", "voice": "다음 이슈로 이어질 가능성도 보여요.", "onscreen": "포인트 3", "broll": "타임라인 그래픽"},
            {"t": "55-60s", "voice": "내일 랭킹도 1분 요약할게요. 구독!", "onscreen": "구독", "broll": "구독 버튼"},
        ],
        "hashtags": ["#연예", "#네이버", "#랭킹", "#쇼츠", "#자동화"],
        "notes": narration,
        "_generator": "template",
    }


# ---------------------------
# 대본 생성: OpenAI 협업(메인)
# ---------------------------
def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    OPENAI_API_KEY 환경변수가 있으면 OpenAI로 대본 생성(구조화 JSON).
    """
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
    max_out = int(os.getenv("OPENAI_MAX_OUTPUT_TOKENS", "900"))

    client = OpenAI()

    schema = {
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

    system = (
        "너는 한국어 유튜브 쇼츠(약 60초) 대본 작가다. "
        "사용자가 준 정보(제목/짧은 설명/링크) 밖의 사실을 단정하지 말고, "
        "정보가 부족하면 '기사 제목/요약 기준'이라고 표현해라. "
        "과장, 루머, 비방은 피하고 맥락/포인트 중심으로 써라. "
        "정확히 6구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s)으로 구성해라. "
        "각 구간 voice는 짧고 말로 읽기 좋게, onscreen은 12자 이내로 써라."
    )

    user_payload = {
        "topic": inputs["topic"],
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "article_brief": inputs.get("article_brief", {}),
    }

    resp = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "shorts_script",
                "strict": True,
                "schema": schema,
            }
        },
        max_output_tokens=max_out,
        reasoning={"effort": "low"},
    )

    data = json.loads(resp.output_text)
    data["_generator"] = "openai"
    return data


# ---------------------------
# 노드 구현
# ---------------------------
def node_load_inputs(ctx: dict[str, Any]):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    brief = fetch_article_brief(article_url)

    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
        "article_brief": brief,
    }


def node_make_script(ctx: dict[str, Any]):
    inp = ctx["inputs"]

    if os.getenv("OPENAI_API_KEY"):
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(inp)
            return
        except Exception as e:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "openai_error.txt").write_text(str(e), encoding="utf-8")

    ctx["shorts"] = build_60s_shorts_script_template(inp["topic"])


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
    txt.append(f"GENERATOR: {shorts.get('_generator')}")
    txt.append(f"TOPIC: {shorts.get('topic')}")
    txt.append(f"SOURCE_URL: {src['source_url']}")
    txt.append(f"RANKING_PAGE: {src['ranking_page']}")
    txt.append(f"ARTICLE_OG_TITLE: {src.get('article_brief', {}).get('og_title', '')}")
    txt.append(f"ARTICLE_OG_DESC: {src.get('article_brief', {}).get('og_description', '')}")
    txt.append("")

    if isinstance(shorts.get("beats"), list) and len(shorts["beats"]) == 6:
        for b in shorts["beats"]:
            txt.append(f'[{b["t"]}] {b["voice"]}')
            txt.append(f'  - ONSCREEN: {b["onscreen"]}')
            txt.append(f'  - BROLL: {b["broll"]}')
        txt.append("")
        txt.append("TITLE_SHORT: " + (shorts.get("title_short") or ""))
        txt.append("DESCRIPTION: " + (shorts.get("description") or ""))
        txt.append("NOTES: " + (shorts.get("notes") or ""))
    else:
        txt.append(shorts.get("notes", "") or shorts.get("narration", ""))

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


# ---------------------------
# 연결
# ---------------------------
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

