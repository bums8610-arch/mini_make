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
from typing import Any, Iterable

from playwright.sync_api import sync_playwright


def log(msg: str) -> None:
    print(msg, flush=True)


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
        ctx: dict = {}
        cur = start
        while cur is not None:
            log(f"[실행] {cur}")
            nxt = self.nodes[cur].fn(ctx)
            cur = nxt if isinstance(nxt, str) else self.links.get(cur)
        return ctx


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"


def _dedupe_items(items: list[dict[str, str]]) -> list[dict[str, str]]:
    seen = set()
    out: list[dict[str, str]] = []
    for it in items:
        key = (it.get("title", "").strip(), it.get("url", "").strip())
        if not key[0] or not key[1]:
            continue
        if key in seen:
            continue
        seen.add(key)
        out.append({"title": key[0], "url": key[1]})
    return out


def _looks_like_entertain_article(url: str) -> bool:
    u = url or ""
    if "entertain.naver.com" not in u:
        return False
    return ("/article/" in u) or ("/home/article/" in u) or ("/ranking/read" in u)


def _walk_json(obj: Any) -> Iterable[Any]:
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from _walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _walk_json(v)


def _extract_items_from_json(data: Any) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for node in _walk_json(data):
        if not isinstance(node, dict):
            continue

        title = None
        url = None

        for tk in ("title", "headline", "articleTitle", "contentTitle", "subject"):
            v = node.get(tk)
            if isinstance(v, str) and v.strip():
                title = v.strip()
                break

        for uk in ("url", "link", "href", "mobileUrl"):
            v = node.get(uk)
            if isinstance(v, str) and v.strip():
                url = v.strip()
                break

        # oid/aid 형태도 대응(있으면 조립)
        if not url:
            oid = node.get("oid") or node.get("officeId")
            aid = node.get("aid") or node.get("articleId")
            if isinstance(oid, (str, int)) and isinstance(aid, (str, int)):
                url = f"https://m.entertain.naver.com/home/article/{oid}/{aid}"

        if isinstance(title, str) and isinstance(url, str):
            if _looks_like_entertain_article(url) and 6 <= len(title) <= 120:
                results.append({"title": title, "url": url})

    return _dedupe_items(results)


def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    """
    1) DOM에서 기사 링크 수집
    2) 실패하면 네트워크 JSON 응답에서 제목/링크 추출
    실패하면 outputs에 디버그 파일 남기고 예외
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    json_urls: list[str] = []
    json_candidates: list[dict[str, str]] = []

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

        # webdriver 흔적 최소화(완벽한 스텔스는 아니지만 도움 됨)
        context.add_init_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined});")

        page = context.new_page()

        def on_response(resp):
            try:
                ct = (resp.headers.get("content-type") or "").lower()
                if "application/json" in ct:
                    json_urls.append(resp.url)
                    try:
                        data = resp.json()
                        json_candidates.extend(_extract_items_from_json(data))
                    except Exception:
                        pass
            except Exception:
                pass

        page.on("response", on_response)

        try:
            log(f"[네이버] goto {RANKING_URL}")
            page.goto(RANKING_URL, wait_until="domcontentloaded", timeout=60000)

            # DOM이 그려질 시간을 조금 줌
            page.wait_for_timeout(3000)

            # DOM에서 넓게 수집
            dom_items = page.evaluate(
                """() => {
                    const links = Array.from(document.querySelectorAll('a'));
                    const out = [];
                    for (const a of links) {
                      const href = a.href || '';
                      if (!href) continue;

                      const t1 = (a.innerText || '').trim();
                      const t2 = (a.textContent || '').trim();
                      const text = (t1 || t2).replace(/\\s+/g,' ').trim();
                      if (!text) continue;

                      const okDomain = href.includes('entertain.naver.com');
                      const okPath = href.includes('/article/') || href.includes('/home/article/') || href.includes('/ranking/read');
                      if (okDomain && okPath && text.length >= 6 && text.length <= 120) {
                        out.push({title: text, url: href});
                      }
                    }
                    // dedupe
                    const uniq = new Map();
                    for (const x of out) uniq.set(x.title + '|' + x.url, x);
                    return Array.from(uniq.values());
                }"""
            )

            dom_items = [{"title": x["title"], "url": x["url"]} for x in (dom_items or [])]
            dom_items = _dedupe_items(dom_items)
            log(f"[네이버] dom_items={len(dom_items)}")

            # JSON 기반 후보도 같이 정리
            json_candidates = _dedupe_items(json_candidates)
            log(f"[네이버] json_candidates={len(json_candidates)} json_urls={len(json_urls)}")

            all_items = _dedupe_items(dom_items + json_candidates)

            if not all_items:
                (OUT_DIR / "naver_debug.html").write_text(page.content(), encoding="utf-8")
                page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                (OUT_DIR / "naver_json_urls.txt").write_text("\n".join(json_urls[:500]), encoding="utf-8")
                raise RuntimeError("기사 후보 0개. outputs/naver_debug.* 및 outputs/naver_json_urls.txt 확인 필요")

            chosen = random.choice(all_items)
            return chosen["title"], chosen["url"], RANKING_URL

        finally:
            try:
                context.close()
            finally:
                browser.close()


def build_60s_shorts_script(topic: str) -> dict:
    narration = "\n".join(
        [
            f"[훅 0-2s] 오늘 연예 랭킹: {topic}",
            "[문제 2-10s] 왜 이게 급상승했는지 핵심만.",
            "[핵심1 10-25s] 제목에서 사람들이 멈춤.",
            "[핵심2 25-40s] 댓글/공유 포인트가 명확함.",
            "[핵심3 40-50s] 다음 이슈로 이어지는 흐름.",
            "[요약 50-57s] 제목-포인트-흐름 3개만 보면 됨.",
            "[CTA 57-60s] 내일 랭킹도 자동으로 뽑아줄게. 구독.",
        ]
    )
    return {"topic": topic, "narration": narration, "hashtags": ["#연예", "#네이버", "#랭킹", "#쇼츠", "#자동화"]}


def node_load_inputs(ctx: dict):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    ctx["inputs"] = {"topic": title, "source_url": article_url, "ranking_page": ranking_url}


def node_make_script(ctx: dict):
    ctx["shorts"] = build_60s_shorts_script(ctx["inputs"]["topic"])


def node_save_files(ctx: dict):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"shorts_{stamp}_{run_id}.json"

    src = ctx["inputs"]
    shorts = ctx["shorts"]

    txt = []
    txt.append(f"DATE(KST): {kst.isoformat()}")
    txt.append(f"TOPIC: {shorts['topic']}")
    txt.append(f"SOURCE_URL: {src['source_url']}")
    txt.append(f"RANKING_PAGE: {src['ranking_page']}")
    txt.append("")
    txt.append(shorts["narration"])
    txt.append("")
    txt.append("HASHTAGS: " + " ".join(shorts["hashtags"]))
    script_path.write_text("\n".join(txt), encoding="utf-8")

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        **src,
        **shorts,
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}


def node_print(ctx: dict):
    log("TOPIC: " + ctx["inputs"]["topic"])
    log("SOURCE_URL: " + ctx["inputs"]["source_url"])
    log("SCRIPT FILE: " + ctx["outputs"]["script_path"])
    log("META FILE: " + ctx["outputs"]["meta_path"])


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
        # 실패 원인을 outputs/error.txt로 남김 + exit code 실패
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        raise
