# mini_make.py
from __future__ import annotations

import os
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path

from playwright.sync_api import sync_playwright


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
            print(f"[실행] {cur}")
            nxt = self.nodes[cur].fn(ctx)
            cur = nxt if isinstance(nxt, str) else self.links.get(cur)
        return ctx


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


RANKING_URL = "https://m.entertain.naver.com/ranking"


def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    """
    네이버 연예 랭킹 페이지를 '브라우저로' 열어서(자바스크립트 실행) 링크를 수집.
    return: (title, article_url, ranking_page_url)
    실패하면 예외 -> Actions 실패(디스코드 실패 알림)
    """
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
            ),
            locale="ko-KR",
        )

        try:
            print(f"[네이버] goto {RANKING_URL}")
            page.goto(RANKING_URL, wait_until="networkidle", timeout=60000)

            # a 태그 전부 뽑아서, '기사 링크'처럼 생긴 것만 필터링
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
                    // 중복 제거
                    const uniq = new Map();
                    for (const x of out) uniq.set(x.text + '|' + x.href, x);
                    return Array.from(uniq.values());
                }"""
            )

            print(f"[네이버] items={len(items)}")

            if not items:
                # 디버그 저장
                (out_dir / "naver_debug.html").write_text(page.content(), encoding="utf-8")
                page.screenshot(path=str(out_dir / "naver_debug.png"), full_page=True)
                raise RuntimeError("렌더링 후에도 기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")

            chosen = random.choice(items)  # ✅ 랜덤 유지
            return chosen["text"], chosen["href"], RANKING_URL

        finally:
            browser.close()


def build_60s_shorts_script(topic: str) -> dict:
    narration = "\n".join([
        f"[훅 0-2s] 오늘 연예 랭킹: {topic}",
        "[문제 2-10s] 왜 이게 급상승했는지 핵심만.",
        "[핵심1 10-25s] 제목에서 사람들이 멈춤.",
        "[핵심2 25-40s] 댓글/공유 포인트가 명확함.",
        "[핵심3 40-50s] 다음 이슈로 이어지는 흐름.",
        "[요약 50-57s] 제목-포인트-흐름 3개만 보면 됨.",
        "[CTA 57-60s] 내일 랭킹도 자동으로 뽑아줄게. 구독.",
    ])
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

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / f"shorts_{stamp}_{run_id}.txt"
    meta_path = out_dir / f"shorts_{stamp}_{run_id}.json"

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
    print("TOPIC:", ctx["inputs"]["topic"])
    print("SOURCE_URL:", ctx["inputs"]["source_url"])
    print("SCRIPT FILE:", ctx["outputs"]["script_path"])
    print("META FILE:", ctx["outputs"]["meta_path"])


flow = Flow()
flow.add(Node("LOAD_INPUTS", node_load_inputs))
flow.add(Node("MAKE_SCRIPT", node_make_script))
flow.add(Node("SAVE_FILES", node_save_files))
flow.add(Node("PRINT", node_print))

flow.connect("LOAD_INPUTS", "MAKE_SCRIPT")
flow.connect("MAKE_SCRIPT", "SAVE_FILES")
flow.connect("SAVE_FILES", "PRINT")

if __name__ == "__main__":
    ctx = flow.run("LOAD_INPUTS")
    print("\n[끝] inputs =", ctx.get("inputs"))
