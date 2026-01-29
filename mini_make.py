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
            print(f"[실행] {cur}", flush=True)
            nxt = self.nodes[cur].fn(ctx)
            cur = nxt if isinstance(nxt, str) else self.links.get(cur)
        return ctx


def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

RANKING_URL = "https://m.entertain.naver.com/ranking"


def pick_topic_from_naver_entertain_random() -> tuple[str, str, str]:
    """
    네이버 연예 랭킹 페이지를 '브라우저로' 열어서(자바스크립트 실행) 링크를 수집.
    return: (title, article_url, ranking_page_url)
    """
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
            print(f"[네이버] goto {RANKING_URL}", flush=True)
            page.goto(RANKING_URL, wait_until="domcontentloaded", timeout=60000)

            # 기사 링크가 생길 때까지 기다림
            page.wait_for_selector(
                "a[href*='/home/article/'], a[href*='/ranking/read'], a[href*='/article/']",
                timeout=15000
            )

            items = page.evaluate(
                """() => {
                    const links = Array.from(document.querySelectorAll('a'));
                    const out = [];
                    for (const a of links) {
                      const text = ((a.innerText || a.textContent || '')).trim().replace(/\\s+/g,' ');
                      const href = a.href || '';
                      if (!text || !href) continue;

                      const okDomain = href.includes('entertain.naver.com');
                      const okPath = href.includes('/article/') || href.includes('/home/article/') || href.includes('/ranking/read');
                      if (okDomain && okPath && text.length >= 6 && text.length <= 120) {
                        out.push({text, href});
                      }
                    }
                    const uniq = new Map();
                    for (const x of out) uniq.set(x.text + '|' + x.href, x);
                    return Array.from(uniq.values());
                }"""
            )

            print(f"[네이버] items={len(items)}", flush=True)

            if not items:
                OUT_DIR.mkdir(parents=True, exist_ok=True)
                (OUT_DIR / "naver_debug.html").write_text(page.content(), encoding="utf-8")
                page.screenshot(path=str(OUT_DIR / "naver_debug.png"), full_page=True)
                raise RuntimeError("렌더링 후에도 기사 링크를 찾지 못했습니다. outputs/naver_debug.* 확인 필요")

            chosen = random.choice(items)
            return chosen["text"], chosen["href"], RANKING_URL

        finally:
            browser.close()


def fetch_article_brief(url: str) -> dict[str, str]:
    """
    기사 페이지에서 og:title / og:description 정도만 뽑아 대본 재료로 사용.
    실패하면 빈 값 반환(파이프라인은 계속 진행)
    """
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            page = browser.new_page(locale="ko-KR")
            page.goto(url, wait_until="domcontentloaded", timeout=60000)
            page.wait_for_timeout(1200)

            def meta(selector: str) -> str:
                el = page.query_selector(selector)
                return (el.get_attribute("content") or "").strip() if el else ""

            og_title = meta("meta[property='og:title']")
            og_desc = meta("meta[property='og:description']") or meta("meta[name='description']")

            browser.close()
            return {"og_title": og_title, "og_description": og_desc}
    except Exception:
        return {"og_title": "", "og_description": ""}


def build_60s_shorts_script_template(topic: str) -> dict:
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


def build_60s_shorts_script_openai(inputs: dict[str, Any]) -> dict[str, Any]:
    """
    OpenAI와 협업: 입력(제목/짧은 설명/링크)을 주면
    6구간(0-60s) 대본/자막/비롤/해시태그를 '구조화 JSON'으로 생성.
    """
    from openai import OpenAI

    model = os.getenv("OPENAI_MODEL", "gpt-5.2")
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
        "너는 한국어 유튜브 쇼츠(약 60초) 전문 대본 작가다. "
        "사용자가 준 정보(제목/짧은 설명/링크) 밖의 사실을 단정하지 말고, "
        "정보가 부족하면 '기사 제목/요약 기준'처럼 표현해라. "
        "과장, 단정적 루머, 비방은 피하고 호기심/맥락 중심으로 써라. "
        "정확히 6개 구간(0-2s, 2-10s, 10-25s, 25-40s, 40-55s, 55-60s)으로 구성해라."
    )

    user_payload = {
        "topic": inputs["topic"],
        "source_url": inputs["source_url"],
        "ranking_page": inputs["ranking_page"],
        "article_brief": inputs.get("article_brief", {}),
    }

    response = client.responses.create(
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

    data = json.loads(response.output_text)
    return data


def node_load_inputs(ctx: dict):
    title, article_url, ranking_url = pick_topic_from_naver_entertain_random()
    brief = fetch_article_brief(article_url)
    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": ranking_url,
        "article_brief": brief,
    }


def node_make_script(ctx: dict):
    # 키가 있으면 OpenAI 사용, 실패하면 템플릿 fallback
    if os.getenv("OPENAI_API_KEY"):
        try:
            ctx["shorts"] = build_60s_shorts_script_openai(ctx["inputs"])
            ctx["shorts"]["_generator"] = "openai"
            return
        except Exception as e:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            (OUT_DIR / "openai_error.txt").write_text(str(e), encoding="utf-8")

    ctx["shorts"] = build_60s_shorts_script_template(ctx["inputs"]["topic"])
    ctx["shorts"]["_generator"] = "template"


def node_save_files(ctx: dict):
    kst = now_kst()
    stamp = kst.strftime("%Y%m%d_%H%M")
    run_id = os.getenv("GITHUB_RUN_ID", "local")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script_path = OUT_DIR / f"shorts_{stamp}_{run_id}.txt"
    meta_path = OUT_DIR / f"shorts_{stamp}_{run_id}.json"

    src = ctx["inputs"]
    shorts = ctx["shorts"]

    txt: list[str] = []
    txt.append(f"DATE(KST): {kst.isofo
