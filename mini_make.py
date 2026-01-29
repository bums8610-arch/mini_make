# mini_make.py
from __future__ import annotations

import os
import re
import json
import random
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urljoin
from html.parser import HTMLParser
import html as html_lib


# ----------------------------
# make.com 같은 "노드/플로우"
# ----------------------------
@dataclass
class Node:
    name: str
    fn: callable  # fn(ctx) -> next node name(str) or None


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


# ----------------------------
# 유틸
# ----------------------------
def now_kst() -> datetime:
    return datetime.now(timezone.utc).astimezone(ZoneInfo("Asia/Seoul"))


def fetch_html(url: str, timeout: int = 10) -> tuple[str, str]:
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        final_url = resp.geturl()
        text = resp.read().decode("utf-8", errors="ignore")
    return final_url, text


# ----------------------------
# 네이버 엔터 랭킹에서 제목 1개 뽑기
# ----------------------------
NAVER_ENT_RANKING_URL = "https://m.entertain.naver.com/ranking"


class AnchorCollector(HTMLParser):
    def __init__(self, base_url: str):
        super().__init__()
        self.base_url = base_url
        self.in_a = False
        self.current_href = ""
        self.current_text_parts: list[str] = []
        self.items: list[tuple[str, str]] = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            self.in_a = True
            self.current_href = ""
            self.current_text_parts = []
            for k, v in attrs:
                if k.lower() == "href":
                    self.current_href = v or ""

    def handle_data(self, data):
        if self.in_a and data:
            self.current_text_parts.append(data.strip())

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.in_a:
            self.in_a = False
            title = " ".join([t for t in self.current_text_parts if t]).strip()
            href = self.current_href.strip()

            if title and href:
                full = urljoin(self.base_url, href)
                if ("/home/article/" in full) or ("/ranking/read" in full):
                    if 6 <= len(title) <= 60:
                        self.items.append((title, full))


def extract_from_next_data(html_text: str) -> list[tuple[str, str]]:
    m = re.search(r'<script[^>]+id="__NEXT_DATA__"[^>]*>(.*?)</script>', html_text, re.DOTALL)
    if not m:
        return []

    raw = html_lib.unescape(m.group(1)).strip()
    try:
        data = json.loads(raw)
    except Exception:
        return []

    results: list[tuple[str, str]] = []

    def walk(obj):
        if isinstance(obj, dict):
            title = None
            link = None

            for tk in ("title", "headline", "articleTitle", "contentTitle", "subject"):
                if tk in obj and isinstance(obj[tk], str):
                    title = obj[tk].strip()
                    if title:
                        break

            for lk in ("url", "link", "href", "mobileUrl"):
                if lk in obj and isinstance(obj[lk], str):
                    link = obj[lk].strip()
                    if link:
                        break

            if not link:
                oid = obj.get("oid") or obj.get("officeId")
                aid = obj.get("aid") or obj.get("articleId")
                if isinstance(oid, (str, int)) and isinstance(aid, (str, int)):
                    link = f"https://m.entertain.naver.com/home/article/{oid}/{aid}"

            if isinstance(title, str) and isinstance(link, str):
                if ("/home/article/" in link) or ("/ranking/read" in link):
                    if 6 <= len(title) <= 60:
                        results.append((title, link))

            for v in obj.values():
                walk(v)

        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)

    uniq = {}
    for t, u in results:
        uniq[(re.sub(r"\s+", " ", t).strip(), u)] = True
    return list(uniq.keys())


def pick_topic_from_naver_entertain() -> tuple[str, str]:
    """
    항상 네이버 엔터 랭킹에서 주제를 가져온다.
    실패하면 예외를 발생시켜 workflow를 실패 처리한다.
    """
    final_url, html_text = fetch_html(NAVER_ENT_RANKING_URL, timeout=12)

    items = extract_from_next_data(html_text)
    if not items:
        p = AnchorCollector(final_url)
        p.feed(html_text)
        items = p.items

    if not items:
        raise RuntimeError("네이버 엔터 랭킹에서 제목을 가져오지 못했습니다(페이지 구조/차단/일시 오류 가능).")

    return random.choice(items)


# ----------------------------
# 쇼츠 대본 생성(예시)
# ----------------------------
def build_60s_shorts_script(topic: str) -> dict:
    hook = f"3초만에 핵심만: {topic}"
    problem = f"사람들이 {topic}에서 자주 놓치는 포인트가 있습니다."
    tip1 = "1) 제목을 한 문장으로 요약해서 시작."
    tip2 = "2) 핵심 3가지만 말하고 나머지는 버림."
    tip3 = "3) 마지막 5초에 다음 행동(구독/댓글)을 딱 1개만 요청."
    recap = "정리: 한 문장 요약 → 핵심 3개 → 행동 1개."
    cta = "이런 방식으로 매일 자동으로 만들고 싶으면 구독."

    narration = "\n".join([
        f"[훅 0-2s] {hook}",
        f"[문제 2-10s] {problem}",
        f"[핵심1 10-23s] {tip1}",
        f"[핵심2 23-36s] {tip2}",
        f"[핵심3 36-45s] {tip3}",
        f"[요약 45-55s] {recap}",
        f"[CTA 55-60s] {cta}",
    ])

    hashtags = ["#연예", "#뉴스", "#쇼츠", "#자동화"]
    return {"topic": topic, "narration": narration, "hashtags": hashtags}


# ----------------------------
# 노드들
# ----------------------------
def node_load_inputs(ctx: dict):
    # 항상 네이버 랭킹에서만 주제 선택 (강제 주제 입력 기능 삭제)
    title, url = pick_topic_from_naver_entertain()
    ctx["inputs"] = {"topic": title, "source": "naver_entertain_ranking", "source_url": url}


def node_make_script(ctx: dict):
    topic = ctx["inputs"]["topic"]
    ctx["shorts"] = build_60s_shorts_script(topic)


def node_save_files(ctx: dict):
    kst = now_kst()
    run_id = os.getenv("GITHUB_RUN_ID", "local")  # 파일 이름 구분용(없으면 local)
    stamp = kst.strftime("%Y%m%d_%H%M")

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    script_path = out_dir / f"shorts_{stamp}_{run_id}.txt"
    meta_path = out_dir / f"shorts_{stamp}_{run_id}.json"

    shorts = ctx["shorts"]
    src = ctx["inputs"]

    txt = []
    txt.append(f"DATE(KST): {kst.isoformat()}")
    txt.append(f"SOURCE: {src.get('source')}")
    txt.append(f"SOURCE_URL: {src.get('source_url')}")
    txt.append(f"TOPIC: {shorts['topic']}")
    txt.append("")
    txt.append(shorts["narration"])
    txt.append("")
    txt.append("HASHTAGS: " + " ".join(shorts["hashtags"]))
    script_path.write_text("\n".join(txt), encoding="utf-8")

    meta = {
        "date_kst": kst.isoformat(),
        "run_id": run_id,
        "source": src.get("source"),
        "source_url": src.get("source_url"),
        **shorts,
        "files": {"script": str(script_path), "meta": str(meta_path)},
    }
    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    ctx["outputs"] = {"script_path": str(script_path), "meta_path": str(meta_path)}


def node_print(ctx: dict):
    print("TOPIC:", ctx["inputs"]["topic"])
    print("TOPIC_URL:", ctx["inputs"]["source_url"])
    print("SCRIPT FILE:", ctx["outputs"]["script_path"])
    print("META FILE:", ctx["outputs"]["meta_path"])


# ----------------------------
# 플로우 연결
# ----------------------------
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

