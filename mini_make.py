# mini_make.py
from __future__ import annotations

import os
import re
import json
import time
import random
import gzip
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from urllib.parse import urljoin
from html.parser import HTMLParser
import html as html_lib


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


def fetch_html(url: str, timeout: int = 12) -> tuple[str, int, str]:
    """
    return (final_url, status, html_text)
    """
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        ),
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Referer": "https://www.naver.com/",
        "Accept-Encoding": "gzip",
    }

    req = urllib.request.Request(url, headers=headers)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            final_url = resp.geturl()
            status = getattr(resp, "status", 200)
            raw = resp.read()
            enc = (resp.headers.get("Content-Encoding") or "").lower()
            if "gzip" in enc:
                raw = gzip.decompress(raw)
            text = raw.decode("utf-8", errors="ignore")
            return final_url, status, text

    except urllib.error.HTTPError as e:
        try:
            raw = e.read()
            enc = (getattr(e, "headers", {}) or {}).get("Content-Encoding", "")
            if enc and "gzip" in str(enc).lower():
                raw = gzip.decompress(raw)
            body = raw.decode("utf-8", errors="ignore")
        except Exception:
            body = ""
        final_url = getattr(e, "url", url)
        status = getattr(e, "code", 0) or 0
        return final_url, status, body


# ✅ 여기: JS 껍데기인 m.entertain 대신, n.news 연예 랭킹을 1순위로
RANKING_URLS = [
    "https://n.news.naver.com/entertain/ranking",
    # 백업(가끔 n.news가 구조 바뀌면)
    "https://m.entertain.naver.com/ranking",
]


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
            s = data.strip()
            if s:
                self.current_text_parts.append(s)

    def handle_endtag(self, tag):
        if tag.lower() == "a" and self.in_a:
            self.in_a = False
            title = " ".join(self.current_text_parts).strip()
            href = (self.current_href or "").strip()
            if not title or not href:
                return

            full = urljoin(self.base_url, href)

            # ✅ n.news 연예 랭킹 기사 패턴
            ok = "n.news.naver.com/entertain/ranking/article/" in full
            # (백업) m.entertain 기사 패턴
            ok = ok or ("m.entertain.naver.com/home/article/" in full) or ("/ranking/read" in full)

            if ok and 6 <= len(title) <= 120:
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
                v = obj.get(tk)
                if isinstance(v, str) and v.strip():
                    title = v.strip()
                    break
            for lk in ("url", "link", "href", "mobileUrl"):
                v = obj.get(lk)
                if isinstance(v, str) and v.strip():
                    link = v.strip()
                    break
            if isinstance(title, str) and isinstance(link, str):
                if "n.news.naver.com/entertain/ranking/article/" in link or "m.entertain.naver.com" in link:
                    results.append((re.sub(r"\s+", " ", title).strip(), link))
            for v in obj.values():
                walk(v)
        elif isinstance(obj, list):
            for v in obj:
                walk(v)

    walk(data)
    uniq = {}
    for t, u in results:
        uniq[(t, u)] = True
    return list(uniq.keys())


def pick_topic_from_naver_entertain() -> tuple[str, str, str]:
    last_status = None
    last_url = None
    last_html = ""

    for base_url in RANKING_URLS:
        for attempt in range(1, 4):
            final_url, status, html_text = fetch_html(base_url, timeout=12)
            last_status, last_url, last_html = status, final_url, html_text

            print(f"[네이버] try={attempt}/3 url={base_url} final={final_url} status={status} html_len={len(html_text)}")

            if status in (403, 429) or status >= 500 or status == 0:
                time.sleep(1 + random.random() * 2)
                continue

            items = extract_from_next_data(html_text)

            if not items:
                p = AnchorCollector(final_url)
                p.feed(html_text)
                items = p.items

            print(f"[네이버] items={len(items)}")

            if items:
                title, url = random.choice(items)  # ✅ 랜덤 유지
                return title, url, final_url

            time.sleep(1 + random.random() * 2)

    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    dbg_path = out_dir / "naver_debug.html"
    try:
        dbg_path.write_text(last_html[:200000], encoding="utf-8")
        print(f"[디버그] saved: {dbg_path}")
    except Exception:
        pass

    raise RuntimeError(f"네이버 연예 랭킹에서 제목을 가져오지 못함. last_status={last_status} last_url={last_url}")


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
    title, article_url, src_url = pick_topic_from_naver_entertain()
    ctx["inputs"] = {
        "topic": title,
        "source_url": article_url,
        "ranking_page": src_url,
    }


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
    print("RANKING_PAGE:", ctx["inputs"]["ranking_page"])
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
