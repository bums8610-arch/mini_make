# mini_make.py
from __future__ import annotations

import os
import json
import random
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin


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
        ctx = {}
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


def env(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v if v is not None and v != "" else default


def fetch_html(url: str, timeout: int = 10) -> tuple[str, str]:
    """
    returns: (final_url, html_text)
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/120.0 Safari/537.36",
        "Accept-Language": "ko-KR,ko;q=0.9,en;q=0.8",
    }
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        final_url = resp.geturl()
        html = resp.read().decode("utf-8", errors="ignore")
    return final_url, html


class NaverRankingParser(HTMLParser):
    """
    네이버 랭킹

