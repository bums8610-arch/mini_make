# mini_make.py
import json
import urllib.request

class Node:
    def __init__(self, name, fn):
        self.name = name
        self.fn = fn  # fn(ctx) -> next node name(str) or None

    def run(self, ctx):
        print(f"[실행] {self.name}")
        return self.fn(ctx)


class Flow:
    def __init__(self):
        self.nodes = []
        self.links = {}

    def add_node(self, node):
        self.nodes.append(node)

    def connect(self, from_name, to_name):
        self.links[from_name] = to_name

    def run(self, start_name, inputs=None):
        by_name = {n.name: n for n in self.nodes}
        ctx = {}

        # make.com처럼 "시나리오 입력값"을 ctx에 넣어둠
        ctx["inputs"] = inputs or {}

        current = start_name
        while current is not None:
            node = by_name[current]
            next_name = node.run(ctx)
            if isinstance(next_name, str):
                current = next_name
            else:
                current = self.links.get(current)

        print("\n[끝] ctx =", ctx)


# ---- 블록들 ----
def start_block(ctx):
    ctx["started"] = True

def http_get_block(ctx):
    inputs = ctx["inputs"]

    # URL/timeout을 inputs에서 읽는다 (없으면 기본값 사용)
    url = inputs.get("url", "https://jsonplaceholder.typicode.com/todos/1")
    timeout = inputs.get("timeout", 10)

    ctx["url"] = url
    ctx["timeout"] = timeout

    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            text = resp.read().decode("utf-8")
        ctx["http_text"] = text
        ctx["http_ok"] = True
    except Exception as e:
        ctx["http_ok"] = False
        ctx["error"] = str(e)

def parse_json_block(ctx):
    if not ctx.get("http_ok"):
        return
    data = json.loads(ctx["http_text"])
    ctx["json"] = data
    ctx["title"] = data.get("title")

def router_block(ctx):
    return "SUCCESS_PRINT" if ctx.get("http_ok") else "FAIL_PRINT"

def success_print_block(ctx):
    print("성공 출력:", ctx.get("title"))

def fail_print_block(ctx):
    print("실패 에러:", ctx.get("error"))

def end_block(ctx):
    print("END")


# ---- 흐름 ----
flow = Flow()
flow.add_node(Node("START", start_block))
flow.add_node(Node("HTTP_GET", http_get_block))
flow.add_node(Node("PARSE_JSON", parse_json_block))
flow.add_node(Node("ROUTER", router_block))
flow.add_node(Node("SUCCESS_PRINT", success_print_block))
flow.add_node(Node("FAIL_PRINT", fail_print_block))
flow.add_node(Node("END", end_block))

flow.connect("START", "HTTP_GET")
flow.connect("HTTP_GET", "PARSE_JSON")
flow.connect("PARSE_JSON", "ROUTER")
flow.connect("SUCCESS_PRINT", "END")
flow.connect("FAIL_PRINT", "END")


if __name__ == "__main__":
    # 여기만 바꿔서 시나리오 설정을 바꾼다 (make.com의 모듈 설정창 느낌)
    scenario_inputs = {
      "url": "https://jsonplaceholder.typicode.com/todos/1",
        "timeout": 10
    }
    flow.run("START", inputs=scenario_inputs)

