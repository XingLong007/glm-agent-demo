"""最小 GLM Agent（v2）：工具注册表 + 推理循环
- 注册表模式：新增工具只需加"实现函数 + schema"两条，不动主循环
- 循环：模型决定是否调工具 -> 本地执行 -> 结果回传 -> 继续推理
- MAX_STEPS 防死循环；工具异常不中断会话（错误信息回传给模型自行调整）
依赖：pip install openai
环境变量：ZHIPUAI_API_KEY（可选 GLM_MODEL，默认 glm-4-flash 免费模型）
"""
import os
import json
from openai import OpenAI

client = OpenAI(api_key=os.environ["ZHIPUAI_API_KEY"],
                base_url="https://open.bigmodel.cn/api/paas/v4")
MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")
MAX_STEPS = 8  # 防死循环：生产环境还应加超时与成本护栏

# ---------- 1. 工具实现 ----------
def get_weather(city: str) -> dict:
    """查询城市天气（模拟数据源，生产环境替换为真实API）"""
    data = {"北京": {"temp_c": 18, "condition": "晴", "wind": "北风3级"},
            "上海": {"temp_c": 22, "condition": "多云", "wind": "东风2级"}}
    base = data.get(city, {"temp_c": None, "condition": "暂无数据"})
    return {"city": city, "source": "mock", **base}

def calc(expression: str) -> str:
    """计算四则运算表达式（白名单校验，防注入）"""
    if not set(expression) <= set("0123456789+-*/(). "):
        return "错误：包含非法字符"
    return str(eval(expression, {"__builtins__": {}}, {}))

# ---------- 2. 工具注册表：schema 与实现绑定 ----------
TOOL_REGISTRY = [
    {"schema": {"type": "function", "function": {
        "name": "get_weather", "description": "查询指定城市的当前天气",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string", "description": "城市名，如 北京"}},
            "required": ["city"]}}},
     "fn": get_weather},
    {"schema": {"type": "function", "function": {
        "name": "calc", "description": "计算四则运算表达式",
        "parameters": {"type": "object", "properties": {
            "expression": {"type": "string", "description": "如 12*(3+4)"}},
            "required": ["expression"]}}},
     "fn": calc},
]
SCHEMAS = [t["schema"] for t in TOOL_REGISTRY]
FN_MAP = {t["schema"]["function"]["name"]: t["fn"] for t in TOOL_REGISTRY}

# ---------- 3. Agent 主循环 ----------
def run(question: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个严谨的助手。需要事实数据时先调用工具，禁止编造数据。"},
        {"role": "user", "content": question},
    ]
    for step in range(1, MAX_STEPS + 1):
        print(f"===== 第 {step} 步 =====")
        resp = client.chat.completions.create(model=MODEL, messages=messages, tools=SCHEMAS)
        msg = resp.choices[0].message
        if not msg.tool_calls:
            print("模型给出最终答案。")
            return msg.content
        messages.append(msg)
        for tc in msg.tool_calls:
            name = tc.function.name
            args = json.loads(tc.function.arguments or "{}")
            print(f"调用工具 {name}({args})")
            try:
                result = FN_MAP[name](**args)
            except Exception as e:
                result = f"工具执行失败: {e}"  # 错误回传模型，由模型决定下一步
            messages.append({"role": "tool", "tool_call_id": tc.id,
                             "content": json.dumps(result, ensure_ascii=False)})
    return "达到最大步数，任务未完成。"

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "北京和上海今天的气温差多少度？"
    print("最终答案:\n" + (run(q) or ""))
