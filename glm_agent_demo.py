#!/usr/bin/env python3
"""最小 GLM Agent Demo：OpenAI 兼容接口 + 工具调用循环。

依赖：
    pip install openai

环境变量（二选一）：
    ZHIPUAI_API_KEY  或  GLM_API_KEY

运行：
    python glm_agent_demo.py
    python glm_agent_demo.py "上海和北京今天天气差多少？"
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI

# 智谱 OpenAI 兼容网关
BASE_URL = "https://open.bigmodel.cn/api/paas/v4/"
# 支持 Function Calling 的通用对话模型，可按账号可用列表替换
MODEL = os.getenv("GLM_MODEL", "glm-4-flash")
# 防止模型反复调工具导致死循环
MAX_STEPS = 8


def get_weather(city: str) -> dict[str, Any]:
    """示例工具：按城市返回模拟天气（不请求真实气象 API）。"""
    mock = {
        "北京": {"temp_c": 18, "condition": "晴", "humidity": 32, "wind": "北风 3 级"},
        "上海": {"temp_c": 22, "condition": "多云", "humidity": 68, "wind": "东南风 2 级"},
        "深圳": {"temp_c": 27, "condition": "阵雨", "humidity": 80, "wind": "南风 4 级"},
        "杭州": {"temp_c": 20, "condition": "阴", "humidity": 71, "wind": "东风 2 级"},
    }
    weather = mock.get(city, {"temp_c": 21, "condition": "晴间多云", "humidity": 50, "wind": "微风"})
    return {"city": city, "source": "mock", **weather}


# 告诉模型「有哪些工具、参数长什么样」。这是 OpenAI tools schema。
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "查询指定城市的当前天气（模拟数据）。需要具体城市名。",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名，例如：北京、上海、深圳",
                    }
                },
                "required": ["city"],
            },
        },
    }
]

# 本地可执行工具表：模型给出的 function.name -> Python 函数
TOOL_IMPLS = {"get_weather": get_weather}


def make_client() -> OpenAI:
    api_key = os.getenv("ZHIPUAI_API_KEY") or os.getenv("GLM_API_KEY")
    if not api_key:
        raise SystemExit("请设置环境变量 ZHIPUAI_API_KEY 或 GLM_API_KEY")
    return OpenAI(api_key=api_key, base_url=BASE_URL)


def run_tool(name: str, arguments_json: str) -> str:
    """执行一次工具调用，始终把结果序列化成字符串再回传给模型。"""
    fn = TOOL_IMPLS.get(name)
    if fn is None:
        return json.dumps({"error": f"未知工具: {name}"}, ensure_ascii=False)
    try:
        args = json.loads(arguments_json or "{}")
    except json.JSONDecodeError:
        return json.dumps({"error": "工具参数不是合法 JSON"}, ensure_ascii=False)
    result = fn(**args)
    return json.dumps(result, ensure_ascii=False)


def agent_loop(client: OpenAI, user_question: str) -> str:
    """工具调用循环：模型决定是否调工具 → 本地执行 → 把结果塞回 messages → 再问模型。

    直到模型不再返回 tool_calls（给出最终自然语言答案），或达到步数上限。
    """
    messages: list[dict[str, Any]] = [
        {
            "role": "system",
            "content": (
                "你是助手。需要天气信息时调用 get_weather；"
                "拿到工具结果后再用中文给出简洁最终答案。"
            ),
        },
        {"role": "user", "content": user_question},
    ]

    for step in range(1, MAX_STEPS + 1):
        print(f"\n===== 第 {step} 步：请求模型 =====")
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS,
            tool_choice="auto",  # 由模型判断：直接回答 or 调工具
        )
        msg = response.choices[0].message

        # 没有 tool_calls：本轮就是最终答案
        if not msg.tool_calls:
            answer = msg.content or ""
            print("模型直接给出最终答案（未再请求工具）。")
            return answer

        # 必须把「带 tool_calls 的 assistant 消息」原样追加进上下文
        messages.append(msg.model_dump(exclude_none=True))

        # 可能一次返回多个并行 tool_calls，逐个执行并回写 role=tool
        for call in msg.tool_calls:
            name = call.function.name
            args = call.function.arguments
            print(f"模型请求工具: {name}({args})")
            tool_output = run_tool(name, args)
            print(f"工具返回: {tool_output}")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.id,  # 必须对应上面那次调用的 id
                    "content": tool_output,
                }
            )

    return f"超过最大步数 {MAX_STEPS}，停止循环。"


def main() -> None:
    question = (
        sys.argv[1]
        if len(sys.argv) > 1
        else "北京今天天气怎么样？适合户外跑步吗？"
    )
    print(f"用户问题: {question}")
    print(f"模型: {MODEL}  |  接口: {BASE_URL}")
    client = make_client()
    answer = agent_loop(client, question)
    print("\n===== 最终答案 =====")
    print(answer)


if __name__ == "__main__":
    main()
