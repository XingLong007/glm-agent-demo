"""最小评测集：用例即断言，对 Agent 行为做自动化回归评测（可接入CI）
断言维度：1) 是否按预期调用了工具（行为断言，而非字面匹配——LLM输出非确定性）
         2) 答案是否包含必要关键词
运行：python eval_agent.py   （需先 export ZHIPUAI_API_KEY）
"""
from agent_demo import run, FN_MAP

CASES = [
    {"id": "TC01", "q": "北京今天适合户外跑步吗？", "expect_tool": "get_weather", "must_include": ["北京"]},
    {"id": "TC02", "q": "请计算12*(3+4)等于多少", "expect_tool": "calc", "must_include": ["84"]},
    {"id": "TC03", "q": "用一句话说明你是什么", "expect_tool": None, "must_include": []},
    {"id": "TC04", "q": "上海今天天气怎么样？", "expect_tool": "get_weather", "must_include": ["上海"]},
]

def instrument():
    """包装工具函数以记录调用（生产环境应改为读取trace日志）"""
    called = []
    for name in list(FN_MAP.keys()):
        orig = FN_MAP[name]
        def wrapper(*a, _orig=orig, _name=name, **k):
            called.append(_name)
            return _orig(*a, **k)
        FN_MAP[name] = wrapper
    return called

def evaluate():
    results = []
    for c in CASES:
        called = instrument()
        ans = run(c["q"]) or ""
        tool_ok = (c["expect_tool"] in called) if c["expect_tool"] else (len(called) == 0)
        kw_ok = all(k in ans for k in c["must_include"])
        results.append({"id": c["id"], "tool_ok": tool_ok, "kw_ok": kw_ok,
                        "pass": tool_ok and kw_ok, "answer": ans[:60]})
    passed = sum(r["pass"] for r in results)
    print(f"\n===== 评测报告：{passed}/{len(results)} 通过 =====")
    for r in results:
        print(f"{r['id']}: {'PASS' if r['pass'] else 'FAIL'} | "
              f"工具断言:{r['tool_ok']} | 关键词:{r['kw_ok']} | {r['answer']}")
    print("\n说明：LLM输出存在非确定性，断言'行为'（是否调用正确工具）比断言"
          "'字面'更稳健——这是评测体系设计的第一原则。")
    return results

if __name__ == "__main__":
    evaluate()
