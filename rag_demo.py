"""最小 RAG：分块 -> 检索 -> 带引用回答（幻觉治理三件套：检索 grounding / 强制引用 / 拒答约束）
检索策略：优先向量检索（embedding-3）；接口不可用/余额不足时自动降级为
词面匹配（字符bigram重叠）——成本感知的兜底路径，保证demo零费用可运行。
依赖：pip install openai numpy
环境变量：ZHIPUAI_API_KEY（聊天走智谱API）
"""
import os
import re
import numpy as np
from openai import OpenAI

client = OpenAI(api_key=os.environ["ZHIPUAI_API_KEY"],
                base_url="https://open.bigmodel.cn/api/paas/v4")
CHAT_MODEL = os.environ.get("GLM_MODEL", "glm-4-flash")
EMBED_MODEL = "embedding-3"

DOCS = [
    "差旅制度（2026版）：国内出差住宿标准一线城市400元/晚，其他城市300元/晚；高铁二等座实报实销。",
    "报销流程：发票需在行程结束后30天内提交OA系统；超期提交需部门总监审批，季度末统一处理。",
    "信息安全红线：客户数据禁止上传外部模型服务；涉密文档仅在内网知识库流转，外发需安全审批。",
    "年假政策：入职满一年享10天年假，满五年15天；未休年假年底清零前可申请按日薪折算。",
]

def chunk(text: str, size: int = 45) -> list:
    """固定长度分块（生产环境应按语义/标题分块并保留10-15%重叠）"""
    return [text[i:i + size] for i in range(0, len(text), size)]

def embed(texts: list):
    """向量编码；接口不可用时返回 None 并触发降级"""
    try:
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
        return np.array([d.embedding for d in resp.data]), True
    except Exception as e:
        print(f"[降级] 向量接口不可用（{type(e).__name__}），检索层切换为词面匹配")
        return None, False

def _grams(s: str, n: int = 2) -> set:
    s = re.sub(r"\s", "", s)
    return {s[i:i + n] for i in range(len(s) - n + 1)}

def lexical_scores(query: str, chunks: list) -> list:
    """词面匹配计分：查询与分块的字符bigram重叠率（轻量中文检索兜底）"""
    qg = _grams(query)
    return [len(qg & _grams(c)) / (len(qg) + 1e-9) for c in chunks]

def retrieve(query: str, chunks: list, matrix, k: int = 2):
    if matrix is not None:
        qv = embed([query])[0][0]
        sims = matrix @ qv / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(qv) + 1e-9)
    else:
        sims = np.array(lexical_scores(query, chunks))
    idx = np.argsort(-sims)[:k]
    return [(chunks[i], float(sims[i])) for i in idx]

def answer(query: str) -> str:
    chunks = [c for d in DOCS for c in chunk(d)]
    matrix, ok = embed(chunks)
    hits = retrieve(query, chunks, matrix if ok else None)
    ctx = "\n".join(f"[{i + 1}] {h[0]}" for i, h in enumerate(hits))
    prompt = ("仅根据以下资料回答，禁止使用资料外知识；每条结论标注引用编号；"
              "资料不足以回答时明确说'资料中未提及'。\n\n" + ctx + f"\n\n问题：{query}")
    resp = client.chat.completions.create(model=CHAT_MODEL,
                                          messages=[{"role": "user", "content": prompt}])
    refs = "\n".join(f"[{i + 1}] 相似度 {h[1]:.2f}" for i, h in enumerate(hits))
    return resp.choices[0].message.content + "\n\n引用来源:\n" + refs

if __name__ == "__main__":
    import sys
    q = sys.argv[1] if len(sys.argv) > 1 else "出差住宿标准是多少？超过30天报销怎么办？"
    print(answer(q))
