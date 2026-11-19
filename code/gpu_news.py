"""뉴스 데이터 정제 2단계: 로컬 AI(RTX 5090)로 시험 부정행위 기사를 고른다.

1) 한국어 임베딩 모델(dragonkue/snowflake-arctic-embed-l-v2.0-ko)로 기사 103,482건(제목 + 본문 앞 400자)을 벡터로 바꾼다.
2) 주제 질의 다섯 개와의 코사인 유사도 최댓값으로 순위를 매겨 위 500건을 고른다.
3) 재순위 모델(Qwen/Qwen3-Reranker-0.6B)이 "시험 부정행위나 그 감독, 방지를 다루는 기사인가"를 판정한다(예 확률).
4) 키워드 방식(제목, 키워드, 본문에 낱말이 함께 나옴)의 결과와 견준다.

실행(GPU 파이썬): HF_HUB_OFFLINE=1 <cuda python> analysis/gpu_news.py
출력: data/processed/news_ai_semantic_top500.csv, data/processed/news_ai_rerank_selected.csv
"""
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import numpy as np
import pandas as pd
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = next((ROOT / "data" / "raw").glob("15155969_*.csv"))
OUT = ROOT / "data" / "processed"
dev = "cuda"
print("GPU:", torch.cuda.get_device_name(0))

t0 = time.time()
df = pd.read_csv(SRC, encoding="utf-8-sig", dtype=str, on_bad_lines="skip")
df["text"] = df["제목"].fillna("") + "\n" + df["본문"].fillna("").str.slice(0, 400)
print(f"기사 {len(df):,}건 읽음 ({time.time() - t0:.0f}초)")

# 1) 임베딩
EMB = "dragonkue/snowflake-arctic-embed-l-v2.0-ko"
tok = AutoTokenizer.from_pretrained(EMB)
enc = AutoModel.from_pretrained(EMB, torch_dtype=torch.float16).to(dev).eval()


@torch.no_grad()
def embed(texts, bs=256, max_len=256):
    out = []
    for i in range(0, len(texts), bs):
        b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(dev)
        v = enc(**b).last_hidden_state[:, 0]  # CLS 풀링(모델 설정과 같음)
        out.append(torch.nn.functional.normalize(v.float(), dim=-1).cpu())
    return torch.cat(out)


t1 = time.time()
D = embed(df["text"].tolist())
print(f"임베딩 {D.shape} ({time.time() - t1:.0f}초)")
queries = ["시험 부정행위", "시험 중 커닝 적발", "인공지능을 이용한 시험 부정행위", "시험 감독과 부정행위 방지 기술", "수능 대입 시험 부정행위"]
Q = embed(["query: " + q for q in queries])
sim = (D @ Q.T).max(dim=1).values.numpy()
df["sim"] = sim
top = df.sort_values("sim", ascending=False).head(500).copy()
del enc
torch.cuda.empty_cache()

# 2) 재순위 판정
RR = "Qwen/Qwen3-Reranker-0.6B"
rtok = AutoTokenizer.from_pretrained(RR, padding_side="left")
rr = AutoModelForCausalLM.from_pretrained(RR, torch_dtype=torch.float16).to(dev).eval()
yes_id, no_id = rtok.convert_tokens_to_ids("yes"), rtok.convert_tokens_to_ids("no")
prefix = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
          "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
instr = ("이 기사가 시험(수능, 대입, 자격시험, 대학 시험, 토익 등)에서 일어나는 부정행위, 커닝, 대리시험이나 "
         "그것을 감독하고 막는 일을 주로 다루는지 판단한다. 시험을 만드는 AI, 기업 감사, 금융 부정은 해당하지 않는다.")
query = "시험 부정행위와 그 감독, 방지"


@torch.no_grad()
def rerank(docs, bs=16):
    ps = []
    for i in range(0, len(docs), bs):
        texts = [prefix + f"<Instruct>: {instr}\n<Query>: {query}\n<Document>: {d}" + suffix for d in docs[i:i + bs]]
        b = rtok(texts, padding=True, truncation=True, max_length=640, return_tensors="pt").to(dev)
        logits = rr(**b).logits[:, -1, :]
        two = torch.stack([logits[:, no_id], logits[:, yes_id]], dim=1).float()
        ps.append(torch.softmax(two, dim=1)[:, 1].cpu())
    return torch.cat(ps).numpy()


t2 = time.time()
top["yes_prob"] = rerank(top["text"].tolist())
print(f"재순위 판정 500건 ({time.time() - t2:.0f}초)")
cols = ["일자", "언론사", "제목", "주소", "sim", "yes_prob"]
top[cols].to_csv(OUT / "news_ai_semantic_top500.csv", index=False, encoding="utf-8-sig")
sel = top[top["yes_prob"] >= 0.5].sort_values("yes_prob", ascending=False)
sel[cols].to_csv(OUT / "news_ai_rerank_selected.csv", index=False, encoding="utf-8-sig")

kw = pd.read_csv(OUT / "news_ai_cheating.csv")
kw_urls = set(kw["주소"])
print(f"AI가 고른 기사 {len(sel)}건, 그 가운데 키워드 방식 44건과 겹치는 것 {len(set(sel['주소']) & kw_urls)}건")
print(f"전체 걸린 시간 {time.time() - t0:.0f}초")
print(sel[["일자", "언론사", "제목", "yes_prob"]].head(80).to_string(index=False))
