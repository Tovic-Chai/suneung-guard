"""언론진흥재단 뉴스빅데이터(인공지능 기사 메타데이터)에서 시험 부정행위 관련 기사를 골라낸다.

실행: py tools/filter_news.py
출력: data/processed/news_ai_cheating.csv (골라낸 기사), data/processed/news_ai_monthly.csv (달별 기사 수)
"""
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parent.parent
SRC = next((ROOT / "data" / "raw").glob("15155969_*.csv"))
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

for enc in ("utf-8-sig", "cp949"):
    try:
        df = pd.read_csv(SRC, encoding=enc, dtype=str, on_bad_lines="skip")
        break
    except UnicodeDecodeError:
        continue
print("읽음", len(df), "행, 인코딩", enc, "열", list(df.columns))
df["일자"] = pd.to_datetime(df["일자"].str[:10], format="%Y-%m-%d", errors="coerce")
text = (df["제목"].fillna("") + " " + df["키워드"].fillna("") + " " + df["본문"].fillna(""))

CHEAT = r"부정행위|커닝|컨닝|대리\s?시험|대리\s?응시|시험\s?부정"
EXAM = r"시험|수능|입시|과제|평가|자격증|토익|고사"
DEVICE = r"안경|글라스|카메라|이어폰|스마트워치|무선|블루투스|전자기기"
df["cheat"] = text.str.contains(CHEAT, regex=True)
df["exam"] = text.str.contains(EXAM, regex=True)
df["device"] = text.str.contains(DEVICE, regex=True)
sel = df[df["cheat"] & df["exam"]].copy()
print("시험 부정행위 관련", len(sel), "건, 그 가운데 기기 언급", int(sel["device"].sum()), "건")

monthly = df.groupby(df["일자"].dt.to_period("M")).agg(ai=("제목", "size"), cheat=("cheat", "sum"))
monthly["cheat_exam"] = sel.groupby(sel["일자"].dt.to_period("M")).size()
monthly = monthly.fillna(0).astype(int)
monthly.to_csv(OUT / "news_ai_monthly.csv", encoding="utf-8-sig")
sel[["일자", "언론사", "제목", "키워드", "주소", "device"]].sort_values("일자").to_csv(
    OUT / "news_ai_cheating.csv", index=False, encoding="utf-8-sig")
print("기간", df["일자"].min().date(), "~", df["일자"].max().date())
print(monthly.tail(18).to_string())
print(sel.sort_values("일자").tail(12)[["일자", "언론사", "제목"]].to_string())
