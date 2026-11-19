# %% [markdown]
# # AI와 초소형 카메라 부정행위, 데이터로 본 문제와 다중 검사 해결안
#
# 2027학년도 수능(2026-11-19)을 앞두고 드러난 AI 부정행위를 공공데이터와 기사 수치, 직접 잰 무선 신호로 분석합니다.
# 모든 수치의 출처는 data/SOURCES.md와 각 표의 출처 열에 있습니다.

# %%
# 구글 코랩에서 돌릴 때만: 한글 글꼴을 깔고 자료 묶음(suneung_data.zip)을 올립니다.
import sys, os
if "google.colab" in sys.modules:
    os.system("apt-get -qq install -y fonts-nanum > /dev/null")
    from google.colab import files
    if not os.path.exists("data"):
        up = files.upload()  # suneung_data.zip 선택
        os.system("unzip -q -o suneung_data.zip")

# %%
import glob
import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(".").resolve()
if not (ROOT / "data").exists() and (ROOT.parent / "data").exists():
    ROOT = ROOT.parent
P = ROOT / "data" / "processed"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

names = {f.name for f in font_manager.fontManager.ttflist}
for fam in ("Malgun Gothic", "NanumGothic", "AppleGothic"):
    if fam in names:
        plt.rcParams["font.family"] = fam
        break
else:
    for p in glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf"):
        font_manager.fontManager.addfont(p)
    plt.rcParams["font.family"] = "NanumGothic"
plt.rcParams["axes.unicode_minus"] = False

INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, LIGHT = "#2a78d6", "#eb6834", "#1baf7a", "#cde2fb"
plt.rcParams.update({
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "figure.dpi": 110, "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def save(fig, name):
    fig.savefig(FIG / name)
    print("그림 저장:", FIG / name)


# %% [markdown]
# ## 1. 문제: 해마다 200건 넘게 적발되지만, 무선기기 부정행위는 거의 잡히지 않는다

# %%
y = pd.read_csv(P / "cheating_by_year.csv")
fig, ax = plt.subplots(figsize=(7, 3.6))
colors = [LIGHT if "계산" in k else BLUE for k in y["값의 종류"]]
bars = ax.bar(y["학년도"].astype(str), y["적발건수"], color=colors, width=0.62)
for b, v, k in zip(bars, y["적발건수"], y["값의 종류"]):
    ax.text(b.get_x() + b.get_width() / 2, v + 4, f"{v}" + ("*" if "계산" in k else ""), ha="center", color=INK2)
ax.set_ylim(0, 300)
ax.set_ylabel("적발 건수")
ax.set_title("수능 부정행위 적발 건수(학년도별)")
ax.grid(axis="x", visible=False)
fig.text(0.01, -0.04, "* 2020학년도는 2020~2024 합계(1,174건)에서 나머지 해를 뺀 계산값. 출처: 교육부 자료(뉴스1, 뉴시스)",
         color=MUTED, fontsize=9)
save(fig, "f1_cheating_by_year.png")
plt.show()

# %%
t = pd.read_csv(P / "cheating_by_type_2020_2024.csv")
t = t.sort_values("건수")
fig, ax = plt.subplots(figsize=(7.5, 3.6))
col = [ORANGE if "무선기기" in u else BLUE for u in t["유형"]]
bars = ax.barh(t["유형"], t["건수"], color=col, height=0.6)
for b, v in zip(bars, t["건수"]):
    ax.text(v + 5, b.get_y() + b.get_height() / 2, f"{v}건", va="center", color=INK2)
ax.set_xlim(0, 430)
ax.set_xlabel("2020~2024학년도 합계(건)")
ax.set_title("5년 1,174건 가운데 무선기기를 써서 잡힌 것은 1건")
ax.grid(axis="y", visible=False)
save(fig, "f2_cheating_by_type.png")
plt.show()
print("무선기기 이용 비율:", round(1 / 1174 * 100, 2), "%")

# %%
# 같은 자료를 1,174칸 점그림(한 칸이 한 건)으로: 무선기기 1건이 얼마나 드문지 한눈에 보이게
order = [("종료령 이후 답안 작성", 376, "#86b6ef"), ("4교시 응시방법 위반", 361, "#5598e7"),
         ("반입금지물품 소지", 336, "#1c5cab"), ("휴대 가능 물품 외 소지", 72, "#104281"),
         ("기타(대리응시 등, 기사에 유형 없음)", 1174 - 376 - 361 - 336 - 72 - 1, "#c3c2b7"), ("무선기기 이용", 1, ORANGE)]
cols_n = 59
cells = [c for _, n, c in order for _ in range(n)]
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(9, 3.7))
for i, c in enumerate(cells):
    x, yv = i % cols_n, -(i // cols_n)
    ax.add_patch(plt.Rectangle((x, yv), 0.86, 0.86, color=c, lw=0))
last = len(cells) - 1
ax.add_patch(plt.Rectangle((last % cols_n - 0.25, -(last // cols_n) - 0.25), 1.36, 1.36, fill=False, ec=ORANGE, lw=1.6))
ax.annotate("무선기기 이용 1건", xy=(last % cols_n + 0.4, -(last // cols_n) - 0.3),
            xytext=(last % cols_n - 6, -(last // cols_n) - 2.6), fontsize=10, color=ORANGE, fontweight="bold",
            arrowprops=dict(arrowstyle="-", color=ORANGE, lw=1.2), annotation_clip=False)
ax.set_xlim(-0.5, cols_n + 0.5)
ax.set_ylim(-(len(cells) // cols_n) - 1, 1.2)
ax.set_aspect("equal")
ax.axis("off")
handles = [Patch(color=c, label=f"{name} {n:,}건") for name, n, c in order]
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02),
           handlelength=1.0, columnspacing=1.6)
ax.set_title("한 칸이 한 건: 2020~2024학년도 부정행위 1,174건")
fig.subplots_adjust(bottom=0.17, top=0.92)
save(fig, "f2b_cheating_waffle.png")
plt.show()

# %% [markdown]
# ## 2. 오늘 수능: 적극적으로 찾자 하루에 98명이 나왔다

# %%
d = pd.read_csv(P / "suneung2027_day_electronics.csv")
parts = {
    "위급재난문자로 울린 휴대전화": int(d.loc[d["구분"].str.startswith("위급재난문자"), "인원"].sum()),
    "와이파이 신호 탐지 등": int(d.loc[d["구분"].str.startswith("와이파이 신호 탐지 등"), "인원"].iloc[0]),
    "스마트기기를 직접 씀(중대)": int(d.loc[d["구분"].str.startswith("스마트기기 직접"), "인원"].iloc[0]),
}
past_avg = 336 / 5  # 2020~2024 반입금지물품 소지 연평균
fig, ax = plt.subplots(figsize=(7.5, 3.4))
labels, vals = list(parts)[::-1], list(parts.values())[::-1]
bars = ax.barh(labels, vals, color=[ORANGE, BLUE, BLUE], height=0.55)
for b, v in zip(bars, vals):
    ax.text(v + 1, b.get_y() + b.get_height() / 2, f"{v}명", va="center", color=INK2)
ax.set_xlim(0, 60)
ax.set_xlabel("인원(명)")
ax.set_title(f"2027학년도 수능 당일 전자기기 관련 적발 {sum(vals)}명")
ax.grid(axis="y", visible=False)
fig.text(0.01, -0.06, f"참고: 2020~2024학년도 '반입금지물품 소지' 적발은 연평균 {past_avg:.1f}건. "
         "오늘 수치는 교육부 11-19 16:37 브리핑 기준 속보(파이낸셜뉴스)", color=MUTED, fontsize=9)
save(fig, "f3_suneung2027_day.png")
plt.show()

# %% [markdown]
# ## 3. 사례: 2026년 AI 부정행위, 현장에서 잡은 것은 일부뿐

# %%
# 날짜가 달까지만 알려진 사례는 그 달 15일에 찍음(토익 2026-05, 대만대 보도 2026-06)
cases = pd.DataFrame([
    ("2026-05-15", "토익 정기시험", "AI 글라스", "현장"),
    ("2026-06-12", "국립대만대 지필고사(보도일)", "AI 안경", "불명"),
    ("2026-10-03", "서울시립대 수리논술", "AI 안경", "현장"),
    ("2026-10-10", "연세대 수리논술", "AI 안경", "현장"),
    ("2026-10-14", "건국대 중간고사", "초소형 카메라와 스마트워치", "사후"),
    ("2026-11-14", "세무사 시험(보도일)", "초소형 카메라와 생성형 AI", "사후"),
    ("2026-11-19", "2027 수능(용인)", "소형카메라와 무선이어폰", "현장"),
], columns=["시기", "시험", "장비", "적발"])
cases["날짜"] = pd.to_datetime(cases["시기"])
cmap = {"현장": BLUE, "사후": ORANGE, "불명": MUTED}
fig, ax = plt.subplots(figsize=(8, 3.8))
for i, r in cases.reset_index(drop=True).iterrows():
    ax.scatter(r["날짜"], i, s=90, color=cmap[r["적발"]], zorder=3, edgecolor="white", linewidth=1.5)
    ax.text(r["날짜"], i + 0.32, f"{r['시험']}  {r['장비']}", fontsize=9, color=INK2, ha="center")
ax.set_yticks([])
ax.set_ylim(-0.8, len(cases) - 0.2)
ax.set_title("현장에서 잡은 것은 7건 가운데 4건, 2건은 나중에 드러남")
for k, c in cmap.items():
    ax.scatter([], [], color=c, s=60, label={"현장": "현장 적발", "사후": "나중에 드러남", "불명": "적발 방식 모름"}[k])
ax.legend(loc="lower right", frameon=False, fontsize=9)
ax.grid(axis="y", visible=False)
save(fig, "f4_cases_timeline.png")
plt.show()
print(cases["적발"].value_counts().to_dict())

# %% [markdown]
# ## 4. 뉴스 데이터 정제: 키워드 방식과 로컬 AI 방식
# 한국언론진흥재단 뉴스빅데이터(인공지능 기사 103,482건, 2024년 1~6월)에서 시험 부정행위 기사를 두 가지 방법으로 골랐습니다.
# 키워드 방식은 tools/filter_news.py, 로컬 AI 방식은 analysis/gpu_news.py(RTX 5090에서 임베딩 74초, 재순위 3초)이고,
# 이 칸은 그 결과 파일만 읽습니다. 마지막 판정은 사람이 제목을 읽고 했습니다.

# %%
news = pd.read_csv(P / "news_ai_cheating.csv")
keep_words = ["챗GPT 통한 부정행위", "옵저뷰", "온라인시험 부정행위", "프록토매틱", "中 대입시험", "中대입 시험",
              "가오카오", "대학 시험지", "대학시험 답안지"]
news["관련"] = news["제목"].fillna("").apply(lambda s: any(w in s for w in keep_words))
rel = news[news["관련"]].copy()


def kind(s):
    if any(w in s for w in ["中", "가오카오"]):
        return "중국 대입시험 감독"
    if any(w in s for w in ["옵저뷰", "온라인시험", "프록토매틱"]):
        return "온라인 시험 감독 기술"
    return "생성형 AI로 시험, 과제"


rel["분류"] = rel["제목"].apply(kind)
steps = pd.DataFrame({"단계": ["인공지능 기사 전체", "부정행위와 시험 낱말이 함께 나온 기사", "제목을 읽고 남긴 관련 기사"],
                      "건수": [103482, len(news), len(rel)]})
print(steps.to_string(index=False))
print(rel["분류"].value_counts().to_string())
print("관련 기사 가운데 착용형 기기(안경, 카메라, 이어폰 등) 언급:", int(rel["device"].sum()), "건")
print(rel[["일자", "언론사", "제목", "분류"]].to_string(index=False))

# %%
# 로컬 AI 방식: 임베딩 유사도 상위 500건을 재순위 모델 점수로 다시 줄 세운 뒤 위 20건을 사람이 확인
sem = pd.read_csv(P / "news_ai_semantic_top500.csv").sort_values("yes_prob", ascending=False).reset_index(drop=True)
human_urls = set(rel["주소"])
new_found = ["프록토매틱'' 선봬"]  # 키워드 방식이 놓쳤고 사람이 관련으로 판정한 기사(에딘트, CES 2024 AI 시험 감독 서비스)
top20 = sem.head(20).copy()
top20["관련"] = top20["주소"].isin(human_urls) | top20["제목"].apply(lambda s: any(w in s for w in new_found))
ai_rel = int(top20["관련"].sum())
in500 = int(sem["주소"].isin(human_urls).sum())
print(f"사람이 고른 {len(rel)}건 가운데 임베딩 상위 500건 안에 든 것 {in500}건")
print(f"재순위 상위 20건 가운데 관련 {ai_rel}건(정밀도 {ai_rel / 20:.0%}), 키워드 방식이 놓친 기사 "
      f"{int((top20['관련'] & ~top20['주소'].isin(human_urls)).sum())}건")

fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True)
for ax, (title, stages, color) in zip(axes, [
    ("키워드 방식", [("인공지능 기사 전체", 103482), ("키워드가 함께 나온 기사", len(news)), ("사람이 읽고 남김", len(rel))], BLUE),
    ("로컬 AI 방식(RTX 5090)", [("인공지능 기사 전체", 103482), ("임베딩 유사도 상위", 500), ("재순위 상위", 20),
                                 ("사람이 읽고 남김", ai_rel)], AQUA),
]):
    names_s, vals_s = [s for s, _ in stages], [v for _, v in stages]
    widths = np.log10(np.array(vals_s, dtype=float) + 1)
    W = np.log10(103483)
    for i, (nm, v, w) in enumerate(zip(names_s, vals_s, widths)):
        ax.barh(-i, w, left=(W - w) / 2, color=color, height=0.42, alpha=0.35 + 0.65 * (i == len(stages) - 1))
        ax.text(W / 2, -i + 0.26, f"{nm}  {v:,}건", ha="center", va="bottom", fontsize=9.5, color=INK)
    ax.set_xlim(0, W)
    ax.set_ylim(-3.5, 0.75)
    ax.set_title(title)
    ax.axis("off")
fig.text(0.5, 0.0, f"정밀도(남긴 것 / 사람이 읽은 것): 키워드 방식 {len(rel)}/{len(news)} = {len(rel) / len(news):.0%}, "
         f"로컬 AI {ai_rel}/20 = {ai_rel / 20:.0%}.  막대 너비는 로그 눈금", ha="center", fontsize=9.5, color=MUTED)
save(fig, "f8_news_cleaning_funnel.png")
plt.show()

# %% [markdown]
# ## 5. 해결안 분석: 수법마다 적어도 두 겹에 걸리는가

# %%
layers = ["차폐 보관\n(신고한 기기)", "EMP\n(숨긴 기기)", "금속탐지기", "와이파이, 블루투스,\n전파 감시"]
methods = ["AI 안경", "초소형 카메라와 무선 이어폰\n(실시간 풀이)", "숨긴 스마트폰", "스마트워치", "저장만 하는 초소형 카메라\n(문제 유출)"]
# 2 = 잡음, 1 = 조건부, 0 = 못 잡음. 근거: tech_params.csv, equipment_prices.csv
M = np.array([
    [1, 2, 2, 2],  # AI 안경: 신고하면 보관, 숨기면 EMP, 금속 부품, 무선 통신
    [1, 2, 2, 2],  # 카메라와 이어폰: 실시간이면 반드시 송신
    [1, 2, 2, 2],
    [1, 2, 2, 2],
    [1, 2, 2, 0],  # 저장형: 송신하지 않아 전파 감시로는 못 잡음
])
why = {(4, 3): "송신 없음"}
fig, ax = plt.subplots(figsize=(8.2, 4.2))
cm = matplotlib.colors.ListedColormap(["#f0efec", LIGHT, BLUE])
ax.imshow(M, cmap=cm, vmin=0, vmax=2, aspect="auto")
txt = {0: "못 잡음", 1: "신고 시", 2: "잡음"}
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, why.get((i, j), txt[M[i, j]]), ha="center", va="center",
                color="white" if M[i, j] == 2 else INK, fontsize=10)
ax.set_xticks(range(len(layers)), layers, fontsize=9)
ax.set_yticks(range(len(methods)), methods, fontsize=9)
ax.tick_params(length=0)
ax.grid(False)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("어느 수법도 한 겹만 피해서는 통과하지 못한다")
save(fig, "f5_layer_matrix.png")
plt.show()
print("수법별로 '잡음'인 겹 수:", dict(zip([m.replace("\n", " ") for m in methods], (M == 2).sum(axis=1))))

# %% [markdown]
# ## 6. EMP 범위 계산: 열린 입구와 차폐 부스
# 가정: 물건을 망가뜨릴 세기 10 kV/m를 0.5 m에서 낸다. 열린 곳에서 세기는 거리에 반비례(먼 거리 근사)한다.
# 근접장에서는 더 빨리 줄어들므로 이 계산은 넉넉하게(위험을 크게) 잡은 값이다.

# %%
E0, r0 = 10_000.0, 0.5
r = np.logspace(np.log10(0.5), np.log10(200), 300)
E_open = E0 * r0 / r
thr = {"휴대폰 수신기 손상(약 300 V/m)": 300, "고출력 전자기 기준(100 V/m)": 100, "일반인 노출 기준, 1 GHz(43.5 V/m)": 43.5}
fig, ax = plt.subplots(figsize=(8, 4.2))
ax.loglog(r, E_open, color=ORANGE, lw=2.2, label="열린 입구")
for sh, c in ((60, BLUE), (80, AQUA)):
    rb = r[r >= 1.0]
    ax.loglog(rb, E0 * r0 / rb / 10 ** (sh / 20), color=c, lw=2.2, label=f"차폐 부스 밖({sh} dB)")
for (k, v), ls in zip(thr.items(), ["--", ":", "-."]):
    ax.axhline(v, color=MUTED, lw=1, ls=ls)
    ax.text(0.55, v * 1.12, k, ha="left", fontsize=8.5, color=INK2)
reach = {k: E0 * r0 / v for k, v in thr.items()}
ax.axvline(reach["휴대폰 수신기 손상(약 300 V/m)"], color=ORANGE, lw=0.8, alpha=0.6)
plain = FuncFormatter(lambda v, _: f"{v:g}")
ax.xaxis.set_major_formatter(plain)
ax.yaxis.set_major_formatter(plain)
ax.set_xlabel("EMP에서 떨어진 거리(m)")
ax.set_ylabel("전기장 세기(V/m)")
ax.set_title(f"열린 입구에서는 {reach['휴대폰 수신기 손상(약 300 V/m)']:.0f} m 안 휴대폰까지 위험, 부스 밖은 기준 아래")
ax.legend(frameon=False, fontsize=9, loc="lower left")
save(fig, "f6_emp_range.png")
plt.show()
for k, v in reach.items():
    print(f"열린 입구에서 '{k}' 아래로 떨어지는 거리: {v:.1f} m")
for sh in (60, 80):
    print(f"{sh} dB 부스 바로 밖(1 m) 세기: {E0 * r0 / 1.0 / 10 ** (sh / 20):.2f} V/m")

# %%
pulse_peak = 43.5 * 32
need = {"전자기기 손상 하한(3 kV/m로 잡음)": 3000, "확실한 손상(10 kV/m)": 10000}
print("ICNIRP 1998 일반인 펄스 순간 최대(1 GHz):", round(pulse_peak), "V/m")
for k, v in need.items():
    print(f"{k}는 그 {v / pulse_peak:.1f}배. 몸에 쏘는 단계는 기준값을 넘으므로 실측 검증이 필요한 과제로 남김")

# %% [markdown]
# ## 7. 비용: 값을 아는 장비만 계산(시험장 1,312곳)

# %%
sites = 1312
items = pd.DataFrame([
    ("금속탐지기 2대", 2 * 10, 2 * 20),
    ("전파탐지기 1대", 60, 80),
    ("와이파이, 블루투스 감시(이미 있는 노트북 또는 휴대폰)", 0, 0),
], columns=["항목", "최소(만 원)", "최대(만 원)"])
per_lo, per_hi = items["최소(만 원)"].sum(), items["최대(만 원)"].sum()
print(items.to_string(index=False))
print(f"시험장 한 곳 {per_lo}~{per_hi}만 원, 전국 {sites}곳 {per_lo * sites / 10000:.1f}~{per_hi * sites / 10000:.1f}억 원")
print("EMP 부스와 차폐 파우치 값은 아직 자료가 없어 빠져 있음")

# %% [markdown]
# ## 8. 직접 잰 무선 신호: 기준선만으로 경보 규칙을 정한다
# 이 PC로 10분 동안 15초마다 40회 주변 와이파이 접속 지점과 블루투스 광고를 기록했다(시험 기기 없음).
# 아무 일도 없는 방에서 경보가 울리면 모두 헛경보다.

# %%
f = sorted(glob.glob(str(ROOT / "data" / "rf" / "scan_*_baseline.csv")))[-1]
b = pd.read_csv(f)
R = int(b["round"].max())
per_round = b.groupby(["round", "kind"])["dev"].nunique().unstack(fill_value=0)
learn = 5  # 처음 5회(약 75초)는 기준선 학습
known = set(b.loc[b["round"] <= learn, "dev"])


seen = b.groupby("round")["dev"].apply(set).to_dict()


def alarms(rule):
    if rule == "직전 회차에 없던 기기가 보이면 경보":
        return sum(len(seen[k] - seen.get(k - 1, set())) for k in range(2, R + 1))
    ev = []
    for dev, g in b[b["round"] > learn].groupby("dev"):
        if dev in known:
            continue
        strong = g[g["rssi_dbm"] >= -75]
        if rule == "처음 75초에 본 기기는 빼고 경보":
            ev.append(dev)
        elif rule == "위 조건에 -75 dBm보다 세고 2회 연속" and len(strong):
            sr = sorted(strong["round"])
            if any(b2 - a2 == 1 for a2, b2 in zip(sr, sr[1:])):
                ev.append(dev)
    return len(ev)


rules = ["직전 회차에 없던 기기가 보이면 경보", "처음 75초에 본 기기는 빼고 경보", "위 조건에 -75 dBm보다 세고 2회 연속"]
res = pd.DataFrame({"규칙": rules, "10분 동안 헛경보": [alarms(x) for x in rules]})
print(f"회차 {R}, 와이파이 기기 {b[b.kind == 'wifi'].dev.nunique()}개(모든 회차에 보인 것 "
      f"{int((b[b.kind == 'wifi'].groupby('dev')['round'].nunique() == R).sum())}개), 블루투스 기기 {b[b.kind == 'ble'].dev.nunique()}개")
print(res.to_string(index=False))

fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw={"width_ratios": [1.5, 1]})
ax = axes[0]
ax.plot(per_round.index * 15 / 60, per_round.get("wifi", 0), color=BLUE, lw=2, label="와이파이 접속 지점")
ax.plot(per_round.index * 15 / 60, per_round.get("ble", 0), color=AQUA, lw=2, label="블루투스 기기")
ax.set_xlabel("시간(분)")
ax.set_ylabel("그 회차에 보인 기기 수")
ax.set_title("아무 일 없는 방에서도 보이는 기기 수가 흔들린다")
ax.set_ylim(0, 13)
ax.legend(frameon=False, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
ax = axes[1]
lab = ["직전 회차에 없던\n기기면 경보", "처음 75초 기기는\n빼고 경보", "+ -75 dBm보다 세고\n2회 연속"]
bars = ax.barh(lab[::-1], res["10분 동안 헛경보"][::-1], color=[BLUE, BLUE, ORANGE], height=0.55)
for bb, v in zip(bars, res["10분 동안 헛경보"][::-1]):
    ax.text(v + 1.0, bb.get_y() + bb.get_height() / 2, f"{v}건", va="center", color=INK2)
ax.set_xlim(0, 56)
ax.set_xlabel("10분 동안 헛경보(건)")
ax.set_title("기준선을 먼저 배우면 헛경보가 사라진다")
ax.grid(axis="y", visible=False)
fig.tight_layout()
save(fig, "f7_rf_baseline.png")
plt.show()

# %%
# 기기마다: 40회 가운데 몇 번 보였나(가로)와 가장 센 신호(세로). 경보 문턱 -75 dBm 위쪽만 경보 후보가 된다.
dv = b.groupby(["kind", "dev"]).agg(seen=("round", "nunique"), rssi=("rssi_dbm", "max"),
                                     first=("round", "min")).reset_index()
dv["seen_pct"] = dv["seen"] / R * 100
fig, ax = plt.subplots(figsize=(7, 4))
ax.axhspan(-75, -40, color=LIGHT, alpha=0.5, lw=0)
ax.text(2, -73, "경보 후보 구역(-75 dBm보다 셈)", fontsize=9, color=INK2, va="bottom")
for k, c, lab in (("wifi", BLUE, "와이파이 접속 지점"), ("ble", AQUA, "블루투스 기기")):
    s = dv[dv["kind"] == k]
    ax.scatter(s["seen_pct"], s["rssi"], s=70, color=c, edgecolor="white", linewidth=1.2, label=lab, zorder=3)
ax.set_xlim(0, 105)
ax.set_ylim(-100, -45)
ax.set_xlabel("40회 가운데 보인 비율(%)")
ax.set_ylabel("가장 센 신호(dBm)")
ax.set_title("깜빡이는 기기는 모두 약한 이웃 신호, 센 신호는 처음부터 있던 것")
ax.legend(frameon=False, fontsize=9, loc="upper left")
save(fig, "f9_rf_devices_scatter.png")
plt.show()
print(dv.sort_values("rssi", ascending=False)[["kind", "seen", "rssi", "first"]].to_string(index=False))

# %% [markdown]
# ## 9. 남은 실험
# 휴대폰 핫스팟(와이파이로 영상을 보내는 초소형 카메라와 같은 방식)과 블루투스 이어폰으로 거리별 탐지율,
# 호일과 전원 뺀 전자레인지로 차폐 감쇠(dB)를 잰다. 순서는 data/rf/EXPERIMENT.md.
