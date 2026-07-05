# -*- coding: utf-8 -*-
"""
☕ 카페인 & 컨디션 분석 대시보드
- Streamlit + Supabase(PostgreSQL) + Plotly
- 학생별 카페인 섭취 패턴과 피로/집중/수면/스트레스의 관계 분석
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# =========================================================
# 0. 페이지 설정 & 상수
# =========================================================
st.set_page_config(page_title="카페인 & 컨디션 분석", page_icon="☕", layout="wide")

TABLE_NAME = "caffeine_survey"   # Supabase 테이블명 (본인 환경에 맞게 수정)

# 카테고리 순서 정의 (정렬용)
AMOUNT_ORDER = ["안마심", "100ml이하", "100~250ml", "250~500ml", "500ml이상"]
TIME_ORDER = ["안마심", "오전", "점심", "오후"]
DRINK_TYPES = ["커피", "에너지드링크", "콜라", "녹차", "안마심"]

# 섭취량을 대략적인 mg으로 환산 (게이미피케이션/분석용 근사치)
AMOUNT_TO_MG = {
    "안마심": 0, "100ml이하": 40, "100~250ml": 95,
    "250~500ml": 160, "500ml이상": 260,
}

NUMERIC_COLS = ["tired_score", "focus_score", "sleep_hours", "stress_score"]

# 점수형 컬럼의 유효 범위 (방어용)
VALID_RANGE = {
    "tired_score": (1, 5), "focus_score": (1, 5),
    "stress_score": (1, 5), "sleep_hours": (0, 24),
}

KR = {
    "tired_score": "피로도", "focus_score": "집중도",
    "sleep_hours": "수면시간", "stress_score": "스트레스",
}


# =========================================================
# 1. Supabase 연결 (실패해도 앱이 죽지 않도록 방어)
# =========================================================
# 기본 연결 정보 (publishable 키는 공개용이라 노출돼도 안전한 편)
# 더 안전하게 쓰려면 .streamlit/secrets.toml 에 SUPABASE_URL / SUPABASE_KEY 를 넣으면
# 아래 기본값 대신 그 값이 우선 사용됩니다.
DEFAULT_SUPABASE_URL = "https://umottuuwdxrmiozzrxpd.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_5jJW9QdYV-BYbdhRz8zOBA_DSld4CGL"


@st.cache_resource(show_spinner=False)
def get_supabase():
    """Supabase 클라이언트 생성. secrets 있으면 우선, 없으면 기본값으로 폴백."""
    try:
        from supabase import create_client
    except Exception:
        return None

    # secrets 우선, 실패하면 코드에 넣어둔 기본값 사용
    try:
        url = st.secrets["SUPABASE_URL"]
        key = st.secrets["SUPABASE_KEY"]
    except Exception:
        url = DEFAULT_SUPABASE_URL
        key = DEFAULT_SUPABASE_KEY

    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


# =========================================================
# 2. 데이터 로딩 & 정제 (핵심 방어 로직)
# =========================================================
def _empty_df():
    cols = ["student_id", "caffeine_yn", "drink_type", "drink_amount",
            "drink_time"] + NUMERIC_COLS
    return pd.DataFrame(columns=cols)


def clean_dataframe(raw: pd.DataFrame) -> pd.DataFrame:
    """어떤 지저분한 데이터가 들어와도 안전한 형태로 정규화."""
    if raw is None or len(raw) == 0:
        return _empty_df()

    df = raw.copy()

    # 필수 컬럼 보장 (없으면 생성)
    for col in _empty_df().columns:
        if col not in df.columns:
            df[col] = np.nan

    # 문자열 컬럼 정리
    for col in ["student_id", "caffeine_yn", "drink_type", "drink_amount", "drink_time"]:
        df[col] = df[col].astype(str).str.strip()
        df[col] = df[col].replace({"nan": np.nan, "None": np.nan, "": np.nan})

    # 숫자 컬럼: 강제 숫자화 + 범위 벗어난 값 제거
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        lo, hi = VALID_RANGE[col]
        df.loc[(df[col] < lo) | (df[col] > hi), col] = np.nan

    # caffeine_yn 정규화
    df["caffeine_yn"] = df["caffeine_yn"].apply(
        lambda x: "예" if str(x).strip() in ("예", "YES", "Yes", "y", "Y", "True", "1")
        else ("아니오" if str(x).strip() in ("아니오", "NO", "No", "n", "N", "False", "0")
              else np.nan)
    )

    # 완전 중복 제거
    df = df.drop_duplicates()

    # 파생: 카페인 근사 mg
    df["caffeine_mg"] = df["drink_amount"].map(AMOUNT_TO_MG).fillna(0)

    return df.reset_index(drop=True)


@st.cache_data(ttl=60, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Supabase에서 로드. 실패하면 로컬 샘플(data.csv)로 폴백."""
    sb = get_supabase()
    if sb is not None:
        try:
            res = sb.table(TABLE_NAME).select("*").execute()
            data = getattr(res, "data", None)
            if data:
                return clean_dataframe(pd.DataFrame(data))
        except Exception as e:
            st.warning(f"Supabase 조회에 실패해 샘플 데이터로 표시합니다. ({type(e).__name__})")

    # 폴백: 로컬 CSV
    try:
        return clean_dataframe(pd.read_csv("data.csv"))
    except Exception:
        return _empty_df()


def insert_row(row: dict) -> tuple[bool, str]:
    """설문 1건 저장. (성공여부, 메시지)"""
    sb = get_supabase()
    if sb is None:
        return False, "Supabase에 연결되지 않았습니다. secrets 설정을 확인하세요."
    try:
        sb.table(TABLE_NAME).insert(row).execute()
        return True, "저장되었습니다."
    except Exception as e:
        return False, f"저장 중 오류: {type(e).__name__} - {e}"


# =========================================================
# 3. 컨디션 점수 (게이미피케이션)
# =========================================================
def compute_wellness(row) -> float:
    """
    0~100 컨디션 스코어.
    집중도↑, 수면 적정(7~8h)↑ = 가점 / 피로↑, 스트레스↑ = 감점.
    NaN은 중립값으로 보정해 계산이 깨지지 않게 함.
    """
    focus = row.get("focus_score")
    tired = row.get("tired_score")
    stress = row.get("stress_score")
    sleep = row.get("sleep_hours")

    focus = 3 if pd.isna(focus) else focus
    tired = 3 if pd.isna(tired) else tired
    stress = 3 if pd.isna(stress) else stress
    sleep = 7 if pd.isna(sleep) else sleep

    # 각 요소를 0~1로 정규화
    focus_n = (focus - 1) / 4                    # 높을수록 좋음
    tired_n = 1 - (tired - 1) / 4                # 낮을수록 좋음
    stress_n = 1 - (stress - 1) / 4              # 낮을수록 좋음
    # 수면: 7.5시간을 최적점으로 하는 삼각형 점수
    sleep_n = max(0.0, 1 - abs(sleep - 7.5) / 4)

    score = (focus_n * 0.30 + tired_n * 0.25 +
             stress_n * 0.25 + sleep_n * 0.20) * 100
    return round(float(np.clip(score, 0, 100)), 1)


def grade_of(score: float):
    """점수 → 등급/이모지/색."""
    if score >= 80:  return "S", "🏆", "#12B76A", "최상의 컨디션"
    if score >= 65:  return "A", "😊", "#66C61C", "좋은 컨디션"
    if score >= 50:  return "B", "🙂", "#F79009", "무난한 컨디션"
    if score >= 35:  return "C", "😥", "#F04438", "주의가 필요해요"
    return "D", "🚨", "#D92D20", "휴식이 시급해요"


def add_wellness(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        df["wellness"] = []
        return df
    df = df.copy()
    df["wellness"] = df.apply(compute_wellness, axis=1)
    return df


# =========================================================
# 4. 자동 인사이트 문구 생성
# =========================================================
def safe_mean(series):
    s = pd.to_numeric(series, errors="coerce").dropna()
    return float(s.mean()) if len(s) else None


def generate_insights(df: pd.DataFrame) -> list[str]:
    """데이터에서 통계적 패턴을 감지해 자연어 인사이트로 변환."""
    out = []
    if len(df) < 3:
        return ["데이터가 충분하지 않아 인사이트를 생성할 수 없습니다. (최소 3건 필요)"]

    caf = df[df["caffeine_yn"] == "예"]
    nocaf = df[df["caffeine_yn"] == "아니오"]

    # 1) 카페인 O vs X 집중도/피로
    if len(caf) >= 2 and len(nocaf) >= 2:
        cf, nf = safe_mean(caf["focus_score"]), safe_mean(nocaf["focus_score"])
        if cf is not None and nf is not None:
            diff = cf - nf
            if abs(diff) >= 0.3:
                direction = "높게" if diff > 0 else "낮게"
                out.append(
                    f"☕ 카페인을 섭취하는 학생의 평균 집중도는 **{cf:.1f}점**으로, "
                    f"섭취하지 않는 학생(**{nf:.1f}점**)보다 {abs(diff):.1f}점 {direction} 나타났습니다."
                )
        cs, ns = safe_mean(caf["sleep_hours"]), safe_mean(nocaf["sleep_hours"])
        if cs is not None and ns is not None and (ns - cs) >= 0.3:
            out.append(
                f"😴 하지만 카페인 섭취 그룹의 평균 수면시간은 **{cs:.1f}시간**으로 "
                f"비섭취 그룹(**{ns:.1f}시간**)보다 짧아, 카페인이 수면을 줄이는 경향이 보입니다."
            )
        cst, nst = safe_mean(caf["stress_score"]), safe_mean(nocaf["stress_score"])
        if cst is not None and nst is not None and (cst - nst) >= 0.3:
            out.append(
                f"⚠️ 카페인 섭취 그룹의 스트레스({cst:.1f})가 비섭취 그룹({nst:.1f})보다 높아, "
                f"집중도 향상 이면에 **스트레스라는 대가**가 함께 나타납니다."
            )

    # 2) 상관관계 중 가장 강한 것
    num = df[NUMERIC_COLS].dropna()
    if len(num) >= 4:
        corr = num.corr()
        pairs = []
        cols = NUMERIC_COLS
        for i in range(len(cols)):
            for j in range(i + 1, len(cols)):
                r = corr.iloc[i, j]
                if pd.notna(r):
                    pairs.append((abs(r), r, cols[i], cols[j]))
        if pairs:
            pairs.sort(reverse=True)
            _, r, a, b = pairs[0]
            rel = "강한 양의" if r > 0 else "강한 음의"
            out.append(
                f"📈 **{KR[a]}**와(과) **{KR[b]}** 사이에는 {rel} 상관관계(r={r:.2f})가 있습니다. "
                + ("한쪽이 오르면 다른 쪽도 오르는 경향입니다."
                   if r > 0 else "한쪽이 오르면 다른 쪽은 낮아지는 경향입니다.")
            )

    # 3) 섭취량-집중/스트레스 동반 상승
    caf_amt = caf.dropna(subset=["focus_score"])
    if len(caf_amt) >= 3:
        caf_amt = caf_amt.copy()
        caf_amt["mg"] = caf_amt["drink_amount"].map(AMOUNT_TO_MG)
        caf_amt = caf_amt.dropna(subset=["mg"])
        if caf_amt["mg"].nunique() >= 2:
            r_focus = caf_amt["mg"].corr(caf_amt["focus_score"])
            r_stress = caf_amt["mg"].corr(pd.to_numeric(caf_amt["stress_score"], errors="coerce"))
            if pd.notna(r_focus) and r_focus > 0.3 and pd.notna(r_stress) and r_stress > 0.3:
                out.append(
                    "🔍 카페인 섭취량이 많아질수록 집중도와 스트레스가 **함께 상승**하는 "
                    "양날의 검 패턴이 관찰됩니다. 적정량 섭취가 중요해 보입니다."
                )

    # 4) 최고 컨디션 음료
    if "wellness" in df.columns:
        cafw = df[(df["caffeine_yn"] == "예") & df["wellness"].notna()]
        if len(cafw) >= 3 and cafw["drink_type"].nunique() >= 2:
            by_drink = cafw.groupby("drink_type")["wellness"].mean()
            if len(by_drink):
                best = by_drink.idxmax()
                out.append(
                    f"🥤 카페인 음료 중 **{best}**를 마신 학생들의 평균 컨디션 점수가 "
                    f"{by_drink.max():.1f}점으로 가장 높았습니다."
                )

    if not out:
        out.append("현재 데이터에서는 뚜렷한 통계적 패턴이 감지되지 않았습니다.")
    return out


# =========================================================
# 5. 공통 차트 헬퍼
# =========================================================
PLOT_TEMPLATE = "plotly_white"
COLOR_CAF = "#7C3AED"
COLOR_NOCAF = "#94A3B8"


def style_fig(fig, height=380):
    fig.update_layout(
        template=PLOT_TEMPLATE, height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="sans-serif", size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


# =========================================================
# 6. 사이드바 (메뉴)
# =========================================================
with st.sidebar:
    st.title("☕ 카페인 분석")
    st.caption("학생 카페인 섭취 & 컨디션 대시보드")
    menu = st.radio(
        "메뉴",
        ["📝 설문 참여", "📊 종합 대시보드", "🔬 심화 분석", "🎮 나의 컨디션 진단", "🔎 데이터 조회"],
        label_visibility="collapsed",
        key="main_menu",
    )
    st.divider()
    sb_ok = get_supabase() is not None
    st.caption("DB 상태: " + ("🟢 Supabase 연결됨" if sb_ok else "🟡 샘플(CSV) 모드"))
    st.caption(f"업데이트: {datetime.now():%H:%M:%S}")


# 데이터 로드 (전 메뉴 공통)
df = add_wellness(load_data())
N = len(df)


def guard_empty():
    if N == 0:
        st.info("표시할 데이터가 없습니다. '설문 등록'에서 데이터를 추가하거나 data.csv를 확인하세요.")
        return True
    return False


# =========================================================
# 7-1. 종합 대시보드
# =========================================================
if menu == "📊 종합 대시보드":
    st.header("📊 종합 대시보드")
    if not guard_empty():
        # --- KPI ---
        caf = df[df["caffeine_yn"] == "예"]
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("총 응답자", f"{N}명")
        rate = len(caf) / N * 100 if N else 0
        c2.metric("카페인 섭취율", f"{rate:.0f}%")
        avg_focus = safe_mean(df["focus_score"])
        c3.metric("평균 집중도", f"{avg_focus:.1f}" if avg_focus is not None else "-")
        avg_sleep = safe_mean(df["sleep_hours"])
        c4.metric("평균 수면", f"{avg_sleep:.1f}h" if avg_sleep is not None else "-")
        avg_well = safe_mean(df["wellness"])
        c5.metric("평균 컨디션", f"{avg_well:.0f}점" if avg_well is not None else "-")

        st.divider()

        # --- 자동 인사이트 ---
        st.subheader("🧠 자동 분석 리포트")
        for line in generate_insights(df):
            st.markdown(f"- {line}")

        st.divider()

        # --- 카페인 O/X 그룹 비교 ---
        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("카페인 섭취 여부별 지표 비교")
            grp = df.groupby("caffeine_yn")[NUMERIC_COLS].mean(numeric_only=True)
            if len(grp):
                melt = grp.reset_index().melt(id_vars="caffeine_yn",
                                              var_name="지표", value_name="평균")
                melt["지표"] = melt["지표"].map(KR)
                fig = px.bar(melt, x="지표", y="평균", color="caffeine_yn",
                             barmode="group", color_discrete_map={"예": COLOR_CAF, "아니오": COLOR_NOCAF},
                             text_auto=".1f")
                fig.update_yaxes(range=[0, max(6, melt["평균"].max() * 1.15)])
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                st.caption("비교할 그룹 데이터가 부족합니다.")

        with col_b:
            st.subheader("수면시간 ↔ 집중도 관계")
            sc = df.dropna(subset=["sleep_hours", "focus_score"])
            if len(sc) >= 2:
                fig = px.scatter(sc, x="sleep_hours", y="focus_score",
                                 color="caffeine_yn", size="wellness",
                                 color_discrete_map={"예": COLOR_CAF, "아니오": COLOR_NOCAF},
                                 hover_data=["student_id", "drink_type"])
                fig.update_yaxes(range=[0, 6])
                st.plotly_chart(style_fig(fig), width="stretch")
            else:
                st.caption("산점도를 그릴 데이터가 부족합니다.")

        # --- 음료 종류별 ---
        st.subheader("음료 종류별 평균 지표")
        drink_df = df[df["drink_type"].isin([d for d in DRINK_TYPES if d != "안마심"])]
        if len(drink_df):
            g = drink_df.groupby("drink_type")[["focus_score", "stress_score", "wellness"]].mean(numeric_only=True)
            g = g.reset_index().melt(id_vars="drink_type", var_name="지표", value_name="평균")
            label_map = {"focus_score": "집중도", "stress_score": "스트레스", "wellness": "컨디션(0-100)"}
            g["지표"] = g["지표"].map(label_map)
            fig = px.bar(g, x="drink_type", y="평균", color="지표", barmode="group", text_auto=".1f")
            fig.update_yaxes(rangemode="tozero")
            st.plotly_chart(style_fig(fig, height=400), width="stretch")
        else:
            st.caption("음료 종류 데이터가 부족합니다.")


# =========================================================
# 7-2. 심화 분석
# =========================================================
elif menu == "🔬 심화 분석":
    st.header("🔬 심화 분석")
    if not guard_empty():
        st.subheader("지표 간 상관관계 히트맵")
        num = df[NUMERIC_COLS].dropna()
        if len(num) >= 3:
            corr = num.corr()
            labels = [KR[c] for c in corr.columns]
            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=labels, y=labels,
                colorscale="RdBu", zmid=0, zmin=-1, zmax=1,
                text=np.round(corr.values, 2), texttemplate="%{text}",
                colorbar=dict(title="상관"),
            ))
            st.plotly_chart(style_fig(fig, height=420), width="stretch")
            st.caption("1에 가까울수록 양의 상관, -1에 가까울수록 음의 상관입니다.")
        else:
            st.caption("상관분석을 위한 유효 데이터가 부족합니다. (3건 이상 필요)")

        st.divider()

        # 섭취량(mg 환산)별 추세
        st.subheader("카페인 섭취량(추정 mg)에 따른 변화")
        caf = df[(df["caffeine_yn"] == "예")].dropna(subset=["caffeine_mg"])
        if len(caf) >= 3:
            metric_sel = st.selectbox(
                "살펴볼 지표", NUMERIC_COLS,
                format_func=lambda c: KR[c],
            )
            sub = caf.dropna(subset=[metric_sel])
            if len(sub) >= 2:
                agg = sub.groupby("drink_amount")[metric_sel].mean().reindex(
                    [a for a in AMOUNT_ORDER if a != "안마심"]).dropna()
                if len(agg):
                    fig = px.line(x=agg.index, y=agg.values, markers=True)
                    fig.update_traces(line_color=COLOR_CAF, line_width=3, marker_size=10)
                    fig.update_xaxes(title="섭취량 구간")
                    fig.update_yaxes(title=KR[metric_sel], rangemode="tozero")
                    st.plotly_chart(style_fig(fig), width="stretch")

                    # 자동 해석
                    vals = agg.values
                    if len(vals) >= 2:
                        trend = "증가" if vals[-1] > vals[0] else ("감소" if vals[-1] < vals[0] else "유지")
                        st.info(f"섭취량이 가장 적은 구간 대비 가장 많은 구간에서 "
                                f"{KR[metric_sel]}이(가) **{trend}**하는 경향을 보입니다 "
                                f"({vals[0]:.1f} → {vals[-1]:.1f}).")
                else:
                    st.caption("구간별 집계 데이터가 부족합니다.")
            else:
                st.caption("선택한 지표의 유효 데이터가 부족합니다.")
        else:
            st.caption("카페인 섭취 데이터가 부족합니다.")

        st.divider()

        # 섭취 시간대별
        st.subheader("섭취 시간대별 컨디션 분포")
        tdf = df[df["drink_time"].isin([t for t in TIME_ORDER if t != "안마심"])]
        tdf = tdf.dropna(subset=["wellness"])
        if len(tdf) >= 2:
            fig = px.box(tdf, x="drink_time", y="wellness", color="drink_time",
                         category_orders={"drink_time": [t for t in TIME_ORDER if t != "안마심"]},
                         points="all")
            fig.update_yaxes(title="컨디션 점수", rangemode="tozero")
            fig.update_layout(showlegend=False)
            st.plotly_chart(style_fig(fig), width="stretch")
        else:
            st.caption("시간대별 데이터가 부족합니다.")


# =========================================================
# 7-3. 나의 컨디션 진단 (게이미피케이션)
# =========================================================
elif menu == "🎮 나의 컨디션 진단":
    st.header("🎮 나의 컨디션 진단")
    if not guard_empty():
        ids = df["student_id"].dropna().unique().tolist()
        if not ids:
            st.info("진단할 학생 ID가 없습니다.")
        else:
            sid = st.selectbox("학생을 선택하세요", ids)
            row = df[df["student_id"] == sid].iloc[0]
            score = row["wellness"]
            g, emoji, color, desc = grade_of(score)

            c1, c2 = st.columns([1, 1.4])
            with c1:
                # 게이지
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    number={"suffix": "점", "font": {"size": 40}},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": color},
                        "steps": [
                            {"range": [0, 35], "color": "#FEE4E2"},
                            {"range": [35, 50], "color": "#FEF0C7"},
                            {"range": [50, 65], "color": "#FEFBE8"},
                            {"range": [65, 80], "color": "#ECFDF3"},
                            {"range": [80, 100], "color": "#D1FADF"},
                        ],
                    },
                ))
                fig.update_layout(height=300, margin=dict(l=20, r=20, t=30, b=10))
                st.plotly_chart(fig, width="stretch")
                st.markdown(f"<h2 style='text-align:center'>{emoji} {g}등급 · {desc}</h2>",
                            unsafe_allow_html=True)

            with c2:
                st.subheader(f"{sid} 상세 지표")
                # 개인 vs 전체 평균 레이더
                metrics = NUMERIC_COLS
                # 정규화(0~1): 좋은 방향이 크도록
                def norm(col, val):
                    if pd.isna(val): return 0.5
                    if col == "focus_score": return (val - 1) / 4
                    if col == "tired_score": return 1 - (val - 1) / 4
                    if col == "stress_score": return 1 - (val - 1) / 4
                    if col == "sleep_hours": return max(0, 1 - abs(val - 7.5) / 4)
                    return 0.5

                personal = [norm(c, row[c]) for c in metrics]
                avg_vals = [norm(c, safe_mean(df[c]) or np.nan) for c in metrics]
                labels = [KR[c] for c in metrics]

                fig = go.Figure()
                fig.add_trace(go.Scatterpolar(r=personal + [personal[0]],
                              theta=labels + [labels[0]], fill="toself",
                              name=sid, line_color=COLOR_CAF))
                fig.add_trace(go.Scatterpolar(r=avg_vals + [avg_vals[0]],
                              theta=labels + [labels[0]], fill="toself",
                              name="전체 평균", line_color=COLOR_NOCAF, opacity=0.6))
                fig.update_layout(polar=dict(radialaxis=dict(range=[0, 1], showticklabels=False)),
                                  height=320, margin=dict(l=30, r=30, t=40, b=20),
                                  legend=dict(orientation="h", y=1.1))
                st.plotly_chart(fig, width="stretch")

            st.divider()
            # 원본 수치 카드
            cols = st.columns(4)
            for i, c in enumerate(NUMERIC_COLS):
                v = row[c]
                cols[i].metric(KR[c], "-" if pd.isna(v) else (f"{v:.1f}h" if c == "sleep_hours" else f"{v:.0f}"))

            # 개인 맞춤 코멘트
            st.subheader("💬 맞춤 코멘트")
            tips = []
            if not pd.isna(row["sleep_hours"]) and row["sleep_hours"] < 6:
                tips.append("수면시간이 부족한 편이에요. 카페인 섭취 시간을 앞당기면 수면에 도움이 될 수 있어요.")
            if not pd.isna(row["stress_score"]) and row["stress_score"] >= 4:
                tips.append("스트레스 지수가 높아요. 고카페인 음료는 스트레스를 키울 수 있으니 양 조절을 권장해요.")
            if not pd.isna(row["focus_score"]) and row["focus_score"] >= 4:
                tips.append("집중도가 우수해요! 다만 수면·스트레스 균형을 함께 챙기면 더 좋아요.")
            if row["caffeine_yn"] == "아니오" and not pd.isna(row["tired_score"]) and row["tired_score"] >= 4:
                tips.append("카페인을 안 마시는데 피로도가 높아요. 수면의 질을 점검해보면 좋아요.")
            if not tips:
                tips.append("전반적으로 균형 잡힌 컨디션이에요. 지금 패턴을 유지해보세요!")
            for t in tips:
                st.markdown(f"- {t}")


# =========================================================
# 7-4. 설문 등록
# =========================================================
elif menu == "📝 설문 참여":
    st.header("📝 카페인 & 컨디션 설문")
    st.markdown(
        "아래 문항에 답하고 **제출**하면 응답이 저장되고, 다른 메뉴에서 전체 분석 결과를 볼 수 있어요. "
        "학생 번호는 자동으로 부여됩니다."
    )

    # 다음 ID 자동 생성 (C0xx 파싱)
    def next_id(df):
        nums = []
        for s in df["student_id"].dropna().astype(str):
            if s.upper().startswith("C") and s[1:].isdigit():
                nums.append(int(s[1:]))
        return f"C{(max(nums) + 1) if nums else 1:03d}"

    new_id = next_id(df)

    c_top1, c_top2 = st.columns([1, 2])
    c_top1.metric("나의 학생 번호", new_id)
    c_top2.info("카페인을 마시지 않는다면 '아니오'를 선택하세요. 음료 관련 문항은 자동으로 '안마심' 처리됩니다.")

    with st.form("survey", clear_on_submit=True):
        st.subheader("① 카페인 섭취")
        caffeine = st.radio("카페인 음료를 마시나요?", ["예", "아니오"], horizontal=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            drink_type = st.selectbox("음료 종류", [d for d in DRINK_TYPES if d != "안마심"])
        with col2:
            drink_amount = st.selectbox("하루 섭취량", [a for a in AMOUNT_ORDER if a != "안마심"])
        with col3:
            drink_time = st.selectbox("주로 마시는 시간", [t for t in TIME_ORDER if t != "안마심"])
        st.caption("↑ 카페인을 마시는 경우에만 응답하세요. '아니오'면 위 3개 문항은 무시됩니다.")

        st.divider()
        st.subheader("② 나의 컨디션")
        cc1, cc2 = st.columns(2)
        with cc1:
            sleep = st.slider("어제 수면시간 (시간)", 0.0, 12.0, 7.0, 0.5)
            tired = st.slider("피로도 (1=쌩쌩 · 5=매우 피곤)", 1, 5, 3)
        with cc2:
            focus = st.slider("수업 집중도 (1=산만 · 5=매우 집중)", 1, 5, 3)
            stress = st.slider("스트레스 (1=편안 · 5=매우 높음)", 1, 5, 3)

        submitted = st.form_submit_button("✅ 설문 제출하기", width="stretch")

    if submitted:
        # 카페인 '아니오'면 음료 관련 필드 정리
        if caffeine == "아니오":
            drink_type = drink_amount = drink_time = "안마심"
        row = {
            "student_id": new_id, "caffeine_yn": caffeine,
            "drink_type": drink_type, "drink_amount": drink_amount,
            "drink_time": drink_time, "sleep_hours": float(sleep),
            "tired_score": int(tired), "focus_score": int(focus),
            "stress_score": int(stress),
        }
        ok, msg = insert_row(row)
        if ok:
            st.success(f"✅ {new_id} 님의 응답이 저장되었어요! 참여해주셔서 감사합니다.")
            st.cache_data.clear()
            st.balloons()
            st.rerun()
        else:
            st.error(f"❌ {msg}")
            st.caption("샘플(CSV) 모드에서는 저장이 지원되지 않습니다. Supabase 연결 후 이용하세요.")


# =========================================================
# 7-5. 데이터 조회
# =========================================================
elif menu == "🔎 데이터 조회":
    st.header("🔎 데이터 조회")
    if not guard_empty():
        c1, c2, c3 = st.columns(3)
        with c1:
            f_caf = st.multiselect("카페인 여부", ["예", "아니오"])
        with c2:
            drink_opts = sorted([d for d in df["drink_type"].dropna().unique()])
            f_drink = st.multiselect("음료 종류", drink_opts)
        with c3:
            time_opts = sorted([t for t in df["drink_time"].dropna().unique()])
            f_time = st.multiselect("섭취 시간대", time_opts)

        view = df.copy()
        if f_caf:
            view = view[view["caffeine_yn"].isin(f_caf)]
        if f_drink:
            view = view[view["drink_type"].isin(f_drink)]
        if f_time:
            view = view[view["drink_time"].isin(f_time)]

        st.caption(f"조회 결과: {len(view)}건 / 전체 {N}건")

        show_cols = ["student_id", "caffeine_yn", "drink_type", "drink_amount",
                     "drink_time", "sleep_hours", "tired_score", "focus_score",
                     "stress_score", "wellness"]
        rename = {**KR, "student_id": "학생ID", "caffeine_yn": "카페인",
                  "drink_type": "음료", "drink_amount": "섭취량",
                  "drink_time": "시간대", "wellness": "컨디션점수"}
        disp = view[show_cols].rename(columns=rename)
        # background_gradient 사용 금지 → 기본 dataframe (matplotlib 의존성 회피)
        st.dataframe(disp, width="stretch", hide_index=True)

        # CSV 다운로드
        csv = disp.to_csv(index=False).encode("utf-8-sig")
        st.download_button("결과 CSV 다운로드", csv,
                           file_name="caffeine_filtered.csv", mime="text/csv")
