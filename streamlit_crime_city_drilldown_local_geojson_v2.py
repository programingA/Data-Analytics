import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import numpy as np
import json
from pathlib import Path

# -------------------------
# 기본 설정 & 약간의 CSS 커스텀
# -------------------------
st.set_page_config(
    page_title="연도별 범죄 통계 대시보드",
    page_icon="📊",
    layout="wide",
)

# 밝은 톤 배경 + 카드 느낌 살짝 주기
st.markdown(
    """
    <style>
    body {
        background-color: #f4f7fb;
    }
    .main {
        background-color: #f4f7fb;
    }
    /* 제목 폰트 살짝 강조 */
    h1, h2, h3 {
        font-family: "Pretendard", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    /* metric 카드 조금 더 카드 느낌 */
    div[data-testid="metric-container"] {
        background-color: #ffffff;
        border-radius: 14px;
        padding: 12px 16px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 연도별 범죄 통계 대시보드")
st.caption(
    "2018~2023년 범죄 데이터를 한눈에!  \n"
    "연도별 추이, 지역별 랭킹, 지도 시각화까지 한 번에 확인해보세요."
)

# --------------------------------------------------
# 사이드바: 연도 / 파일 / GeoJSON / 기본 설정
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ 기본 설정")

    year = st.selectbox(
        "연도 선택",
        [2023, 2022, 2021, 2020, 2019, 2018],
        index=0,
    )

    # 연도별 정규화된 CSV (2023 기준 long 포맷)
    FILE_MAP = {
        2018: "data/crime_2018_aligned.csv",
        2019: "data/crime_2019_aligned.csv",
        2020: "data/crime_2020_aligned.csv",
        2021: "data/crime_2021_aligned.csv",
        2022: "data/crime_2022_aligned.csv",
        2023: "data/crime_2023_aligned.csv",
    }

    use_embedded = st.toggle(
        f"{year}년: 로컬 CSV 사용",
        value=True,
        help=f"체크 시 {FILE_MAP.get(year, '')} 를 읽어서 사용합니다.",
    )

    geojson_path = st.text_input(
        "시군구 GeoJSON 경로",
        value="geo/sig.json",  # 예: ./geo/sig.json
        help="SIG_CD / SIG_KOR_NM 속성이 있는 전국 시군구 GeoJSON",
    )

    top_n = st.number_input(
        "상위 N 지역 (막대그래프)",
        min_value=3,
        max_value=50,
        value=10,
        step=1,
    )


# --------------------------------------------------
# CSV 로딩 유틸
# --------------------------------------------------
def read_csv_safely(path_or_buffer):
    encodings_to_try = ["utf-8-sig", "cp949", "euc-kr", "utf-8", "latin1"]
    last_error = None
    for enc in encodings_to_try:
        try:
            df = pd.read_csv(path_or_buffer, encoding=enc)
            df.columns = [c.replace("\ufeff", "").strip() for c in df.columns]
            return df
        except Exception as e:
            last_error = e
    st.error(f"CSV 읽기에 실패했습니다: {last_error}")
    return None


# --------------------------------------------------
# 현재 선택된 연도 CSV 로딩
# --------------------------------------------------
if use_embedded:
    app_dir = Path(__file__).resolve().parent
    csv_rel = FILE_MAP.get(year)
    if csv_rel is None:
        st.error(f"{year}년 파일명이 정의되어 있지 않습니다.")
        st.stop()
    csv_path = app_dir / csv_rel
    if not csv_path.exists():
        st.error(f"{year}년 CSV 파일을 찾을 수 없습니다: {csv_path}")
        st.stop()
    df_raw = read_csv_safely(csv_path)
else:
    uploaded = st.file_uploader(f"{year}년 CSV 업로드 (aligned 포맷)", type=["csv"])
    if uploaded is None:
        st.info("직접 업로드하거나, '로컬 CSV 사용'을 켜 주세요.")
        st.stop()
    df_raw = read_csv_safely(uploaded)

if df_raw is None:
    st.stop()

st.markdown(f"### 📂 현재 분석 연도: **{year}년**")
st.dataframe(df_raw, use_container_width=True)

# --------------------------------------------------
# 데이터 전처리
# --------------------------------------------------
expected_cols = ["범죄대분류", "범죄중분류", "시도", "세부지역", "지역원본", "발생건수"]
missing = [c for c in expected_cols if c not in df_raw.columns]
if missing:
    st.error(f"다음 컬럼이 없습니다: {missing}\n\naligned 포맷을 확인해 주세요.")
    st.stop()

df = df_raw.copy()

# 숫자형 변환
if df["발생건수"].dtype == "object":
    df["발생건수"] = (
        df["발생건수"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .pipe(pd.to_numeric, errors="coerce")
    )

df["발생건수"] = df["발생건수"].fillna(0)

# --------------------------------------------------
# ⭐ 연도별 총합 꺾은선 그래프용 데이터 (embedded일 때)
# --------------------------------------------------
year_line_df = None
if use_embedded:
    totals = []
    for y, path_rel in FILE_MAP.items():
        path = Path(__file__).resolve().parent / path_rel
        if not path.exists():
            continue
        tmp = read_csv_safely(path)
        if tmp is None or "발생건수" not in tmp.columns:
            continue
        s = tmp["발생건수"]
        if s.dtype == "object":
            s = (
                s.astype(str)
                .str.replace(",", "", regex=False)
                .pipe(pd.to_numeric, errors="coerce")
            )
        total_val = float(s.fillna(0).sum())
        totals.append({"연도": y, "발생건수": total_val})

    if totals:
        year_line_df = pd.DataFrame(totals).sort_values("연도")

# --------------------------------------------------
# 사이드바: 필터 (시도 / 분류)
# --------------------------------------------------
with st.sidebar:
    st.header("📌 필터")

    sido_list = sorted(df["시도"].dropna().astype(str).unique().tolist())
    selected_sido = st.multiselect(
        "광역시·도 선택",
        sido_list,
        default=sido_list,
    )

    major_list = sorted(df["범죄대분류"].dropna().astype(str).unique().tolist())
    sel_major = st.selectbox(
        "범죄대분류",
        options=["전체"] + major_list,
        index=0,
    )

    if sel_major == "전체":
        minor_pool = df
    else:
        minor_pool = df[df["범죄대분류"].astype(str) == sel_major]

    minor_list = sorted(minor_pool["범죄중분류"].dropna().astype(str).unique().tolist())
    sel_minor = st.selectbox(
        "범죄중분류",
        options=["전체"] + minor_list,
        index=0,
    )

# 필터 적용
flt = df.copy()
if selected_sido:
    flt = flt[flt["시도"].astype(str).isin(selected_sido)]

if sel_major != "전체":
    flt = flt[flt["범죄대분류"].astype(str) == sel_major]

if sel_minor != "전체":
    flt = flt[flt["범죄중분류"].astype(str) == sel_minor]

if flt.empty:
    st.warning("선택된 조건에 해당하는 데이터가 없습니다.")
    st.stop()

# 표시용 지역 이름
def make_region_label(row):
    sido = str(row["시도"])
    detail = str(row["세부지역"]) if pd.notna(row["세부지역"]) else ""
    if detail and detail.lower() != "nan":
        return f"{sido} {detail}"
    return sido

flt["표시지역"] = flt.apply(make_region_label, axis=1)

# --------------------------------------------------
# 상단: KPI + (신규) 연도별 총합 꺾은선 그래프
# --------------------------------------------------
st.markdown("### ✨ 요약 지표 (KPI) & 연도별 추이")

k1, k2, k3 = st.columns(3)
k1.metric("레코드 수", f"{len(flt):,}")
k2.metric("총 발생건수", f"{int(flt['발생건수'].sum()):,}")
k3.metric("고유 지역 수", f"{flt['표시지역'].nunique():,}")

if year_line_df is not None and not year_line_df.empty:
    line_chart = (
        alt.Chart(year_line_df)
        .mark_line(point=True)
        .encode(
            x=alt.X("연도:O", title="연도"),
            y=alt.Y("발생건수:Q", title="발생건수 총합", axis=alt.Axis(format=",")),
            tooltip=[
                alt.Tooltip("연도:O", title="연도"),
                alt.Tooltip("발생건수:Q", title="총합", format=","),
            ],
            color=alt.value("#4f46e5"),
        )
        .properties(
            height=260,
            title="연도별 범죄 발생 총합 추이 (2018~2023)",
        )
    )
    st.altair_chart(line_chart, use_container_width=True)
else:
    st.info("연도별 꺾은선 그래프는 로컬 aligned CSV가 모두 있을 때 표시됩니다.")

st.markdown("---")

# --------------------------------------------------
# 시도별 집계 (버블맵용)
# --------------------------------------------------
sido_sum = (
    flt.groupby("시도", dropna=False)["발생건수"]
    .sum()
    .reset_index()
)

sido_centroids = {
    "서울": (37.5665, 126.9780),
    "부산": (35.1796, 129.0756),
    "대구": (35.8714, 128.6014),
    "인천": (37.4563, 126.7052),
    "광주": (35.1595, 126.8526),
    "대전": (36.3504, 127.3845),
    "울산": (35.5384, 129.3114),
    "세종": (36.4800, 127.2890),
    "경기도": (37.4363, 127.5500),
    "강원도": (37.8228, 128.1555),
    "충북": (36.6357, 127.4914),
    "충남": (36.5184, 126.8000),
    "전북": (35.7175, 127.1530),
    "전남": (34.8679, 126.9910),
    "경북": (36.4919, 128.8889),
    "경남": (35.4606, 128.2132),
    "제주": (33.4996, 126.5312),
}

plot_df = []
for _, row in sido_sum.iterrows():
    name = str(row["시도"])
    val = float(row["발생건수"])
    if name in sido_centroids:
        lat, lon = sido_centroids[name]
        plot_df.append({"시도": name, "lat": lat, "lon": lon, "발생건수": val})

plot_df = pd.DataFrame(plot_df)

left, right = st.columns([0.54, 0.46])

with left:
    st.subheader("🏆 지역별 범죄 총합 랭킹")

    ranked = (
        flt.groupby("표시지역", dropna=False)["발생건수"]
        .sum()
        .reset_index()
        .sort_values("발생건수", ascending=False)
    )

    chart = (
        alt.Chart(ranked.head(int(top_n)))
        .mark_bar(cornerRadiusTopLeft=6, cornerRadiusTopRight=6)
        .encode(
            x=alt.X("발생건수:Q", title="발생건수(합계)", axis=alt.Axis(format=",")),
            y=alt.Y("표시지역:N", sort="-x", title="지역"),
            tooltip=[
                alt.Tooltip("표시지역:N", title="지역"),
                alt.Tooltip("발생건수:Q", format=",", title="발생건수"),
            ],
            color=alt.value("#6366f1"),
        )
        .properties(height=420)
    )

    st.altair_chart(chart, use_container_width=True)

with right:
    st.subheader("🗺️ 시도별 버블맵")

    if plot_df.empty:
        st.info("표시할 시도 데이터가 없습니다.")
    else:
        vals = plot_df["발생건수"].to_numpy()
        vmin, vmax = float(vals.min()), float(vals.max())
        if vmax == vmin:
            vmax = vmin + 1.0

        def radius_scale(v):
            t = (v - vmin) / (vmax - vmin)
            t = float(np.clip(t, 0, 1))
            return 10000 * (0.4 + 1.6 * np.sqrt(t))

        def color_scale(v):
            t = (v - vmin) / (vmax - vmin)
            t = float(np.clip(t, 0, 1))
            # 보라 ~ 파랑 계열
            return [99 + int(40 * t), 102 + int(80 * t), 241, 180]

        plot_df["radius"] = plot_df["발생건수"].apply(radius_scale)
        plot_df["color"] = plot_df["발생건수"].apply(color_scale)

        view_state = pdk.ViewState(
            latitude=36.5,
            longitude=127.8,
            zoom=5.5,
        )

        layer = pdk.Layer(
            "ScatterplotLayer",
            data=plot_df,
            get_position="[lon, lat]",
            get_radius="radius",
            get_fill_color="color",
            pickable=True,
            auto_highlight=True,
        )

        tooltip = {"html": "<b>{시도}</b><br/>발생건수: {발생건수}"}

        st.pydeck_chart(
            pdk.Deck(
                layers=[layer],
                initial_view_state=view_state,
                tooltip=tooltip,
                map_style="light",
            ),
            use_container_width=True,
        )

st.markdown("---")

# --------------------------------------------------
# 선택한 시도의 구/시/군 지도 (세부지역 있을 때만)
# --------------------------------------------------
st.subheader("🧭 선택한 시·도의 시/군/구 지도 (세부지역 데이터가 있을 때만)")

district_sum = (
    flt.groupby(["시도", "세부지역"], dropna=False)["발생건수"]
    .sum()
    .reset_index()
)

has_detail_level = False
if not district_sum.empty:
    # 세부지역이 하나라도 있으면 지도 시도
    has_detail_level = district_sum["세부지역"].notna().any()

if not has_detail_level:
    st.info(
        "이 연도 / 필터 조합에서는 시/군/구 단위 세부지역 데이터가 없어\n"
        "상단 시도 버블맵만 표시합니다."
    )
else:
        # 아래 기존 GeoJSON 처리 로직 실행
    def normalize_name(name: str) -> str:
        n = str(name).strip()
        if not n or n.lower() == "nan":
            return ""

        # 0) 세종특별자치시 / 제주특별자치도 같은 것 먼저 자르기
        #    예: "세종특별자치시" → "세종"
        #        "제주특별자치도" → "제주"
        for token in ["특별자치시", "특별자치도"]:
            if token in n:
                n = n.split(token)[0]
                break

        # 1) "천안시 서북구" 같이 공백이 있으면 첫 단어만 사용
        #    예: "천안시 서북구" → "천안시"
        if " " in n:
            n = n.split()[0]

        # 2) 공백이 없는데 "천안시서북구"처럼 붙어있으면
        #    "시" 앞까지 자르기 → "천안시"
        if " " not in n and "시" in n and "구" in n:
            if n.index("시") < n.index("구"):
                cut = n.index("시")
                n = n[:cut + 1]

        # 3) 맨 끝 접미사 한 번만 제거
        #    예: "천안시" → "천안", "세종시" → "세종"
        for suffix in ["특별시", "광역시", "도", "시", "군", "구"]:
            if n.endswith(suffix):
                n = n[: -len(suffix)]
                break

        return n



    gj_path = Path(geojson_path).expanduser().resolve()
    if not gj_path.exists():
        st.info(f"GeoJSON 파일을 찾을 수 없습니다: {gj_path}")
    else:
        try:
            with open(gj_path, "r", encoding="utf-8") as f:
                gj = json.load(f)

            features = gj.get("features", [])
            if not features:
                st.warning("GeoJSON에 feature가 없습니다.")
            else:
                sample_props = features[0]["properties"]
                prop_keys = list(sample_props.keys())

                default_idx = prop_keys.index("SIG_KOR_NM") if "SIG_KOR_NM" in prop_keys else 0
                col_district = st.selectbox(
                    "GeoJSON 시군구명 프로퍼티",
                    options=prop_keys,
                    index=default_idx,
                )

                code2sido = {
                    "11": "서울",
                    "26": "부산",
                    "27": "대구",
                    "28": "인천",
                    "29": "광주",
                    "30": "대전",
                    "31": "울산",
                    "36": "세종",
                    "41": "경기도",
                    "42": "강원도",
                    "43": "충북",
                    "44": "충남",
                    "45": "전북",
                    "46": "전남",
                    "47": "경북",
                    "48": "경남",
                    "50": "제주",
                }

                # 1) 원본 맵 (그냥 참고용)
                val_map_raw = {
                    (str(r["시도"]), str(r["세부지역"]).strip()): float(r["발생건수"])
                    for _, r in district_sum.iterrows()
                }

                # 2) 정규화된 이름 기준 맵
                val_map_norm = {}
                for (sido_name, detail_name), v in val_map_raw.items():
                    norm_key = (sido_name, normalize_name(detail_name))
                    # 같은 normalize 키로 여러 레코드가 뭉칠 수 있으니 합산
                    val_map_norm[norm_key] = val_map_norm.get(norm_key, 0.0) + v

                # 🔧 여기서 val_map → val_map_norm 으로 수정
                vals = np.array(list(val_map_norm.values())) if len(val_map_norm) > 0 else np.array([0])
                vmin, vmax = float(vals.min()), float(vals.max())
                if vmax == vmin:
                    vmax = vmin + 1.0

                def to_color(v):
                    t = (v - vmin) / (vmax - vmin)
                    t = float(np.clip(t, 0, 1))
                    # 블루/퍼플 계열
                    return [80, 120 + int(80 * t), 220 + int(20 * t), 210]

                features_colored = []
                for f in features:
                    props = f["properties"]

                    sig_cd = str(props.get("SIG_CD", ""))[:2]
                    sido = code2sido.get(sig_cd)

                    if (sido is None) or (sido not in selected_sido):
                        continue

                    dname = str(props.get(col_district, "")).strip()

                    # 정규화해서 매칭
                    norm_key = (sido, normalize_name(dname))
                    val = float(val_map_norm.get(norm_key, 0.0))

                    props["__value__"] = val
                    props["__label__"] = f"{sido} {dname}" if sido else dname

                    if val <= 0:
                        props["__fill__"] = [0, 0, 0, 0]
                    else:
                        props["__fill__"] = to_color(val)

                    features_colored.append(f)

                if not features_colored:
                    st.warning(
                        "선택한 시·도와 매칭되는 구/시/군 데이터가 거의 없습니다.\n"
                        "세부지역 이름(예: 강남구)과 GeoJSON의 시군구명이 일치하는지 확인해 주세요."
                    )
                else:
                    if len(selected_sido) == 1:
                        only = selected_sido[0]
                        center_map = {
                            "서울": (37.5665, 126.9780, 9.2),
                            "부산": (35.1796, 129.0756, 9.0),
                            "대구": (35.8714, 128.6014, 9.0),
                            "인천": (37.4563, 126.7052, 9.0),
                            "광주": (35.1595, 126.8526, 9.0),
                            "대전": (36.3504, 127.3845, 9.2),
                            "울산": (35.5384, 129.3114, 9.2),
                            "세종": (36.4800, 127.2890, 9.5),
                            "경기도": (37.4363, 127.5500, 8.2),
                            "강원도": (37.8228, 128.1555, 7.8),
                            "충북": (36.6357, 127.4914, 8.4),
                            "충남": (36.5184, 126.8000, 8.0),
                            "전북": (35.7175, 127.1530, 8.2),
                            "전남": (34.8679, 126.9910, 7.8),
                            "경북": (36.4919, 128.8889, 7.8),
                            "경남": (35.4606, 128.2132, 8.0),
                            "제주": (33.4996, 126.5312, 9.0),
                        }
                        lat, lon, zm = center_map.get(only, (36.5, 127.8, 6.0))
                    else:
                        lat, lon, zm = 36.5, 127.8, 6.3

                    view_state = pdk.ViewState(
                        latitude=lat,
                        longitude=lon,
                        zoom=zm,
                    )

                    layer = pdk.Layer(
                        "GeoJsonLayer",
                        {"type": "FeatureCollection", "features": features_colored},
                        stroked=True,
                        filled=True,
                        get_fill_color="properties.__fill__",
                        get_line_color=[255, 255, 255, 140],
                        line_width_min_pixels=1.0,
                        pickable=True,
                        auto_highlight=True,
                    )

                    tooltip = {
                        "html": "<b>{__label__}</b><br/>발생건수: {__value__}",
                    }

                    st.pydeck_chart(
                        pdk.Deck(
                            layers=[layer],
                            initial_view_state=view_state,
                            tooltip=tooltip,
                            map_style="light",
                        ),
                        use_container_width=True,
                    )

                    st.markdown("#### 📋 시도-세부지역별 발생건수 (집계)")
                    st.dataframe(
                        district_sum.sort_values("발생건수", ascending=False),
                        use_container_width=True,
                    )

        except Exception as e:
            st.error(f"GeoJSON 처리 중 오류: {e}")
            st.info("SIG_CD / SIG_KOR_NM 속성 및 파일 경로를 다시 확인해 주세요.")

st.caption(
    "💡 TIP: 상단 꺾은선 그래프에서 연도별 추이를 보고, 아래에서 특정 연도를 골라 지역/범죄 유형별로 파고들어 보세요."
)
