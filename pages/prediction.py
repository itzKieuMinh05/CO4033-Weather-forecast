import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import GLOBAL_CSS, load_models, page_header, section, sidebar_header


st.set_page_config(
    page_title="Dự báo - WeatherVN",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, .stApp, .stMarkdown, .stText, .stButton, .stSelectbox, .stNumberInput, .stDataFrame {
    font-family: 'Manrope', sans-serif !important;
}

.mono-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    font-weight: 600;
    color: #64748b;
    letter-spacing: 0.7px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

.panel {
    background: #ffffff;
    border: 1px solid #eef2ff;
    border-radius: 16px;
    padding: 16px;
}

.result-card {
    border-radius: 16px;
    padding: 16px;
    border: 1px solid transparent;
}

.result-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    opacity: 0.8;
}

.result-value {
    font-size: 30px;
    font-weight: 800;
    line-height: 1.1;
    margin: 8px 0 4px;
}

.result-sub {
    font-size: 13px;
    opacity: 0.9;
}

.hint-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 4px solid #3b82f6;
    border-radius: 12px;
    padding: 12px;
    font-size: 13px;
    color: #334155;
    line-height: 1.6;
}

[data-testid="stNumberInput"] label { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


PRESETS = {
    "Tùy chỉnh": dict(
        humidity=80,
        pressure=1008,
        cloudcover=60,
        visibility=8000,
        wsin=0.0,
        wcos=1.0,
        hour=14,
        month=7,
        day=15,
        weekday=2,
        is_day=1,
        temp_lag1=28.0,
        hum_lag1=78,
        pres_lag1=1007,
    ),
    "Ngày mưa": dict(
        humidity=95,
        pressure=1002,
        cloudcover=95,
        visibility=3000,
        wsin=0.5,
        wcos=0.5,
        hour=15,
        month=9,
        day=20,
        weekday=3,
        is_day=1,
        temp_lag1=27.0,
        hum_lag1=92,
        pres_lag1=1003,
    ),
    "Nắng nóng": dict(
        humidity=40,
        pressure=1015,
        cloudcover=10,
        visibility=10000,
        wsin=0.2,
        wcos=0.9,
        hour=13,
        month=5,
        day=10,
        weekday=1,
        is_day=1,
        temp_lag1=38.0,
        hum_lag1=42,
        pres_lag1=1016,
    ),
    "Nắng đẹp": dict(
        humidity=55,
        pressure=1012,
        cloudcover=20,
        visibility=10000,
        wsin=-0.2,
        wcos=0.8,
        hour=10,
        month=3,
        day=5,
        weekday=0,
        is_day=1,
        temp_lag1=26.0,
        hum_lag1=57,
        pres_lag1=1013,
    ),
    "Bão": dict(
        humidity=98,
        pressure=995,
        cloudcover=100,
        visibility=1000,
        wsin=0.8,
        wcos=0.3,
        hour=18,
        month=10,
        day=12,
        weekday=5,
        is_day=0,
        temp_lag1=25.0,
        hum_lag1=96,
        pres_lag1=997,
    ),
}

INPUT_KEY_MAP = {
    "humidity": "n_hum",
    "pressure": "n_pres",
    "cloudcover": "n_cloud",
    "visibility": "n_vis",
    "wsin": "n_wsin",
    "wcos": "n_wcos",
    "hour": "n_hour",
    "month": "n_month",
    "day": "n_day",
    "weekday": "n_wday",
    "is_day": "n_isday",
    "temp_lag1": "n_temp",
    "hum_lag1": "n_hlag",
    "pres_lag1": "n_plag",
}


def apply_preset(preset_name):
    values = PRESETS[preset_name]
    st.session_state["_pv"] = values.copy()
    for field, input_key in INPUT_KEY_MAP.items():
        st.session_state[input_key] = values[field]


def init_state():
    if "_pv" not in st.session_state:
        apply_preset("Tùy chỉnh")
    if "preset_name" not in st.session_state:
        st.session_state["preset_name"] = "Tùy chỉnh"
    for field, input_key in INPUT_KEY_MAP.items():
        if input_key not in st.session_state:
            st.session_state[input_key] = st.session_state["_pv"][field]


def on_preset_change():
    apply_preset(st.session_state["preset_name"])


def sync_derived_inputs():
    # Keep lag features aligned with primary inputs when compact mode is used.
    st.session_state["n_hlag"] = st.session_state["n_hum"]
    st.session_state["n_plag"] = st.session_state["n_pres"]


def build_input_frame():
    return pd.DataFrame(
        [
            {
                "humidity": st.session_state["n_hum"],
                "pressure": st.session_state["n_pres"],
                "cloudcover": st.session_state["n_cloud"],
                "visibility": st.session_state["n_vis"],
                "wind_dir_sin": st.session_state["n_wsin"],
                "wind_dir_cos": st.session_state["n_wcos"],
                "hour": st.session_state["n_hour"],
                "day": st.session_state["n_day"],
                "month": st.session_state["n_month"],
                "weekday": st.session_state["n_wday"],
                "is_day": st.session_state["n_isday"],
                "temp_lag_1": st.session_state["n_temp"],
                "temp_lag1": st.session_state["n_temp"],
                "humidity_lag_1": st.session_state["n_hlag"],
                "humidity_lag1": st.session_state["n_hlag"],
                "hum_lag1": st.session_state["n_hlag"],
                "pressure_lag_1": st.session_state["n_plag"],
                "pressure_lag1": st.session_state["n_plag"],
                "pres_lag1": st.session_state["n_plag"],
            }
        ]
    )


def align_features(model, data):
    if not hasattr(model, "feature_names_in_"):
        return data
    d = data.copy()
    for c in model.feature_names_in_:
        if c not in d.columns:
            d[c] = 0
    return d[list(model.feature_names_in_)]


def find_heatwave_idx(model, label_encoder):
    labels = label_encoder.inverse_transform(model.classes_)
    for i, label in enumerate(labels):
        if "heatwave" in str(label).lower():
            return i
    return None


def find_positive_rain_idx(model):
    for i, c in enumerate(model.classes_):
        s = str(c).lower()
        if c == 1 or s in ["1", "1.0", "true", "rain", "yes"]:
            return i
    return 1 if len(model.classes_) > 1 else 0


def weather_meta(label):
    v = str(label).lower()
    if "heatwave" in v:
        return "🔥", "Nắng nóng", "#c2410c", "#fff7ed", "#fdba74"
    if "storm" in v:
        return "⛈️", "Bão", "#6d28d9", "#faf5ff", "#c4b5fd"
    if "rain" in v:
        return "🌧️", "Mưa", "#1d4ed8", "#eff6ff", "#93c5fd"
    return "☀️", "Bình thường", "#047857", "#ecfdf5", "#86efac"


def predict_all(models):
    rf = models["rf"]
    xgb = models["xgb"]
    rf_rain = models["rf_rain"]
    xgb_rain = models["xgb_rain"]
    le = models["le"]

    X_base = build_input_frame()
    Xrf = align_features(rf, X_base)
    Xxgb = align_features(xgb, X_base)
    Xrf_rain = align_features(rf_rain, X_base)
    Xxgb_rain = align_features(xgb_rain, X_base)

    rf_prob = rf.predict_proba(Xrf)[0]
    xgb_prob = xgb.predict_proba(Xxgb)[0]
    avg_prob = (rf_prob + xgb_prob) / 2.0

    classes = le.inverse_transform(rf.classes_)
    pred_idx = int(np.argmax(avg_prob))
    pred_label = classes[pred_idx]

    heat_idx_rf = find_heatwave_idx(rf, le)
    heat_idx_xgb = find_heatwave_idx(xgb, le)
    heat_rf = float(rf_prob[heat_idx_rf]) if heat_idx_rf is not None else 0.0
    heat_xgb = float(xgb_prob[heat_idx_xgb]) if heat_idx_xgb is not None else 0.0
    heat_prob = (heat_rf + heat_xgb) / 2.0

    rain_rf_prob = rf_rain.predict_proba(Xrf_rain)[0]
    rain_xgb_prob = xgb_rain.predict_proba(Xxgb_rain)[0]
    rain_idx_rf = find_positive_rain_idx(rf_rain)
    rain_idx_xgb = find_positive_rain_idx(xgb_rain)
    rain_rf = float(rain_rf_prob[rain_idx_rf])
    rain_xgb = float(rain_xgb_prob[rain_idx_xgb])
    rain_prob = (rain_rf + rain_xgb) / 2.0

    return {
        "pred_label": pred_label,
        "classes": classes,
        "avg_prob": avg_prob,
        "heat_prob": heat_prob,
        "rain_prob": rain_prob,
        "rain_forecast": "Trời mưa" if rain_prob >= 0.40 else "Không mưa",
        "rain_rf": rain_rf,
        "rain_xgb": rain_xgb,
    }


init_state()


def sidebar_content():
    st.markdown(
        """
        <div style="padding:10px 2px 8px;font-size:11px;font-weight:700;color:#334155;
                    text-transform:uppercase;letter-spacing:.8px;font-family:'JetBrains Mono',monospace;">
            Mẫu thời tiết
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.selectbox(
        "Mẫu",
        list(PRESETS.keys()),
        key="preset_name",
        on_change=on_preset_change,
        label_visibility="collapsed",
    )
    if st.button("Áp dụng mẫu", use_container_width=True):
        apply_preset(st.session_state["preset_name"])
        st.rerun()
    if st.button("Đặt lại", use_container_width=True):
        st.session_state["preset_name"] = "Tùy chỉnh"
        apply_preset("Tùy chỉnh")
        st.rerun()


sidebar_header(sidebar_content)

page_header(
    "🌧️",
    "linear-gradient(135deg,#dbeafe,#e0f2fe)",
    "Dự báo nắng nóng và lượng mưa",
    "Nhập nhanh 4 thông số chính để ước lượng mưa và cảnh báo nắng nóng.",
)

models = load_models()
required = ["rf", "xgb", "rf_rain", "xgb_rain", "le"]
if not all(k in models for k in required):
    st.error(
        "Thiếu file bắt buộc: rf_model.pkl, xgb_model.pkl, rf_rain_model.pkl, xgb_rain_model.pkl, label_encoder.pkl"
    )
    st.stop()

left_col, right_col = st.columns([5, 4], gap="large")

with left_col:
    section("Nhập những thông tin thời tiết để dự báo")

    c1, c2 = st.columns(2)
    c1.markdown('<div class="mono-label">Nhiệt độ</div>', unsafe_allow_html=True)
    c1.number_input("Nhiệt độ (°C)", min_value=10.0, max_value=45.0, step=0.5, key="n_temp")
    c2.markdown('<div class="mono-label">Độ ẩm</div>', unsafe_allow_html=True)
    c2.number_input("Độ ẩm (%)", min_value=0, max_value=100, step=1, key="n_hum")

    c3, c4 = st.columns(2)
    c3.markdown('<div class="mono-label">Áp suất</div>', unsafe_allow_html=True)
    c3.number_input("Áp suất (hPa)", min_value=990, max_value=1025, step=1, key="n_pres")
    c4.markdown('<div class="mono-label">Mây che phủ</div>', unsafe_allow_html=True)
    c4.number_input("Mây che phủ (%)", min_value=0, max_value=100, step=1, key="n_cloud")

    sync_derived_inputs()

    h = st.session_state["n_hum"]
    t = st.session_state["n_temp"]
    st.progress(int(min(max(h, 0), 100)), text=f"Độ ẩm hiện tại: {h}%")
    st.progress(int(min(max(((t - 10) / 35) * 100, 0), 100)), text=f"Mức nhiệt độ: {t:.1f}°C")
    st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("Tinh chỉnh nâng cao (tùy chọn)", expanded=False):
        st.caption("Các thông số bên dưới giữ giá trị preset, chỉ cần sửa khi cần.")
        a1, a2, a3 = st.columns(3)
        a1.number_input("Tầm nhìn (m)", min_value=0, max_value=10000, step=500, key="n_vis")
        a2.number_input("Wind sin", min_value=-1.0, max_value=1.0, step=0.05, key="n_wsin", format="%.2f")
        a3.number_input("Wind cos", min_value=-1.0, max_value=1.0, step=0.05, key="n_wcos", format="%.2f")

        tcols = st.columns(5)
        tcols[0].number_input("Giờ", min_value=0, max_value=23, step=1, key="n_hour")
        tcols[1].number_input("Ngày", min_value=1, max_value=31, step=1, key="n_day")
        tcols[2].number_input("Tháng", min_value=1, max_value=12, step=1, key="n_month")
        tcols[3].number_input("Thứ", min_value=0, max_value=6, step=1, key="n_wday")
        tcols[4].selectbox("Thời điểm", [1, 0], format_func=lambda x: "Ban ngày" if x == 1 else "Ban đêm", key="n_isday")

with right_col:
    section("Kết quả dự báo", "Cập nhật theo thời gian thực")
    try:
        pred = predict_all(models)
        icon, label_txt, color, bg, border = weather_meta(pred["pred_label"])

        st.markdown(
            f"""
            <div class="result-card" style="background:{bg};border-color:{border};">
                <div class="result-title" style="color:{color};">Tổng hợp thời tiết cực đoan</div>
                <div class="result-value" style="color:{color};">{icon} {label_txt}</div>
                <div class="result-sub" style="color:{color};">
                    Nhãn chính: <b>{pred['pred_label']}</b>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        hcol, rcol = st.columns(2)
        heat_pct = pred["heat_prob"] * 100
        rain_pct = pred["rain_prob"] * 100
        heat_level = "Cao" if heat_pct >= 60 else ("Trung bình" if heat_pct >= 35 else "Thấp")
        rain_level = "Trời mưa" if rain_pct >= 40 else "Không mưa"

        with hcol:
            st.markdown(
                f"""
                <div class="result-card" style="background:#fff7ed;border-color:#fdba74;">
                    <div class="result-title" style="color:#c2410c;">Nguy cơ nắng nóng</div>
                    <div class="result-value" style="color:#c2410c;">{heat_pct:.1f}%</div>
                    <div class="result-sub" style="color:#9a3412;">Mức độ: <b>{heat_level}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(min(max(heat_pct, 0), 100)))

        with rcol:
            st.markdown(
                f"""
                <div class="result-card" style="background:#eff6ff;border-color:#93c5fd;">
                    <div class="result-title" style="color:#1d4ed8;">Dự báo lượng mưa</div>
                    <div class="result-value" style="color:#1d4ed8;">{rain_pct:.1f}%</div>
                    <div class="result-sub" style="color:#1e3a8a;">Kết luận: <b>{rain_level}</b></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(int(min(max(rain_pct, 0), 100)))

        st.markdown('<div style="height:8px"></div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class="hint-box">
                Dự báo lượng mưa được dẫn bởi nhiệt độ và độ ẩm bạn vừa nhập.
                Giá trị hiện tại: <b>Nhiệt độ = {st.session_state['n_temp']:.1f}°C</b>,
                <b>Độ ẩm = {st.session_state['n_hum']}%</b>.
                Kết quả ensemble dùng RF ({pred['rain_rf']*100:.1f}%) và
                XGBoost ({pred['rain_xgb']*100:.1f}%).
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div style="height:10px"></div>', unsafe_allow_html=True)
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.markdown('<div class="mono-label">Xác suất các lớp cực đoan (trung bình RF + XGB)</div>', unsafe_allow_html=True)
        for cls, p in sorted(zip(pred["classes"], pred["avg_prob"]), key=lambda x: float(x[1]), reverse=True):
            st.progress(int(min(max(p * 100, 0), 100)), text=f"{cls}: {p*100:.1f}%")
        st.markdown('</div>', unsafe_allow_html=True)

    except Exception as e:
        import traceback

        st.error(f"Lỗi dự báo: {e}")
        with st.expander("Chi tiết lỗi"):
            st.code(traceback.format_exc())
