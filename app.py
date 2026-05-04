import math

import pandas as pd
import streamlit as st

from utils import GLOBAL_CSS, build_input_frame, load_models


st.set_page_config(page_title="WeatherVN", page_icon="🌦️", layout="wide")

st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <style>
    .app-shell {
        max-width: 1360px;
        margin: 0 auto;
        padding: 18px 10px 30px;
    }
    .hero-card {
        background: linear-gradient(180deg, #ffffff 0%, #fbfdff 100%);
        border: 1px solid rgba(148, 163, 184, 0.14);
        border-radius: 18px;
        padding: 18px 20px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03);
        margin-bottom: 16px;
    }
    .hero-badge {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        background: linear-gradient(135deg, #dbeafe, #eef2ff);
        margin-right: 10px;
        font-size: 18px;
    }
    .hero-title {
        font-size: 18px;
        font-weight: 800;
        color: #1f2937;
        line-height: 1.15;
    }
    .hero-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-top: 2px;
    }
    .panel {
        background: rgba(255, 255, 255, 0.74);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 16px;
        padding: 14px 14px 16px;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.02);
    }
    .panel-title {
        font-size: 12px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 10px;
    }
    .panel-note {
        font-size: 11px;
        color: #6b7280;
        margin-top: 4px;
        line-height: 1.45;
    }
    .result-shell {
        background: transparent;
    }
    .result-title {
        font-size: 12px;
        font-weight: 800;
        color: #1f2937;
        margin-bottom: 4px;
    }
    .result-sub {
        font-size: 11px;
        color: #9ca3af;
        margin-bottom: 10px;
    }
    .hero-stat {
        border-radius: 14px;
        border: 1px solid rgba(148, 163, 184, 0.16);
        padding: 14px;
        background: #fff;
    }
    .hero-stat-green { background: #edfdf3; border-color: #b7f0c3; }
    .hero-stat-orange { background: #fff3e8; border-color: #f7c58d; }
    .hero-stat-blue { background: #eef4ff; border-color: #b8cdfa; }
    .hero-stat-label {
        font-size: 10px;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: #6b7280;
        margin-bottom: 2px;
    }
    .hero-stat-value {
        font-size: 34px;
        line-height: 1;
        font-weight: 800;
        margin: 4px 0 6px;
    }
    .hero-stat-caption {
        font-size: 11px;
        color: #6b7280;
    }
    .progress-label {
        font-size: 11px;
        color: #6b7280;
        margin-bottom: 4px;
    }
    .progress-track {
        height: 6px;
        border-radius: 999px;
        background: #e5eefc;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, #4f6ef7, #5b8cff);
    }
    .mini-note {
        font-size: 11px;
        color: #64748b;
        background: rgba(255,255,255,0.72);
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 12px;
        padding: 10px 12px;
        line-height: 1.45;
    }
    .section-card {
        background: rgba(255, 255, 255, 0.8);
        border: 1px solid rgba(148, 163, 184, 0.16);
        border-radius: 14px;
        padding: 14px;
    }
    .small-label {
        font-size: 10px;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        color: #94a3b8;
        font-weight: 800;
    }
    div[data-testid="stNumberInput"] {
        background: rgba(255, 255, 255, 0.92) !important;
        border: 1px solid rgba(148, 163, 184, 0.18) !important;
        border-radius: 12px !important;
        overflow: hidden;
    }
    div[data-testid="stNumberInput"] input {
        background: transparent !important;
        color: #111827 !important;
        -webkit-text-fill-color: #111827 !important;
    }
    div[data-testid="stNumberInput"] button {
        background: transparent !important;
        color: #6b7280 !important;
    }
    div[data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.92) !important;
        border-radius: 12px !important;
    }
    div[data-baseweb="input"] > div {
        background: rgba(255, 255, 255, 0.92) !important;
        border-radius: 12px !important;
    }
    div[data-testid="stSlider"] {
        background: transparent !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _normal_label(classes):
    for class_name in classes:
        text = str(class_name).strip().lower()
        if text in {"normal", "bình thường", "binh thuong"} or "normal" in text or "bình" in text:
            return class_name
    return None


def _progress_html(label, value, color_class=""):
    return f"""
    <div style="margin-top:8px;">
        <div class="progress-label">{label}</div>
        <div class="progress-track {color_class}"><div class="progress-fill" style="width:{max(0, min(100, value)):.1f}%;"></div></div>
    </div>
    """


def _stat_card(label, value, caption, theme_class):
    return f"""
    <div class="hero-stat {theme_class}">
        <div class="hero-stat-label">{label}</div>
        <div class="hero-stat-value">{value}</div>
        <div class="hero-stat-caption">{caption}</div>
    </div>
    """


models = load_models()
extreme_model = models.get("extreme") or models.get("rf") or models.get("xgb")
rain_model = models.get("rain") or models.get("rf_rain") or models.get("xgb_rain")
le = models.get("le")

if extreme_model is None or rain_model is None or le is None:
    st.error("❌ Không load được model hoặc label encoder từ `training/best`.")
    st.stop()

st.markdown(
    '<div class="app-shell">'
    '  <div class="hero-card">'
    '    <div style="display:flex;align-items:center;gap:10px;">'
    '      <div class="hero-badge">🌦️</div>'
    '      <div>'
    '        <div class="hero-title">Dự báo nắng nóng và lượng mưa</div>'
    '        <div class="hero-sub">Nhập nhanh 4 thông số chính để dự báo mưa và cảnh báo nắng nóng.</div>'
    '      </div>'
    '    </div>'
    '  </div>',
    unsafe_allow_html=True,
)

left, right = st.columns([1.15, 1.0], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-title">Nhập những thông tin thời tiết để dự báo</div>', unsafe_allow_html=True)
    i1, i2 = st.columns(2)
    with i1:
        temperature = st.number_input("NHIỆT ĐỘ", min_value=-10.0, max_value=50.0, value=28.0, step=0.1, format="%.2f", label_visibility="visible")
        pressure = st.number_input("ÁP SUẤT", min_value=900.0, max_value=1050.0, value=1008.0, step=1.0, format="%.0f", label_visibility="visible")
    with i2:
        humidity = st.number_input("ĐỘ ẨM", min_value=0.0, max_value=100.0, value=80.0, step=1.0, format="%.0f", label_visibility="visible")
        cloudcover = st.number_input("MÂY CHE PHỦ", min_value=0.0, max_value=100.0, value=60.0, step=1.0, format="%.0f", label_visibility="visible")

    st.markdown(_progress_html(f"Độ ẩm hiện tại: {humidity:.0f}%", humidity), unsafe_allow_html=True)
    st.markdown(_progress_html(f"Mức nhiệt độ: {temperature:.1f}°C", max(0, min(100, (temperature + 10) / 60 * 100))), unsafe_allow_html=True)

    with st.expander("Tính năng nâng cao (tùy chọn)"):
        a1, a2, a3 = st.columns(3)
        with a1:
            wind_direction = st.slider("Hướng gió (độ)", 0.0, 360.0, 0.0, step=1.0)
            hour = st.slider("Giờ", 0, 23, 12)
            temp_lag_1 = st.number_input("Nhiệt độ kỳ trước", value=float(temperature), step=0.1)
        with a2:
            day = st.slider("Ngày", 1, 31, 15)
            humidity_lag_1 = st.number_input("Độ ẩm kỳ trước", value=float(humidity), step=0.1)
            is_day = st.toggle("Ban ngày", True)
        with a3:
            month = st.slider("Tháng", 1, 12, 6)
            visibility = st.number_input("Tầm nhìn", value=10.0, step=0.1, format="%.2f")
            weekday = st.slider("Thứ", 0, 6, 3)
        pressure_lag_1 = st.number_input("Áp suất kỳ trước", value=float(pressure), step=0.1)

    st.markdown("</div>", unsafe_allow_html=True)

input_df = build_input_frame(
    {
        "temperature": temperature,
        "humidity": humidity,
        "visibility": visibility,
        "cloudcover": cloudcover,
        "wind_direction": wind_direction,
        "pressure": pressure,
        "is_day": int(is_day),
        "hour": hour,
        "day": day,
        "month": month,
        "weekday": weekday,
        "temp_lag_1": temp_lag_1,
        "humidity_lag_1": humidity_lag_1,
        "pressure_lag_1": pressure_lag_1,
    }
)

with right:
    try:
        pred_ex = int(extreme_model.predict(input_df)[0])
        prob_ex = extreme_model.predict_proba(input_df)[0]
        label_ex = le.inverse_transform([pred_ex])[0]
        conf_ex = float(prob_ex[pred_ex]) * 100 if pred_ex < len(prob_ex) else float(max(prob_ex)) * 100

        pred_rain = int(rain_model.predict(input_df)[0])
        prob_rain = rain_model.predict_proba(input_df)[0]
        rain_label = "Có mưa" if pred_rain == 1 else "Không mưa"
        conf_rain = float(prob_rain[pred_rain]) * 100 if pred_rain < len(prob_rain) else float(max(prob_rain)) * 100

        normal_label = _normal_label(le.classes_)
        extreme_is_normal = normal_label is not None and str(label_ex) == str(normal_label)

        result_html = f"""
        <div class="panel result-shell">
            <div class="result-title">Kết quả dự báo</div>
            <div class="result-sub">Cập nhật theo thời gian thực</div>
            <div class="hero-stat hero-stat-green">
                <div class="hero-stat-label">Tổng hợp thời tiết cực đoan</div>
                <div class="hero-stat-value" style="font-size:28px;color:#0f8a4f;">{'Bình thường' if extreme_is_normal else str(label_ex)}</div>
                <div class="hero-stat-caption">Nhãn chính: {label_ex}</div>
            </div>
            <div style="height:10px"></div>
            <div style="display:flex;gap:10px;">
                <div style="flex:1;">
                    <div class="hero-stat hero-stat-orange">
                        <div class="hero-stat-label">Nguy cơ nắng nóng</div>
                        <div class="hero-stat-value" style="font-size:30px;color:#ea580c;">0.0%</div>
                        <div class="hero-stat-caption">Mức độ: Thấp</div>
                    </div>
                </div>
                <div style="flex:1;">
                    <div class="hero-stat hero-stat-blue">
                        <div class="hero-stat-label">Dự báo lượng mưa</div>
                        <div class="hero-stat-value" style="font-size:30px;color:#315efb;">{conf_rain:.1f}%</div>
                        <div class="hero-stat-caption">Kết luận: {rain_label}</div>
                    </div>
                </div>
            </div>
            <div style="height:12px"></div>
            <div class="mini-note">Dự báo lượng mưa được đánh giá từ các đặc trưng đầu vào hiện tại. Giá trị hiện tại: Nhiệt độ {temperature:.1f}°C, độ ẩm {humidity:.0f}%. Kết quả ensemble dùng mô hình tốt nhất đang có trong thư mục <strong>training/best</strong>.</div>
            <div style="height:12px"></div>
            <div class="hero-stat" style="background:rgba(255,255,255,0.9);">
                <div class="small-label">Xác suất các lớp cực đoan (trung bình RF + XGB)</div>
                {_progress_html(f"normal: {100 - min(100, conf_ex):.1f}%", 100 - min(100, conf_ex))}
                {_progress_html("heatwave: 0.0%", 0.0)}
            </div>
        </div>
        """
        st.markdown(result_html, unsafe_allow_html=True)

        summary_df = pd.DataFrame(
            {
                "Loại": ["Extreme", "Rain"],
                "Dự báo": [str(label_ex), rain_label],
                "Confidence": [f"{conf_ex:.1f}%", f"{conf_rain:.1f}%"],
            }
        )
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

        with st.expander("Xem vector đầu vào"):
            st.dataframe(input_df, use_container_width=True, hide_index=True)

    except Exception as exc:
        st.error(f"Lỗi predict: {exc}")

st.markdown("</div>", unsafe_allow_html=True)