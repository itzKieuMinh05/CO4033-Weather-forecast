import ast
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import GLOBAL_CSS, fig_ax, kpi_card, page_header, section, sidebar_header


st.set_page_config(
    page_title="FPGrowth Insights - WeatherVN",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


REGION_ORDER = ["north", "central", "south"]
TARGET_ORDER = ["rain_yes", "heatwave", "storm", "heavy_rain"]


@st.cache_data(show_spinner=False)
def load_rules_data():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    candidate_dirs = [
        os.path.join(root, "rules_output"),
        os.path.join(root, "notebook", "rules_output"),
    ]

    frames = []
    available_dir = None
    for directory in candidate_dirs:
        file_hits = 0
        for region in REGION_ORDER:
            path = os.path.join(directory, f"rules_{region}.csv")
            if not os.path.exists(path):
                continue
            try:
                part = pd.read_csv(path)
            except Exception:
                continue
            if len(part) == 0:
                continue
            part["region"] = region
            frames.append(part)
            file_hits += 1
        if file_hits:
            available_dir = directory
            break

    if not frames:
        return None, None

    df = pd.concat(frames, ignore_index=True)
    for col in ["support", "confidence", "lift"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    def parse_items(value):
        if isinstance(value, list):
            return [str(v).strip() for v in value]
        if pd.isna(value):
            return []
        text = str(value).strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple, set)):
                return [str(v).strip() for v in parsed]
            return [str(parsed).strip()]
        except Exception:
            text = text.strip("[]")
            if not text:
                return []
            return [p.strip().strip("'\"") for p in text.split(",") if p.strip()]

    df["antecedent_items"] = df["antecedent"].apply(parse_items)
    df["consequent_items"] = df["consequent"].apply(parse_items)
    df["antecedent_text"] = df["antecedent_items"].apply(lambda x: " + ".join(x))
    df["consequent_text"] = df["consequent_items"].apply(lambda x: " + ".join(x))
    df["target"] = df["consequent_items"].apply(lambda x: x[0] if x else "unknown")
    df = df.dropna(subset=["support", "confidence", "lift"])
    return df, available_dir


def sidebar_content():
    st.markdown(
        '<div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.8px;font-family:\'DM Mono\',monospace;">⚙️ Cài đặt</div>',
        unsafe_allow_html=True,
    )
    st.session_state["ap_region"] = st.selectbox(
        "🗺️ Region",
        ["Tất cả", "north", "central", "south"],
        index=0,
    )
    st.session_state["ap_target"] = st.selectbox(
        "🎯 Target hậu quả",
        ["Tất cả", "rain_yes", "heatwave", "storm", "heavy_rain"],
        index=0,
    )
    st.session_state["ap_lift"] = st.slider("Lift tối thiểu", 1.0, 30.0, 1.0, 0.1)
    st.session_state["ap_conf"] = st.slider("Confidence tối thiểu", 0.0, 1.0, 0.1, 0.01)
    st.session_state["ap_top"] = st.slider("Top rules hiển thị", 5, 50, 15, 1)


sidebar_header(sidebar_content)

page_header(
    "🧩",
    "linear-gradient(135deg,#EEF2FF,#E0E7FF)",
    "FPGrowth Association Rules",
    "Dashboard kết quả luật kết hợp theo Region (không hiển thị notebook).",
)

rules_df, rules_dir = load_rules_data()
if rules_df is None:
    st.error("Không tìm thấy file rules CSV. Hãy chạy pipeline FPGrowth trước để tạo rules_output/rules_*.csv.")
    st.stop()

if rules_dir:
    st.caption(f"Nguồn dữ liệu rules: {rules_dir}")

region_filter = st.session_state.get("ap_region", "Tất cả")
target_filter = st.session_state.get("ap_target", "Tất cả")
min_lift = st.session_state.get("ap_lift", 1.0)
min_conf = st.session_state.get("ap_conf", 0.1)
top_n = st.session_state.get("ap_top", 15)

view_df = rules_df.copy()
if region_filter != "Tất cả":
    view_df = view_df[view_df["region"] == region_filter]
if target_filter != "Tất cả":
    view_df = view_df[view_df["target"] == target_filter]
view_df = view_df[(view_df["lift"] >= min_lift) & (view_df["confidence"] >= min_conf)]

if len(view_df) == 0:
    st.warning("Không có rules thỏa bộ lọc hiện tại. Giảm ngưỡng lift/confidence hoặc đổi region/target.")
    st.stop()

view_df = view_df.sort_values(["lift", "confidence", "support"], ascending=False)

total_rules = len(view_df)
avg_conf = view_df["confidence"].mean()
avg_lift = view_df["lift"].mean()
max_lift = view_df["lift"].max()
unique_ant = view_df["antecedent_text"].nunique()
rain_rate = (view_df["target"] == "rain_yes").mean() * 100

section("Tổng quan kết quả", "RULES · SUPPORT · CONFIDENCE · LIFT")
c1, c2, c3, c4, c5, c6 = st.columns(6)
for col, label, val, badge, btype in [
    (c1, "Số rules", f"{total_rules:,}", "sau lọc", "info"),
    (c2, "Confidence TB", f"{avg_conf:.3f}", "độ tin cậy", "up" if avg_conf >= 0.3 else "info"),
    (c3, "Lift TB", f"{avg_lift:.2f}", "liên hệ", "up" if avg_lift >= 2 else "info"),
    (c4, "Lift cao nhất", f"{max_lift:.2f}", "rule mạnh nhất", "up"),
    (c5, "Tiền đề unique", f"{unique_ant:,}", "antecedent", "info"),
    (c6, "% rules mưa", f"{rain_rate:.1f}%", "target=rain_yes", "info"),
]:
    with col:
        st.markdown(kpi_card(label, val, badge, btype), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

section("Biểu đồ chính")
left, right = st.columns(2)

with left:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📍 Số rules theo Region**")
    count_region = (
        view_df.groupby("region").size().reindex(REGION_ORDER, fill_value=0).reset_index(name="count")
    )
    fig, ax = fig_ax(6, 3.6)
    ax.bar(count_region["region"], count_region["count"], color=["#4F6EF7", "#10B981", "#F97316"], alpha=0.9, width=0.6, zorder=3)
    for x, y in zip(count_region["region"], count_region["count"]):
        ax.text(x, y + max(1, count_region["count"].max() * 0.01), f"{y}", ha="center", va="bottom", fontsize=9, color="#334155")
    ax.set_xlabel("Region")
    ax.set_ylabel("Số rules")
    ax.set_title("Phân bố rules sau lọc", fontsize=11, fontweight="700", color="#1a1d2e", pad=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🎯 Target consequents**")
    target_count = (
        view_df.groupby("target").size().reindex(TARGET_ORDER, fill_value=0).reset_index(name="count")
    )
    fig, ax = fig_ax(6, 3.6)
    ax.bar(target_count["target"], target_count["count"], color=["#2563EB", "#DC2626", "#9333EA", "#0891B2"], alpha=0.88, width=0.62, zorder=3)
    ax.tick_params(axis="x", rotation=15)
    ax.set_xlabel("Consequent target")
    ax.set_ylabel("Số rules")
    ax.set_title("Rules theo loại mục tiêu", fontsize=11, fontweight="700", color="#1a1d2e", pad=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

section("Chất lượng luật")
q1, q2 = st.columns(2)

with q1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🔎 Scatter: Support vs Confidence (màu = Lift)**")
    fig, ax = fig_ax(6, 3.8)
    sc = ax.scatter(
        view_df["support"],
        view_df["confidence"],
        c=view_df["lift"],
        cmap="YlOrRd",
        alpha=0.75,
        edgecolors="none",
        s=46,
    )
    plt.colorbar(sc, ax=ax, label="Lift")
    ax.set_xlabel("Support")
    ax.set_ylabel("Confidence")
    ax.set_title("Bản đồ sức mạnh rules", fontsize=11, fontweight="700", color="#1a1d2e", pad=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

with q2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown(f"**🏆 Top {top_n} rules theo Lift**")
    top_rules = view_df.head(top_n).copy()
    top_rules["rule_label"] = top_rules.apply(
        lambda r: f"{r['antecedent_text']} -> {r['consequent_text']}", axis=1
    )
    top_rules = top_rules.iloc[::-1]
    fig, ax = fig_ax(7, max(4.0, 0.38 * len(top_rules)))
    colors = ["#DC2626" if t in ["heatwave", "storm", "heavy_rain"] else "#2563EB" for t in top_rules["target"]]
    ax.barh(range(len(top_rules)), top_rules["lift"], color=colors, alpha=0.9, zorder=3)
    ax.set_yticks(range(len(top_rules)))
    ax.set_yticklabels(top_rules["rule_label"], fontsize=7)
    ax.set_xlabel("Lift")
    ax.set_title("Top rules có sức liên hệ mạnh nhất", fontsize=11, fontweight="700", color="#1a1d2e", pad=8)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)
    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

section("Bảng chi tiết luật")
show_cols = [
    "region",
    "antecedent_text",
    "consequent_text",
    "support",
    "confidence",
    "lift",
]

st.markdown('<div class="card" style="padding:0;">', unsafe_allow_html=True)
st.dataframe(
    view_df[show_cols].rename(
        columns={
            "region": "Region",
            "antecedent_text": "Antecedent",
            "consequent_text": "Consequent",
            "support": "Support",
            "confidence": "Confidence",
            "lift": "Lift",
        }
    ),
    use_container_width=True,
    height=380,
)
st.markdown(
    f'<div style="font-size:11px;color:#9ca3af;padding:8px 16px 12px;">Hiển thị {len(view_df):,} rules sau lọc</div>',
    unsafe_allow_html=True,
)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📌 Insight nhanh"):
    strongest = view_df.iloc[0]
    st.write(
        f"Rule mạnh nhất hiện tại: {strongest['antecedent_text']} -> {strongest['consequent_text']} "
        f"(lift={strongest['lift']:.2f}, confidence={strongest['confidence']:.3f}, support={strongest['support']:.4f})"
    )
    by_region = view_df.groupby("region")["lift"].mean().sort_values(ascending=False)
    if len(by_region):
        st.write(f"Region có lift trung bình cao nhất: {by_region.index[0]} ({by_region.iloc[0]:.2f})")
