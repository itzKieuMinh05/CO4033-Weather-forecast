import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

CITY_COLUMN_CANDIDATES = ["city", "province", "location"]
FEATURE_CANDIDATES = ["temperature", "humidity", "pressure", "wind_speed", "cloudcover"]


def _select_city_column(df):
    for col in CITY_COLUMN_CANDIDATES:
        if col in df.columns:
            return col
    return None


def _find_optimal_k(X_scaled, k_cap=10):
    max_k = min(k_cap, len(X_scaled) - 1)
    if max_k < 2:
        return None, [], [], []

    k_values = list(range(2, max_k + 1))
    inertia = []
    sil_scores = []

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertia.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    best_k = k_values[int(np.argmax(sil_scores))]
    return best_k, k_values, inertia, sil_scores

st.set_page_config(page_title="Clustering — WeatherVN", page_icon="🗺️", layout="wide", initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def _sb():
    st.markdown('<div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.8px;font-family:\'DM Mono\',monospace;">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    df_ = st.selectbox("📂 File dữ liệu", ["weather_vn_cleaned.csv","test_data.csv","train_data.csv"])
    ak  = st.checkbox("Tự chọn k theo Silhouette", value=True)
    kv  = st.slider("Số clusters (k) thủ công", 2, 8, 3, disabled=ak)
    sp  = st.slider("% mẫu hiển thị", 5, 50, 25)
    ro  = st.selectbox("Lọc Region", ["Tất cả","north","central","south"])
    se  = st.checkbox("Hiện Elbow + Silhouette", value=True)
    st.session_state['cl_file'] = df_
    st.session_state['cl_auto_k'] = ak
    st.session_state['cl_k']    = kv
    st.session_state['cl_sp']   = sp
    st.session_state['cl_ro']   = ro
    st.session_state['cl_se']   = se

sidebar_header(_sb)
data_file  = st.session_state.get('cl_file', 'weather_vn_cleaned.csv')
auto_k     = st.session_state.get('cl_auto_k', True)
k_val      = st.session_state.get('cl_k',    3)
sample_pct = st.session_state.get('cl_sp',   25)
region_opt = st.session_state.get('cl_ro',   'Tất cả')
show_elbow = st.session_state.get('cl_se',   True)


page_header("🗺️","linear-gradient(135deg,#FAF5FF,#EDE9FE)",
            "Phân tích Clustering",
            "KMeans · PCA · Phân bổ thời tiết theo vùng miền")

models  = load_models()
df_full = load_csv(data_file)
if df_full is None and data_file != "test_data.csv":
    df_full = load_csv("test_data.csv")

if df_full is None:
    st.warning("⚠️ Không tìm thấy file dữ liệu."); st.stop()

city_col = _select_city_column(df_full)
if city_col is None:
    st.error("Không tìm thấy cột tỉnh/thành: city/province/location.")
    st.stop()

# Đồng bộ với notebook/clustering.py: ưu tiên đúng bộ feature dùng cho phân cụm.
feats = [f for f in FEATURE_CANDIDATES if f in df_full.columns]
if not feats:
    st.error(f"Không tìm thấy bất kỳ cột nào trong: {FEATURE_CANDIDATES}")
    st.stop()

base_cols = [city_col] + feats
if 'region' in df_full.columns:
    base_cols.append('region')

df_use = df_full[base_cols].dropna(subset=[city_col]).copy()
city_metrics = df_use.groupby(city_col, as_index=False)[feats].mean().dropna()

if 'region' in df_use.columns:
    city_region = (
        df_use.groupby(city_col)['region']
        .agg(lambda s: s.mode().iat[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )
    city_metrics = city_metrics.merge(city_region, on=city_col, how='left')

X_raw = city_metrics[feats]

if len(X_raw) == 0:
    st.error("Không có dữ liệu sau khi loại bỏ NaN."); st.stop()

# Đồng bộ notebook: luôn fit StandardScaler từ dữ liệu hiện tại.
X_sc = StandardScaler().fit_transform(X_raw)

best_k, k_values, inertia_vals, sil_vals = _find_optimal_k(X_sc, k_cap=10)
if best_k is None:
    st.error("Số lượng tỉnh/thành không đủ để clustering (cần ít nhất 3).")
    st.stop()

if auto_k:
    selected_k = best_k
else:
    selected_k = max(2, min(k_val, max(k_values)))
    if selected_k != k_val:
        st.info(f"k thủ công được điều chỉnh về {selected_k} do dữ liệu hiện tại.")

st.caption(f"k gợi ý theo Silhouette: {best_k} | k đang dùng: {selected_k} | Số tỉnh/thành: {len(city_metrics)}")

# KMeans
with st.spinner("⏳ Đang phân cụm..."):
    km = KMeans(n_clusters=selected_k, random_state=42, n_init=10)
    labels = km.fit_predict(X_sc)

    # PCA n_components must be <= n_features
    n_components = min(2, X_sc.shape[1])
    pca   = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_sc)
    var   = pca.explained_variance_ratio_

    # Pad to 2D if only 1 component
    if X_pca.shape[1] == 1:
        X_pca = np.hstack([X_pca, np.zeros((len(X_pca), 1))])
        var   = np.append(var, 0.0)

df_cl = city_metrics.copy()
df_cl['cluster'] = labels

# Region filter
if region_opt != "Tất cả" and 'region' in df_cl.columns:
    df_view = df_cl[df_cl['region'] == region_opt]
else:
    df_view = df_cl

n = len(df_view)
if n == 0:
    st.warning(f"Không có dữ liệu cho region: {region_opt}"); st.stop()

# Biểu đồ PCA chính hiển thị full dữ liệu theo notebook để bảo toàn output trực quan.
X_pca_plot = X_pca
labels_plot = labels
city_plot = df_cl[city_col].to_numpy()

PAL     = plt.cm.tab10(np.linspace(0, 0.9, selected_k))
PAL_HEX = [f'#{int(c[0]*255):02x}{int(c[1]*255):02x}{int(c[2]*255):02x}' for c in PAL]

# ── PCA scatter ───────────────────────────────────────
section(f"PCA Scatter — {len(X_pca_plot):,} tỉnh/thành")
pca_c, stat_c = st.columns([3, 2])

with pca_c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    scatter = ax.scatter(
        X_pca_plot[:, 0],
        X_pca_plot[:, 1],
        c=labels_plot,
        cmap="tab10",
        s=70,
        alpha=0.85,
        edgecolors="k",
        linewidth=0.3,
    )

    for i, c_name in enumerate(city_plot):
        ax.annotate(c_name, (X_pca_plot[i, 0], X_pca_plot[i, 1]), fontsize=8, alpha=0.9)

    ax.set_xlabel(f"PC1 ({var[0]:.1%})")
    ax.set_ylabel(f"PC2 ({var[1]:.1%})")
    ax.set_title("KMeans Clusters of Provinces/Cities")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    handles, legend_labels = scatter.legend_elements()
    ax.legend(handles, legend_labels, title="Cluster", loc="best")
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with stat_c:
    summary = df_view.groupby('cluster')[feats].mean().round(3)

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**📊 Cluster centroids (thang đo gốc)**")
    st.dataframe(summary, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("**📦 Số mẫu mỗi Cluster**")
    counts = df_view['cluster'].value_counts().sort_index()
    for ci, cnt in counts.items():
        pct = cnt / len(df_view) * 100
        color_hex = PAL_HEX[ci % len(PAL_HEX)]
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:10px;padding:5px 0;
                    border-bottom:1px solid #F4F6FB;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:{color_hex};flex-shrink:0;"></div>
            <span style="font-size:13px;color:#1a1d2e;font-weight:600;">Cluster {ci}</span>
            <span style="font-size:12px;color:#9ca3af;margin-left:auto;">{cnt:,} ({pct:.1f}%)</span>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# ── Region breakdown ──────────────────────────────────
if 'region' in df_cl.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Phân bổ Region × Cluster")
    rc1, rc2 = st.columns(2)
    reg_colors = ['#4F6EF7','#10B981','#F97316']

    with rc1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Số lượng theo Cluster & Region**")
        cross = pd.crosstab(df_view['cluster'], df_view['region'])
        fig, ax = fig_ax(6, 3.5)
        cross.plot(kind='bar', ax=ax, color=reg_colors[:len(cross.columns)],
                   alpha=0.88, width=0.65, edgecolor='none', zorder=3)
        ax.set_xlabel("Cluster", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Số lượng", color=MPL_LABEL, fontsize=10)
        ax.set_title("Phân bổ Region", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with st.expander("📍 Danh sách tỉnh/thành theo Cluster", expanded=False):
        city_lists = df_view.groupby('cluster')[city_col].apply(list)
        for c_id, city_list in city_lists.items():
            st.write(f"Cluster {c_id} ({len(city_list)} tỉnh/thành): {', '.join(city_list)}")

    with rc2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("**Tỷ lệ (%) theo Cluster & Region**")
        cn = pd.crosstab(df_view['cluster'], df_view['region'], normalize='index') * 100
        fig, ax = fig_ax(6, 3.5)
        bottom = np.zeros(len(cn))
        for i, reg in enumerate(cn.columns):
            bars = ax.bar(cn.index, cn[reg], bottom=bottom,
                          label=reg, color=reg_colors[i % 3], alpha=0.88, width=0.55, zorder=3)
            for rect, v in zip(bars, cn[reg]):
                if v > 6:
                    ax.text(rect.get_x()+rect.get_width()/2,
                            rect.get_y()+rect.get_height()/2,
                            f'{v:.0f}%', ha='center', va='center',
                            fontsize=8, color='white', fontweight='700')
            bottom += cn[reg].values
        ax.set_title("Tỷ lệ Region (%)", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.set_xlabel("Cluster", color=MPL_LABEL, fontsize=10)
        ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
        ax.tick_params(axis='x', rotation=0)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Radar chart ───────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Radar — Profile Cluster", "ĐẶC TRƯNG TRUNG BÌNH MỖI CỤM")
s_norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-9)
cats   = list(s_norm.columns)
N      = len(cats)

if N >= 3:
    angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist() + [0]
    n_profiles = len(summary)
    fig, axes = plt.subplots(1, n_profiles, figsize=(4.2*n_profiles, 4.2),
                              facecolor=MPL_BG, subplot_kw=dict(polar=True))
    if n_profiles == 1:
        axes = [axes]
    for ci, ax in zip(summary.index.tolist(), axes):
        ax.set_facecolor('#FAFBFF')
        profile = s_norm.loc[ci].tolist()
        vals = profile + [profile[0]]
        c    = PAL[ci]
        ax.fill(angles, vals, alpha=0.2, color=c)
        ax.plot(angles, vals, color=c, lw=2.5)
        ax.scatter(angles[:-1], vals[:-1], color=c, s=50, zorder=5)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(cats, size=8, color='#4b5563')
        ax.set_yticklabels([]); ax.grid(color='#E5E7EB', lw=0.8)
        ax.set_title(f'Cluster {ci}', color=MPL_TITLE, fontsize=11, fontweight='700', pad=12)
        for spine in ax.spines.values(): spine.set_color('#E5E7EB')
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True); plt.close()
else:
    st.info("Cần ít nhất 3 features để vẽ radar chart.")

# ── Extreme heatmap per cluster ───────────────────────
if 'extreme' in df_cl.columns:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Extreme Weather per Cluster (%)")
    col_ext, _ = st.columns([2, 1])
    with col_ext:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ext_pct = (df_view.groupby('cluster')['extreme']
                   .value_counts(normalize=True).mul(100).round(2)
                   .unstack(fill_value=0))
        fig, ax = fig_ax(7, 3.2)
        sns.heatmap(ext_pct, annot=True, fmt='.1f', cmap='YlOrRd',
                    ax=ax, linewidths=0.3, linecolor='white',
                    annot_kws={'size': 10, 'weight': '600'},
                    cbar_kws={'label': '%', 'shrink': 0.8})
        ax.set_title("Tỷ lệ loại thời tiết theo Cluster (%)",
                     color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Elbow ─────────────────────────────────────────────
if show_elbow:
    st.markdown("<br>", unsafe_allow_html=True)
    section("Elbow + Silhouette", "CHỌN K TỐI ƯU")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = fig_ax(6, 3.6)
        ax.plot(k_values, inertia_vals, marker='o', color=ACCENT, lw=2.2, ms=7, zorder=5)
        ax.fill_between(k_values, inertia_vals, alpha=0.07, color=ACCENT)
        y_sel = inertia_vals[k_values.index(selected_k)]
        ax.scatter([selected_k], [y_sel], color='#F97316', s=120, zorder=10, label=f'k={selected_k}')
        ax.set_xlabel("k", color=MPL_LABEL, fontsize=11)
        ax.set_ylabel("Inertia", color=MPL_LABEL, fontsize=11)
        ax.set_title("Elbow Method", color=MPL_TITLE, fontsize=12, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB')
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

    with c2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        fig, ax = fig_ax(6, 3.6)
        ax.plot(k_values, sil_vals, marker='o', color='#10B981', lw=2.2, ms=7, zorder=5)
        ax.fill_between(k_values, sil_vals, alpha=0.08, color='#10B981')
        y_best = sil_vals[k_values.index(best_k)]
        ax.scatter([best_k], [y_best], color='#F97316', s=120, zorder=10, label=f'k tốt nhất={best_k}')
        ax.set_xlabel("k", color=MPL_LABEL, fontsize=11)
        ax.set_ylabel("Silhouette", color=MPL_LABEL, fontsize=11)
        ax.set_title("Silhouette by k", color=MPL_TITLE, fontsize=12, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB')
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)