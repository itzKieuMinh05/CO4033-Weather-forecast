import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import label_binarize

st.set_page_config(page_title="Đánh giá — WeatherVN", page_icon="🔍", layout="wide", initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)

def _sb():
    st.markdown('<div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.8px;font-family:\'DM Mono\',monospace;">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    tf = st.selectbox("📂 File test", ["test_data.csv","weather_vn_cleaned.csv","train_data.csv"])
    st.session_state['ev_file'] = tf

sidebar_header(_sb)
test_file = st.session_state.get('ev_file', 'test_data.csv')


page_header("🔍","linear-gradient(135deg,#FFF7ED,#FFEDD5)",
            "Đánh giá Mô hình",
            "So sánh Random Forest vs XGBoost trên tập test độc lập")

models = load_models()
df_test = load_csv(test_file)

if not all(k in models for k in ["rf","xgb","le"]):
    st.error("❌ Thiếu model. Cần: `rf_model.pkl`, `xgb_model.pkl`, `label_encoder.pkl`")
    st.stop()
if df_test is None:
    st.warning(f"⚠️ Không tìm thấy `{test_file}`"); st.stop()
if 'extreme' not in df_test.columns:
    st.error("Cần cột `extreme` trong file test."); st.stop()

le, rf, xgb = models['le'], models['rf'], models['xgb']

# FIX: drop only columns that actually exist
drop_existing = [c for c in DROP_COLS if c in df_test.columns]
X_raw = df_test.drop(columns=drop_existing).select_dtypes(include=[np.number])
X_raw = X_raw.ffill().bfill().fillna(0)

# FIX: safely encode labels — handle unseen classes
try:
    y_test = le.transform(df_test['extreme'])
except Exception:
    # Filter to only known classes
    known_mask = df_test['extreme'].isin(le.classes_)
    df_test = df_test[known_mask].copy()
    if len(df_test) == 0:
        st.error("Không còn mẫu hợp lệ sau khi lọc theo classes của label encoder.")
        st.stop()
    X_raw = df_test.drop(columns=[c for c in DROP_COLS if c in df_test.columns]).select_dtypes(include=[np.number])
    X_raw = X_raw.ffill().bfill().fillna(0)
    y_test = le.transform(df_test['extreme'])

def align(model, X):
    if hasattr(model, 'feature_names_in_'):
        d = X.copy()
        for c in model.feature_names_in_:
            if c not in d.columns:
                d[c] = 0
        return d[list(model.feature_names_in_)]
    return X

def get_metrics(model, name):
    X    = align(model, X_raw.copy())
    pred = model.predict(X)
    prob = model.predict_proba(X)
    acc  = accuracy_score(y_test, pred)
    rec  = recall_score(y_test, pred, average='macro', zero_division=0)
    prec = precision_score(y_test, pred, average='macro', zero_division=0)
    f1   = f1_score(y_test, pred, average='macro', zero_division=0)
    cls  = model.classes_
    n_cls = len(cls)
    try:
        if n_cls == 2:
            auc = roc_auc_score(y_test, prob[:, 1])
        else:
            # FIX: pass classes as list of ints, not encoded labels
            y_bin = label_binarize(y_test, classes=list(cls))
            auc = roc_auc_score(y_bin, prob, average='macro', multi_class='ovr')
    except Exception:
        auc = float('nan')
    return dict(Model=name, AUC=auc, Accuracy=acc, Recall=rec, Precision=prec, F1=f1), pred

with st.spinner("Đang tính metrics..."):
    m_rf,  pred_rf  = get_metrics(rf,  "Random Forest")
    m_xgb, pred_xgb = get_metrics(xgb, "XGBoost")

winner = "XGBoost" if m_xgb['AUC'] >= m_rf['AUC'] else "Random Forest"
w = m_xgb if winner == "XGBoost" else m_rf

# ── KPI row ───────────────────────────────────────────
section("Tổng quan kết quả")
c1,c2,c3,c4,c5,c6 = st.columns(6)
kpis = [
    (c1,"RF — AUC",       f"{m_rf['AUC']:.4f}",  "","info"),
    (c2,"RF — Accuracy",  f"{m_rf['Accuracy']:.4f}", "","info"),
    (c3,"RF — F1",        f"{m_rf['F1']:.4f}",   "","info"),
    (c4,"XGB — AUC",      f"{m_xgb['AUC']:.4f}", "","info"),
    (c5,"XGB — Accuracy", f"{m_xgb['Accuracy']:.4f}", "","info"),
    (c6,"XGB — F1",       f"{m_xgb['F1']:.4f}",  "","info"),
]
for col, label, val, badge, btype in kpis:
    with col: st.markdown(kpi_card(label, val, badge, btype), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Winner card + comparison chart ───────────────────
chart_c, winner_c = st.columns([3,1])

with chart_c:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**So sánh RF vs XGBoost**")
    metrics_bar = ["AUC","Accuracy","Recall","Precision","F1"]
    x = np.arange(len(metrics_bar)); w_bar = 0.3
    fig, ax = fig_ax(7, 4)
    b1 = ax.bar(x-w_bar/2, [m_rf[m]  for m in metrics_bar], w_bar,
                label='Random Forest', color='#4F6EF7', alpha=0.88, zorder=3)
    b2 = ax.bar(x+w_bar/2, [m_xgb[m] for m in metrics_bar], w_bar,
                label='XGBoost',       color='#F97316', alpha=0.88, zorder=3)
    for bars in [b1, b2]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x()+bar.get_width()/2, h+0.006,
                        f'{h:.3f}', ha='center', va='bottom',
                        color=MPL_LABEL, fontsize=8, fontweight='600')
    ax.set_xticks(x); ax.set_xticklabels(metrics_bar, fontsize=10)
    ax.set_ylim(0, 1.14)
    ax.set_title("Metrics comparison", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
    ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
    ax.axhline(1.0, color='#E5E7EB', ls='--', lw=1)
    st.pyplot(fig, use_container_width=True); plt.close()
    st.markdown('</div>', unsafe_allow_html=True)

with winner_c:
    diff_auc = abs(m_xgb['AUC'] - m_rf['AUC'])
    color_w  = "#4F6EF7" if winner == "Random Forest" else "#F97316"
    st.markdown(f"""
    <div class="card" style="text-align:center;border-top:4px solid {color_w};">
        <div style="font-size:10px;color:#9ca3af;font-family:'DM Mono',monospace;
                    letter-spacing:1.5px;text-transform:uppercase;margin-bottom:12px;">
            Mô hình tốt nhất
        </div>
        <div style="font-size:36px;">🏆</div>
        <div style="font-size:17px;font-weight:800;color:{color_w};margin-top:10px;">
            {winner}
        </div>
        <div style="margin-top:16px;">
    """, unsafe_allow_html=True)
    for metric in ["AUC","Accuracy","F1"]:
        val = w[metric]
        st.markdown(f"""
        <div style="display:flex;justify-content:space-between;padding:6px 0;
                    border-bottom:1px solid #F0F2F8;">
            <span style="font-size:12px;color:#6b7280;">{metric}</span>
            <span style="font-size:12px;font-weight:700;color:#1a1d2e;">
                {"N/A" if np.isnan(val) else f"{val:.4f}"}
            </span>
        </div>""", unsafe_allow_html=True)
    st.markdown(f"""
        <div style="margin-top:12px;background:{'#EEF2FF' if winner=='Random Forest' else '#FFF7ED'};
                    border-radius:8px;padding:8px;font-size:11px;color:{color_w};font-weight:600;">
            ▲ {diff_auc:.4f} AUC so với model còn lại
        </div>
    </div></div>""", unsafe_allow_html=True)

# ── Confusion matrices ────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Confusion Matrix", "MA TRẬN NHẦM LẪN")
cm_c1, cm_c2 = st.columns(2)

cm_labels = np.unique(np.concatenate([y_test, pred_rf, pred_xgb]))
classes_names = le.inverse_transform(cm_labels)

for col_c, name, pred, cmap_name in [
    (cm_c1, "Random Forest", pred_rf,  'Blues'),
    (cm_c2, "XGBoost",       pred_xgb, 'Oranges'),
]:
    with col_c:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**{name}**")
        cm = confusion_matrix(y_test, pred, labels=cm_labels)
        fig, ax = fig_ax(5, 4.2)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap_name,
                    xticklabels=classes_names, yticklabels=classes_names,
                    ax=ax, linewidths=0.4, linecolor='white',
                    annot_kws={'size':12,'weight':'bold'})
        ax.set_title('Confusion Matrix', color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.set_xlabel('Dự đoán', color=MPL_LABEL, fontsize=10)
        ax.set_ylabel('Thực tế', color=MPL_LABEL, fontsize=10)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)

# ── Feature importance ────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Feature Importance", "TOP 15 ĐẶC TRƯNG QUAN TRỌNG NHẤT")
fi_c1, fi_c2 = st.columns(2)

for col_fi, model, name, color in [
    (fi_c1, rf,  "Random Forest", "#4F6EF7"),
    (fi_c2, xgb, "XGBoost",       "#F97316"),
]:
    with col_fi:
        if not hasattr(model, 'feature_importances_'):
            continue
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"**{name}**")
        feats = list(model.feature_names_in_) if hasattr(model, 'feature_names_in_') else list(X_raw.columns)
        fi = pd.Series(model.feature_importances_, index=feats).nlargest(15)
        fig, ax = fig_ax(5.5, 5.2)
        alphas = np.linspace(0.5, 1.0, len(fi))
        colors_fi = [(*plt.matplotlib.colors.to_rgb(color), a) for a in alphas[::-1]]
        bars = ax.barh(fi.index[::-1], fi.values[::-1], color=colors_fi,
                       height=0.65, zorder=3)
        for bar, val in zip(bars, fi.values[::-1]):
            ax.text(bar.get_width()+0.001, bar.get_y()+bar.get_height()/2,
                    f'{val:.4f}', va='center', color=MPL_LABEL, fontsize=8, fontweight='600')
        ax.set_title('Top 15 features', color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.set_xlim(0, fi.max()*1.28)
        ax.tick_params(labelsize=8)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown('</div>', unsafe_allow_html=True)