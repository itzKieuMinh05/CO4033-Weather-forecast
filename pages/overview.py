import streamlit as st
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *
import pandas as pd, numpy as np, matplotlib.pyplot as plt, seaborn as sns

st.set_page_config(page_title="Tổng quan — WeatherVN", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
st.markdown(GLOBAL_CSS, unsafe_allow_html=True)
st.markdown("""
<style>
.insight-box {
    background:#F0F4FF; border-left:4px solid #4F6EF7;
    border-radius:0 10px 10px 0; padding:10px 14px;
    font-size:12px; color:#374151; margin-top:8px;
}
</style>""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────
def _sidebar():
    st.markdown('<div style="padding:12px 4px 8px;font-size:11px;font-weight:700;color:#374151;text-transform:uppercase;letter-spacing:.8px;font-family:\'DM Mono\',monospace;">⚙️ Cài đặt</div>', unsafe_allow_html=True)
    data_file     = st.selectbox("📂 File dữ liệu",
        ["weather_vn_cleaned.csv","test_data.csv","train_data.csv"])
    region_filter = st.selectbox("🗺️ Lọc Region",
        ["Tất cả","north","central","south"])
    st.session_state['ov_file']   = data_file
    st.session_state['ov_region'] = region_filter

sidebar_header(_sidebar)
data_file     = st.session_state.get('ov_file',   "weather_vn_cleaned.csv")
region_filter = st.session_state.get('ov_region', "Tất cả")

page_header("📊","linear-gradient(135deg,#EEF2FF,#E0E7FF)",
            "Tổng quan Dữ liệu",
            "Phân tích phân bố thời tiết · Nhiệt độ · Mưa · Gió theo vùng miền")

# ── Load data ─────────────────────────────────────────
df = load_csv(data_file)
if df is None:
    for fb in ["test_data.csv","train_data.csv","weather_vn_cleaned.csv"]:
        df = load_csv(fb)
        if df is not None:
            st.info(f"Không tìm thấy **{data_file}**, dùng **{fb}**."); break
if df is None:
    st.warning("⚠️ Không tìm thấy file dữ liệu. Đặt file CSV cùng thư mục với app.py.")
    st.stop()

df_view = df.copy()
if region_filter != "Tất cả" and 'region' in df.columns:
    df_view = df[df['region'] == region_filter].copy()
    if len(df_view) == 0:
        st.warning(f"Không có dữ liệu cho region: {region_filter}"); df_view = df.copy()

# ── KPIs ──────────────────────────────────────────────
total    = len(df_view)
rain     = df_view['rain'].mean()*100       if 'rain'        in df_view.columns else 0
ext      = (df_view['extreme']!='normal').mean()*100 if 'extreme' in df_view.columns else 0
cities   = df_view['city'].nunique()        if 'city'        in df_view.columns else "—"
avg_tmp  = df_view['temperature'].mean()    if 'temperature' in df_view.columns else None
avg_wind = df_view['wind_speed'].mean()     if 'wind_speed'  in df_view.columns else None

c1,c2,c3,c4,c5,c6 = st.columns(6)
for col, label, val, badge, btype in [
    (c1,"Tổng bản ghi",f"{total:,}",     f"{'toàn bộ' if region_filter=='Tất cả' else region_filter}","info"),
    (c2,"Tỷ lệ mưa",   f"{rain:.1f}%",   "có mưa","up" if rain>30 else "info"),
    (c3,"Cực đoan",     f"{ext:.1f}%",    "heatwave/storm","down" if ext>5 else "info"),
    (c4,"Thành phố",   str(cities),       "trong dataset","info"),
    (c5,"Nhiệt TB",    f"{avg_tmp:.1f}°C" if avg_tmp else "—","trung bình","down" if avg_tmp and avg_tmp>35 else "info"),
    (c6,"Gió TB",      f"{avg_wind:.1f} m/s" if avg_wind else "—","tốc độ gió","info"),
]:
    with col: st.markdown(kpi_card(label, val, badge, btype), unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Row 1 ─────────────────────────────────────────────
section("Phân bố thời tiết","EXTREME · RAINFALL · PATTERNS")
r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🌀 Loại thời tiết theo Region**")
    if 'extreme' in df_view.columns and 'region' in df_view.columns:
        cross = pd.crosstab(df_view['region'], df_view['extreme'], normalize='index')*100
        fig, ax = fig_ax(6, 3.6)
        pal = {'normal':'#4F6EF7','heatwave':'#F97316','storm':'#9333EA','heavy_rain':'#06B6D4'}
        bottom = np.zeros(len(cross))
        for cn in cross.columns:
            c_ = pal.get(cn,'#94A3B8')
            bars = ax.bar(cross.index, cross[cn], bottom=bottom, label=cn,
                          color=c_, alpha=0.9, width=0.5, zorder=3)
            for rect, v in zip(bars, cross[cn]):
                if v > 5:
                    ax.text(rect.get_x()+rect.get_width()/2,
                            rect.get_y()+rect.get_height()/2,
                            f'{v:.0f}%', ha='center', va='center',
                            fontsize=9, color='white', fontweight='700')
            bottom += cross[cn].values
        ax.set_xlabel("Region", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Tỷ lệ (%)", color=MPL_LABEL, fontsize=10)
        ax.set_title("Phân bổ loại thời tiết (%)", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='#E5E7EB')
        ax.set_ylim(0, 115)
        st.pyplot(fig, use_container_width=True); plt.close()
        if 'heatwave' in cross.columns:
            hottest = cross['heatwave'].idxmax()
            st.markdown(f'<div class="insight-box">🔥 <b>{hottest}</b> có heatwave cao nhất: <b>{cross.loc[hottest,"heatwave"]:.1f}%</b></div>', unsafe_allow_html=True)
    else: st.info("Cần cột `extreme` và `region`")
    st.markdown('</div>', unsafe_allow_html=True)

with r1c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🌧️ Xác suất mưa theo Tháng**")
    if 'rain' in df_view.columns and 'month' in df_view.columns:
        monthly = df_view.groupby('month')['rain'].mean()*100
        peak_m  = monthly.idxmax()
        fig, ax = fig_ax(6, 3.6)
        clrs = ['#4F6EF7' if v >= monthly.max()*0.8 else '#93C5FD' for v in monthly.values]
        ax.bar(monthly.index, monthly.values, color=clrs, alpha=0.85, width=0.65, zorder=3)
        ax.plot(monthly.index, monthly.values, color=ACCENT, lw=2.5, marker='o', ms=5, zorder=5)
        ax.fill_between(monthly.index, monthly.values, alpha=0.08, color=ACCENT)
        ax.set_xticks(range(1,13))
        ax.set_xticklabels([f'T{i}' for i in range(1,13)], fontsize=9)
        ax.set_xlabel("Tháng", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Tỷ lệ mưa (%)", color=MPL_LABEL, fontsize=10)
        ax.set_title("Xác suất mưa theo tháng", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        st.pyplot(fig, use_container_width=True); plt.close()
        st.markdown(f'<div class="insight-box">💧 Tháng mưa nhiều nhất: <b>T{peak_m}</b> ({monthly[peak_m]:.1f}%) · Ít nhất: <b>T{monthly.idxmin()}</b> ({monthly.min():.1f}%)</div>', unsafe_allow_html=True)
    else: st.info("Cần cột `rain` và `month`")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 2 ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🌡️ Nhiệt độ TB theo Region & Tháng (°C)**")
    if all(c in df_view.columns for c in ['temperature','month','region']):
        pivot = df_view.pivot_table('temperature','region','month',aggfunc='mean')
        fig, ax = fig_ax(6, 3.0)
        sns.heatmap(pivot.round(1), annot=True, fmt='.1f', cmap='YlOrRd', ax=ax,
                    linewidths=0.3, linecolor='white',
                    annot_kws={'size':8,'weight':'600'}, cbar_kws={'shrink':0.8,'pad':0.02})
        ax.set_xlabel("Tháng", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Region", color=MPL_LABEL, fontsize=10)
        ax.set_title("Heatmap nhiệt độ TB", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.tick_params(labelsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()
        hr = pivot.max(axis=1).idxmax(); hm = pivot.loc[hr].idxmax()
        st.markdown(f'<div class="insight-box">🌡️ Nóng nhất: <b>{hr}</b>, tháng <b>T{hm}</b> ({pivot.loc[hr,hm]:.1f}°C)</div>', unsafe_allow_html=True)
    else: st.info("Cần cột `temperature`, `month`, `region`")
    st.markdown('</div>', unsafe_allow_html=True)

with r2c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**💨 Phân bố Wind Speed theo Region**")
    if 'wind_speed' in df_view.columns:
        fig, ax = fig_ax(6, 3.0)
        regs = sorted(df_view['region'].dropna().unique()) if 'region' in df_view.columns else ['all']
        stats_w = []
        for i, reg in enumerate(regs):
            d = df_view[df_view['region']==reg]['wind_speed'].dropna() if 'region' in df_view.columns else df_view['wind_speed'].dropna()
            if len(d)==0: continue
            ax.hist(d, bins=40, alpha=0.5, color=PALETTE[i%len(PALETTE)],
                    label=str(reg), density=True, histtype='stepfilled', zorder=3)
            ax.hist(d, bins=40, alpha=0.9, color=PALETTE[i%len(PALETTE)],
                    density=True, histtype='step', linewidth=1.5, zorder=4)
            stats_w.append((reg, d.mean(), d.max()))
        ax.set_xlabel("Wind Speed (m/s)", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("Density", color=MPL_LABEL, fontsize=10)
        ax.set_title("Phân bố tốc độ gió", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.legend(facecolor='white', edgecolor='#E5E7EB', fontsize=9)
        st.pyplot(fig, use_container_width=True); plt.close()
        if stats_w:
            wd = max(stats_w, key=lambda x: x[1])
            st.markdown(f'<div class="insight-box">💨 Gió mạnh nhất tb: <b>{wd[0]}</b> ({wd[1]:.1f} m/s tb, max {wd[2]:.1f} m/s)</div>', unsafe_allow_html=True)
    else: st.info("Cần cột `wind_speed`")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Row 3 ─────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Phân tích chuyên sâu","HOURLY PATTERN · CORRELATION")
r3c1, r3c2 = st.columns(2)

with r3c1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🕐 Xác suất mưa theo Giờ trong ngày**")
    if 'rain' in df_view.columns and 'hour' in df_view.columns:
        hourly = df_view.groupby('hour')['rain'].mean()*100
        fig, ax = fig_ax(6, 3.2)
        tc_ = []
        for h in hourly.index:
            if 5<=h<12:   tc_.append('#FCD34D')
            elif 12<=h<18: tc_.append('#4F6EF7')
            elif 18<=h<22: tc_.append('#9333EA')
            else:          tc_.append('#1a1d2e')
        ax.bar(hourly.index, hourly.values, color=tc_, alpha=0.8, width=0.8, zorder=3)
        ax.plot(hourly.index, hourly.values, color='#EF4444', lw=2, marker='o', ms=3, zorder=5)
        ax.set_xticks(range(0,24,2))
        ax.set_xlabel("Giờ", color=MPL_LABEL, fontsize=10)
        ax.set_ylabel("% mưa", color=MPL_LABEL, fontsize=10)
        ax.set_title("Pattern mưa theo giờ", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color='#FCD34D',label='Sáng (5-12h)'),
                            Patch(color='#4F6EF7',label='Chiều (12-18h)'),
                            Patch(color='#9333EA',label='Tối (18-22h)'),
                            Patch(color='#1a1d2e',label='Đêm')],
                  fontsize=8, facecolor='white', edgecolor='#E5E7EB', ncol=2)
        st.pyplot(fig, use_container_width=True); plt.close()
        peak_h = hourly.idxmax()
        st.markdown(f'<div class="insight-box">⏰ Mưa nhiều nhất lúc <b>{peak_h:02d}:00</b> ({hourly[peak_h]:.1f}%) · Ít nhất lúc <b>{hourly.idxmin():02d}:00</b> ({hourly.min():.1f}%)</div>', unsafe_allow_html=True)
    else: st.info("Cần cột `rain` và `hour`")
    st.markdown('</div>', unsafe_allow_html=True)

with r3c2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("**🔗 Tương quan giữa các biến số**")
    num_cols = [c for c in ['temperature','humidity','pressure','wind_speed','cloudcover','visibility'] if c in df_view.columns]
    if len(num_cols) >= 3:
        corr = df_view[num_cols].corr()
        fig, ax = fig_ax(6, 3.2)
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlBu_r', ax=ax,
                    linewidths=0.4, linecolor='white', vmin=-1, vmax=1,
                    annot_kws={'size':8,'weight':'600'}, cbar_kws={'shrink':0.8})
        ax.set_title("Correlation Matrix", color=MPL_TITLE, fontsize=11, fontweight='700', pad=8)
        ax.tick_params(labelsize=8, rotation=30)
        st.pyplot(fig, use_container_width=True); plt.close()
        cv = corr.abs().unstack(); cv = cv[cv<1.0].sort_values(ascending=False)
        if len(cv)>0:
            tp = cv.index[0]; tv = cv.iloc[0]
            dr = "thuận" if corr.loc[tp]>0 else "nghịch"
            st.markdown(f'<div class="insight-box">🔗 Tương quan mạnh nhất: <b>{tp[0]}</b> ↔ <b>{tp[1]}</b> ({dr}, r={tv:.2f})</div>', unsafe_allow_html=True)
    else: st.info("Cần ít nhất 3 cột số")
    st.markdown('</div>', unsafe_allow_html=True)

# ── Data table ────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
section("Dữ liệu mẫu")
show_cols = [c for c in ['time','city','region','temperature','humidity',
                          'pressure','wind_speed','cloudcover','rain','extreme']
             if c in df_view.columns]
st.markdown('<div class="card" style="padding:0;">', unsafe_allow_html=True)
if 'city' in df_view.columns:
    sel_city = st.selectbox("🔍 Lọc thành phố",
        ["Tất cả"]+sorted(df_view['city'].dropna().unique().tolist()), key="city_filter")
    df_show = df_view[df_view['city']==sel_city] if sel_city!="Tất cả" else df_view
else:
    df_show = df_view
st.dataframe(df_show[show_cols].head(200), use_container_width=True, height=280)
st.markdown(f'<div style="font-size:11px;color:#9ca3af;padding:8px 16px 12px;">Hiển thị 200/{len(df_show):,} bản ghi</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

with st.expander("📐 Thống kê mô tả đầy đủ"):
    st.dataframe(df_view.select_dtypes(include=[np.number]).describe().round(3), use_container_width=True)