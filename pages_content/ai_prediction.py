"""
pages_content/ai_prediction.py
--------------------------
หน้า "AI Prediction" ของ CIS Dashboard
 
วิธีทดสอบหน้านี้แบบเดี่ยว (ไม่ต้องรอทีมคนอื่น):
    streamlit run preview_my_page.py
    (แล้วเลือกโมดูลนี้จาก dropdown ในไฟล์ preview_my_page.py)
 
ข้อมูลที่ใช้ได้ใน ctx (ดูนิยามเต็มใน common.py -> class PageContext):
    ctx.selected_ticker, ctx.stock_info, ctx.stock_daily, ctx.fin_stock, ctx.sector_peers,
    ctx.scores_df, ctx.fin_df, ctx.feat_imp_df, ctx.backtest_df, ctx.risk_hist_df,
    ctx.health_yearly_df, ctx.fair_value_yearly_df,
    ctx.current_price, ctx.change_pct, ctx.change_val, ctx.change_color, ctx.change_sign, ctx.arrow_sign
 
ห้ามแก้ CSS ส่วนกลางหรือ helper function ใน common.py จากไฟล์นี้ — ถ้าจำเป็นต้องแก้ ให้แจ้ง Layout Lead ก่อน
 
=== CHANGELOG (v4 — Font & Prediction box redesign) ===
- ขยายฟอนต์ทั้งหน้า: KPI (label 14 / ค่า 36 / คำอธิบายย่อย 15), Model Performance (ค่า 26 / label 14.5),
  Explainable AI Summary (17), คำอธิบายใน Prediction (18), ตัวเลข/legend ของกราฟทุกอัน (12.5-13)
  และปรับสีข้อความรอง (muted) จาก #64748B เป็น #94A3B8 ให้อ่านชัดขึ้นบนพื้นเข้ม
- ช่อง Prediction แยกเป็น 2 กล่อง: ซ้าย = gauge % ใหญ่ (ตัดข้อความ "Probability of ..." ออกจากในวงกลม)
  ขวา = กล่องคำอธิบายสีอ่อนกว่า มีหัวข้อ "PROBABILITY OF UP" (อังกฤษล้วน) และแยกคำอธิบายเป็น 2 บรรทัด
- เปลี่ยน 🔮 เป็น 📈 ที่หัวข้อ Prediction / เอา 🧠 ออกจากหัวข้อ Model Explanation
- ประโยคอธิบายใช้ความน่าจะเป็นของ "ทิศทางที่ทำนาย" (dir_prob) เพื่อให้ตรงเมื่อ prob_up < 50
  (เดิมจะโชว์ "มีโอกาสขาลง 40%" ทั้งที่ 40% คือโอกาสขึ้น)
- ย้ายโครง KPI card ไปใช้ helper _kpi_card() แทนการเขียน HTML ซ้ำ 5 ก้อน
 
=== CHANGELOG (v3 — Layout/UX redesign) ===
- รวมสถานะทำนายเป็นแหล่งความจริงเดียว: ใช้เกณฑ์จาก prob_up (>=70 / >=50 / อื่นๆ) กำหนด status_color
  เดียวกันทั้งหน้า (Direction, Probability, Score, Recommendation, กราฟ Forecast)
- ลบการ์ด "MODEL & DATA SUMMARY" ออกจากหน้าแรก ย้ายรายละเอียดไปไว้ใน expander ท้ายส่วน Model Performance
- เพิ่มแถบ KPI ใหญ่ด้านบนสุด (Price, Direction, Probability, Score, Recommendation)
- จัดลำดับส่วนใหม่: Overview -> Prediction -> Forecast -> Model Explanation -> Model Performance
- กราฟ Forecast: แยก Actual / Model Forecast / Prediction Range ให้ต่างกันชัดเจน
- ลดจำนวนสีที่ใช้: สถานะ 3 สี (เขียว/เหลือง/แดง) + ฟ้า 1 สีสำหรับข้อมูลราคาจริง/กราฟทั่วไป
- Explainable AI Summary ตัดบรรทัดที่ซ้ำกับ Model Performance ออก
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
 
from common import fmt_mb, fmt_ratio, safe, show_chart, render_nav_footer, COMPANY_NAMES, SECTOR_MAP
 
GREEN = "#10B981"
AMBER = "#F59E0B"
RED = "#EF4444"
BLUE = "#38BDF8"
MUTED = "#94A3B8"  # สีข้อความรอง (เดิม #64748B จางเกินไปบนพื้นเข้ม)
 
 
def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"
 
 
def _section_title(text):
    return f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-radius:12px 12px 0 0; padding:14px 18px 2px 18px;">
<div><span style="font-size:16.5px; font-weight:bold; color:{MUTED}; letter-spacing:0.5px;">{text}</span></div></div>"""
 
 
def _kpi_card(label, value_html, sub_html="", value_color="#F8FAFC", value_size=36, border="#1E293B"):
    """การ์ด KPI ใบเดียว (HTML บรรทัดเดียว ห้ามมีบรรทัดว่าง/เยื้องหน้า ไม่งั้น markdown จะตีเป็น code block)"""
    return (
        f'<div style="background-color:#0F172A; border:1px solid {border}; border-radius:12px; padding:18px 10px; '
        f'text-align:center; min-height:140px; display:flex; flex-direction:column; justify-content:center;">'
        f'<div style="font-size:14px; font-weight:bold; color:{MUTED}; letter-spacing:1px;">{label}</div>'
        f'<div style="font-size:{value_size}px; font-weight:bold; color:{value_color}; line-height:1.2; margin-top:4px;">{value_html}</div>'
        f'{sub_html}'
        f'</div>'
    )
 
 
def _kpi_sub(text, color=MUTED, bold=False):
    weight = "bold" if bold else "normal"
    return f'<div style="font-size:15px; font-weight:{weight}; color:{color}; margin-top:2px;">{text}</div>'
 
 
def _metric_cell(label, value):
    return (
        f'<div style="background:#151E2F; border:1px solid #1E293B; border-radius:8px; padding:14px 6px; text-align:center;">'
        f'<div style="font-size:14.5px; color:{MUTED};">{label}</div>'
        f'<div style="font-size:26px; font-weight:bold; color:#F8FAFC; line-height:1.35;">{value}</div></div>'
    )
 
 
def render(ctx):
    # ============================================================
    # แหล่งความจริงเดียวของสถานะทำนาย — ใช้ทุกจุดในหน้า ไม่คำนวณซ้ำแยกชุด
    # ============================================================
    prob_up = safe(ctx.stock_info.get('prob_up'), 50)
    down_prob = round(100 - prob_up, 1)
    ai_score = int(round(safe(ctx.stock_info.get('ai_score'), 50)))
    acc_val = safe(ctx.stock_info.get('accuracy'), 50)
    baseline_val = safe(ctx.stock_info.get('baseline_accuracy'), acc_val)
    precision_val = safe(ctx.stock_info.get('precision'), 0)
    recall_val = safe(ctx.stock_info.get('recall'), 0)
    signal = ctx.stock_info.get('ai_signal', '-')
    # [High FIX] ค่าคงที่จำลองตอนข้อมูลไม่พอ ไม่ใช่ผลโมเดลจริง — ต้องเตือนผู้ใช้ตรงๆ ไม่ใช่โชว์เหมือนผลจริง
    is_fallback = bool(ctx.stock_info.get('is_fallback', False))
 
    # [Medium FIX] เดิม reliability_low = acc_val < baseline_val (strict less-than) เพียงอย่างเดียว
    # ไม่จับกรณี "โมเดล degenerate" ที่ precision/recall = 0 (ไม่เคยทำนาย "ขึ้น" เลย) แต่ accuracy ดัน
    # เท่ากับ baseline พอดี (กรณี HANA) — เพิ่มเงื่อนไข <= และเช็ค precision/recall=0 ด้วย
    reliability_low = (acc_val <= baseline_val) or (precision_val == 0) or (recall_val == 0) or is_fallback
    baseline_note = "สูงกว่า" if acc_val > baseline_val else "ต่ำกว่าหรือเท่ากับ"
 
    # [High FIX] เดิม status_color (สี KPI การ์ดหลัก SCORE/RECOMMENDATION + gauge) อิงจาก prob_up
    # อย่างเดียว ทำให้หุ้นที่ accuracy ต่ำกว่า baseline มาก (เช่น THCOM, JMART) ยังขึ้นสีเขียว STRONG BUY
    # เหมือนเป็นสัญญาณที่เชื่อถือได้ — ตอนนี้ให้ reliability_low ลดระดับสีลง ห้ามขึ้นเขียวถ้าความน่าเชื่อถือต่ำ
    if is_fallback:
        status_color = RED
    elif prob_up >= 70:
        status_color = AMBER if reliability_low else GREEN
    elif prob_up >= 50:
        status_color = AMBER
    else:
        status_color = RED
 
    direction_th = "ขาขึ้น" if prob_up >= 50 else "ขาลง"
    dir_prob = prob_up if prob_up >= 50 else down_prob  # ความน่าจะเป็นของ "ทิศทางที่ทำนาย" ใช้ในประโยคอธิบาย
 
    st.markdown(f"""<div style="margin-bottom:16px;">
<div style="font-size:15px; color:{MUTED}; margin-bottom:2px;">Home / Module 4 / AI Prediction</div>
<h2 style="margin:0; font-size:26px; font-weight:bold; color:#F8FAFC; letter-spacing:0.5px;">AI PREDICTION</h2>
</div>""", unsafe_allow_html=True)
 
    # ============================================================
    # 1) OVERVIEW — แถบ KPI ใหญ่ ตอบคำถาม "สรุปแล้วตัวเลขคืออะไร"
    # ============================================================
    k1, k2, k3, k4, k5 = st.columns(5)
 
    with k1:
        st.markdown(_kpi_card(
            "PRICE", f"{ctx.current_price:,.2f}",
            _kpi_sub(f"{ctx.change_val:+.2f} ({ctx.change_pct:+.2f}%) {ctx.arrow_sign}", ctx.change_color, bold=True)
        ), unsafe_allow_html=True)
 
    with k2:
        st.markdown(_kpi_card(
            "DIRECTION (10D)", direction_th, _kpi_sub("10 Trading Days"), value_color=status_color
        ), unsafe_allow_html=True)
 
    with k3:
        st.markdown(_kpi_card(
            "PROBABILITY", f"{prob_up:.0f}%", _kpi_sub(f"Down: {down_prob:.0f}%"), value_color=status_color
        ), unsafe_allow_html=True)
 
    with k4:
        st.markdown(_kpi_card(
            "SCORE", f'{ai_score}<span style="font-size:18px; color:{MUTED};">/100</span>',
            _kpi_sub("Prediction Score"), value_color=status_color
        ), unsafe_allow_html=True)
 
    with k5:
        st.markdown(_kpi_card(
            "RECOMMENDATION", signal, value_color=status_color, value_size=24, border=status_color
        ), unsafe_allow_html=True)
 
    st.markdown("<div style='margin-top:24px;'></div>", unsafe_allow_html=True)
 
    # ============================================================
    # 2) PREDICTION — 2 กล่อง: ซ้าย = gauge % ใหญ่ / ขวา = กล่องคำอธิบาย (สีอ่อนกว่า)
    # ============================================================
    st.markdown(_section_title("📈 PREDICTION"), unsafe_allow_html=True)
 
    _t = min(1, max(0, prob_up / 100))
    _gx = 50 - 40 * np.cos(np.pi * _t)
    _gy = 50 - 40 * np.sin(np.pi * _t)
 
    warn_line = ""
    if is_fallback:
        # [High FIX] แยกข้อความเตือนกรณี fallback (ไม่ได้เทรนโมเดลจริงเลย) ออกจากกรณี reliability_low ทั่วไป
        # เพราะเป็นคนละปัญหา — ตรงนี้คือ "ไม่มีข้อมูลพอจะเทรน" ไม่ใช่ "เทรนแล้วแต่แม่นยำต่ำ"
        warn_line = (
            f'<div style="font-size:15px; color:{RED}; background:rgba(239,68,68,0.1); border:1px solid {RED}; '
            f'border-radius:8px; padding:10px 14px; margin-top:14px; line-height:1.55;">'
            f'⚠ ข้อมูลย้อนหลังของหุ้นนี้ไม่พอสำหรับการเทรนโมเดล ตัวเลขที่แสดงเป็น<b>ค่าจำลองคงที่</b> ไม่ใช่ผลจากโมเดลจริง — '
            f'โปรดอย่าใช้ตัวเลขนี้ประกอบการตัดสินใจ</div>'
        )
    elif reliability_low:
        warn_line = (
            f'<div style="font-size:15px; color:{RED}; background:rgba(239,68,68,0.1); border:1px solid {RED}; '
            f'border-radius:8px; padding:10px 14px; margin-top:14px; line-height:1.55;">'
            f'⚠ ความแม่นยำของโมเดลต่ำกว่าหรือใกล้เคียงเกณฑ์เปรียบเทียบ (baseline) สำหรับหุ้นตัวนี้ — ควรใช้ผลทำนายนี้ด้วยความระมัดระวังเป็นพิเศษ</div>'
        )
 
    st.markdown(f"""<div style="background-color:#0F172A; border:1px solid #1E293B; border-top:none; border-radius:0 0 12px 12px; padding:16px;">
<div style="display:flex; gap:16px; flex-wrap:wrap; a
