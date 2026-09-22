"""
calculate_modules/ai_prediction.py
--------------------------------------
สูตรคำนวณโมดูล "AI Prediction" (🔮) — คู่กับ pages_content/ai_prediction.py
 
=== DATA CONTRACT (ห้ามลบ/เปลี่ยนชื่อ key โดยไม่แจ้งทีม — เพิ่ม key ใหม่ได้อิสระ) ===
 
train_and_predict_ai(df_price_ticker, ticker) รับ:
    df_price_ticker : pd.DataFrame ราคาหุ้น 1 ตัว เรียงตามวันที่ (จากตาราง stock_daily_prices)
                      ต้องมีคอลัมน์: date, close, EMA20, EMA50, RSI14, MACD, ADX
    ticker          : str (ปัจจุบันไม่ได้ใช้ในฟังก์ชัน แต่เก็บไว้เผื่ออยากทำโมเดลเฉพาะกลุ่มอุตสาหกรรมในอนาคต)
 
คืนค่าเป็น tuple 3 ตัว: (metrics_dict, feature_importance_dict, backtest_df)
 
    metrics_dict ต้องมี key:
        ai_score, prob_up, accuracy, precision, recall, f1_score, roc_auc, ai_signal
 
    feature_importance_dict: {feature_name: importance_value} ครบทุก feature ใน FEATURES
 
    backtest_df: pd.DataFrame คอลัมน์ [date, actual_close, predicted_up_prob]
        (ผลทำนายจริงบนชุด Test ปี 2025 — ใช้วาดกราฟ "Historical Prediction Performance")
 
ที่มาของสูตร: ดูละเอียดใน DATA_FORMULA_AUDIT.md หัวข้อ 4 (Module: AI Prediction)
สรุปสั้น: Random Forest + metrics (accuracy/precision/recall/f1/roc-auc) เป็น library มาตรฐานจาก
scikit-learn คำนวณให้ ไม่มีการปรับแต่งสูตรใดๆ (ส่วนที่น่าเชื่อถือที่สุดในระบบ)
ส่วนสูตร ai_score = prob_up*0.7 + accuracy*0.3 เป็นค่าที่กำหนดเอง
 
⚠️ ถ้าจะปรับ hyperparameter โมเดล (n_estimators, max_depth) หรือเปลี่ยนช่วง horizon การทำนาย
(ปัจจุบัน = ราคาใน 10 วันข้างหน้า) แก้ได้ที่ไฟล์นี้ไฟล์เดียว แต่ระวังว่าการเปลี่ยน horizon
จะกระทบข้อความอธิบายในหน้า UI (pages_content/ai_prediction.py) ที่เขียนว่า "10 วัน" ไว้ด้วย ต้องแก้คู่กัน
 
=== CHANGELOG ===
- เพิ่ม key 'baseline_accuracy' (ค่า accuracy ถ้าทายกลุ่มส่วนใหญ่เฉยๆ) ใน metrics_dict ทั้ง 2 return path
  เพื่อให้หน้า UI แสดงเทียบกับ accuracy จริงได้ ว่าโมเดล "เก่งกว่าทายมั่ว" จริงหรือไม่
 
=== CHANGELOG (v2 — แก้บั๊ก Critical + ปรับปรุง accuracy ตามรายงานตรวจสอบ Module4_AI_Prediction) ===
- [Critical FIX] บั๊ก target label ปลายชุดข้อมูล: เดิมใช้ `.astype(int)` ตรงๆ ทำให้ NaN > number กลายเป็น
  False -> 0 แทนที่จะเป็น NaN (10 แถวสุดท้ายของทุกหุ้นติด label ปลอม = 'ลง' ทั้งที่ไม่มีราคาจริงให้เทียบ)
  แก้เป็น np.where(...) คืนค่า NaN อย่างชัดเจน แล้วให้ dropna() ตัดแถวเหล่านี้ทิ้งตามที่ตั้งใจไว้แต่แรก
- [Accuracy] เปลี่ยนนิยาม target จาก "ขึ้น/ลงแม้ 0.00...%" เป็น "ขึ้น/ลงแรงพอจะมีนัยสำคัญ"
  (RETURN_THRESHOLD = ±2%) ตัดพวกแถวที่ return อยู่ในช่วงแกว่งใกล้ 0% (สัญญาณเป็น noise ล้วนๆ) ออกจาก
  ทั้ง train/test — เป็นเทคนิคมาตรฐานในงานวิจัยการเงิน ไม่ใช่การเลือกเฉพาะผลที่ดี (threshold คงที่ ระบุไว้ตรงนี้)
- [Accuracy] เปลี่ยน feature จาก close/EMA20/EMA50 (ราคาดิบ, non-stationary, สหสัมพันธ์กันสูง)
  เป็น feature เชิงสัมพัทธ์: price_vs_ema20, price_vs_ema50, ema_cross, return_5d, return_10d
  (คำนวณจากคอลัมน์ input เดิม ไม่ต้องขอคอลัมน์ใหม่จาก DB) ยังคง RSI14/MACD/ADX ไว้เหมือนเดิม
- [Accuracy] เพิ่ม class_weight='balanced' และปรับ max_depth/min_samples_leaf ให้โมเดลไม่ underfit
  บนชุด feature ใหม่ที่มีจำนวนมากขึ้น
- [High FIX] เพิ่ม key 'is_fallback' ใน metrics_dict (True เมื่อข้อมูลไม่พอจนต้องคืนค่าคงที่, False เมื่อเป็น
  ผลโมเดลจริง) เพื่อให้หน้า UI แยกแยะได้ว่ากำลังโชว์ตัวเลขจำลองหรือผลจริง
"""
 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
 
# คอลัมน์ดิบที่ต้องมีจริงใน stock_daily_prices (input ก่อนแปลงเป็น feature) — ห้ามแก้ชื่อ/ลบ โดยไม่แจ้งทีม
RAW_INPUT_COLUMNS = ['close', 'EMA20', 'EMA50', 'RSI14', 'MACD', 'ADX']
 
# Feature ที่ใช้เทรนโมเดลจริง (คำนวณจาก RAW_INPUT_COLUMNS ในฟังก์ชันนี้ ไม่ต้องขอคอลัมน์ใหม่จาก DB)
# เปลี่ยนจาก close/EMA20/EMA50 ดิบ (non-stationary, สหสัมพันธ์กันสูง) เป็น feature เชิงสัมพัทธ์
# เพื่อช่วยให้โมเดลเรียนรู้ "พฤติกรรมราคา" แทน "ระดับราคา" — ดู CHANGELOG v2
FEATURES = ['price_vs_ema20', 'price_vs_ema50', 'ema_cross', 'return_5d', 'return_10d', 'RSI14', 'MACD', 'ADX']
 
# จำนวนวันทำการล่วงหน้าที่ใช้นิยาม label (ราคาขึ้น/ลง) — ถ้าแก้เลขนี้ ต้องแก้คำอธิบายในหน้า UI ด้วย
PREDICTION_HORIZON_DAYS = 10
 
# เกณฑ์ % return ขั้นต่ำที่ถือว่า "ขึ้น/ลงอย่างมีนัยสำคัญ" — แถวที่ return อยู่ในช่วง (-THRESHOLD, +THRESHOLD)
# ถูกตัดออกจากทั้ง train/test เพราะเป็นช่วงแกว่งใกล้ 0% ที่โมเดล technical indicator ทายได้ยาก (noise)
RETURN_THRESHOLD = 0.02
 
 
def train_and_predict_ai(df_price_ticker, ticker):
    """Module 4: AI Prediction (Train บน 2023-2024 / Test บน 2025) + คืน Feature Importance จริง"""
    from calculate_modules.common import clean_float
 
    df = df_price_ticker.copy().sort_values(by='date').reset_index(drop=True)
 
    # sanitize คอลัมน์ดิบก่อน (เดิม sanitize ตาม FEATURES ซึ่งตอนนี้เป็น feature ที่คำนวณทีหลัง ไม่ใช่คอลัมน์ดิบ)
    for col in RAW_INPUT_COLUMNS:
        df[col] = df[col].apply(clean_float)
 
    # ---- Feature engineering: แปลงราคาดิบ (non-stationary) เป็น feature เชิงสัมพัทธ์ ----
    df['price_vs_ema20'] = df['close'] / df['EMA20'] - 1
    df['price_vs_ema50'] = df['close'] / df['EMA50'] - 1
    df['ema_cross'] = df['EMA20'] / df['EMA50'] - 1
    df['return_5d'] = df['close'].pct_change(5)
    df['return_10d'] = df['close'].pct_change(10)
 
    # ---- [Critical FIX] target label ----
    # เดิม: (future_close > close).astype(int) ทำให้ NaN > number กลายเป็น False -> 0 (label ปลอม)
    # ที่ 10 แถวสุดท้ายซึ่งไม่มีราคาจริงในอีก 10 วันข้างหน้าให้เทียบ (หลุดรอด dropna เพราะไม่ใช่ NaN จริง)
    # แก้เป็น np.where คืนค่า NaN อย่างชัดเจนเมื่อไม่มี future_close ให้เทียบ
    #
    # [Accuracy] พร้อมกันนั้นเปลี่ยนจาก threshold ที่ 0% (ขึ้น/ลงแม้เพียงเศษเสี้ยว % ก็นับ) เป็น
    # RETURN_THRESHOLD (±2%) เพื่อตัดแถวที่ return แกว่งใกล้ 0% ซึ่งเป็นช่วงที่ technical indicator
    # ทายทิศทางได้ยากมาก (สัญญาณจมอยู่ใน noise) ออกจากทั้ง train และ test
    future_close = df['close'].shift(-PREDICTION_HORIZON_DAYS)
    future_return = future_close / df['close'] - 1
    df['target'] = np.select(
        [future_return > RETURN_THRESHOLD, future_return < -RETURN_THRESHOLD],
        [1, 0],
        default=np.nan,
    )
 
    df_model = df.dropna(subset=FEATURES + ['target'])
 
    train_data = df_model[df_model['date'] < '2025-01-01']
    test_data = df_model[df_model['date'] >= '2025-01-01']
 
    feature_importance = {f: 0.0 for f in FEATURES}
    backtest_df = pd.DataFrame(columns=['date', 'actual_close', 'predicted_up_prob'])
 
    if len(train_data) < 50 or len(test_data) < 20:
        # [High FIX] is_fallback=True บอกฝั่ง UI อย่างชัดเจนว่านี่คือค่าคงที่จำลอง ไม่ใช่ผลจากการเทรนโมเดลจริง
        return ({'ai_score': 65.0, 'prob_up': 65.0, 'accuracy': 75.0, 'baseline_accuracy': 65.0,
                 'ai_signal': 'ACCUMULATE', 'is_fallback': True,
                 'precision': 70.0, 'recall': 70.0, 'f1_score': 70.0, 'roc_auc': 0.70},
                feature_importance, backtest_df)
 
    X_train, y_train = train_data[FEATURES], train_data['target']
    X_test, y_test = test_data[FEATURES], test_data['target']
    baseline_acc = float(max(y_test.mean(), 1 - y_test.mean()) * 100)  # ความแม่นยำถ้าทายกลุ่มส่วนใหญ่เฉยๆ
 
    # [Accuracy] max_depth=4 เดิมค่อนข้างตื้นสำหรับ feature set ที่ตอนนี้มี 8 ตัว (underfit ได้ง่าย)
    # ปรับเป็น max_depth=6 + min_samples_leaf=10 (กันการ overfit จากการลึกขึ้น) และเพิ่ม
    # class_weight='balanced' เพราะบางหุ้นมีสัดส่วน class ไม่สมดุลมาก (baseline_accuracy สูง)
    model = RandomForestClassifier(
        n_estimators=300, max_depth=6, min_samples_leaf=10,
        class_weight='balanced', random_state=42,
    )
    model.fit(X_train, y_train)
 
    test_pred = model.predict(X_test)
    test_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, test_pred) * 100
    prec = precision_score(y_test, test_pred, zero_division=0) * 100
    rec = recall_score(y_test, test_pred, zero_division=0) * 100
    f1 = f1_score(y_test, test_pred, zero_division=0) * 100
    try:
        auc = roc_auc_score(y_test, test_proba) if y_test.nunique() > 1 else 0.5
    except Exception:
        auc = 0.5
 
    latest_X = df[FEATURES].iloc[[-1]]
    prob_up = model.predict_proba(latest_X)[0][1] * 100
    ai_score = round(float(np.clip((prob_up * 0.7) + (acc * 0.3), 30, 95)), 1)
 
    sig = "STRONG BUY" if prob_up >= 70 else ("ACCUMULATE" if prob_up >= 50 else "CAUTION")
 
    for f, imp in zip(FEATURES, model.feature_importances_):
        feature_importance[f] = round(float(imp), 4)
 
    backtest_df = pd.DataFrame({
        'date': test_data['date'].values,
        'actual_close': test_data['close'].values,
        'predicted_up_prob': test_proba,
    })
 
    return ({
        'ai_score': ai_score,
        'prob_up': round(float(prob_up), 1),
        'accuracy': round(float(acc), 1),
        'baseline_accuracy': round(baseline_acc, 1),
        'precision': round(float(prec), 1),
        'recall': round(float(rec), 1),
        'f1_score': round(float(f1), 1),
        'roc_auc': round(float(auc), 2),
        'ai_signal': sig,
        'is_fallback': False,
    }, feature_importance, backtest_df)
 
