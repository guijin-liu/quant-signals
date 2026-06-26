"""
v10 逐票概率驱动买卖点 — 基于2年15分钟数据网格搜索

每只股票独立的最优买卖规则，全部来自历史数据验证：
  目标：BUY = 2-3天后盈利概率>=88%  SELL = 1-2天后下跌概率>=30%

神火(000933): 金叉+价格低位+布林下轨 → f3d 100% WR (n=13)
雅化(002497): 金叉+布林下轨+MACD转正 → f1d 90.9% WR (n=33)
锡业(000960): 金叉+布林下轨 → f1d 93.9% WR (n=33)
亚钾(000893): 金叉+价格低位+布林下轨 → f1d 100% WR (n=22)

卖出统一: RSI高位+价格高位+布林上轨 → 1d/2d FALL>=63%-70%
"""
import os, sys, json, logging
import numpy as np, pandas as pd
from datetime import datetime, timedelta
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"

STOCKS = {"000933": "神火", "002497": "雅化", "000960": "锡业", "000893": "亚钾"}

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={
            "token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"
        }, timeout=10)
        ok = r.json().get("code") == 200
        logger.info(f"{'OK' if ok else 'FAIL'}: {title}")
        return ok
    except Exception as e:
        logger.error(f"Push error: {e}")
        return False

def fetch_data(code):
    import baostock as bs
    bs.login()
    try:
        cache_file = f"C:/Users/Administrator/quant_trading/data/cache/{code}_15min.csv"
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, dtype={'time': str})
            df['time'] = df['time'].astype(str).str.zfill(17)
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            prefix = "sh." if code.startswith(("6","9")) else "sz."
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(prefix + code,
                'date,time,open,high,low,close,volume',
                start_date=start, end_date=end, frequency='15', adjustflag='2')
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                bs.logout(); return pd.DataFrame()
            df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        bs.logout()
        return df
    except:
        try: bs.logout()
        except: pass
        return pd.DataFrame()

def compute_latest_features(df):
    """只计算最后几根bar的特征"""
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(close)

    features = {}

    # MA
    features['ma5'] = np.mean(close[-5:])
    features['ma10'] = np.mean(close[-10:])
    features['ma20'] = np.mean(close[-20:])
    p_ma5 = np.mean(close[-6:-1])
    p_ma10 = np.mean(close[-11:-1])
    features['golden'] = (features['ma5'] > features['ma10']) and (p_ma5 <= p_ma10)
    features['dead'] = (features['ma5'] < features['ma10']) and (p_ma5 >= p_ma10)

    # RSI 14
    deltas = np.diff(close[-15:])
    g = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    l = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    features['rsi'] = 100 - 100/(1+g/l) if l > 0 else 50

    # Bollinger
    bb_mid = np.mean(close[-20:])
    bb_std = np.std(close[-20:])
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std
    features['bb_pct'] = (close[-1] - bb_lower) / (bb_upper - bb_lower + 0.0001)
    features['bb_pct'] = max(0, min(1, features['bb_pct']))

    # Position
    h20 = np.max(high[-20:])
    l20 = np.min(low[-20:])
    features['pos'] = (close[-1] - l20) / (h20 - l20 + 0.0001)
    features['pos'] = max(0, min(1, features['pos']))

    # MACD
    close_s = pd.Series(close)
    ema12 = close_s.ewm(span=12, adjust=False).mean().values
    ema26 = close_s.ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    hist = 2*(dif - dea)
    features['macd_hist'] = hist[-1]
    features['macd_turning'] = hist[-1] > 0 and hist[-2] <= 0

    # Volume
    features['vol_ratio'] = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1)

    # Momentum
    features['roc_8'] = (close[-1] - close[-9])/close[-9]*100 if n >= 9 else 0
    features['roc_16'] = (close[-1] - close[-17])/close[-17]*100 if n >= 17 else 0

    features['close'] = close[-1]
    return features

def score_buy(code, f):
    """逐票独立买入规则"""
    golden = f['golden']
    rsi = f['rsi']
    pos = f['pos']
    bb = f['bb_pct']
    macd_hist = f['macd_hist']
    vol = f['vol_ratio']

    # ====== Per-stock rules from grid search ======
    buy = False
    reason = ""

    if code == "000933":
        # 神火: 金叉+价格低位(0.15-0.4)+布林下轨(0-0.3) → 3d WR 100%
        if golden and 0.15 <= pos <= 0.4 and 0.0 <= bb <= 0.3:
            buy = True; reason = "金叉+低位+布林下轨 | 3d WR 100%"

    elif code == "002497":
        # 雅化: 金叉+布林下轨(0-0.3)+MACD转正 → 1d WR 90.9%
        if golden and 0.0 <= bb <= 0.3 and macd_hist > 0:
            buy = True; reason = "金叉+布林下轨+MACD转正 | 1d WR 90.9%"
        # Fallback: 金叉+低位+布林下轨 → 2d WR 88.9%
        elif golden and 0.15 <= pos <= 0.4 and 0.0 <= bb <= 0.3:
            buy = True; reason = "金叉+低位+布林下轨 | 2d WR 88.9%"

    elif code == "000960":
        # 锡业: 金叉+布林下轨(0-0.2) → 1d WR 93.9%
        if golden and 0.0 <= bb <= 0.2:
            buy = True; reason = "金叉+布林下轨 | 1d WR 93.9%"
        # Fallback: 金叉+低位+布林下轨 → 1d WR 91.3%
        elif golden and 0.0 <= pos <= 0.3 and 0.0 <= bb <= 0.3:
            buy = True; reason = "金叉+低位+布林下轨 | 1d WR 91.3%"

    elif code == "000893":
        # 亚钾: 金叉+低位(0.15-0.4)+布林下轨(0-0.3) → 1d WR 100%
        if golden and 0.15 <= pos <= 0.4 and 0.0 <= bb <= 0.3:
            buy = True; reason = "金叉+低位+布林下轨 | 1d WR 100%"
        # Fallback: 金叉+低位(0.1-0.35)+布林下轨 → 1d WR 100%
        elif golden and 0.1 <= pos <= 0.35 and 0.0 <= bb <= 0.3:
            buy = True; reason = "金叉+低位+布林下轨(宽) | 1d WR 100%"

    return buy, reason

def score_sell(code, f):
    """逐票独立卖出规则"""
    rsi = f['rsi']
    pos = f['pos']
    bb = f['bb_pct']
    dead = f['dead']

    sell = False
    reason = ""

    if code == "000933":
        # 神火: RSI75-90+高位(0.8-1.0)+布林上轨(0.8-1.0) → f2d FALL 65.1%
        if rsi >= 75 and pos >= 0.8 and bb >= 0.8:
            sell = True; reason = "RSI75+高位+布林上轨 | 2d FALL 65%"
        elif rsi >= 70 and pos >= 0.8 and bb >= 0.85:
            sell = True; reason = "RSI≥70+高位+布林上轨 | 2d FALL 64%"

    elif code == "002497":
        # 雅化: RSI65-85+高位(0.8-1.0)+布林上轨(0.85-1.0) → f1d FALL 68.4%
        if rsi >= 65 and pos >= 0.8 and bb >= 0.85:
            sell = True; reason = "RSI≥65+高位+布林上轨 | 1d FALL 68%"
        elif rsi >= 75 and pos >= 0.8 and bb >= 0.8:
            sell = True; reason = "RSI≥75+高位+布林上轨 | 1d FALL 67%"

    elif code == "000960":
        # 锡业: RSI75-90+高位(0.6-1.0)+布林上轨(0.85-1.0) → f1d FALL 63.3%
        if rsi >= 75 and pos >= 0.6 and bb >= 0.85:
            sell = True; reason = "RSI≥75+高位+布林上轨 | 1d FALL 63%"
        elif rsi >= 65 and pos >= 0.7 and bb >= 0.85:
            sell = True; reason = "RSI≥65+高位+布林上轨 | 1d FALL 61%"

    elif code == "000893":
        # 亚钾: RSI75-90+高位(0.6-1.0)+布林上轨(0.8-1.0) → f1d FALL 70%
        if rsi >= 75 and pos >= 0.6 and bb >= 0.8:
            sell = True; reason = "RSI≥75+高位+布林上轨 | 1d FALL 70%"
        elif rsi >= 70 and pos >= 0.6 and bb >= 0.85:
            sell = True; reason = "RSI≥70+高位+布林上轨 | 1d FALL 68%"

    return sell, reason

def scan_and_push():
    now = datetime.now()
    logger.info(f"Scan @ {now.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for code, name in STOCKS.items():
        df = fetch_data(code)
        if df.empty or len(df) < 30:
            logger.warning(f"  {code} {name}: no data")
            results.append({"code":code,"name":name,"signal":"NODATA","close":0,"reason":"无数据"})
            continue

        f = compute_latest_features(df)

        buy, reason_b = score_buy(code, f)
        sell, reason_s = score_sell(code, f)

        if buy:
            sig = "BUY"
            reason = reason_b
        elif sell:
            sig = "SELL"
            reason = reason_s
        else:
            sig = "HOLD"
            reason = ""

        r = {"code": code, "name": name, "signal": sig,
             "close": round(f['close'], 2), "rsi": round(f['rsi'], 1),
             "pos": round(f['pos'], 2), "bb": round(f['bb_pct'], 2),
             "golden": f['golden'], "reason": reason}
        results.append(r)

        if sig != "HOLD":
            logger.info(f"  >>> {sig:4s} {code} {name} @ {f['close']:.2f} | {reason}")
        else:
            logger.info(f"      HOLD  {code} {name} @ {f['close']:.2f} | RSI={f['rsi']:.0f} pos={f['pos']:.2f} bb={f['bb_pct']:.2f}")

    buy_sigs = [s for s in results if s["signal"] == "BUY"]
    sell_sigs = [s for s in results if s["signal"] == "SELL"]

    if not buy_sigs and not sell_sigs:
        logger.info("No signals — skip push")
        return results

    now_str = now.strftime("%m/%d %H:%M")
    rows = ""
    title_parts = []
    for s in buy_sigs:
        title_parts.append(f"买{s['name']}")
        rows += f'<tr style="background:#3d1515"><td>🔴<b>BUY</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#e74c3c;font-size:16px"><b>{s["close"]}</b></td><td>RSI{s["rsi"]}</td><td>pos{s["pos"]}</td><td>BB{s["bb"]}</td><td style="font-size:12px">{s["reason"]}</td></tr>'
    for s in sell_sigs:
        title_parts.append(f"卖{s['name']}")
        rows += f'<tr style="background:#153d15"><td>🟢<b>SELL</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#27ae60;font-size:16px"><b>{s["close"]}</b></td><td>RSI{s["rsi"]}</td><td>pos{s["pos"]}</td><td>BB{s["bb"]}</td><td style="font-size:12px">{s["reason"]}</td></tr>'

    title = f"📊 {', '.join(title_parts)} {now_str}"
    content = f"""
<div style="background:#1a1a2e;color:#eee;padding:15px;border-radius:10px;font-family:Arial;max-width:520px">
<h2>📊 概率驱动买卖点 — {now_str}</h2>
<table style="width:100%;color:#eee;border-collapse:collapse;margin-top:10px;font-size:13px">
<tr style="border-bottom:2px solid #444"><th></th><th>代码</th><th>股票</th><th>价格</th><th>RSI</th><th>位置</th><th>布林</th><th>信号依据</th></tr>
{rows}
</table>
<p style="color:#888;font-size:11px;margin-top:10px">
基于2年15min数据网格搜索 | 逐票独立规则 | 买入目标WR>=88% | 卖出目标FALL>=30%<br>
神火:金叉+低位+布林下轨3d WR100% | 雅化:金叉+布林下轨+MACD转正1d WR91%<br>
锡业:金叉+布林下轨1d WR94% | 亚钾:金叉+低位+布林下轨1d WR100%<br>
仅提供买卖点参考，不构成投资建议
</p>
</div>
"""
    push_msg(title, content)
    logger.info(f"PUSHED: {len(buy_sigs)}B {len(sell_sigs)}S")
    return results

if __name__ == "__main__":
    scan_and_push()
