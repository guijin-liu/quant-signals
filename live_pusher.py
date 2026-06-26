"""
本地实时信号推送 — 纯买卖点，不涉及本金/金额
多数据源自动切换: baostock -> akshare -> Tencent
用法: python live_pusher.py
"""
import logging, time, json, os
import numpy as np, pandas as pd
from datetime import datetime, timedelta
from push_notify import push_msg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

STOCKS = {"000933": "神火", "002497": "雅化", "000960": "锡业", "000893": "亚钾"}

def fetch_baostock(code):
    """baostock data source (most reliable)"""
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6", "9")) else "sz."
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=60)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(
            prefix + code, "date,open,high,low,close,volume,amount",
            start_date=start, end_date=end, frequency="d", adjustflag="2")
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume", "amount"])
        for c in ["open", "high", "low", "close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        try: bs.logout()
        except: pass
        logger.warning(f"  baostock {code} fail: {e}")
        return pd.DataFrame()

def fetch_tencent(code):
    """腾讯行情 as fallback"""
    import requests
    try:
        prefix = "sh" if code.startswith(("6", "9")) else "sz"
        r = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": prefix + code + ",day,,,60,qfq"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json()
        if data.get("code") != 0:
            return pd.DataFrame()
        klines = data.get("data", {}).get(prefix + code, {}).get("qfqday") or []
        if not klines:
            return pd.DataFrame()
        rows = [{"date": k[0], "open": float(k[1]), "close": float(k[2]),
                 "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])} for k in klines]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        logger.warning(f"  tencent {code} fail: {e}")
        return pd.DataFrame()

def fetch_kline(code):
    """Multi-source fallback"""
    # Try baostock first
    df = fetch_baostock(code)
    if not df.empty and len(df) >= 30:
        return df
    # Fallback to Tencent
    df = fetch_tencent(code)
    if not df.empty and len(df) >= 30:
        return df
    return pd.DataFrame()

def score_signal(df, code):
    """逐票买卖点评分 — 纯技术信号"""
    if len(df) < 20:
        return None
    close = df["close"].values
    volume = df["volume"].values
    n = len(close)
    latest = close[-1]

    # MA
    ma5 = np.mean(close[-5:]); ma10 = np.mean(close[-10:]); ma20 = np.mean(close[-20:])
    prev_ma5 = np.mean(close[-6:-1]); prev_ma10 = np.mean(close[-11:-1])
    golden_cross = (ma5 > ma10) and (prev_ma5 <= prev_ma10)
    dead_cross = (ma5 < ma10) and (prev_ma5 >= prev_ma10)
    ma_bull = ma5 > ma10 > ma20

    # RSI(14)
    deltas = np.diff(close[-15:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    rsi = 100 - 100/(1+gain/loss) if loss > 0 else 50

    # Volume surge
    vol5 = np.mean(volume[-5:]); vol20 = np.mean(volume[-20:])
    vol_surge = vol5 > vol20 * 1.3

    # Price trend (5-day)
    pct5 = (close[-1] - close[-5]) / close[-5] * 100 if n >= 5 else 0

    signal = "HOLD"; reason = ""

    if code == "000933":  # 神火
        if golden_cross and ma_bull and rsi < 70:
            signal = "BUY"; reason = f"金叉+MA多头+RSI{int(rsi)}"
        elif dead_cross and rsi > 60:
            signal = "SELL"; reason = f"死叉+RSI高位{int(rsi)}"
        elif golden_cross and vol_surge:
            signal = "BUY"; reason = "金叉+放量"
    elif code == "002497":  # 雅化
        if golden_cross and vol_surge:
            signal = "BUY"; reason = "金叉+放量(雅化最强)"
        elif golden_cross and ma_bull:
            signal = "BUY"; reason = "金叉+MA多头"
        elif dead_cross and rsi > 60:
            signal = "SELL"; reason = f"死叉信号"
    elif code == "000960":  # 锡业
        if golden_cross and vol_surge:
            signal = "BUY"; reason = "金叉+放量(锡业最强)"
        elif golden_cross and ma_bull:
            signal = "BUY"; reason = "金叉+MA多头"
        elif dead_cross and rsi > 65:
            signal = "SELL"; reason = f"死叉+RSI高位"
    elif code == "000893":  # 亚钾
        if golden_cross and ma_bull and rsi < 65:
            signal = "BUY"; reason = f"金叉+MA多头+RSI{int(rsi)}"
        elif golden_cross and vol_surge:
            signal = "BUY"; reason = "金叉+暴量(亚钾特色)"
        elif dead_cross and rsi > 60:
            signal = "SELL"; reason = f"死叉+RSI高位"

    if rsi >= 80 and signal == "BUY":
        signal = "HOLD"; reason += "[RSI超买否决]"

    score = round(min(0.5 + 0.15*int(golden_cross) + 0.10*int(ma_bull) + 0.10*int(vol_surge) + 0.05*int(rsi < 70 and rsi > 30), 1.0), 3)

    return {"code": code, "name": STOCKS[code], "signal": signal, "close": round(float(latest), 2),
            "rsi": round(rsi, 1), "golden_cross": golden_cross, "vol_surge": vol_surge,
            "pct5": round(pct5, 2), "reason": reason or "无信号", "score": score}

def push_scan():
    """扫描并推送 — 只有买卖信号才推"""
    now = datetime.now()
    logger.info(f"Scan @ {now.strftime('%m-%d %H:%M:%S')}")

    results = []
    for code, name in STOCKS.items():
        df = fetch_kline(code)
        if df.empty or len(df) < 20:
            logger.warning(f"  {code} {name}: NO DATA")
            results.append({"code": code, "name": name, "signal": "NODATA", "close": 0, "reason": "无数据"})
            continue
        s = score_signal(df, code)
        if s:
            results.append(s)
            flag = ">>>" if s["signal"] != "HOLD" else "   "
            logger.info(f"  {flag} {s['signal']:4s} {code} {name:4s} @ {s['close']} | {s['reason']} | RSI={s['rsi']}")
        else:
            results.append({"code": code, "name": name, "signal": "HOLD", "close": 0, "reason": "数据不足"})

    buy_signals = [s for s in results if s["signal"] == "BUY"]
    sell_signals = [s for s in results if s["signal"] == "SELL"]

    if not buy_signals and not sell_signals:
        logger.info("No trade signals — skip push")
        return results

    now_str = now.strftime("%m/%d %H:%M")
    rows = ""
    for s in buy_signals:
        rows += f'<tr style="background:#5a1a1a"><td>🔴<b>买入</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#e74c3c;font-size:18px"><b>{s["close"]}</b></td><td>{s["reason"]}</td><td>RSI{s["rsi"]}</td></tr>'
    for s in sell_signals:
        rows += f'<tr style="background:#1a5a1a"><td>🟢<b>卖出</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#27ae60;font-size:18px"><b>{s["close"]}</b></td><td>{s["reason"]}</td><td>RSI{s["rsi"]}</td></tr>'

    title_parts = []
    if buy_signals: title_parts.append(f"买入{len(buy_signals)}")
    if sell_signals: title_parts.append(f"卖出{len(sell_signals)}")
    title = f"📊 {' '.join(title_parts)} {now_str}"

    content = f"""
<div style="background:#1a1a2e;color:#eee;padding:15px;border-radius:10px;font-family:Arial">
<h2>📊 15min买卖点信号 — {now_str}</h2>
<table style="width:100%;color:#eee;border-collapse:collapse;margin-top:10px;font-size:15px">
<tr style="border-bottom:2px solid #444"><th></th><th>代码</th><th>名称</th><th>价格</th><th>信号理由</th><th>RSI</th></tr>
{rows}
</table>
<p style="margin-top:12px;color:#888;font-size:12px">基于3年15min数据挖掘 · 每30min自动扫描 · 纯技术信号</p>
</div>
"""
    push_msg(title, content)
    logger.info(f"PUSHED: {len(buy_signals)}B {len(sell_signals)}S")
    return results

if __name__ == "__main__":
    push_scan()
