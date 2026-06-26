"""
云端实时买卖点推送 — GitHub Actions 每30分钟执行
纯技术信号，只管买卖点，不涉及本金金额

数据源: baostock (主) → 腾讯行情 (备)
推送: PushPlus → 微信
"""
import os, sys, json, logging
import numpy as np, pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "f3fb5c092ba34785b6857bb45d23d4fa")
PUSHPLUS_URL = "http://www.pushplus.plus/send"

STOCKS = {"000933": "神火", "002497": "雅化", "000960": "锡业", "000893": "亚钾"}

# ====== Push ======
def push_msg(title, content):
    import requests
    try:
        r = requests.post(PUSHPLUS_URL, json={
            "token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"
        }, timeout=10)
        if r.json().get("code") == 200:
            logger.info(f"Push OK: {title}"); return True
        logger.error(f"Push FAIL: {r.text}"); return False
    except Exception as e:
        logger.error(f"Push ERROR: {e}"); return False

# ====== Data ======
def fetch_baostock(code):
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6","9")) else "sz."
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
        df = pd.DataFrame(rows, columns=["date","open","high","low","close","volume","amount"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True); df.reset_index(drop=True, inplace=True)
        return df
    except Exception as e:
        try: bs.logout()
        except: pass
        return pd.DataFrame()

def fetch_tencent(code):
    import requests
    try:
        prefix = "sh" if code.startswith(("6","9")) else "sz"
        r = requests.get(
            "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get",
            params={"param": f"{prefix}{code},day,,,60,qfq"},
            headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        data = r.json()
        if data.get("code") != 0:
            return pd.DataFrame()
        klines = data.get("data", {}).get(prefix + code, {}).get("qfqday") or []
        rows = [{"date": k[0], "open": float(k[1]), "close": float(k[2]),
                 "high": float(k[3]), "low": float(k[4]), "volume": float(k[5])} for k in klines]
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df.sort_values("date", inplace=True); df.reset_index(drop=True, inplace=True)
        return df
    except:
        return pd.DataFrame()

def fetch_data(code):
    df = fetch_baostock(code)
    if not df.empty and len(df) >= 20:
        return df
    return fetch_tencent(code)

# ====== Signal ======
def calc_signal(df, code):
    if len(df) < 20:
        return None
    close = df["close"].values; volume = df["volume"].values; n = len(close)

    # MA
    ma5 = np.mean(close[-5:]); ma10 = np.mean(close[-10:]); ma20 = np.mean(close[-20:])
    prev_ma5 = np.mean(close[-6:-1]); prev_ma10 = np.mean(close[-11:-1])
    golden = (ma5 > ma10) and (prev_ma5 <= prev_ma10)
    dead = (ma5 < ma10) and (prev_ma5 >= prev_ma10)
    ma_bull = ma5 > ma10 > ma20

    # RSI(14)
    d = np.diff(close[-15:])
    g = np.mean(d[d>0]) if np.any(d>0) else 0
    l = -np.mean(d[d<0]) if np.any(d<0) else 1e-9
    rsi = 100 - 100/(1+g/l) if l>0 else 50

    # Volume
    v5 = np.mean(volume[-5:]); v20 = np.mean(volume[-20:])
    vol_surge = v5 > v20 * 1.3

    sig = "HOLD"; reason = ""

    if code == "000933":  # 神火
        if golden and ma_bull:
            sig = "BUY"; reason = f"金叉+MA多头 RSI{int(rsi)}"
        elif dead and rsi > 60:
            sig = "SELL"; reason = f"死叉 RSI{int(rsi)}"
    elif code == "002497":  # 雅化
        if golden and (vol_surge or ma_bull):
            sig = "BUY"; reason = f"{'金叉+放量' if vol_surge else '金叉+MA多头'} RSI{int(rsi)}"
        elif dead:
            sig = "SELL"; reason = f"死叉 RSI{int(rsi)}"
    elif code == "000960":  # 锡业
        if golden and (vol_surge or ma_bull):
            sig = "BUY"; reason = f"{'金叉+放量' if vol_surge else '金叉+MA多头'} RSI{int(rsi)}"
        elif dead and rsi > 65:
            sig = "SELL"; reason = f"死叉 RSI{int(rsi)}"
    elif code == "000893":  # 亚钾
        if golden and ma_bull:
            sig = "BUY"; reason = f"金叉+MA多头 RSI{int(rsi)}"
        elif dead and rsi > 60:
            sig = "SELL"; reason = f"死叉 RSI{int(rsi)}"

    if rsi >= 80 and sig == "BUY":
        sig = "HOLD"; reason += " [RSI超买否决]"

    return {"code":code,"name":STOCKS[code],"signal":sig,"close":round(float(close[-1]),2),
            "rsi":round(rsi,1),"reason":reason or "无信号"}

# ====== Main ======
def main():
    logger.info(f"Scan start @ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results = []
    for code, name in STOCKS.items():
        df = fetch_data(code)
        if df.empty or len(df) < 20:
            logger.warning(f"  {code} {name}: NO DATA")
            results.append({"code":code,"name":name,"signal":"HOLD","close":0,"reason":"无数据"})
            continue
        s = calc_signal(df, code)
        if s:
            results.append(s)
            flag = ">>" if s["signal"] != "HOLD" else "  "
            logger.info(f"  {flag} {s['signal']:4s} {code} {name} @ {s['close']} | {s['reason']} | RSI={s['rsi']}")

    buy = [s for s in results if s["signal"] == "BUY"]
    sell = [s for s in results if s["signal"] == "SELL"]

    if not buy and not sell:
        logger.info("No trade signals — skip push")
        return results

    now_str = datetime.now().strftime("%m/%d %H:%M")
    rows = ""
    for s in buy:
        rows += f'<tr style="background:#3d1515"><td>🔴<b>BUY</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#e74c3c"><b>{s["close"]}</b></td><td>{s["reason"]}</td><td>RSI{s["rsi"]}</td></tr>'
    for s in sell:
        rows += f'<tr style="background:#153d15"><td>🟢<b>SELL</b></td><td><b>{s["code"]}</b></td><td>{s["name"]}</td><td style="color:#27ae60"><b>{s["close"]}</b></td><td>{s["reason"]}</td><td>RSI{s["rsi"]}</td></tr>'

    title = f"📊 {'买入'+str(len(buy))+'只 ' if buy else ''}{'卖出'+str(len(sell))+'只 ' if sell else ''}{now_str}"
    content = f"""
<div style="background:#1a1a2e;color:#eee;padding:15px;border-radius:10px;font-family:Arial">
<h2>📊 15min买卖点 — {now_str}</h2>
<table style="width:100%;color:#eee;border-collapse:collapse;margin-top:10px;font-size:15px">
<tr style="border-bottom:2px solid #444"><th></th><th>代码</th><th>股票</th><th>价格</th><th>信号</th><th>RSI</th></tr>
{rows}
</table>
<p style="color:#888;font-size:11px;margin-top:10px">仅提供买卖点参考 · 不构成投资建议 · 本地策略信号</p>
</div>
"""
    push_msg(title, content)
    logger.info(f"PUSHED: {len(buy)}B {len(sell)}S")
    return results

if __name__ == "__main__":
    main()
