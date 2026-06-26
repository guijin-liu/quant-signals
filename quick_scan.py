#!/usr/bin/env python
"""
盘前快速扫描 — 每天9:25执行（集合竞价结束）
推美股收盘 + 昨天A股收盘 + 持仓开盘预判
纯信号提示，不涉及本金
"""
import os, sys, json, logging
import numpy as np, pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"

STOCKS = {"000933": "神火", "002497": "雅化", "000960": "锡业", "000893": "亚钾"}

def push_msg(title, content):
    import requests
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

def fetch_one(code):
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6","9")) else "sz."
        end = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
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
    except:
        try: bs.logout()
        except: pass
        return pd.DataFrame()

def main():
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%Y-%m-%d %H:%M")
    weekday = ["周一","周二","周三","周四","周五","周六","周日"][now.weekday()]

    logger.info(f"Premarket scan {time_str} {weekday}")

    # ====== 扫描每只股票 ======
    rows = ""
    summary_buy = 0
    summary_sell = 0

    for code, name in STOCKS.items():
        df = fetch_one(code)
        if df.empty or len(df) < 30:
            rows += f'<tr><td>{code}</td><td>{name}</td><td>—</td><td>—</td><td>—</td><td style="color:#888">数据不可用</td></tr>'
            continue

        close = df["close"].values
        vol = df["volume"].values
        n = len(close)
        latest = close[-1]
        prev = close[-2] if n >= 2 else latest
        chg = (latest - prev) / prev * 100

        # 5/10/20 MA
        ma5  = np.mean(close[-5:])
        ma10 = np.mean(close[-10:])
        ma20 = np.mean(close[-20:])

        # RSI
        d = np.diff(close[-15:])
        g = np.mean(d[d>0]) if np.any(d>0) else 0
        l = -np.mean(d[d<0]) if np.any(d<0) else 1e-9
        rsi = 100 - 100/(1+g/l) if l>0 else 50

        # Trend
        pct5  = (close[-1] - close[-5]) / close[-5] * 100 if n >= 5 else 0
        pct20 = (close[-1] - close[-20]) / close[-20] * 100 if n >= 20 else 0

        # Volume trend
        v5  = np.mean(vol[-5:])
        v20 = np.mean(vol[-20:])
        vol_trend = v5 / v20 if v20 > 0 else 1

        # Signal
        prev_ma5  = np.mean(close[-6:-1]); prev_ma10 = np.mean(close[-11:-1])
        golden = (ma5 > ma10) and (prev_ma5 <= prev_ma10)
        dead   = (ma5 < ma10) and (prev_ma5 >= prev_ma10)
        ma_bull = ma5 > ma10 > ma20
        ma_bear = ma5 < ma10 < ma20

        # Pre-market outlook
        outlook = ""
        outlook_color = "#888"
        if golden and ma_bull:
            outlook = "🟢 金叉+MA多头，偏多"
            outlook_color = "#e74c3c"
            summary_buy += 1
        elif golden:
            outlook = "🟡 金叉信号，关注"
            outlook_color = "#f39c12"
            summary_buy += 1
        elif dead and ma_bear:
            outlook = "🔴 死叉+MA空头，偏空"
            outlook_color = "#27ae60"
            summary_sell += 1
        elif dead:
            outlook = "🟠 死叉信号，谨慎"
            outlook_color = "#e67e22"
            summary_sell += 1
        elif ma_bull and rsi < 70:
            outlook = "✅ MA多头，持仓"
            outlook_color = "#95a5a6"
        elif ma_bear and rsi < 30:
            outlook = "⚠️ 超卖区域，观望"
            outlook_color = "#f39c12"
        else:
            outlook = "➖ 震荡"
            outlook_color = "#888"

        chg_color = "#e74c3c" if chg > 0 else ("#27ae60" if chg < 0 else "#888")
        rsi_color = "#e74c3c" if rsi > 70 else ("#27ae60" if rsi < 30 else "#eee")

        rows += f'''
<tr>
<td><b>{code}</b></td><td>{name}</td>
<td style="color:#eee"><b>{latest:.2f}</b></td>
<td style="color:{chg_color}">{chg:+.2f}%</td>
<td style="color:{rsi_color}">{rsi:.0f}</td>
<td>{pct5:+.1f}%</td>
<td>{pct20:+.1f}%</td>
<td style="color:{outlook_color}">{outlook}</td>
</tr>'''

    # ====== Build message ======
    title = f"📅 盘前扫描 {date_str} {weekday}"

    content = f"""
<div style="background:#1a1a2e;color:#eee;padding:15px;border-radius:10px;font-family:Arial;max-width:500px">
<h2>📅 盘前扫描 — {date_str} {weekday}</h2>

<h3 style="color:#e74c3c">持仓盘前判断</h3>
<table style="width:100%;color:#eee;border-collapse:collapse;font-size:13px">
<tr style="border-bottom:2px solid #444">
<th>代码</th><th>名称</th><th>昨收</th><th>涨跌</th><th>RSI</th><th>5日</th><th>20日</th><th>预判</th>
</tr>{rows}
</table>

<p style="margin-top:15px;padding:10px;background:#222;border-radius:5px">
<b>买入信号 {summary_buy} 只 | 卖出信号 {summary_sell} 只</b><br>
<span style="color:#888;font-size:11px">基于MA金叉/死叉+多头排列+RSI，仅供参考</span>
</p>

<p style="color:#888;font-size:11px;margin-top:10px">
9:25集合竞价 | 每30分钟盘中扫描 | 不做投资建议
</p>
</div>
"""
    push_msg(title, content)
    logger.info(f"Premarket pushed: {summary_buy}B {summary_sell}S")

if __name__ == "__main__":
    main()
