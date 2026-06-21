"""
腾讯云函数部署包 — 量化信号自动推送
直接上传此文件到腾讯云函数即可

配置:
  运行环境: Python 3.9+
  执行方法: index.main_handler
  定时触发器: cron(0 0 1 ? * MON-FRI *)  # 周一到周五 9:00 (北京时间)
  环境变量: PUSHPLUS_TOKEN=f3fb5c092ba34785b6857bb45d23d4fa

免费额度: 100万次/月调用, 完全够用
"""

import os
import json
import logging
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# === 配置 ===
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "f3fb5c092ba34785b6857bb45d23d4fa")
PUSHPLUS_URL = "http://www.pushplus.plus/send"

STOCKS = {
    "000933": "神火股份",
    "002497": "雅化集团",
    "000960": "锡业股份",
    "000893": "亚钾国际",
}


def push_msg(title, content):
    """推送到微信"""
    try:
        r = requests.post(
            PUSHPLUS_URL,
            json={"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"},
            timeout=10,
        )
        result = r.json()
        if result.get("code") == 200:
            logger.info(f"推送成功: {title}")
            return True
        logger.error(f"推送失败: {result}")
        return False
    except Exception as e:
        logger.error(f"推送异常: {e}")
        return False


def push_signal_summary(signals):
    """推送信号汇总HTML"""
    now = datetime.now().strftime("%m-%d %H:%M")
    buy_count = sum(1 for s in signals if s.get("signal") == "BUY")

    rows = ""
    for s in signals:
        emoji = {"BUY": "买", "SELL": "卖", "HOLD": "持"}.get(s.get("signal", ""), "")
        rows += (
            "<tr><td><b>{}</b></td><td>{}</td>"
            "<td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>"
        ).format(emoji, s.get('code',''), s.get('name',''), '{:.2f}'.format(s.get('close',0)),
                 '{:.3f}'.format(s.get('score',0)), s.get('resonance',''))

    content = (
        "<div style='background:#1a1a2e;color:#eee;padding:15px;border-radius:10px'>"
        "<h2>量化信号 v7 — {}</h2>"
        "<table style='width:100%;color:#eee;border-collapse:collapse'>"
        "<tr style='border-bottom:1px solid #333'><th></th><th>代码</th><th>名称</th><th>价格</th><th>评分</th><th>信号理由</th></tr>"
        "{}"
        "</table>"
        "<p style='margin-top:12px'>买入: <b style='color:#e74c3c'>{}</b>只 | 策略: 逐票独立概率型</p>"
        "</div>"
    ).format(now, rows, buy_count)

    title = "信号 v7 | 买入{}只".format(buy_count) if buy_count > 0 else "信号 v7 | 无可买入"
    return push_msg(title, content)


def fetch_kline(symbol, freq="15", days=30):
    """用腾讯接口获取K线 (不依赖baostock)"""
    prefix = "sh" if symbol.startswith(("6", "9")) else "sz"

    try:
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{prefix}{symbol},day,,,{min(days*8, 400)},qfq"}

        r = requests.get(url, params=params, headers={
            "User-Agent": "Mozilla/5.0",
        }, timeout=15)
        data = r.json()

        if data.get("code") != 0:
            logger.warning(f"{symbol}: 腾讯接口返回异常")
            return pd.DataFrame()

        stock_key = f"{prefix}{symbol}"
        stock_data = data.get("data", {}).get(stock_key, {})
        klines = stock_data.get("qfqday") or stock_data.get("day") or []

        if not klines:
            return pd.DataFrame()

        rows = []
        for line in klines:
            rows.append({
                "datetime": line[0],
                "open": float(line[1]),
                "close": float(line[2]),
                "high": float(line[3]),
                "low": float(line[4]),
                "volume": float(line[5]),
            })

        df = pd.DataFrame(rows)
        df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime", inplace=True)
        df.reset_index(drop=True, inplace=True)
        df["pct_change"] = df["close"].pct_change() * 100

        return df

    except Exception as e:
        logger.error(f"获取{symbol}失败: {e}")
        return pd.DataFrame()


def score_signal(df, code):
    """v7 逐票独立概率策略 — 每只票用历史最优信号规则
    - 000933 神火: MACD金叉 → 5年27次, 胜率59%
    - 002497 雅化: MA多头+RSI30-65 → 放宽(原MA多+金叉只有2次/5年)
    - 000960 锡业: RSI<30超卖抄底 → 5年12次, 胜率91.7%
    - 000893 亚钾: MA多头+RSI30-65 → 5年200次, 胜率最高73%
    """
    if len(df) < 30:
        return {"signal": "HOLD", "close": 0, "score": 0, "resonance": 0}

    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    volume = df["volume"].values
    n = len(close)

    # 基础指标
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:]) if n >= 10 else ma5
    ma20 = np.mean(close[-20:]) if n >= 20 else ma10

    # RSI(14)
    deltas = np.diff(close[-15:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 50

    # MACD简易(EMA12-EMA26)
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().iloc[-1]
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().iloc[-1]
    prev_ema12 = pd.Series(close[:-1]).ewm(span=12, adjust=False).mean().iloc[-1] if n>12 else ema12
    prev_ema26 = pd.Series(close[:-1]).ewm(span=26, adjust=False).mean().iloc[-1] if n>26 else ema26
    macd_dif = ema12 - ema26
    prev_dif = prev_ema12 - prev_ema26
    dea = pd.Series([prev_dif,macd_dif]).ewm(span=9,adjust=False).mean().iloc[-1]
    golden_cross = (macd_dif > dea) and (prev_dif <= dea)

    # 放量
    vol_5 = np.mean(volume[-5:])
    vol_20 = np.mean(volume[-20:]) if n >= 20 else vol_5
    vol_surge = vol_5 > vol_20 * 2.0

    # 布林
    std20 = np.std(close[-20:]) if n >= 20 else np.std(close)
    boll_mid = ma20
    boll_pct_b = (close[-1] - (boll_mid - 2*std20)) / (4*std20 + 1e-9) if std20>0 else 0.5

    # 价格vs MA20
    pct_5d = (close[-1] - close[-6]) / close[-6] * 100 if n >= 6 else 0
    close_above_ma20 = close[-1] > ma20

    # ====== 逐票独立规则 ======
    buy=False; reasons=[]

    if code == "000933":  # 神火: MACD金叉最优(32次/5年,胜率59%)
        if golden_cross and rsi < 75 and boll_pct_b < 0.9:
            buy=True; reasons.append("MACD金叉")
        if vol_surge and golden_cross:
            reasons.append("金叉+放量(极强)")

    elif code == "002497":  # 雅化: MA多头+RSI30-65 (原金叉只有2次/5年)
        if ma5 > ma10 > ma20 and 30 < rsi < 65:
            buy=True; reasons.append("MA多头+RSI适中")
        if golden_cross:
            reasons.append("金叉(加分)")

    elif code == "000960":  # 锡业: RSI<30抄底(12次/5年,胜率91.7%)
        if rsi < 30:
            buy=True; reasons.append("RSI超卖抄底")
        if rsi < 35 and close_above_ma20:
            buy=True; reasons.append("RSI低+站上MA20")

    elif code == "000893":  # 亚钾: MA多头+RSI30-65(200次/5年,胜率73%持15天)
        if ma5 > ma10 > ma20 and 30 < rsi < 65:
            buy=True; reasons.append("MA多头+RSI适中")
        if not buy and ma5 > ma10 and rsi < 50 and boll_pct_b < 0.5:
            buy=True; reasons.append("MA短期多头+RSI低+BB中下")

    # 通用否决: RSI超买不买
    if rsi >= 80:
        buy=False; reasons.append("RSI超买否决")

    score = 0.5 + 0.1*len(reasons) + 0.1*(1 if golden_cross else 0) + 0.05*(1 if vol_surge else 0)
    score = min(score, 1.0)

    if buy:
        signal="BUY"
    elif score < 0.25:
        signal="SELL"
    else:
        signal="HOLD"

    resonance_str = ",".join(reasons) if reasons else "无"

    return {
        "signal": signal,
        "close": round(float(close[-1]), 2),
        "score": round(score, 3),
        "resonance": resonance_str,
    }


def main_handler(event, context):
    """腾讯云函数入口"""
    logger.info("量化信号扫描开始...")

    signals = []
    for code, name in STOCKS.items():
        df = fetch_kline(code)
        if df.empty:
            s = {"code": code, "name": name, "signal": "HOLD", "close": 0, "score": 0, "resonance": "无"}
        else:
            s = score_signal(df, code)
            s.update({"code": code, "name": name})
        signals.append(s)
        logger.info("  {} {}: {} price={} score={:.3f}".format(code,name,s['signal'],s['close'],s['score']))

    push_signal_summary(signals)

    return {
        "statusCode": 200,
        "body": json.dumps({
            "time": datetime.now().isoformat(),
            "signals": signals,
        }, ensure_ascii=False),
    }


# 本地测试
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = main_handler({}, {})
    print(result)
