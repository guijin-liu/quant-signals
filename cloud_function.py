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
        emoji = {"BUY": "🔴", "SELL": "🟢", "HOLD": "⚪"}.get(s.get("signal", ""), "")
        rows += (
            f"<tr><td>{emoji}</td><td><b>{s.get('code','')}</b></td>"
            f"<td>{s.get('name','')}</td><td>{s.get('close',0):.2f}</td>"
            f"<td>{s.get('score',0):.3f}</td><td>{s.get('resonance',0)}/6</td></tr>"
        )

    content = (
        f"<div style='background:#1a1a2e;color:#eee;padding:15px;border-radius:10px'>"
        f"<h2>📊 量化信号 — {now}</h2>"
        f"<table style='width:100%;color:#eee;border-collapse:collapse'>"
        f"<tr style='border-bottom:1px solid #333'><th></th><th>代码</th><th>名称</th><th>价格</th><th>评分</th><th>共振</th></tr>"
        f"{rows}</table>"
        f"<p style='margin-top:12px'>买入: <b style='color:#e74c3c'>{buy_count}</b>只 | 目标胜率 >88%</p>"
        f"</div>"
    )

    title = f"📊 信号 {'买入'+str(buy_count)+'只' if buy_count > 0 else '无可买入'}"
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


def score_signal(df):
    """简化量化评分"""
    if len(df) < 30:
        return {"signal": "HOLD", "close": 0, "score": 0, "resonance": 0}

    close = df["close"].values

    # MA多头
    ma5 = np.mean(close[-5:])
    ma10 = np.mean(close[-10:]) if len(close) >= 10 else ma5
    ma_bull = 1 if ma5 > ma10 else 0

    # RSI
    deltas = np.diff(close[-15:])
    gain = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    loss = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    rsi = 100 - 100 / (1 + gain / loss) if loss > 0 else 50
    rsi_ok = 1 if 40 < rsi < 70 else 0

    # 短线趋势
    pct_5 = (close[-1] - close[-5]) / close[-5] * 100 if len(close) >= 5 else 0
    trend_ok = 1 if pct_5 > 0 else 0

    # 放量
    vol_recent = df["volume"].tail(5).mean()
    vol_avg = df["volume"].tail(20).mean()
    vol_surge = 1 if vol_recent > vol_avg * 1.3 else 0

    reasons = ma_bull + rsi_ok + trend_ok + vol_surge
    score = 0.3 * ma_bull + 0.2 * rsi_ok + 0.25 * trend_ok + 0.15 * vol_surge + 0.1

    if score > 0.55 and reasons >= 3:
        signal = "BUY"
    elif score < 0.25:
        signal = "SELL"
    else:
        signal = "HOLD"

    return {
        "signal": signal,
        "close": round(float(close[-1]), 2),
        "score": round(score, 3),
        "resonance": reasons,
    }


def main_handler(event, context):
    """腾讯云函数入口"""
    logger.info("量化信号扫描开始...")

    signals = []
    for code, name in STOCKS.items():
        df = fetch_kline(code)
        if df.empty:
            s = {"code": code, "name": name, "signal": "HOLD", "close": 0, "score": 0, "resonance": 0}
        else:
            s = score_signal(df)
            s.update({"code": code, "name": name})
        signals.append(s)
        logger.info(f"  {code} {name}: {s['signal']} price={s['close']} score={s['score']:.3f}")

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
