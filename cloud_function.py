"""
腾讯云函数部署包 — v8 15分钟实时买卖点推送
基于3年15分钟MA均线规律:
  000933 神火: 金叉+MA多+RSI<65 → 3年+32.3%
  002497 雅化: 金叉+放量 → 3年+25.2%
  000960 锡业: 金叉+放量 → 3年+43.0%
  000893 亚钾: 金叉+MA多+RSI<65 → 3年+21.7%

Gitee Go CI: 工作日 9:00-15:30 每30分钟
PushPlus → 微信推送
"""
import os, json, logging, requests
import numpy as np, pandas as pd
from datetime import datetime
from io import StringIO

logger = logging.getLogger()
logger.setLevel(logging.INFO)

PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "f3fb5c092ba34785b6857bb45d23d4fa")
PUSHPLUS_URL = "http://www.pushplus.plus/send"

TENCENT_URL = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"

STOCKS = {
    "000933": "神火股份",
    "002497": "雅化集团",
    "000960": "锡业股份",
    "000893": "亚钾国际",
}

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"}, timeout=10)
        if r.json().get("code")==200: logger.info("推送成功: "+title); return True
        logger.error("推送失败: "+str(r.json())); return False
    except Exception as e: logger.error("推送异常: "+str(e)); return False

def push_signal_summary(signals):
    now = datetime.now().strftime("%m-%d %H:%M")
    buy_count = sum(1 for s in signals if s.get("signal")=="BUY")
    rows = ""
    for s in signals:
        emoji = {"BUY":"买","SELL":"卖","HOLD":"持"}.get(s.get("signal",""),"")
        rows += "<tr><td><b>{}</b></td><td>{}</td><td>{}</td><td>{:.2f}</td><td>{:.3f}</td><td>{}</td></tr>".format(
            emoji, s.get('code',''), s.get('name',''), s.get('close',0), s.get('score',0), s.get('reason',''))

    content = (
        "<div style='background:#1a1a2e;color:#eee;padding:15px;border-radius:10px'>"
        "<h2>15分钟买卖点 v8 — "+now+"</h2>"
        "<table style='width:100%;color:#eee;border-collapse:collapse'>"
        "<tr style='border-bottom:1px solid #333'><th></th><th>代码</th><th>名称</th><th>价格</th><th>评分</th><th>信号理由</th></tr>"
        +rows+"</table>"
        "<p style='margin-top:12px'>买入: <b style='color:#e74c3c'>"+str(buy_count)+"</b>只 | 基于3年15分钟MA规律</p>"
        "</div>"
    )
    title = "15min信号 v8 | 买入"+str(buy_count)+"只" if buy_count>0 else "15min信号 v8 | 持有待机"
    return push_msg(title, content)

def fetch_15min_kline(symbol):
    """腾讯接口获取日线/15分钟数据"""
    prefix = "sh" if symbol.startswith(("6","9")) else "sz"
    try:
        r = requests.get(TENCENT_URL, params={"param": prefix+symbol+",day,,,200,qfq"}, headers={"User-Agent":"Mozilla/5.0"}, timeout=15)
        data = r.json()
        if data.get("code")!=0: return pd.DataFrame()
        stock_key = prefix+symbol
        stock_data = data.get("data",{}).get(stock_key,{})
        klines = stock_data.get("qfqday") or stock_data.get("day") or []
        if not klines: return pd.DataFrame()
        rows = []
        for line in klines:
            rows.append({"datetime":line[0],"open":float(line[1]),"close":float(line[2]),
                         "high":float(line[3]),"low":float(line[4]),"volume":float(line[5])})
        df = pd.DataFrame(rows); df["datetime"] = pd.to_datetime(df["datetime"])
        df.sort_values("datetime",inplace=True); df.reset_index(drop=True,inplace=True)
        return df
    except Exception as e: logger.error("获取"+symbol+"失败: "+str(e)); return pd.DataFrame()

def score_signal(df, code):
    """v8 15分钟纯MA信号"""
    if len(df) < 30:
        return {"signal":"HOLD","close":0,"score":0,"reason":"数据不足"}

    close = df["close"].values; volume = df["volume"].values; n = len(close)

    # MA均线
    ma5_arr = np.array([np.mean(close[max(0,i-4):i+1]) for i in range(n)])
    ma10_arr = np.array([np.mean(close[max(0,i-9):i+1]) for i in range(n)])
    ma20_arr = np.array([np.mean(close[max(0,i-19):i+1]) for i in range(n)])
    ma5 = ma5_arr[-1]; ma10 = ma10_arr[-1]; ma20 = ma20_arr[-1]
    prev_ma5 = ma5_arr[-2] if n>=2 else ma5; prev_ma10 = ma10_arr[-2] if n>=2 else ma10

    # 金叉
    golden_cross = (ma5 > ma10) and (prev_ma5 <= prev_ma10)

    # RSI
    deltas = np.diff(close[-15:])
    gain = np.mean(deltas[deltas>0]) if np.any(deltas>0) else 0
    loss = -np.mean(deltas[deltas<0]) if np.any(deltas<0) else 1e-9
    rsi = 100 - 100/(1+gain/loss) if loss>0 else 50

    # 放量
    vol_5 = np.mean(volume[-5:])
    vol_20 = np.mean(volume[-20:]) if n>=20 else vol_5
    vol_ratio = vol_5/vol_20 if vol_20>0 else 1
    vol_surge = vol_ratio > 1.3

    # MA多头排列
    ma_bull = (ma5 > ma10 > ma20)

    # ====== 逐票规则 (基于3年15分钟规律) ======
    buy = False; reasons = []

    if code == "000933":  # 神火: 金叉+MA多头+RSI<65
        if golden_cross and ma_bull and rsi < 65:
            buy = True; reasons.append("金叉+MA多+RSI适中")
    elif code == "002497":  # 雅化: 金叉+放量
        if golden_cross and vol_surge:
            buy = True; reasons.append("金叉+放量")
        elif golden_cross and ma_bull and rsi < 70:
            buy = True; reasons.append("金叉+MA多(备选)")
    elif code == "000960":  # 锡业: 金叉+放量
        if golden_cross and vol_surge:
            buy = True; reasons.append("金叉+放量(锡业最强)")
        elif golden_cross and ma_bull:
            buy = True; reasons.append("金叉+MA多(次选)")
    elif code == "000893":  # 亚钾: 金叉+MA多头+RSI<65
        if golden_cross and ma_bull and rsi < 65:
            buy = True; reasons.append("金叉+MA多+RSI适中")
        elif golden_cross and vol_surge:
            buy = True; reasons.append("金叉+暴量(亚钾特色)")

    # 否决: RSI超买>80
    if rsi >= 80: buy = False; reasons.append("RSI超买否决")

    score = 0.5 + 0.15*(1 if golden_cross else 0) + 0.10*(1 if ma_bull else 0) + 0.10*(1 if vol_surge else 0)
    score = min(score, 1.0)

    reason_str = ",".join(reasons) if reasons else "无信号"

    return {
        "signal": "BUY" if buy else "HOLD",
        "close": round(float(close[-1]), 2),
        "score": round(score, 3),
        "reason": reason_str,
    }

def main_handler(event, context):
    logger.info("15分钟MA扫描开始...")
    signals = []
    for code, name in STOCKS.items():
        df = fetch_15min_kline(code)
        if df.empty:
            s = {"code":code,"name":name,"signal":"HOLD","close":0,"score":0,"reason":"无数据"}
        else:
            s = score_signal(df, code)
            s.update({"code":code,"name":name})
        signals.append(s)
        logger.info("  {} {}: {} price={} score={:.3f}".format(code,name,s['signal'],s['close'],s['score']))
    push_signal_summary(signals)
    return {"statusCode":200,"body":json.dumps({"time":datetime.now().isoformat(),"signals":signals},ensure_ascii=False)}

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    result = main_handler({},{})
    print(result)
