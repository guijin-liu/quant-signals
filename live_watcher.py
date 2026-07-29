"""v20 实时买卖点监控 — 60秒轮询，腾讯K线+妙想财务"""
import os, logging, requests, time
import numpy as np, pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("live")

P_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
P_URL = "http://www.pushplus.plus/send"

from stock_pool import STOCK_POOL_BACKUP
import mx_fetcher

STOCKS = {c: i["name"] for c, i in STOCK_POOL_BACKUP.items()}
log.info(f"股票池: {len(STOCKS)} 只")
state = {}

def push(title, body):
    try:
        r = requests.post(P_URL, json={"token": P_TOKEN, "title": title, "content": body, "template": "html"}, timeout=10)
        log.info(f"{'OK' if r.json().get('code')==200 else 'FAIL'}: {title}")
    except: pass

def alert(code, name, sig, f, reason, target=0, tp=0):
    color = "#e74c3c" if sig == "BUY" else "#27ae60"
    emoji = "🔴" if sig == "BUY" else "🟢"
    action = "买入" if sig == "BUY" else "卖出"
    extra = f"<tr><td><b>目标</b></td><td style='color:#2ecc71;font-size:18px'>{target} (+{tp}%)</td></tr>" if sig == "BUY" else ""

    # 妙想财务确认（仅买入时）
    fin_line = ""
    if sig == "BUY":
        fin = mx_fetcher.get_financial_quality(code, name)
        if fin:
            _, fin_label = mx_fetcher.score_financial_quality(fin, {})
            fin_line = f"<tr><td><b>📊 妙想</b></td><td style='color:#27ae60'>{fin_label}</td></tr>"

    push(f"{emoji}{action} {name} {f['close']:.2f}",
         f"<div style='background:#{1000 if sig=='BUY' else 100}00;color:#eee;padding:14px;border-left:4px solid {color}'><h2 style='color:{color}'>{emoji} {action}信号</h2>"
         f"<table style='width:100%;line-height:2.2'><tr><td><b>股票</b></td><td style='font-size:18px'>{name} {code}</td></tr>"
         f"<tr><td><b>现价</b></td><td style='color:{color};font-size:22px;font-weight:bold'>{f['close']:.2f}</td></tr>"
         f"<tr><td><b>RSI</b></td><td>{f['rsi']:.0f} | 位置{f['pos']:.0%} | BB{f['bb_pct']:.0%}</td></tr>"
         f"<tr><td><b>理由</b></td><td style='color:#f39c12'>{reason}</td></tr>{extra}{fin_line}</table>"
         f"<p style='color:#888;font-size:11px'>{datetime.now().strftime('%m/%d %H:%M:%S')}</p></div>")

import mx_fetcher

def fetch(code):
    """腾讯15min K线"""
    return mx_fetcher.fetch_kline(code, '15')

def feat(df):
    c, h, l, v = df['close'].values, df['high'].values, df['low'].values, df['volume'].values
    f = {'close': c[-1], 'ma5': np.mean(c[-5:]), 'ma10': np.mean(c[-10:]), 'ma20': np.mean(c[-20:])}
    f['golden'] = (f['ma5'] > f['ma10']) and (np.mean(c[-6:-1]) <= np.mean(c[-11:-1]))
    g = np.mean(d := np.diff(c[-15:]), where=d>0) if (d>0).any() else 0
    ln = -np.mean(d, where=d<0) if (d<0).any() else 1e-9
    f['rsi'] = 100 - 100/(1+g/ln) if ln>0 else 50
    bs, bm = np.std(c[-20:]), np.mean(c[-20:])
    f['bb_pct'] = max(0, min(1, (c[-1]-(bm-2*bs))/(4*bs+1e-4)))
    f['pos'] = max(0, min(1, (c[-1]-np.min(l[-20:]))/(np.max(h[-20:])-np.min(l[-20:])+1e-4)))
    f['vol_ratio'] = np.mean(v[-5:])/(np.mean(v[-20:])+1)
    e12 = pd.Series(c).ewm(span=12, adjust=False).mean().values
    e26 = pd.Series(c).ewm(span=26, adjust=False).mean().values
    ml = e12 - e26; ms = pd.Series(ml).ewm(span=9, adjust=False).mean().values
    f['macd_turn'] = 2*(ml[-1]-ms[-1])>0 and 2*(ml[-2]-ms[-2])<=0
    return f

def buy(f):
    g, bb, cl, vol, mt, pos = f['golden'], f['bb_pct'], f['close'], f.get('vol_ratio',1), f.get('macd_turn',0), f['pos']
    rules = [(g and bb<=0.15 and mt and vol>0.7, "核心DNA", 2.0),
             (g and bb<=0.20 and mt and vol>1.0, "BB窄+MACD转正+量增", 2.2),
             (g and bb<=0.25 and pos<=0.4 and vol>1.0, "金叉+低位+量增", 1.8),
             (g and bb<=0.30 and vol>1.0, "金叉+BB+量增", 1.7),
             (g and bb<=0.30, "金叉+BB", 1.5)]
    for cond, reason, tp in rules:
        if cond: return True, reason, round(cl*(1+tp/100),2), tp
    return False, "", 0, 0

def sell(code, f):
    r, p, bb = f['rsi'], f['pos'], f['bb_pct']
    if code in ("159941","513100"):
        if r>=70 and bb>=0.85: return True, "RSI70+布林上轨(纳指)"
        if r>=75 and p>=0.7: return True, "RSI75+高位(纳指)"
    rules = {"000933": [(75,0.8,0.8),(70,0.8,0.85)], "002497": [(65,0.8,0.85),(75,0.8,0.8)],
             "000960": [(75,0.6,0.85),(65,0.7,0.85)], "000893": [(75,0.6,0.8),(70,0.6,0.85)]}
    for ri, pi, bi in rules.get(code, []):
        if r>=ri and p>=pi and bb>=bi: return True, f"RSI{ri}+高位+布林上轨"
    if r>=70 and p>=0.7 and bb>=0.8: return True, "RSI高位+布林上轨"
    if r>=75 and p>=0.6: return True, "RSI超买+高位"
    return False, ""

def scan(notify=True):
    changes = 0
    for code, name in STOCKS.items():
        df = fetch(code)
        if df.empty or len(df)<20: continue
        f = feat(df)
        b, rb, tgt, tp = buy(f)
        s, rs = sell(code, f)
        sig = "BUY" if b else ("SELL" if s else "HOLD")
        old = state.get(code, "NONE")
        if sig != old:
            changes += 1
            if notify and old != "NONE":
                alert(code, name, sig, f, rb if b else rs, tgt, tp)
            state[code] = sig
    return changes

def main():
    log.info("初始化...")
    scan(notify=False)
    init_b = sum(1 for v in state.values() if v=="BUY")
    init_s = sum(1 for v in state.values() if v=="SELL")
    push("🟢 实时监控启动", f"{len(STOCKS)}只股票 60秒轮询 | {init_b}B {init_s}S<br>{datetime.now():%m/%d %H:%M:%S}")
    log.info(f"就绪: {init_b}B {init_s}S")

    n = 0
    while True:
        try:
            n += 1
            now = datetime.now()
            in_trade = now.weekday() < 5 and 925 <= now.hour*100+now.minute <= 1505
            if in_trade:
                ch = scan()
                if n%30==0 or ch>0: log.info(f"#{n} 变化{ch}")
            elif n%30==0:
                log.info(f"待机 ({now:%H:%M})")
            time.sleep(60)
        except KeyboardInterrupt:
            push("🔴 监控已停止", f"共{n}轮<br>{datetime.now():%m/%d %H:%M:%S}")
            break
        except Exception as e:
            log.error(f"异常: {e}"); time.sleep(10)

if __name__ == "__main__":
    main()
