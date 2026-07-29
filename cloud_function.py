"""v16 量化买卖点 — 妙想主力+腾讯K线 + 5+15min双框架 + 全指标 + 7折卖点"""
import os, sys, json, logging, requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone

import mx_fetcher
from stock_pool import STOCK_POOL_BACKUP

BEIJING_TZ = timezone(timedelta(hours=8))
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={"token":PUSHPLUS_TOKEN,"title":title,"content":content,"template":"html"}, timeout=10)
        ok = r.json().get("code") == 200
        logger.info(f"PUSH {'OK' if ok else 'FAIL'}: {title}")
        return ok
    except: return False

def is_trading_time():
    now = datetime.now(BEIJING_TZ)
    if now.weekday() >= 5: return False
    h, m = now.hour, now.minute
    if (h == 9 and m >= 15) or (h == 10) or (h == 11 and m <= 30): return True
    if (h == 13) or (h == 14) or (h == 15 and m == 0): return True
    return False


# ═══════════════════════════════════════
# 预筛 — 妙想估值主力，腾讯备用
# ═══════════════════════════════════════

def batch_pre_screen(stocks: dict) -> list:
    """妙想估值预筛（缓存1h），失败降级腾讯"""
    codes = list(stocks.keys())
    quotes = mx_fetcher.get_quotes(codes, stocks)

    candidates = []
    for code, name in stocks.items():
        q = quotes.get(code)
        if not q: continue
        price = q.get("price", 0)
        change_pct = q.get("change_pct", 0)
        vol_ratio = q.get("vol_ratio", 0)
        if price <= 0: continue
        if change_pct < -3 or change_pct > 9.5: continue
        if vol_ratio < 1.0: continue

        volume = q.get("volume", 0)
        amount = price * volume if volume else 0
        candidates.append((code, name, price, vol_ratio, change_pct, amount,
                          q.get("pe", 0), q.get("pb", 0), q.get("mcap", 0),
                          q.get("source", "")))

    candidates.sort(key=lambda x: x[5], reverse=True)
    mx_n = sum(1 for c in candidates if c[9] == "mx-data")
    logger.info(f"预筛: {len(stocks)}只→{len(candidates)}只 (妙想{mx_n}只)")
    return candidates


# ═══════════════════════════════════════
# 全指标特征计算
# ═══════════════════════════════════════

def compute_features(df):
    close = df['close'].values; high = df['high'].values
    low = df['low'].values; vol = df['volume'].values
    n = len(close)
    if n < 26: return None
    f = {}
    f['close'] = close[-1]
    f['ma5']  = np.mean(close[-5:])
    f['ma10'] = np.mean(close[-10:])
    f['ma20'] = np.mean(close[-20:])
    f['ma60'] = np.mean(close[-min(60,n):])
    p_ma5 = np.mean(close[-6:-1]); p_ma10 = np.mean(close[-11:-1])
    f['golden'] = (f['ma5'] > f['ma10']) and (p_ma5 <= p_ma10)
    f['dead']   = (f['ma5'] < f['ma10']) and (p_ma5 >= p_ma10)
    deltas = np.diff(close[-15:])
    g = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    l = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    f['rsi'] = 100 - 100/(1+g/l) if l > 0 else 50
    bb_m = np.mean(close[-20:]); bb_s = np.std(close[-20:])
    bb_u = bb_m + 2*bb_s; bb_l = bb_m - 2*bb_s
    f['bb_pct'] = max(0.0, min(1.0, (close[-1] - bb_l) / (bb_u - bb_l + 0.0001)))
    f['bb_width'] = (bb_u - bb_l) / bb_m
    h20 = np.max(high[-20:]); l20 = np.min(low[-20:])
    f['pos'] = max(0.0, min(1.0, (close[-1] - l20) / (h20 - l20 + 0.0001)))
    f['vol_ratio'] = np.mean(vol[-5:]) / (np.mean(vol[-20:]) + 1)
    f['vol_trend'] = np.mean(vol[-10:]) / (np.mean(vol[-30:]) + 1) if n >= 30 else 1.0
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26; dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    f['macd'] = 2 * (dif[-1] - dea[-1])
    f['macd_direction'] = 'up' if f['macd'] > 2*(dif[-2]-dea[-2]) else 'down'
    h9 = np.max(high[-9:]); l9 = np.min(low[-9:])
    rsv = (close[-1] - l9) / (h9 - l9 + 0.0001) * 100
    f['k'] = 2/3 * 50 + 1/3 * rsv
    f['d'] = 2/3 * 50 + 1/3 * f['k']; f['j'] = 3 * f['k'] - 2 * f['d']
    trs = []
    for i in range(-14, 0):
        h, lv, pc = high[i], low[i], close[i-1] if i > -14 else close[i-1]
        trs.append(max(h-lv, abs(h-pc), abs(lv-pc)))
    f['atr'] = np.mean(trs)
    obv_changes = [vol[i] if close[i] > close[i-1] else (-vol[i] if close[i] < close[i-1] else 0) for i in range(-10, 0)]
    f['obv_up'] = sum(obv_changes) > 0
    f['above_ma20'] = close[-1] > f['ma20']
    f['above_ma60'] = close[-1] > f['ma60']
    return f


# ═══════════════════════════════════════
# 买卖点评分
# ═══════════════════════════════════════

def score_buy(f15, f5=None):
    if not f15: return False, "", 0, 0
    close = f15['close']; score = 0; reasons = []
    if f15['golden']: score += 2; reasons.append("金叉")
    elif f15['ma5'] <= f15['ma10']: return False, "无金叉", 0, 0
    if f15['bb_pct'] <= 0.25: score += 2; reasons.append("BB低位")
    elif f15['bb_pct'] <= 0.35: score += 1; reasons.append("BB中低位")
    elif f15['bb_pct'] > 0.7: return False, "BB高位", 0, 0
    if 40 <= f15['rsi'] <= 60: score += 1; reasons.append("RSI健康")
    if f15['j'] < 30: score += 1; reasons.append("J值超卖")
    elif f15['j'] > 80: return False, "J值超买", 0, 0
    if f15['vol_ratio'] > 1.0: score += 1; reasons.append("放量")
    if f15['obv_up']: score += 1; reasons.append("OBV向上")
    if f15['above_ma60']: score += 1; reasons.append("多头趋势")
    if f5 and f5.get('golden'): score += 1; reasons.append("5min确认")
    if close > f15['ma20'] * 1.05: return False, "追高(>MA20+5%)", 0, 0
    if score < 3: return False, f"共振不足({score}分)", 0, 0
    bb_w = f15['bb_width']; atr_pct = f15['atr'] / close
    vol_boost = min(1.5, f15['vol_ratio'])
    predicted_gain = atr_pct * 100 * (2.0 + max(0, 0.15-bb_w)*20 + (vol_boost-1)*0.5)
    predicted_gain = round(max(1.0, min(8.0, predicted_gain)), 1)
    target = round(close * (1 + predicted_gain/100), 2)
    signal = "强" if score >= 4 else "标准"
    return True, f"{signal}({score}分):{'+'.join(reasons)}", target, predicted_gain

def score_sell(f15, predicted_gain_pct):
    rsi = f15['rsi']; pos = f15['pos']; bb = f15['bb_pct']
    if rsi >= 75 and pos >= 0.65: return True, "RSI超买+高位"
    if rsi >= 70 and bb >= 0.85: return True, "RSI高+BB上轨"
    if f15['dead'] and f15['close'] < f15['ma20']: return True, "死叉+破MA20"
    return False, ""


# ═══════════════════════════════════════
# 主扫描
# ═══════════════════════════════════════

def scan_and_push():
    now = datetime.now(BEIJING_TZ)
    if not is_trading_time():
        logger.info(f"非交易时间 {now.strftime('%m/%d %H:%M')}")
        return []
    scan_time = now.strftime('%m/%d %H:%M')

    # 1. 股票池
    all_stocks = {c: i["name"] for c, i in STOCK_POOL_BACKUP.items()}
    n_all = len(all_stocks)
    logger.info(f"v16 @ {scan_time} — {n_all}只")

    # 2. 预筛 — 妙想主力
    candidates = batch_pre_screen(all_stocks)
    if not candidates:
        push_msg(f"☁️ 无候选 {scan_time}", f'<div>妙想预筛{n_all}只无候选</div>')
        return []

    # 3. 深度分析
    results = []
    for idx, c in enumerate(candidates):
        code, name, price = c[0], c[1], c[2]
        vol_ratio = c[3]; change_pct = c[4]; amount = c[5]
        pe = c[6]; pb = c[7]; mcap = c[8]
        vol_rank = idx + 1

        df15 = mx_fetcher.fetch_kline(code, '15')
        if df15.empty or len(df15) < 26: continue
        f15 = compute_features(df15)
        if not f15: continue

        df5 = mx_fetcher.fetch_kline(code, '5')
        f5 = compute_features(df5) if (not df5.empty and len(df5) >= 26) else None

        buy, reason_b, target, gain_pct = score_buy(f15, f5)
        sell, reason_s = score_sell(f15, gain_pct)
        sig = "BUY" if buy else ("SELL" if sell else "HOLD")
        reason = reason_b if buy else (reason_s if sell else "")

        sell_price = round(f15['close'] * (1 + gain_pct * 0.7 / 100), 2) if buy else 0
        amt_str = f"{amount/1e8:.2f}亿" if amount > 1e8 else f"{amount/1e4:.0f}万"

        r = {"code":code,"name":name,"signal":sig,"close":round(f15['close'],2),
             "rsi":round(f15['rsi'],1),"bb":round(f15['bb_pct'],2),
             "reason":reason,"target":target,"gain_pct":gain_pct,
             "sell_price":sell_price,"amount":amount,"vol_ratio":vol_ratio,
             "change_pct":change_pct,"amt_str":amt_str,"vol_rank":vol_rank,
             "pe":pe,"pb":pb,"mcap":mcap}
        results.append(r)

        # ── 个股推送 ──
        if sig == "BUY":
            fin = mx_fetcher.get_financial_quality(code, name)
            fin_score, fin_label = mx_fetcher.score_financial_quality(fin, {"pe": pe, "pb": pb})
            mcap_str = f"{mcap:.0f}亿" if mcap else ""

            logger.info(f"  >>> BUY {code} {name} @{f15['close']:.2f} +{gain_pct}% 卖点{sell_price} | {reason} | 妙想:{fin_label}")
            push_msg(f"🔴买入 {name} {f15['close']:.2f} +{gain_pct}%",
                     f'<div style="font-size:15px;padding:12px;line-height:2">'
                     f'<h3 style="color:#e74c3c">🔴 买入信号 — {name}({code})</h3>'
                     f'<table style="width:100%">'
                     f'<tr><td>现价</td><td><b style="color:#e74c3c;font-size:20px">{r["close"]}</b></td>'
                     f'<td>涨跌</td><td style="color:#e74c3c">{change_pct:+.2f}%</td></tr>'
                     f'<tr><td>目标价</td><td><b style="color:#2ecc71">{target}</b> (+{gain_pct}%)</td>'
                     f'<td>7折卖点</td><td style="color:#f39c12"><b>{sell_price}</b> (+{round(gain_pct*0.7,1)}%)</td></tr>'
                     f'<tr><td>PE</td><td>{pe:.1f}</td><td>PB</td><td>{pb:.2f}</td></tr>'
                     f'<tr><td>市值</td><td>{mcap_str}</td><td>量比</td><td>{vol_ratio:.1f}</td></tr>'
                     f'<tr><td>成交额</td><td>{amt_str}</td><td>排名</td><td>#{vol_rank}</td></tr>'
                     f'<tr><td>RSI</td><td>{f15["rsi"]:.0f}</td>'
                     f'<td>KDJ</td><td>K:{f15["k"]:.0f} D:{f15["d"]:.0f} J:{f15["j"]:.0f}</td></tr>'
                     f'<tr><td>信号</td><td style="color:#f39c12">{reason}</td>'
                     f'<td>妙想</td><td style="color:#27ae60">{fin_label}</td></tr>'
                     f'</table>'
                     f'<p style="color:#888;font-size:11px">{scan_time} | v16 妙想+腾讯</p></div>')

        elif sig == "SELL":
            logger.info(f"  >>> SELL {code} {name} @{f15['close']:.2f} | {reason}")
            push_msg(f"🟢卖出 {name} {f15['close']:.2f}",
                     f'<div style="font-size:15px;padding:12px;line-height:2">'
                     f'<h3 style="color:#27ae60">🟢 卖出信号 — {name}({code})</h3>'
                     f'<table style="width:100%">'
                     f'<tr><td>现价</td><td><b style="color:#27ae60;font-size:20px">{r["close"]}</b></td>'
                     f'<td>涨跌</td><td>{change_pct:+.2f}%</td></tr>'
                     f'<tr><td>RSI</td><td>{f15["rsi"]:.0f}</td>'
                     f'<td>BB%</td><td>{f15["bb_pct"]:.2f}</td></tr>'
                     f'<tr><td>成交额</td><td>{amt_str}</td><td>排名</td><td>#{vol_rank}</td></tr>'
                     f'<tr><td>PE</td><td>{pe:.1f}</td><td>PB</td><td>{pb:.2f}</td></tr>'
                     f'<tr><td>理由</td><td style="color:#f39c12">{reason}</td></tr>'
                     f'</table>'
                     f'<p style="color:#888;font-size:11px">{scan_time} | v16 妙想+腾讯</p></div>')

    # 4. 汇总推送
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')

    top10 = sorted(results, key=lambda r: r.get('amount', 0), reverse=True)[:10]
    top_rows = ""
    for r in top10:
        emoji = {"BUY": "🔴", "SELL": "🟢", "HOLD": "⚪"}[r["signal"]]
        top_rows += (f'<tr><td>#{r["vol_rank"]}</td><td>{emoji}</td>'
                     f'<td><b>{r["code"]}</b></td><td>{r["name"]}</td>'
                     f'<td>{r["close"]}</td><td>{r["amt_str"]}</td>'
                     f'<td>PE{r.get("pe",0):.0f}</td><td>{r.get("reason","")[:25]}</td></tr>')

    mkt = mx_fetcher.get_market_brief()
    mkt_line = f"上证{mkt.get('sh_idx', 0):+.2f}% | 成交{mkt.get('sz_amount', 0)/10000:.2f}万亿" if mkt else ""

    push_msg(f"☁️{scan_time} B{buy_count} S{sell_count}",
             f'<div style="font-size:14px;padding:10px">'
             f'<b>{mkt_line}</b><br>'
             f'妙想预筛: {n_all}只→{len(candidates)}候选→{len(results)}分析<br>'
             f'<b style="color:#e74c3c">买入{buy_count}</b> | <b style="color:#27ae60">卖出{sell_count}</b><br>'
             f'<br><b>排名 | 代码 | 名称 | 价格 | 成交额 | PE | 信号</b><br>'
             f'<table style="width:100%;font-size:11px;border-collapse:collapse">'
             f'{top_rows}</table>'
             f'<br><span style="color:#888;font-size:10px">v16 妙想主力+腾讯K线 | 5+15min双框架 | 7折卖点</span></div>')

    return results


if __name__ == "__main__":
    scan_and_push()
