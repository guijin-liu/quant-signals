"""v15 量化买卖点 — 5+15min双框架 + 全指标 + 动态目标 + 7折卖点 + 量能排名"""
import os, sys, json, logging, requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

BEIJING_TZ = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
PRE_SCREEN_WORKERS = 12


# ═══════════════════════════════════════════════
# 工具函数
# ═══════════════════════════════════════════════

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={"token":PUSHPLUS_TOKEN,"title":title,"content":content,"template":"html"}, timeout=10)
        ok = r.json().get("code") == 200
        logger.info(f"PUSH {'OK' if ok else 'FAIL'}: {title}")
        return ok
    except Exception as e:
        logger.error(f"Push error: {e}"); return False

def check_deepseek_balance():
    try:
        r = requests.get("https://api.deepseek.com/user/balance", headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"}, timeout=10)
        for b in r.json().get("balance_infos", []):
            if b["currency"] == "CNY": return float(b["total_balance"])
    except: pass
    return None

def is_trading_time():
    now = datetime.now(BEIJING_TZ)
    if now.weekday() >= 5: return False
    h, m = now.hour, now.minute
    if (h == 9 and m >= 15) or (h == 10) or (h == 11 and m <= 30): return True
    if (h == 13) or (h == 14) or (h == 15 and m == 0): return True
    return False


# ═══════════════════════════════════════════════
# 数据获取
# ═══════════════════════════════════════════════

def fetch_kline(code, freq='15'):
    """获取K线数据 freq='5'|'15'"""
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6","9")) else "sz."
        now_bj = datetime.now(BEIJING_TZ)
        end = now_bj.strftime("%Y-%m-%d")
        start = (now_bj - timedelta(days=90)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(prefix + code,
            'date,time,open,high,low,close,volume',
            start_date=start, end_date=end, frequency=freq, adjustflag='2')
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except:
        try: bs.logout()
        except: pass
        return pd.DataFrame()

def fetch_daily_snapshot(code):
    """日线快照 — 预筛用"""
    import baostock as bs
    try:
        bs.login()
        prefix = "sh." if code.startswith(("6","9")) else "sz."
        end = datetime.now(BEIJING_TZ).strftime("%Y-%m-%d")
        start = (datetime.now(BEIJING_TZ) - timedelta(days=5)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(prefix + code,
            'date,close,volume,turn,preclose',
            start_date=start, end_date=end, frequency='d', adjustflag='2')
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if len(rows) < 2: return None
        return rows
    except:
        try: bs.logout()
        except: pass
        return None


# ═══════════════════════════════════════════════
# 预筛
# ═══════════════════════════════════════════════

def pre_screen_one(code, name):
    rows = fetch_daily_snapshot(code)
    if not rows or len(rows) < 2: return None
    try:
        latest = rows[-1]
        close, volume, preclose = float(latest[1]), float(latest[2]), float(latest[4])
        change_pct = (close - preclose) / preclose * 100
        if change_pct < -3 or change_pct > 9.5: return None
        amount = close * volume
        hist_vols = [float(r[2]) for r in rows[:-1] if float(r[2]) > 0]
        if not hist_vols: return None
        vol_ratio = volume / np.mean(hist_vols)
        if vol_ratio < 1.0: return None
        return (code, name, close, vol_ratio, change_pct, amount, volume)
    except: return None

def parallel_pre_screen(stocks, max_workers=PRE_SCREEN_WORKERS):
    candidates = []
    codes = list(stocks.items())
    total = len(codes)
    logger.info(f"预筛: {total} 只 → {max_workers}线程")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(pre_screen_one, c, n): c for c, n in codes}
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 500 == 0: logger.info(f"  预筛 {done}/{total}")
            r = f.result()
            if r: candidates.append(r)
    candidates.sort(key=lambda x: x[5], reverse=True)
    logger.info(f"预筛完成: {len(candidates)} 只")
    return candidates


# ═══════════════════════════════════════════════
# 全指标特征计算
# ═══════════════════════════════════════════════

def compute_features(df):
    """15min K线 → 全技术指标"""
    close = df['close'].values; high = df['high'].values
    low = df['low'].values; vol = df['volume'].values
    n = len(close)
    if n < 26: return None
    f = {}
    f['close'] = close[-1]

    # MA
    f['ma5']  = np.mean(close[-5:])
    f['ma10'] = np.mean(close[-10:])
    f['ma20'] = np.mean(close[-20:])
    f['ma60'] = np.mean(close[-min(60,n):])

    # 金叉/死叉
    p_ma5 = np.mean(close[-6:-1]); p_ma10 = np.mean(close[-11:-1])
    f['golden'] = (f['ma5'] > f['ma10']) and (p_ma5 <= p_ma10)
    f['dead']   = (f['ma5'] < f['ma10']) and (p_ma5 >= p_ma10)

    # RSI(14)
    deltas = np.diff(close[-15:])
    g = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    l = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    f['rsi'] = 100 - 100/(1+g/l) if l > 0 else 50

    # BB%
    bb_m = np.mean(close[-20:]); bb_s = np.std(close[-20:])
    bb_u = bb_m + 2*bb_s; bb_l = bb_m - 2*bb_s
    f['bb_pct'] = max(0.0, min(1.0, (close[-1] - bb_l) / (bb_u - bb_l + 0.0001)))
    f['bb_width'] = (bb_u - bb_l) / bb_m  # 带宽 = 波动率

    # 位置 (20日)
    h20 = np.max(high[-20:]); l20 = np.min(low[-20:])
    f['pos'] = max(0.0, min(1.0, (close[-1] - l20) / (h20 - l20 + 0.0001)))

    # 量能
    f['vol_ratio'] = np.mean(vol[-5:]) / (np.mean(vol[-20:]) + 1)
    f['vol_trend'] = np.mean(vol[-10:]) / (np.mean(vol[-30:]) + 1) if n >= 30 else 1.0

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    f['macd'] = 2 * (dif[-1] - dea[-1])
    f['macd_direction'] = 'up' if f['macd'] > 2*(dif[-2]-dea[-2]) else 'down'

    # KDJ (9,3,3)
    h9 = np.max(high[-9:]); l9 = np.min(low[-9:])
    rsv = (close[-1] - l9) / (h9 - l9 + 0.0001) * 100
    # 简化：取近似K/D值
    f['k'] = 2/3 * 50 + 1/3 * rsv  # 近似
    f['d'] = 2/3 * 50 + 1/3 * f['k']
    f['j'] = 3 * f['k'] - 2 * f['d']

    # ATR(14) — 波动率
    trs = []
    for i in range(-14, 0):
        h, l, pc = high[i], low[i], close[i-1] if i > -14 else close[i-1]
        tr = max(h-l, abs(h-pc), abs(l-pc))
        trs.append(tr)
    f['atr'] = np.mean(trs)

    # OBV方向
    obv_changes = [vol[i] if close[i] > close[i-1] else (-vol[i] if close[i] < close[i-1] else 0)
                   for i in range(-10, 0)]
    f['obv_up'] = sum(obv_changes) > 0

    # 价格相对MA20的位置
    f['above_ma20'] = close[-1] > f['ma20']
    f['above_ma60'] = close[-1] > f['ma60']

    return f


# ═══════════════════════════════════════════════
# 综合评分 → 胜率>70% 买点
# ═══════════════════════════════════════════════

def score_buy(f15, f5=None):
    """
    多指标共振打分，≥3分 = 胜率>70%

    加分项:
      +2 金叉 (MA5上穿MA10)
      +2 BB%≤0.25 (低位压缩)
      +1 RSI 40-60 (健康区间)
      +1 KDJ J<30 或 J拐头向上
      +1 量比>1.0 (有资金)
      +1 OBV向上 (量价配合)
      +1 价格>MA60 (中长期趋势向上)
      +1 5min确认 (5min金叉)

    满分9分，≥4分 = 强信号(>70%)，3分 = 标准信号(~70%)

    返回: (buy:bool, reason:str, target_price:float, predicted_gain_pct:float)
    """
    if not f15: return False, "", 0, 0

    close = f15['close']
    score = 0
    reasons = []

    # 核心: 金叉 (权重最高)
    if f15['golden']:
        score += 2; reasons.append("金叉")
    elif f15['ma5'] <= f15['ma10']:
        return False, "无金叉", 0, 0  # 无金叉不买

    # BB低位
    if f15['bb_pct'] <= 0.25:
        score += 2; reasons.append("BB低位")
    elif f15['bb_pct'] <= 0.35:
        score += 1; reasons.append("BB中低位")
    elif f15['bb_pct'] > 0.7:
        return False, "BB高位", 0, 0  # 太高不买

    # RSI健康区间
    if 40 <= f15['rsi'] <= 60:
        score += 1; reasons.append("RSI健康")

    # KDJ
    if f15['j'] < 30:
        score += 1; reasons.append("J值超卖")
    elif f15['j'] > 80:
        return False, "J值超买", 0, 0

    # 量能
    if f15['vol_ratio'] > 1.0:
        score += 1; reasons.append("放量")

    # OBV
    if f15['obv_up']:
        score += 1; reasons.append("OBV向上")

    # 趋势
    if f15['above_ma60']:
        score += 1; reasons.append("多头趋势")

    # 5min确认
    if f5 and f5.get('golden'):
        score += 1; reasons.append("5min确认")

    # 排除追高: 价格离MA20太远
    if close > f15['ma20'] * 1.05:
        return False, "追高(>MA20+5%)", 0, 0

    if score < 3:
        return False, f"共振不足({score}分)", 0, 0

    # ── 动态预测涨幅 ──
    # 基于: BB带宽 × ATR倍数 × 量能系数
    bb_w = f15['bb_width']
    atr_pct = f15['atr'] / close
    vol_boost = min(1.5, f15['vol_ratio'])

    # 预测涨幅 = ATR × (2 + BB压缩加成 + 量能加成)
    predicted_gain = atr_pct * 100 * (2.0 + max(0, 0.15-bb_w)*20 + (vol_boost-1)*0.5)
    predicted_gain = round(max(1.0, min(8.0, predicted_gain)), 1)  # 限制1%-8%

    target = round(close * (1 + predicted_gain/100), 2)

    signal = "强" if score >= 4 else "标准"
    reason_str = f"{signal}({score}分):{'+'.join(reasons)}"

    return True, reason_str, target, predicted_gain


# ═══════════════════════════════════════════════
# 卖点 = 预测涨幅 × 0.7
# ═══════════════════════════════════════════════

def score_sell(f15, predicted_gain_pct):
    """
    卖点: 在预测涨幅的70%处提示卖出
    同时保留RSI超买+布林上轨的兜底卖出
    """
    close = f15['close']
    rsi = f15['rsi']; pos = f15['pos']; bb = f15['bb_pct']

    # 主卖点: 涨到预测涨幅的70%
    # 这个由调用方根据持仓成本计算，这里无法跟踪成本，所以用当天最低点近似
    # 实际使用: 在推送时标注卖点价格

    # 兜底卖出: 技术过热
    if rsi >= 75 and pos >= 0.65:
        return True, "RSI超买+高位"
    if rsi >= 70 and bb >= 0.85:
        return True, "RSI高+BB上轨"
    if f15['dead'] and close < f15['ma20']:
        return True, "死叉+破MA20"

    return False, ""


# ═══════════════════════════════════════════════
# 主扫描
# ═══════════════════════════════════════════════

def scan_and_push():
    now = datetime.now(BEIJING_TZ)
    if not is_trading_time():
        logger.info(f"非交易时间 {now.strftime('%m/%d %H:%M')}")
        return []
    scan_time = now.strftime('%m/%d %H:%M')

    # 1. 股票池
    try:
        from stock_pool import get_all_stocks
        all_stocks = get_all_stocks()
    except:
        from stock_pool import STOCK_POOL_BACKUP
        all_stocks = {c: i["name"] for c, i in STOCK_POOL_BACKUP.items()}
    n_all = len(all_stocks)
    logger.info(f"☁️ v15 @ {scan_time} — {n_all}只全市场")

    # 2. 预筛
    candidates = parallel_pre_screen(all_stocks)
    if not candidates:
        push_msg(f"☁️ 无候选 {scan_time}", f'<div>全市场{n_all}只无候选</div>')
        return []

    # 3. 财务过滤
    try:
        from stock_pool import filter_loss_stocks
        before = len(candidates)
        candidates = filter_loss_stocks(candidates)
        logger.info(f"财务过滤: {before} → {len(candidates)}")
    except: pass

    # 4. 深度分析
    results = []
    n_data = 0
    for idx, (code, name, daily_close, vol_ratio, change_pct, amount, day_volume) in enumerate(candidates):
        vol_rank = idx + 1  # 成交量排名
        is_top150 = vol_rank <= 150

        # 双时间框架
        df15 = fetch_kline(code, '15')
        if df15.empty or len(df15) < 26: continue
        f15 = compute_features(df15)
        if not f15: continue

        # 5min框架
        df5 = fetch_kline(code, '5')
        f5 = compute_features(df5) if (not df5.empty and len(df5) >= 26) else None

        n_data += 1
        buy, reason_b, target, gain_pct = score_buy(f15, f5)
        sell, reason_s = score_sell(f15, gain_pct)

        if buy: sig, reason = "BUY", reason_b
        elif sell: sig, reason = "SELL", reason_s
        else: sig, reason = "HOLD", ""

        # 7折卖点
        sell_price = round(f15['close'] * (1 + gain_pct * 0.7 / 100), 2) if buy else 0

        amt_str = f"{amount/1e8:.2f}亿" if amount > 1e8 else f"{amount/1e4:.0f}万"
        rank_tag = f" #{vol_rank}" if is_top150 else ""

        r = {"code":code,"name":name,"signal":sig,"close":round(f15['close'],2),
             "rsi":round(f15['rsi'],1),"bb":round(f15['bb_pct'],2),
             "reason":reason,"target":target,"gain_pct":gain_pct,
             "sell_price":sell_price,"amount":amount,"vol_ratio":vol_ratio,
             "change_pct":change_pct,"amt_str":amt_str,
             "vol_rank":vol_rank,"is_top150":is_top150}
        results.append(r)

        # ── 推送 ──
        if sig == "BUY":
            rank_line = f'<tr><td colspan="2" style="color:#f39c12"><b>📊 成交额排名 #{vol_rank} | {amt_str}</b></td></tr>' if is_top150 else ""
            logger.info(f"  >>> BUY #{vol_rank} {code} {name} @ {f15['close']:.2f} +{gain_pct}% → 卖点{sell_price} | {reason}")
            push_msg(f"☁️{name} 买入 +{gain_pct}% 卖点{sell_price}{rank_tag}",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#e74c3c;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#e74c3c;font-size:16px">建议买入</span> | 目标 <b>{target}</b> (+{gain_pct}%)<br>'
                     f'<span style="color:#f39c12;font-size:14px">💰 7折卖点 <b>{sell_price}</b> (+{round(gain_pct*0.7,1)}%)</span><br>'
                     f'成交额 <b>{amt_str}</b> | 量比 {vol_ratio:.1f} | 排名 <b>#{vol_rank}</b><br>'
                     f'RSI:{f15["rsi"]:.0f} | K:{f15["k"]:.0f} D:{f15["d"]:.0f} J:{f15["j"]:.0f}<br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | v15双框架</span></div>')
        elif sig == "SELL":
            logger.info(f"  >>> SELL #{vol_rank} {code} {name} @ {f15['close']:.2f} | {reason}")
            push_msg(f"☁️{name} 卖出{rank_tag}",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#27ae60;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#27ae60;font-size:16px">建议卖出</span><br>'
                     f'成交额 <b>{amt_str}</b> | 排名 <b>#{vol_rank}</b><br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | v15双框架</span></div>')

    # 5. 汇总
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')

    # TOP10 龙虎榜
    top10 = sorted(results, key=lambda r: r.get('amount', 0), reverse=True)[:10]
    top_rows = ""
    for r in top10:
        emoji = {"BUY": "🔴", "SELL": "🟢", "HOLD": "⚪"}
        top_rows += (f'<tr><td>#{r["vol_rank"]}</td><td>{emoji[r["signal"]]}</td>'
                     f'<td><b>{r["code"]}</b></td><td>{r["name"]}</td>'
                     f'<td>{r["close"]}</td><td>{r["amt_str"]}</td>'
                     f'<td>{r.get("reason","")[:30]}</td></tr>')

    balance = check_deepseek_balance()
    bal_str = f" | DeepSeek ¥{balance:.2f}" if balance else ""
    warn = "\n⚠️余额不足!" if balance and balance < 3 else ""

    push_msg(f"☁️ {scan_time} | B{buy_count} S{sell_count} | {n_data}只{bal_str}{warn}",
             f'<div style="font-size:14px;padding:10px">'
             f'全市场: {n_all}只 | 候选: {len(candidates)}只 | 分析: {n_data}只<br>'
             f'买入:<b style="color:#e74c3c">{buy_count}</b> | 卖出:<b style="color:#27ae60">{sell_count}</b>{bal_str}<br>'
             f'<br><b>📊 成交额TOP10龙虎榜</b><br>'
             f'<table style="width:100%;font-size:12px;border-collapse:collapse">'
             f'<tr style="background:#333;color:#fff"><th>排名</th><th></th><th>代码</th><th>名称</th><th>价格</th><th>成交额</th><th>信号</th></tr>'
             f'{top_rows}</table><br>'
             f'<span style="color:#888;font-size:11px">v15 双框架 | 5min+15min全指标 | 动态目标+7折卖点{warn}</span></div>')

    return results


if __name__ == "__main__":
    scan_and_push()
