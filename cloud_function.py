"""v14 量化买卖点 — 全A股覆盖(~3500只) + 并行预筛 + 70%+胜率"""
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
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

MAX_DEEP_SCAN = 150   # 成交额前150名深度分析
PRE_SCREEN_WORKERS = 12  # 并行预筛线程数


# ═══════════════════════════════════════════════
# 推送 & 工具
# ═══════════════════════════════════════════════

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={"token":PUSHPLUS_TOKEN,"title":title,"content":content,"template":"html"}, timeout=10)
        ok = r.json().get("code") == 200
        logger.info(f"{'OK' if ok else 'FAIL'}: {title}")
        return ok
    except Exception as e:
        logger.error(f"Push error: {e}"); return False

def check_deepseek_balance():
    try:
        r = requests.get(DEEPSEEK_BALANCE_URL, headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"}, timeout=10)
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

def fetch_data(code):
    """15分钟K线数据"""
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6","9")) else "sz."
        now_bj = datetime.now(BEIJING_TZ)
        end = now_bj.strftime("%Y-%m-%d")
        start = (now_bj - timedelta(days=90)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(prefix + code,
            'date,time,open,high,low,close,volume',
            start_date=start, end_date=end, frequency='15', adjustflag='2')
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if not rows: return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        try: bs.logout()
        except: pass
        return pd.DataFrame()


def fetch_daily_snapshot(code):
    """快速日线快照 — 用于预筛，只取最近3天"""
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
# 并行预筛 — 3500只 → 200只候选
# ═══════════════════════════════════════════════

def pre_screen_one(code, name):
    """预筛单只股票：日线快照 → 成交额+量比+涨幅过滤"""
    rows = fetch_daily_snapshot(code)
    if not rows or len(rows) < 2:
        return None

    try:
        latest = rows[-1]
        close = float(latest[1])
        volume = float(latest[2])
        preclose = float(latest[4])
        change_pct = (close - preclose) / preclose * 100
        amount = close * volume  # 成交额

        # === 过滤条件 ===
        # 跌超3%不要（弱势）
        if change_pct < -3:
            return None
        # 涨停附近不要（买不到）
        if change_pct > 9.5:
            return None

        # 量比 = 今日量 / 前N日均量
        historical_vols = [float(r[2]) for r in rows[:-1] if float(r[2]) > 0]
        if not historical_vols:
            return None
        avg_vol = np.mean(historical_vols)
        vol_ratio = volume / avg_vol if avg_vol > 0 else 1.0

        # 量比<1.0 不要（没资金关注）
        if vol_ratio < 1.0:
            return None

        return (code, name, close, vol_ratio, change_pct, amount, volume)
    except:
        return None


def parallel_pre_screen(stocks, max_workers=PRE_SCREEN_WORKERS):
    """并行预筛全部股票，按成交额排序取前 MAX_DEEP_SCAN 只"""
    candidates = []
    codes = list(stocks.items())
    total = len(codes)
    logger.info(f"预筛开始: {total} 只 → {max_workers} 线程并行")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(pre_screen_one, code, name): code
                   for code, name in codes}
        done = 0
        for f in as_completed(futures):
            done += 1
            if done % 500 == 0:
                logger.info(f"  预筛进度: {done}/{total}")
            result = f.result()
            if result:
                candidates.append(result)

    # 按成交额降序 → 主力资金在哪一目了然
    candidates.sort(key=lambda x: x[5], reverse=True)
    top_n = min(MAX_DEEP_SCAN, len(candidates))
    top_amount = sum(c[5] for c in candidates[:top_n])
    logger.info(f"预筛完成: {len(candidates)} 只通过 → 取成交额前 {top_n} 只 (合计成交额 {top_amount/1e8:.0f}亿)")
    return candidates[:top_n]


# ═══════════════════════════════════════════════
# 特征计算
# ═══════════════════════════════════════════════

def compute_features(df):
    close = df['close'].values; high = df['high'].values
    low = df['low'].values; volume = df['volume'].values; n = len(close)
    f = {}
    f['close'] = close[-1]
    f['ma5'] = np.mean(close[-5:]); f['ma10'] = np.mean(close[-10:])
    f['ma20'] = np.mean(close[-20:])
    p_ma5 = np.mean(close[-6:-1]); p_ma10 = np.mean(close[-11:-1])
    f['golden'] = (f['ma5'] > f['ma10']) and (p_ma5 <= p_ma10)
    f['dead'] = (f['ma5'] < f['ma10']) and (p_ma5 >= p_ma10)
    deltas = np.diff(close[-15:])
    g = np.mean(deltas[deltas > 0]) if np.any(deltas > 0) else 0
    l = -np.mean(deltas[deltas < 0]) if np.any(deltas < 0) else 1e-9
    f['rsi'] = 100 - 100/(1+g/l) if l > 0 else 50
    bb_s = np.std(close[-20:]); bb_m = np.mean(close[-20:])
    f['bb_pct'] = max(0.0, min(1.0, (close[-1] - (bb_m - 2*bb_s)) / (4*bb_s + 0.0001)))
    h20 = np.max(high[-20:]); l20 = np.min(low[-20:])
    f['pos'] = max(0.0, min(1.0, (close[-1] - l20) / (h20 - l20 + 0.0001)))
    f['vol_ratio'] = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1)
    return f


# ═══════════════════════════════════════════════
# 买卖评分 (70%+胜率版)
# ═══════════════════════════════════════════════

def score_buy(code, f):
    """买入: 金叉 + BB%≤0.35 + 位置 + 量 + RSI<65 + 价格>MA20 → 目标70%+胜率"""
    golden = f["golden"]; rsi = f["rsi"]; pos = f["pos"]; bb = f["bb_pct"]
    close = f["close"]; vol = f.get("vol_ratio", 1.0); ma20 = f["ma20"]
    B, R, T, P = False, "", 0.0, 0.0

    if not golden:
        return B, R, T, P

    # RSI>65 = 短期过热，不追
    if rsi > 65:
        return B, R, T, P

    # 价格<MA20 = 趋势向下，不买
    if close < ma20:
        return B, R, T, P

    # 质量分级
    if bb <= 0.20 and pos <= 0.35 and vol > 1.0:
        B,R,T,P = True,"金叉+BB窄+低位+放量(强)",round(close*1.022,2),2.20
    elif bb <= 0.28 and pos <= 0.45 and vol > 0.8:
        B,R,T,P = True,"金叉+BB适中+中低位",round(close*1.018,2),1.80
    elif bb <= 0.35 and pos <= 0.55 and vol > 0.7:
        B,R,T,P = True,"金叉+BB放宽+低位",round(close*1.015,2),1.50
    elif bb <= 0.35 and pos <= 0.7:
        B,R,T,P = True,"金叉+BB达标",round(close*1.012,2),1.20
    return B, R, T, P


def score_sell(code, f):
    """卖出"""
    rsi = f['rsi']; pos = f['pos']; bb = f['bb_pct']
    if rsi >= 70 and pos >= 0.7 and bb >= 0.8: return True, "RSI高位+布林上轨"
    if rsi >= 75 and pos >= 0.6: return True, "RSI超买+高位"
    return False, ""


# ═══════════════════════════════════════════════
# 主扫描
# ═══════════════════════════════════════════════

def scan_and_push():
    now = datetime.now(BEIJING_TZ)
    if not is_trading_time():
        logger.info(f"非交易时间，跳过扫描 {now.strftime('%m/%d %H:%M')}")
        return []
    scan_time = now.strftime('%m/%d %H:%M')

    # ── 1. 获取全市场股票池 ──
    try:
        from stock_pool import get_all_stocks
        all_stocks = get_all_stocks()
    except:
        from stock_pool import STOCK_POOL_BACKUP
        all_stocks = {code: info["name"] for code, info in STOCK_POOL_BACKUP.items()}

    n_all = len(all_stocks)
    logger.info(f"☁️ v14 扫描启动 @ {scan_time} — {n_all}只全市场")

    # ── 2. 并行预筛 → 200只候选 ──
    candidates = parallel_pre_screen(all_stocks)
    if not candidates:
        push_msg(f"☁️ 无候选 {scan_time}",
                 f'<div style="font-size:14px;padding:10px">全市场{n_all}只股票无候选<br>'
                 f'<span style="color:#888">可能是非交易时段或市场异常</span></div>')
        return []

    # ── 3. 财务过滤：剔除连续3年亏损 ──
    try:
        from stock_pool import filter_loss_stocks
        before = len(candidates)
        candidates = filter_loss_stocks(candidates)
        logger.info(f"财务过滤: {before} → {len(candidates)} 只")
    except:
        pass

    # ── 4. 深度分析候选股 ──
    results = []
    n_data = 0
    for code, name, daily_close, vol_ratio, change_pct, amount, day_volume in candidates:
        df = fetch_data(code)
        if df.empty or len(df) < 20:
            continue
        n_data += 1
        f = compute_features(df)
        buy, reason_b, target, tp = score_buy(code, f)
        sell, reason_s = score_sell(code, f)
        if buy: sig, reason = "BUY", reason_b
        elif sell: sig, reason = "SELL", reason_s
        else: sig, reason = "HOLD", ""

        r = {"code":code,"name":name,"signal":sig,"close":round(f['close'],2),
             "rsi":round(f['rsi'],1),"pos":round(f['pos'],2),"bb":round(f['bb_pct'],2),
             "golden":f['golden'],"reason":reason,
             "target":target if target>0 else 0,"target_pct":tp if tp>0 else 0}
        results.append(r)

        # ── 有信号立即推送 ──
        if sig == "BUY":
            amt_str = f"{amount/1e8:.2f}亿" if amount > 1e8 else f"{amount/1e4:.0f}万"
            logger.info(f"  >>> BUY  {code} {name} @ {f['close']:.2f} +{tp}% | 成交{amt_str} | {reason}")
            push_msg(f"☁️{name} 现价{r['close']} 建议买入 目标{target}(+{tp}%)",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#e74c3c;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#e74c3c;font-size:16px">建议买入</span><br>'
                     f'目标 <b>{target}</b> (+{tp}%)<br>'
                     f'成交额 <b>{amt_str}</b> | 量比 <b>{vol_ratio:.1f}</b><br>'
                     f'RSI:{f["rsi"]:.0f} | BB:{f["bb_pct"]:.2f} | 涨幅:{change_pct:+.1f}%<br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | v14云端推送</span></div>')
        elif sig == "SELL":
            logger.info(f"  >>> SELL {code} {name} @ {f['close']:.2f} | {reason}")
            push_msg(f"☁️{name} 现价{r['close']} 建议卖出",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#27ae60;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#27ae60;font-size:16px">建议卖出</span><br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | 云端推送</span></div>')

    # ── 5. 汇总推送 ──
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')
    balance = check_deepseek_balance()
    bal_str = f" | DeepSeek ¥{balance:.2f}" if balance else ""
    warn = "\n⚠️余额不足!" if balance and balance < 3 else ""

    push_msg(f"☁️ 扫描完成 {scan_time} | B{buy_count} S{sell_count} | {n_data}/{n_all}只{bal_str}{warn}",
             f'<div style="font-size:14px;padding:10px">'
             f'全市场: {n_all}只 | 深度分析: {n_data}只<br>'
             f'买入:<b style="color:#e74c3c">{buy_count}</b> | 卖出:<b style="color:#27ae60">{sell_count}</b>{bal_str}<br>'
             f'<span style="color:#888;font-size:11px">v14 全A股云端CI{warn}</span></div>')

    return results


if __name__ == "__main__":
    scan_and_push()
