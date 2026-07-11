"""v13 量化买卖点 — 41只股票 逐票独立+ETF+通用 (Gitee同步)"""
import os, sys, json, logging, requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone

BEIJING_TZ = timezone(timedelta(hours=8))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_KEY", "")
DEEPSEEK_BALANCE_URL = "https://api.deepseek.com/user/balance"

try:
    from stock_pool import STOCK_POOL
    STOCKS = {code: info["name"] for code, info in STOCK_POOL.items()}
except:
    STOCKS = {"000933":"神火","002497":"雅化","000960":"锡业","000893":"亚钾"}

def push_msg(title, content):
    try:
        r = requests.post(PUSHPLUS_URL, json={"token":PUSHPLUS_TOKEN,"title":title,"content":content,"template":"html"}, timeout=10)
        ok = r.json().get("code") == 200
        logger.info(f"{'OK' if ok else 'FAIL'}: {title}")
        return ok
    except Exception as e:
        logger.error(f"Push error: {e}"); return False

def check_deepseek_balance():
    """返回 CNY 余额，失败返回 None"""
    try:
        r = requests.get(DEEPSEEK_BALANCE_URL,
                        headers={"Authorization": f"Bearer {DEEPSEEK_KEY}"}, timeout=10)
        data = r.json()
        for b in data.get("balance_infos", []):
            if b["currency"] == "CNY":
                return float(b["total_balance"])
    except: pass
    return None

def fetch_data(code):
    """纯云端数据获取 — 不依赖本地缓存"""
    import baostock as bs
    bs.login()
    try:
        prefix = "sh." if code.startswith(("6","9")) else "sz."
        now_bj = datetime.now(BEIJING_TZ)
        end = now_bj.strftime("%Y-%m-%d")
        # 云端用90天数据即可，太久太慢
        start = (now_bj - timedelta(days=90)).strftime("%Y-%m-%d")
        rs = bs.query_history_k_data_plus(prefix + code,
            'date,time,open,high,low,close,volume',
            start_date=start, end_date=end, frequency='15', adjustflag='2')
        rows = []
        while (rs.error_code == '0') & rs.next():
            rows.append(rs.get_row_data())
        bs.logout()
        if not rows:
            logger.warning(f"{code} 无数据")
            return pd.DataFrame()
        df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
        for c in ['open','high','low','close','volume']:
            df[c] = pd.to_numeric(df[c], errors='coerce')
        return df
    except Exception as e:
        try: bs.logout()
        except: pass
        logger.error(f"{code} fetch error: {e}")
        return pd.DataFrame()

def compute_features(df):
    """特征计算 — BB%修正: 0=下轨, 1=上轨"""
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
    # BB% = (close - lower) / (upper - lower), clipped to 0-1
    bb_s = np.std(close[-20:]); bb_m = np.mean(close[-20:])
    f['bb_pct'] = max(0.0, min(1.0, (close[-1] - (bb_m - 2*bb_s)) / (4*bb_s + 0.0001)))
    h20 = np.max(high[-20:]); l20 = np.min(low[-20:])
    f['pos'] = max(0.0, min(1.0, (close[-1] - l20) / (h20 - l20 + 0.0001)))
    f['vol_ratio'] = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1)
    # MACD快速判断
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    macd_h = 2 * ((ema12[-1] - ema26[-1]) - pd.Series(ema12-ema26).ewm(span=9, adjust=False).mean().values[-1])
    f['macd_turning'] = macd_h > 0 and (2 * ((ema12[-2] - ema26[-2]) - pd.Series(ema12-ema26).ewm(span=9, adjust=False).mean().values[-2])) <= 0
    return f

def score_buy(code, f):
    """买入: 金叉优先 + BB%≤0.35 + 量不萎缩 (去掉MACD, 目标65-75%胜率)"""
    golden = f["golden"]; rsi = f["rsi"]; pos = f["pos"]; bb = f["bb_pct"]
    close = f["close"]; vol = f.get("vol_ratio", 1.0)
    B, R, T, P = False, "", 0.0, 0.0

    if not golden:
        return B, R, T, P  # 无金叉，不买

    # 质量分级 — BB越窄+位置越低+量越大 = 信号越强
    if bb <= 0.20 and pos <= 0.35 and vol > 1.0:
        # 强信号: BB极窄+低位+放量
        B,R,T,P = True,"金叉+BB窄+低位+放量(强)",round(close*1.022,2),2.20
    elif bb <= 0.28 and pos <= 0.45 and vol > 0.8:
        # 中信号: BB适中+中低位+量正常
        B,R,T,P = True,"金叉+BB适中+中低位",round(close*1.018,2),1.80
    elif bb <= 0.35 and pos <= 0.55 and vol > 0.7:
        # 标准信号: BB放宽+不追高
        B,R,T,P = True,"金叉+BB放宽+低位",round(close*1.015,2),1.50
    elif bb <= 0.35 and pos <= 0.7:
        # 宽松信号: BB达标但位置稍高
        B,R,T,P = True,"金叉+BB达标",round(close*1.012,2),1.20
    return B, R, T, P


def score_sell(code, f):
    """卖出: 逐票独立 + ETF + 通用"""
    rsi = f['rsi']; pos = f['pos']; bb = f['bb_pct']

    # === 纳指ETF 卖出 ===
    if code in ("159941", "513100"):
        if rsi >= 70 and bb >= 0.85: return True, "RSI70+布林上轨(纳指)"
        if rsi >= 75 and pos >= 0.7: return True, "RSI75+高位(纳指)"

    if code == "000933":
        if rsi >= 75 and pos >= 0.8 and bb >= 0.8: return True, "RSI75+高位+布林上轨"
        if rsi >= 70 and pos >= 0.8 and bb >= 0.85: return True, "RSI70+高位+布林上轨"
    elif code == "002497":
        if rsi >= 65 and pos >= 0.8 and bb >= 0.85: return True, "RSI65+高位+布林上轨"
        if rsi >= 75 and pos >= 0.8 and bb >= 0.8: return True, "RSI75+高位+布林上轨"
    elif code == "000960":
        if rsi >= 75 and pos >= 0.6 and bb >= 0.85: return True, "RSI75+高位+布林上轨"
        if rsi >= 65 and pos >= 0.7 and bb >= 0.85: return True, "RSI65+高位+布林上轨"
    elif code == "000893":
        if rsi >= 75 and pos >= 0.6 and bb >= 0.8: return True, "RSI75+高位+布林上轨"
        if rsi >= 70 and pos >= 0.6 and bb >= 0.85: return True, "RSI70+高位+布林上轨"
    else:
        if rsi >= 70 and pos >= 0.7 and bb >= 0.8: return True, "RSI高位+布林上轨"
        if rsi >= 75 and pos >= 0.6: return True, "RSI超买+高位"
    return False, ""

def is_trading_time():
    """判断是否交易时间：工作日 9:15-15:00（北京时间）"""
    now = datetime.now(BEIJING_TZ)
    if now.weekday() >= 5:
        return False
    h, m = now.hour, now.minute
    if (h == 9 and m >= 15) or (h == 10) or (h == 11 and m <= 30):
        return True
    if (h == 13) or (h == 14) or (h == 15 and m == 0):
        return True
    return False

def scan_and_push():
    now = datetime.now(BEIJING_TZ)
    if not is_trading_time():
        logger.info(f"非交易时间，跳过扫描 {now.strftime('%m/%d %H:%M')}")
        return []
    scan_time = now.strftime('%m/%d %H:%M')
    logger.info(f"Scan @ {scan_time} — 云端CI推送")
    
    # ====== 先发一条诊断推送，确认云端通道畅通 ======
    n_total = len(STOCKS)
    push_msg(f"☁️ 云端扫描启动 {scan_time}", 
             f'<div style="font-size:14px;padding:10px">'
             f'<b>云端CI扫描中</b> — {n_total}只股票<br>'
             f'<span style="color:#888;font-size:11px">GitHub Actions | 关机也能推送</span></div>')

    results = []
    n_data = 0
    for code, name in STOCKS.items():
        df = fetch_data(code)
        if df.empty or len(df) < 20:
            results.append({"code":code,"name":name,"signal":"NODATA","close":0})
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

        # === 有信号立即单独推送 ===
        if sig == "BUY":
            logger.info(f"  >>> BUY  {code} {name} @ {f['close']:.2f} +{tp}% | {reason}")
            t = r['target']; tp_val = r['target_pct']
            push_msg(f"☁️{name} 现价{r['close']} 建议买入 目标{t}(+{tp_val}%)",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#e74c3c;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#e74c3c;font-size:16px">建议买入</span><br>'
                     f'目标 <b>{t}</b> (+{tp_val}%)<br>'
                     f'T+1可卖 | {reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | 云端推送</span></div>')
        elif sig == "SELL":
            logger.info(f"  >>> SELL {code} {name} @ {f['close']:.2f} | {reason}")
            push_msg(f"☁️{name} 现价{r['close']} 建议卖出",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#27ae60;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#27ae60;font-size:16px">建议卖出</span><br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{scan_time} | 云端推送</span></div>')

    # ====== 扫描完成汇总推送 ======
    buy_count = sum(1 for r in results if r['signal'] == 'BUY')
    sell_count = sum(1 for r in results if r['signal'] == 'SELL')
    balance = check_deepseek_balance()
    bal_str = f" | DeepSeek ¥{balance:.2f}" if balance else ""
    warn = "\n⚠️余额不足请充值!" if balance and balance < 3 else ""
    push_msg(f"☁️ 扫描完成 {scan_time} | B{buy_count} S{sell_count} | {n_data}/{n_total}只{bal_str}{warn}",
             f'<div style="font-size:14px;padding:10px">'
             f'数据: {n_data}/{n_total} | 买入:<b style="color:#e74c3c">{buy_count}</b> | 卖出:<b style="color:#27ae60">{sell_count}</b>{bal_str}<br>'
             f'<span style="color:#888;font-size:11px">云端CI运行正常{warn}</span></div>')

    return results

if __name__ == "__main__":
    scan_and_push()
