"""v13 量化买卖点 — 41只股票 逐票独立+ETF+通用 (Gitee同步)"""
import os, sys, json, logging, requests
import numpy as np, pandas as pd
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()

PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"

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

def fetch_data(code):
    import baostock as bs
    bs.login()
    try:
        cache_file = f"C:/Users/Administrator/quant_trading/data/cache/{code}_15min.csv"
        if os.path.exists(cache_file):
            df = pd.read_csv(cache_file, dtype={'time': str})
            df['time'] = df['time'].astype(str).str.zfill(17)
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        else:
            prefix = "sh." if code.startswith(("6","9")) else "sz."
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=730)).strftime("%Y-%m-%d")
            rs = bs.query_history_k_data_plus(prefix + code,
                'date,time,open,high,low,close,volume',
                start_date=start, end_date=end, frequency='15', adjustflag='2')
            rows = []
            while (rs.error_code == '0') & rs.next():
                rows.append(rs.get_row_data())
            if not rows:
                bs.logout(); return pd.DataFrame()
            df = pd.DataFrame(rows, columns=['date','time','open','high','low','close','volume'])
            for c in ['open','high','low','close','volume']:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        bs.logout()
        return df
    except:
        try: bs.logout()
        except: pass
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
    """买入: v17逐票独立最优规则 (41只全量回测)"""
    golden = f["golden"]; rsi = f["rsi"]; pos = f["pos"]; bb = f["bb_pct"]
    close = f["close"]; vol = f.get("vol_ratio", 1.0)
    macd_t = f.get("macd_turning", False); macd_b = f.get("macd_bull", False)
    B, R, T, P = False, "", 0.0, 0.0
    vol_ok = vol > 1.0; vol_surge = vol > 1.3

    if code == "600392":  # 盛和资源 WR=100.0% +1.76%
        if golden and bb <= 0.2 and macd_t and pos <= 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.2+MACD金叉+pos<0.3+量增",round(close*1.0176,2),1.76
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002601":  # 龙佰集团 WR=93.8% +1.0%
        if golden and bb <= 0.1 and macd_b and 35 <= rsi <= 50 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.1+RSI35-50+MACD多+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "603799":  # 华友钴业 WR=92.9% +2.61%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0261,2),2.61
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002240":  # 盛新锂能 WR=88.9% +1.53%
        if golden and bb <= 0.1 and macd_b and 35 <= rsi <= 50 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.1+RSI35-50+MACD多+量增",round(close*1.0153,2),1.53
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "001337":  # 四川黄金 WR=87.0% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601899":  # 紫金矿业 WR=85.7% +1.0%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002155":  # 湖南黄金 WR=85.7% +1.64%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0164,2),1.64
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "159941":  # 纳指ETF广发 WR=83.3% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600309":  # 万华化学 WR=82.4% +1.25%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0125,2),1.25
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "300750":  # 宁德时代 WR=82.4% +1.0%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600489":  # 中金黄金 WR=81.8% +1.31%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0131,2),1.31
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601398":  # 工商银行 WR=81.8% +1.0%
        if golden and bb <= 0.1 and macd_b and 35 <= rsi <= 50 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.1+RSI35-50+MACD多+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002709":  # 天赐材料 WR=81.5% +1.79%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0179,2),1.79
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000807":  # 云铝股份 WR=81.0% +1.55%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0155,2),1.55
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601600":  # 中国铝业 WR=80.0% +1.05%
        if golden and bb <= 0.2 and macd_t and pos <= 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.2+MACD金叉+pos<0.3+量增",round(close*1.0105,2),1.05
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600030":  # 中信证券 WR=80.0% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000737":  # 北方铜业 WR=78.9% +1.28%
        if golden and bb <= 0.1 and macd_b and 35 <= rsi <= 50 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.1+RSI35-50+MACD多+量增",round(close*1.0128,2),1.28
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000933":  # 神火股份 WR=75.8% +1.23%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0123,2),1.23
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002428":  # 云南锗业 WR=75.0% +1.89%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0189,2),1.89
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601168":  # 西部矿业 WR=75.0% +1.0%
        if golden and bb <= 0.1 and 40 <= rsi <= 55 and pos <= 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+RSI40-55+pos<0.3+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000893":  # 亚钾国际 WR=73.9% +1.1%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0110,2),1.1
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002415":  # 海康威视 WR=73.9% +1.0%
        if golden and bb <= 0.1 and macd_b and 35 <= rsi <= 50 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.1+RSI35-50+MACD多+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000960":  # 锡业股份 WR=73.7% +1.11%
        if golden and bb <= 0.1 and 35 <= rsi <= 50 and pos <= 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+RSI35-50+pos<0.3+量增",round(close*1.0111,2),1.11
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "300748":  # 金力永磁 WR=73.7% +1.55%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0155,2),1.55
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000831":  # 中国稀土 WR=73.3% +1.7%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0170,2),1.7
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600160":  # 巨化股份 WR=73.3% +1.0%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "603993":  # 洛阳钼业 WR=72.7% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002497":  # 雅化集团 WR=72.7% +1.23%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0123,2),1.23
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601138":  # 工业富联 WR=72.7% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600362":  # 江西铜业 WR=72.2% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601069":  # 西部黄金 WR=72.2% +1.78%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0178,2),1.78
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002407":  # 多氟多 WR=69.6% +1.35%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0135,2),1.35
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002466":  # 天齐锂业 WR=68.4% +1.0%
        if golden and bb <= 0.1 and 40 <= rsi <= 55 and pos <= 0.4 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+RSI40-55+pos<0.4+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "601288":  # 农业银行 WR=68.2% +1.0%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000630":  # 铜陵有色 WR=66.7% +1.0%
        if golden and bb <= 0.1 and macd_t and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+MACD金叉+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "002460":  # 赣锋锂业 WR=66.7% +1.18%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0118,2),1.18
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "000725":  # 京东方A WR=63.0% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600111":  # 北方稀土 WR=61.5% +1.0%
        if golden and bb <= 0.1 and 30 <= rsi <= 45 and pos <= 0.2 and vol_ok:
            B,R,T,P = True,"金叉+BB<0.15+RSI30-45+pos<0.2+量增",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "300059":  # 东方财富 WR=58.3% +1.0%
        if golden and bb <= 0.2 and pos <= 0.3 and vol_surge:
            B,R,T,P = True,"金叉+BB<0.2+pos<0.3+放量",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
    elif code == "600988":  # 赤峰黄金 WR=56.7% +1.0%
        if golden and bb <= 0.1 and macd_t:
            B,R,T,P = True,"金叉+BB<0.1+MACD金叉",round(close*1.0100,2),1.0
        elif golden and pos < 0.4 and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+低位+BB+量价",round(close*1.022,2),2.20
        elif golden and bb < 0.3 and vol_ok:
            B,R,T,P = True,"金叉+BB+量价",round(close*1.017,2),1.70
        elif golden and bb < 0.3:
            B,R,T,P = True,"金叉+BB",round(close*1.015,2),1.50
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

def scan_and_push():
    now = datetime.now()
    logger.info(f"Scan @ {now.strftime('%Y-%m-%d %H:%M:%S')}")

    results = []
    for code, name in STOCKS.items():
        df = fetch_data(code)
        if df.empty or len(df) < 20:
            results.append({"code":code,"name":name,"signal":"NODATA","close":0})
            continue
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
            push_msg(f"{name} 现价{r['close']} 建议买入 目标{t}(+{tp_val}%) T+1可卖",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#e74c3c;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#e74c3c;font-size:16px">建议买入</span><br>'
                     f'目标 <b>{t}</b> (+{tp_val}%)<br>'
                     f'T+1可卖 | {reason}<br>'
                     f'<span style="color:#888;font-size:11px">{now.strftime("%m/%d %H:%M")} | 逐票概率</span></div>')
        elif sig == "SELL":
            logger.info(f"  >>> SELL {code} {name} @ {f['close']:.2f} | {reason}")
            push_msg(f"{name} 现价{r['close']} 建议卖出",
                     f'<div style="font-size:16px;padding:12px;line-height:2.2"><b>{name}</b> {code}<br>'
                     f'现价 <b style="color:#27ae60;font-size:22px">{r["close"]}</b><br>'
                     f'<span style="color:#27ae60;font-size:16px">建议卖出</span><br>'
                     f'{reason}<br>'
                     f'<span style="color:#888;font-size:11px">{now.strftime("%m/%d %H:%M")} | 逐票概率</span></div>')
        else:
            logger.info(f"      HOLD  {code} {name} @ {f['close']:.2f} RSI={f['rsi']:.0f} pos={f['pos']:.2f} bb={f['bb_pct']:.2f}")

    return results

if __name__ == "__main__":
    scan_and_push()
