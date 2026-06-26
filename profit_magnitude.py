"""
逐票买点获利幅度分析 — 不只胜率，还要看3-5天能赚多少
"""
import baostock as bs
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

bs.login()
CACHE_DIR = "C:/Users/Administrator/quant_trading/data/cache"

STOCKS = {"000933": "神火", "002497": "雅化", "000960": "锡业", "000893": "亚钾"}

for code, name in STOCKS.items():
    cache_file = f"{CACHE_DIR}/{code}_15min.csv"
    df = pd.read_csv(cache_file, dtype={'time': str})
    df['time'] = df['time'].astype(str).str.zfill(17)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    volume = df['volume'].values
    n = len(close)

    # Features
    ma5  = pd.Series(np.convolve(close, np.ones(5)/5, mode='same'))
    ma10 = pd.Series(np.convolve(close, np.ones(10)/10, mode='same'))
    ma20 = pd.Series(np.convolve(close, np.ones(20)/20, mode='same'))
    p_ma5  = ma5.shift(1); p_ma10 = ma10.shift(1)
    golden = (ma5 > ma10) & (p_ma5 <= p_ma10)

    # RSI
    deltas = np.diff(close)
    rsi = np.full(n, 50.0)
    for i in range(14, n):
        g = np.sum(np.maximum(deltas[i-14:i], 0))
        l = np.sum(np.abs(np.minimum(deltas[i-14:i], 0)))
        rsi[i] = 100 - 100/(1+g/l) if l > 0 else 100

    # BB
    bb_mid = ma20.values
    bb_std = np.array([np.std(close[max(0,i-19):i+1]) for i in range(n)])
    bb_upper = bb_mid + 2*bb_std
    bb_lower = bb_mid - 2*bb_std
    bb_pct = np.clip((close - bb_lower) / (bb_upper - bb_lower + 0.0001), 0, 1)

    # Position
    h20 = pd.Series(high).rolling(20).max().values
    l20 = pd.Series(low).rolling(20).min().values
    pos = np.clip((close - l20) / (h20 - l20 + 0.0001), 0, 1)

    # MACD
    ema12 = pd.Series(close).ewm(span=12, adjust=False).mean().values
    ema26 = pd.Series(close).ewm(span=26, adjust=False).mean().values
    dif = ema12 - ema26
    dea = pd.Series(dif).ewm(span=9, adjust=False).mean().values
    macd_hist = 2*(dif - dea)
    macd_turning = (macd_hist > 0) & (np.roll(macd_hist, 1) <= 0)

    vol20_pad = np.array([np.mean(volume[max(0,i-19):i+1]) for i in range(n)])
    vol5_pad  = np.array([np.mean(volume[max(0,i-4):i+1]) for i in range(n)])
    vol_ratio = vol5_pad / (vol20_pad + 1)

    # ====== Analyze each buy pattern's profit distribution ======
    print(f"\n{'='*70}")
    print(f"  {name} ({code}) — 买点获利幅度分析 (1日=16根15min)")
    print(f"{'='*70}")

    patterns = {
        "金叉+低位(pos<0.4)+布林下轨(BB<0.3)": golden & (pos < 0.4) & (bb_pct < 0.3),
        "金叉+布林下轨(BB<0.3)": golden & (bb_pct < 0.3),
        "金叉+布林下轨(BB<0.3)+MACD转正": golden & (bb_pct < 0.3) & macd_turning,
        "金叉+RSI40-55+低位+布林下轨": golden & (rsi >= 40) & (rsi <= 55) & (pos < 0.4) & (bb_pct < 0.3),
        "金叉+RSI30-50+低位+布林下轨": golden & (rsi >= 30) & (rsi <= 50) & (pos < 0.4) & (bb_pct < 0.3),
        "金叉+低位+布林下轨+放量(vol>1.2)": golden & (pos < 0.4) & (bb_pct < 0.3) & (vol_ratio > 1.2),
    }

    for pat_name, mask in patterns.items():
        sig_count = mask.sum()
        if sig_count < 8:
            continue

        # Forward returns at 16bar(1d), 32(2d), 48(3d), 64(4d), 80(5d)
        horizons = [(16, "1日"), (32, "2日"), (48, "3日"), (64, "4日"), (80, "5日")]

        best_h = None
        best_metrics = None

        for h_bars, h_name in horizons:
            fwd_ret = np.full(n, np.nan)
            for i in range(n - h_bars):
                fwd_ret[i] = (close[i + h_bars] - close[i]) / close[i] * 100

            rets = fwd_ret[mask]
            rets = rets[~np.isnan(rets)]
            if len(rets) < 8:
                continue

            wr = (rets > 0).mean() * 100
            avg = rets.mean()
            median = np.median(rets)
            p25 = np.percentile(rets, 25)
            p75 = np.percentile(rets, 75)
            best_case = rets.max()
            worst_case = rets.min()
            avg_win = rets[rets > 0].mean() if (rets > 0).any() else 0
            avg_loss = rets[rets <= 0].mean() if (rets <= 0).any() else 0

            if wr >= 80:
                best_h = h_name
                best_metrics = {
                    'n': len(rets), 'wr': round(wr, 1),
                    'avg': round(avg, 2), 'median': round(median, 2),
                    'p25': round(p25, 2), 'p75': round(p75, 2),
                    'best': round(best_case, 2), 'worst': round(worst_case, 2),
                    'avg_win': round(avg_win, 2), 'avg_loss': round(avg_loss, 2),
                }

            if wr >= 70:
                print(f"    [{h_name}] {pat_name[:50]:50s}")
                print(f"        n={len(rets):3d}  WR={wr:.1f}%  avg={avg:+.2f}%  med={median:+.2f}%  赢均={avg_win:+.2f}%  亏均={avg_loss:+.2f}%  best={best_case:+.2f}%  worst={worst_case:+.2f}%")

    print()

bs.logout()
