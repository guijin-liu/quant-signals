#!/usr/bin/env python
"""
30秒买点回放视频 v2 — 清晰标注买入信号+后市走势
先快速滚动到买点 → 买点处慢放 → 显示后市涨跌
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from datetime import datetime
import os

VIDEO_SEC = 30; FPS = 30; FRAMES = VIDEO_SEC * FPS
SLOW = 5; WINDOW = 20  # slow factor, bars around buy
DPI = 100; FIG_W, FIG_H = 16, 9
OUT = os.path.expanduser("~/Desktop/quant_buy_points.mp4")
CACHE = "C:/Users/Administrator/quant_trading/data/cache"

STOCKS = {"000933":"神火","002497":"雅化","000960":"锡业","000893":"亚钾"}

# ---- Load ----
print("Loading...")
all_data = {}
for code, name in STOCKS.items():
    df = pd.read_csv(f"{CACHE}/{code}_15min.csv", dtype={'time':str})
    df['time'] = df['time'].astype(str).str.zfill(17)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df[df['date'] >= '2025-06-01'].copy()
    df.reset_index(drop=True, inplace=True)
    all_data[code] = df

# ---- Find Signals ----
print("Finding signals...")
signals = []
for code, df in all_data.items():
    c = df['close'].values; h = df['high'].values; l = df['low'].values
    n = len(c); ma5 = np.convolve(c, np.ones(5)/5, mode='same')
    ma10 = np.convolve(c, np.ones(10)/10, mode='same')
    ma20 = np.convolve(c, np.ones(20)/20, mode='same')
    golden = (ma5 > ma10) & (np.roll(ma5,1) <= np.roll(ma10,1))
    # RSI
    d = np.diff(c); rsi = np.full(n, 50.)
    for i in range(14,n):
        g = np.sum(np.maximum(d[i-14:i],0)); loss = np.sum(np.abs(np.minimum(d[i-14:i],0)))
        rsi[i] = 100-100/(1+g/loss) if loss>0 else 100
    # BB
    bb_s = np.array([np.std(c[max(0,i-19):i+1]) for i in range(n)])
    bb_pct = np.clip((c - ma20 - 2*bb_s) / (4*bb_s + 0.0001) + 0.5, 0, 1)
    # Pos
    h20 = pd.Series(h).rolling(20).max().values; l20 = pd.Series(h).rolling(20).min().values
    pos = np.clip((c - l20) / (h20 - l20 + 0.0001), 0, 1)

    for i in range(60, n):
        if golden[i] and bb_pct[i] < 0.35 and pos[i] < 0.45:
            fwd = min(i+32, n-1); fwd_r = (c[fwd]-c[i])/c[i]*100
            signals.append({'code':code,'name':STOCKS[code],'i':i,'p':c[i],
                           'rsi':rsi[i],'bb':bb_pct[i],'fwd':fwd_r,'win':fwd_r>0})

signals = sorted(signals, key=lambda x: (x['win'], abs(x['fwd'])), reverse=True)[:10]
signals.sort(key=lambda x: x['win'], reverse=True)  # wins first
print(f"Selected {len(signals)} signals, wins: {sum(s['win'] for s in signals)}")

# ---- Build precise frame schedule ----
# Key frames: signal_i * (bars before signal that get normal speed)
# Then signal window gets slow speed
# We have 10 signals to show, each gets ~90 frames
FRAMES_PER_SIGNAL = FRAMES // len(signals)

sig_frames = []  # (signal_dict, bar_offset from -WINDOW to +16)
for sig in signals:
    pre_bars = WINDOW
    post_bars = 16  # 1 trading day ahead
    # Distribute: WINDOW bars at normal speed, then WINDOW+16 at slow speed
    # pre: 1 frame/bar, during: SLOW frames/bar, post: 1 frame/bar
    pre_frames = pre_bars * 1
    slow_frames = (WINDOW + post_bars) * SLOW
    total_needed = pre_frames + slow_frames
    scale = FRAMES_PER_SIGNAL / total_needed

    for offset in range(-pre_bars, WINDOW + post_bars + 1):
        idx = sig['i'] + offset
        if idx < 0 or idx >= len(all_data[sig['code']]):
            continue
        is_in_slow = abs(offset) < WINDOW
        n_frames = int(SLOW * scale) if is_in_slow else int(1 * scale)
        n_frames = max(1, n_frames)
        for _ in range(n_frames):
            sig_frames.append((sig['code'], idx, offset, sig))

    # Add some gap frames
    for _ in range(8):
        sig_frames.append((sig['code'], sig['i'], 0, sig))

# Ensure exactly FRAMES
if len(sig_frames) < FRAMES:
    # pad last
    last = sig_frames[-1]
    while len(sig_frames) < FRAMES:
        sig_frames.append(last)
else:
    sig_frames = sig_frames[:FRAMES]

print(f"Total frames: {len(sig_frames)}")

# ---- Render ----
print("Rendering...")
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor='#0a0a12')
ax.set_facecolor('#0a0a12')
writer = animation.FFMpegWriter(fps=FPS, bitrate=1800, codec='libx264')

import time; t0 = time.time()

with writer.saving(fig, OUT, dpi=DPI):
    for fi, (code, idx, offset, sig) in enumerate(sig_frames):
        ax.clear(); ax.set_facecolor('#0a0a12')
        df = all_data[code]; name = STOCKS[code]
        is_buy_point = (offset == 0)
        is_slow = abs(offset) < WINDOW

        # Window: 40 bars before to 20 after
        start_i = max(0, idx - 50)
        end_i   = min(len(df), idx + 25)
        w = df.iloc[start_i:end_i].copy().reset_index(drop=True)
        rel_buy = idx - start_i
        close_w = w['close'].values; high_w = w['high'].values; low_w = w['low'].values
        x = np.arange(len(w))

        # MA lines
        if len(close_w) >= 20:
            ma5w = np.convolve(close_w, np.ones(5)/5, mode='same')
            ma10w = np.convolve(close_w, np.ones(10)/10, mode='same')
            ma20w = np.convolve(close_w, np.ones(20)/20, mode='same')
            ax.plot(x, ma5w, color='#f39c12', lw=1.5, alpha=0.8, label='MA5')
            ax.plot(x, ma10w, color='#e74c3c', lw=1.2, alpha=0.7, label='MA10')
            ax.plot(x, ma20w, color='#9b59b6', lw=1.2, alpha=0.7, label='MA20')

        # Price line
        color_line = '#2196F3'
        ax.plot(x, close_w, color=color_line, lw=1.8, alpha=0.95)
        ax.fill_between(x, close_w, alpha=0.04, color=color_line)

        # Bollinger bands when slow
        if is_slow and len(close_w) >= 20:
            bb_s = np.array([np.std(close_w[max(0,i-19):i+1]) for i in range(len(close_w))])
            bb_m = ma20w
            ax.plot(x, bb_m+2*bb_s, color='#555', lw=0.5, ls='--', alpha=0.4)
            ax.plot(x, bb_m-2*bb_s, color='#555', lw=0.5, ls='--', alpha=0.4)

        # Mark buy point
        if 0 <= rel_buy < len(w):
            bp = close_w[rel_buy]
            ax.axvline(x=rel_buy, color='#e74c3c', lw=2, ls='--', alpha=0.6)

            # BIG BUY annotation
            if is_buy_point or abs(offset) < 3:
                ax.annotate(f'BUY\n{bp:.2f}',
                    xy=(rel_buy, bp),
                    xytext=(rel_buy, bp + (max(high_w)-min(low_w))*0.18),
                    fontsize=16, color='white', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#c0392b', edgecolor='white', lw=2),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))
                # Draw target line
                target = bp * (1 + abs(sig['fwd'])/100 * 0.7) if sig['fwd'] > 0 else bp * 1.01
                ax.axhline(y=target, color='#2ecc71', lw=1.5, ls='--', alpha=0.7)
                ax.text(len(w)-1, target, f' Target {target:.2f}', color='#2ecc71', fontsize=10, va='bottom')
            else:
                ax.plot(rel_buy, bp, marker='v', color='#e74c3c', ms=10, zorder=5,
                       markeredgecolor='white', markeredgewidth=1)

        # Forward outcome line
        fwd_i = min(sig['i'] + 32, len(df) - 1)
        fwd_rel = fwd_i - start_i
        if 0 <= fwd_rel < len(w):
            fwd_p = df['close'].iloc[fwd_i]
            outcome_color = '#2ecc71' if sig['win'] else '#e74c3c'
            ax.plot([rel_buy, fwd_rel], [bp, fwd_p], color=outcome_color,
                   lw=2.5, ls='-', alpha=0.7, marker='o', ms=6,
                   markeredgecolor='white', markeredgewidth=1)
            result_text = f"+{sig['fwd']:.1f}%" if sig['fwd'] > 0 else f"{sig['fwd']:.1f}%"
            ax.annotate(result_text, xy=(fwd_rel, fwd_p),
                       fontsize=14, color=outcome_color, fontweight='bold',
                       xytext=(5, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a0a12', edgecolor=outcome_color, alpha=0.9))

        # Current position marker
        cur_rel = idx - start_i
        if 0 <= cur_rel < len(w):
            ax.plot(cur_rel, w['close'].iloc[cur_rel], 'o', color='white', ms=8, zorder=10)

        # Title
        dt_str = df['date'].iloc[idx] if idx < len(df) else ""
        slow_tag = " [SLOW]" if is_slow else ""
        buy_tag = " >>> BUY SIGNAL!" if is_buy_point else ""
        ax.set_title(f"{name} {code} | {dt_str} | 15min K-line{slow_tag}{buy_tag}",
                    fontsize=15, color='#ddd', fontweight='bold')

        # Info panel
        info_y = 0.95
        ax.text(0.01, info_y, f"Price {w['close'].iloc[cur_rel]:.2f}" if 0<=cur_rel<len(w) else "",
               transform=ax.transAxes, fontsize=11, color='#aaa', va='top', fontfamily='monospace')
        if is_slow:
            metrics = f"RSI {sig['rsi']:.0f} | BB% {sig['bb']:.2f} | 后市{'盈利' if sig['win'] else '亏损'} {sig['fwd']:+.1f}%"
            ax.text(0.01, info_y-0.04, metrics, transform=ax.transAxes, fontsize=10, color='#f39c12', va='top')

        # Progress bar
        progress = fi / FRAMES
        ax.axhline(y=ax.get_ylim()[0], color='#333', lw=4)
        ax.text(0.5, -0.03, f"v10.4 | 逐票概率买点 | 信号 {int(fi/FRAMES_PER_SIGNAL)+1}/{len(signals)}",
               transform=ax.transAxes, fontsize=8, color='#555', ha='center')

        # Stock watermark
        ax.text(0.98, 0.98, name, transform=ax.transAxes, fontsize=28,
               color='white', alpha=0.08, ha='right', va='top', fontweight='bold')

        ax.set_xlim(-1, len(w)+1)
        m = (max(high_w)-min(low_w))*0.15
        ax.set_ylim(min(low_w)-m, max(high_w)+m*2)
        ax.set_xticks([]); ax.set_yticks([])
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333'); ax.spines['bottom'].set_color('#333')

        writer.grab_frame()
        if fi % 90 == 0:
            e = time.time()-t0; eta = e/(fi+1)*(FRAMES-fi) if fi>0 else 0
            print(f"  {fi}/{FRAMES} ({100*fi/FRAMES:.0f}%) ETA {eta:.0f}s")

e = time.time()-t0
print(f"\nDone! {OUT}")
print(f"Time: {e:.0f}s | Size: {os.path.getsize(OUT)/1024/1024:.1f}MB | Duration: 30s")
