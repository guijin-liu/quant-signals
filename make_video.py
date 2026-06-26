#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
30秒买点回放视频 v4 — 简洁版，只标注买点
快速滚动K线 → 到买点慢放 → 大标签BUY + 后市涨跌线
"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os, time

VIDEO_SEC = 30; FPS = 30; FRAMES = VIDEO_SEC * FPS
SLOW = 8; PRE = 8       # slow zone bars
DPI = 72; FIG_W, FIG_H = 16, 9
OUT = os.path.expanduser("~/Desktop/quant_buy_points.mp4")
CACHE = "C:/Users/Administrator/quant_trading/data/cache"
NAMES = {"000933":"神火","002497":"雅化","000960":"锡业","000893":"亚钾"}

# ---- Load ----
print("Loading...")
dfs = {}
for c, n in NAMES.items():
    df = pd.read_csv(f"{CACHE}/{c}_15min.csv", dtype={'time': str})
    df['time'] = df['time'].astype(str).str.zfill(17)
    for col in ['open','high','low','close','volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df[df['date'] >= '2025-06-01'].copy().reset_index(drop=True)
    dfs[c] = df

# ---- Find signals ----
print("Finding buy signals...")
signals = []
for code, df in dfs.items():
    c = df['close'].values; n = len(c)
    if n < 60: continue
    ma5  = np.convolve(c, np.ones(5)/5, mode='same')
    ma10 = np.convolve(c, np.ones(10)/10, mode='same')
    ma20 = np.convolve(c, np.ones(20)/20, mode='same')
    p_ma5 = np.roll(ma5,1); p_ma10 = np.roll(ma10,1)
    golden = (ma5 > ma10) & (p_ma5 <= p_ma10)
    # RSI
    d = np.diff(c); rsi = np.full(n, 50.)
    for i in range(14,n):
        g = np.sum(np.maximum(d[i-14:i],0))
        l = np.sum(np.abs(np.minimum(d[i-14:i],0)))
        rsi[i] = 100-100/(1+g/l) if l>0 else 100
    # BB
    bbs = np.array([np.std(c[max(0,i-19):i+1]) for i in range(n)])
    bb_pct = np.clip((c - ma20 - 2*bbs) / (4*bbs + 0.0001) + 0.5, 0, 1)
    # Pos
    h20 = pd.Series(df['high']).rolling(20).max().values
    l20 = pd.Series(df['low']).rolling(20).min().values
    pos = np.clip((c - l20) / (h20 - l20 + 0.0001), 0, 1)

    for i in range(60, n):
        if not golden[i]: continue
        match = False
        if code == "000933" and bb_pct[i] < 0.3: match = True
        elif code == "002497" and ((40<=rsi[i]<=55 and pos[i]<0.4 and bb_pct[i]<0.3) or bb_pct[i]<0.3): match = True
        elif code == "000960" and ((pos[i]<0.4 and bb_pct[i]<0.3) or bb_pct[i]<0.2): match = True
        elif code == "000893" and ((40<=rsi[i]<=55 and pos[i]<0.4 and bb_pct[i]<0.3) or (pos[i]<0.4 and bb_pct[i]<0.3)): match = True
        if not match: continue
        fwd = min(i+32, n-1)
        fwd_r = (c[fwd]-c[i])/c[i]*100
        signals.append({'code':code,'name':NAMES[code],'i':i,'p':c[i],
                       'rsi':rsi[i],'fwd':fwd_r,'win':fwd_r>0})

# Pick 10 best: balanced wins/losses, stock diversity
wins  = sorted([s for s in signals if s['win']], key=lambda x: x['fwd'], reverse=True)
losses = sorted([s for s in signals if not s['win']], key=lambda x: x['fwd'], reverse=True)
selected = []
for sc in ["000960","002497","000933","000893"]:
    selected.extend([s for s in wins if s['code']==sc][:2])
    selected.extend([s for s in losses if s['code']==sc][:1])
selected.sort(key=lambda x: x['i'])
if len(selected) > 12: selected = selected[:12]
print(f"Selected {len(selected)}: {sum(s['win'] for s in selected)}W {sum(1 for s in selected if not s['win'])}L")

# ---- Frame schedule ----
SEG = FRAMES // len(selected)  # frames per signal segment
schedule = []

for si, sig in enumerate(selected):
    total = SEG - 2
    # fast approach (15 bars)
    app_bars = 15; app_f = int(app_bars * 1.0)
    # slow zone (PRE + 15 bars after)
    slow_bars = PRE + 15; slow_f = int(slow_bars * SLOW)
    # post (8 bars)
    post_bars = 8; post_f = int(post_bars * 1.0)
    need = app_f + slow_f + post_f
    scale = total / need

    # Approach
    for off in range(-PRE-app_bars, -PRE):
        idx = sig['i'] + off
        if 0 <= idx < len(dfs[sig['code']]):
            nf = max(1, int(1.0 * scale))
            for _ in range(nf):
                schedule.append((sig['code'], idx, off, sig))
    # Slow zone (the money zone)
    for off in range(-PRE, 15):
        idx = sig['i'] + off
        if 0 <= idx < len(dfs[sig['code']]):
            nf = max(1, int(SLOW * scale))
            for _ in range(nf):
                schedule.append((sig['code'], idx, off, sig))
    # Post
    for off in range(15, 15+post_bars):
        idx = sig['i'] + off
        if 0 <= idx < len(dfs[sig['code']]):
            nf = max(1, int(1.0 * scale))
            for _ in range(nf):
                schedule.append((sig['code'], idx, off, sig))

# Trim/pad
if len(schedule) < FRAMES:
    last = schedule[-1]
    while len(schedule) < FRAMES:
        schedule.append(last)
else:
    schedule = schedule[:FRAMES]

print(f"Schedule: {len(schedule)} frames")

# ---- Render ----
print("Rendering...")
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei','SimHei','DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), facecolor='#080810')
ax.set_facecolor('#080810')
writer = animation.FFMpegWriter(fps=FPS, bitrate=2000, codec='libx264')

t0 = time.time()
with writer.saving(fig, OUT, dpi=DPI):
    for fi, (code, idx, offset, sig) in enumerate(schedule):
        ax.clear(); ax.set_facecolor('#080810')
        df = dfs[code]; name = NAMES[code]
        is_buy = (offset == 0)
        in_slow = abs(offset) < PRE

        # Display window
        s = max(0, idx - 55)
        e = min(len(df), idx + 22)
        w = df.iloc[s:e].copy().reset_index(drop=True)
        rel_cur = idx - s
        rel_buy = sig['i'] - s

        cw = w['close'].values; hw = w['high'].values; lw = w['low'].values
        x = np.arange(len(w))
        if len(cw) < 10: writer.grab_frame(); continue

        # === SIMPLE K-LINE ===
        # Draw vertical lines (high-low)
        for i_w in range(len(w)):
            body_c = '#2ecc71' if cw[i_w] >= w['open'].values[i_w] else '#e74c3c'
            ax.plot([i_w, i_w], [lw[i_w], hw[i_w]], color=body_c, lw=0.6, alpha=0.7)
            o, cl = w['open'].values[i_w], cw[i_w]
            b_bot, b_top = min(o,cl), max(o,cl)
            ax.add_patch(plt.Rectangle((i_w-0.25, b_bot), 0.5, max(b_top-b_bot, 0.01),
                          facecolor=body_c, edgecolor=body_c, lw=0.3, alpha=0.85))

        # MA overlay (semi-transparent)
        if len(cw) >= 20:
            for per, col, lw_val in [(5,'#ff9800',1.2),(10,'#e91e63',1.0),(20,'#9c27b0',0.8)]:
                m = np.convolve(cw, np.ones(per)/per, mode='same')
                ax.plot(x, m, color=col, lw=lw_val, alpha=0.7)

        # BB bands when slow
        if in_slow and len(cw) >= 20:
            bbmid = np.convolve(cw, np.ones(20)/20, mode='same')
            bbsd = np.array([np.std(cw[max(0,i-19):i+1]) for i in range(len(cw))])
            ax.plot(x, bbmid+2*bbsd, color='#555', lw=0.4, ls='--', alpha=0.3)
            ax.plot(x, bbmid-2*bbsd, color='#555', lw=0.4, ls='--', alpha=0.3)
            ax.fill_between(x, bbmid-2*bbsd, bbmid+2*bbsd, alpha=0.02, color='white')

        # === BUY POINT ANNOTATION ===
        if 0 <= rel_buy < len(w):
            bp = cw[rel_buy]
            if is_buy:
                # BIG FLASHING BUY SIGNAL
                ax.axvline(x=rel_buy, color='#ff1744', lw=3, alpha=0.9)
                y_range = max(hw) - min(lw)
                ax.annotate('买入信号', xy=(rel_buy, bp),
                    xytext=(rel_buy, bp + y_range*0.22),
                    fontsize=28, color='white', fontweight='bold', ha='center',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='#d50000', alpha=0.95),
                    arrowprops=dict(arrowstyle='->', color='white', lw=3))
                # Price label below signal
                ax.annotate(f'{bp:.2f}', xy=(rel_buy, bp - y_range*0.05),
                    fontsize=14, color='white', ha='center', fontweight='bold')
                # RSI info
                ax.annotate(f'RSI {sig["rsi"]:.0f}', xy=(rel_buy, bp + y_range*0.15),
                    fontsize=11, color='#ff9800', ha='center')
            elif 0 < offset <= PRE:
                # Just past buy - show marker
                ax.axvline(x=rel_buy, color='#e74c3c', lw=1.5, ls='--', alpha=0.5)
                ax.plot(rel_buy, bp, 'v', color='#ff1744', ms=8, zorder=5)
            else:
                ax.axvline(x=rel_buy, color='#e74c3c', lw=0.4, ls='--', alpha=0.2)

        # === OUTCOME LINE ===
        if offset > 0:
            fwd_i = min(sig['i'] + 32, len(df) - 1)
            f_rel = fwd_i - s
            if 0 <= f_rel < len(w):
                oc = '#00e676' if sig['win'] else '#ff1744'
                ax.plot([rel_buy, f_rel], [bp, df['close'].iloc[fwd_i]],
                       color=oc, lw=3, ls='-', alpha=0.8, marker='o', ms=8,
                       markeredgecolor='white', markeredgewidth=1.5)
                txt = f"+{sig['fwd']:.1f}%" if sig['win'] else f"{sig['fwd']:.1f}%"
                ax.annotate(txt, xy=(f_rel, df['close'].iloc[fwd_i]),
                    fontsize=15, color='white', fontweight='bold', ha='center',
                    xytext=(0, 18), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='#43a047' if sig['win'] else '#c62828', alpha=0.9))

        # Current position dot
        if 0 <= rel_cur < len(w):
            ax.plot(rel_cur, cw[rel_cur], 'o', color='white', ms=8, zorder=10, markeredgecolor='#222', markeredgewidth=1.5)

        # === UI ===
        dt_str = str(df['date'].iloc[idx]) if idx < len(df) else ""
        title_parts = [f"{name} {code}", f"15min", dt_str]
        if is_buy: title_parts.append(">>> 买入信号!")
        elif in_slow: title_parts.append("[慢放]")
        ax.set_title(" | ".join(title_parts), fontsize=14, color='#ccc')

        # Top-right: outcome badge
        badge_c = '#4caf50' if sig['win'] else '#f44336'
        badge_t = f"{'+' if sig['win'] else ''}{sig['fwd']:.1f}%"
        ax.text(0.98, 0.96, f"后市 {badge_t}", transform=ax.transAxes,
               fontsize=14, color='white', fontweight='bold', ha='right', va='top',
               bbox=dict(boxstyle='round,pad=0.3', facecolor=badge_c, alpha=0.85))

        # Bottom-left: metrics
        ax.text(0.02, 0.04, f"RSI{sig['rsi']:.0f} | Price{cw[rel_cur]:.2f}" if 0<=rel_cur<len(w) else "",
               transform=ax.transAxes, fontsize=9, color='#777', va='bottom')

        # Progress & watermark
        ax.text(0.5, 0.01, f"v10.4 逐票概率买点 | {int(fi/SEG)+1}/{len(selected)}",
               transform=ax.transAxes, fontsize=7, color='#444', ha='center', va='bottom')
        ax.text(0.98, 0.5, NAMES[code], transform=ax.transAxes, fontsize=35,
               color='white', alpha=0.03, ha='right', va='center', rotation=-15, fontweight='bold')

        ax.set_xlim(-1.5, len(w)+1.5)
        m = (max(hw)-min(lw))*0.18
        ax.set_ylim(min(lw)-m, max(hw)+m*2.5)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)

        writer.grab_frame()
        if fi % 120 == 0:
            e = time.time()-t0
            print(f"  {fi}/{len(schedule)} ({100*fi/len(schedule):.0f}%) ETA {(e/(fi+1)*(len(schedule)-fi)):.0f}s" if fi>0 else f"  Starting...")

e = time.time()-t0
sz = os.path.getsize(OUT)/1024/1024
print(f"\nDone! {sz:.1f}MB | 30s | {FIG_W}x{FIG_H}")
print(f"Desktop/quant_buy_points.mp4")
print(f"Signals: {len(selected)} ({sum(s['win'] for s in selected)}W {sum(1 for s in selected if not s['win'])}L)")
