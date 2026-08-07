"""刘圭金2号 — 全量指标 + 组合挖掘回溯引擎

流程：
  动态池(≈400只) → 东财半年15分钟K线 + 资金流120日 → 全量指标矩阵(40+列)
  → 胜率标签(未来8根内最高涨幅≥1%为胜) → 三级挖掘(单条件→两两组合+贪心→资金流分离验证)
  → 输出 ≥85% 胜率规则 rules2.json

防过拟合：n≥30 + 95%二项分布置信下限≥80%。
重点：分析主力资金流入/流出对胜率的分离作用(fund_split)。
"""
import os, sys, json, logging, glob, time
import numpy as np
import pandas as pd

import em_client
from dynamic_pool import load_pool_window

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════
# 1. 指标原子函数（返回全长度向量）
# ═══════════════════════════════════════════════

def _sma(a, n):
    return pd.Series(a).rolling(n, min_periods=1).mean().values

def _ema(a, n):
    return pd.Series(a).ewm(span=n, adjust=False).mean().values

def _rsi(close, n=14):
    c = pd.Series(close)
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1/n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1/n, adjust=False).mean()
    rs = up / (dn + 1e-9)
    return (100 - 100 / (1 + rs)).values

def _bb(close, n=20, k=2.0):
    c = pd.Series(close)
    mid = c.rolling(n, min_periods=1).mean()
    std = c.rolling(n, min_periods=1).std(ddof=0)
    return mid.values, (mid + k*std).values, (mid - k*std).values

def _macd(close, fast=12, slow=26, sig=9):
    c = pd.Series(close)
    dif = c.ewm(span=fast, adjust=False).mean() - c.ewm(span=slow, adjust=False).mean()
    dea = dif.ewm(span=sig, adjust=False).mean()
    return dif.values, dea.values, (2*(dif-dea)).values

def _kdj(high, low, close, n=9):
    h = pd.Series(high).rolling(n, min_periods=1).max()
    l = pd.Series(low).rolling(n, min_periods=1).min()
    rsv = (pd.Series(close) - l) / (h - l + 1e-9) * 100
    k = rsv.ewm(alpha=1/3, adjust=False).mean()
    d = k.ewm(alpha=1/3, adjust=False).mean()
    j = 3*k - 2*d
    return k.values, d.values, j.values

def _atr(high, low, close, n=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=1).mean().values

def _cci(high, low, close, n=14):
    tp = (pd.Series(high) + pd.Series(low) + pd.Series(close)) / 3
    ma = tp.rolling(n, min_periods=1).mean()
    md = (tp - ma).abs().rolling(n, min_periods=1).mean()
    return ((tp - ma) / (0.015 * md + 1e-9)).values

def _wr(high, low, close, n=14):
    h = pd.Series(high).rolling(n, min_periods=1).max()
    l = pd.Series(low).rolling(n, min_periods=1).min()
    return ((h - pd.Series(close)) / (h - l + 1e-9) * 100).values

def _bias(close, n):
    c = pd.Series(close)
    return ((c - c.rolling(n, min_periods=1).mean()) / c.rolling(n, min_periods=1).mean() * 100).values

def _dmi(high, low, close, n=14):
    h, l, c = pd.Series(high), pd.Series(low), pd.Series(close)
    up = h.diff()
    dn = -l.diff()
    plus_dm = pd.Series(np.where((up > dn) & (up > 0), up, 0), index=h.index)
    minus_dm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0), index=l.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/n, adjust=False).mean()
    pdi = 100 * plus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-9)
    mdi = 100 * minus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr + 1e-9)
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi + 1e-9)
    adx = dx.ewm(alpha=1/n, adjust=False).mean()
    return pdi.values, mdi.values, adx.values

def _roc(close, n=12):
    c = pd.Series(close)
    return (c.pct_change(n) * 100).values

def _mom(close, n=10):
    c = pd.Series(close)
    return (c - c.shift(n)).values

def _psy(close, n=12):
    c = pd.Series(close)
    up = (c.diff() > 0).astype(float)
    return (up.rolling(n, min_periods=1).mean() * 100).values


# ═══════════════════════════════════════════════
# 2. 全量指标矩阵（每票一段K线 → DataFrame，一行一根bar）
# ═══════════════════════════════════════════════

def compute_feature_matrix(close, high, low, vol):
    """返回 DataFrame(一根bar一行)。含 40+ 列指标 + 资金流列(由 fund 参数合并)。"""
    close = np.asarray(close, dtype=float)
    high = np.asarray(high, dtype=float)
    low = np.asarray(low, dtype=float)
    vol = np.asarray(vol, dtype=float)
    n = len(close)
    if n < 70:
        return pd.DataFrame()

    mid, bup, blo = _bb(close)
    dif, dea, hist = _macd(close)
    k, d, j = _kdj(high, low, close)
    pdi, mdi, adx = _dmi(high, low, close)

    f = pd.DataFrame(index=range(n))
    f["close"] = close
    f["ma5"] = _sma(close, 5)
    f["ma10"] = _sma(close, 10)
    f["ma20"] = _sma(close, 20)
    f["ma60"] = _sma(close, 60)
    f["ema12"] = _ema(close, 12)
    f["ema26"] = _ema(close, 26)
    f["dif"] = dif
    f["dea"] = dea
    f["macd_hist"] = hist
    f["rsi"] = _rsi(close)
    f["bb_pct"] = np.clip((close - blo) / (bup - blo + 1e-9), 0, 1)
    f["bb_width"] = (bup - blo) / (mid + 1e-9)
    h20 = pd.Series(high).rolling(20, min_periods=1).max().values
    l20 = pd.Series(low).rolling(20, min_periods=1).min().values
    f["pos"] = np.clip((close - l20) / (h20 - l20 + 1e-9), 0, 1)
    f["vol_ratio"] = pd.Series(vol).rolling(5, min_periods=1).mean() / (pd.Series(vol).rolling(20, min_periods=1).mean() + 1e-9)
    f["vol_trend"] = pd.Series(vol).rolling(10, min_periods=1).mean() / (pd.Series(vol).rolling(30, min_periods=1).mean() + 1e-9)
    f["k"], f["d"], f["j"] = k, d, j
    f["atr"] = _atr(high, low, close)
    f["cci"] = _cci(high, low, close)
    f["wr"] = _wr(high, low, close)
    f["bias6"] = _bias(close, 6)
    f["bias12"] = _bias(close, 12)
    f["pdi"], f["mdi"], f["adx"] = pdi, mdi, adx
    f["roc"] = _roc(close)
    f["mom"] = _mom(close)
    f["psy"] = _psy(close)

    # 派生布尔/结构特征
    f["golden"] = (f["ma5"] > f["ma10"]) & (f["ma5"].shift(1) <= f["ma10"].shift(1))
    f["dead"] = (f["ma5"] < f["ma10"]) & (f["ma5"].shift(1) >= f["ma10"].shift(1))
    f["macd_golden"] = (f["dif"] > f["dea"]) & (f["dif"].shift(1) <= f["dea"].shift(1))
    f["kdj_golden"] = (f["k"] > f["d"]) & (f["k"].shift(1) <= f["d"].shift(1))
    f["ma_bull"] = (f["ma5"] > f["ma10"]) & (f["ma10"] > f["ma20"]) & (f["ma20"] > f["ma60"])
    f["ma_squeeze"] = (f["ma5"] - f["ma20"]).abs() / (f["close"] + 1e-9)
    f["above_ma20"] = close > f["ma20"]
    f["above_ma60"] = close > f["ma60"]
    f["vol_break"] = f["vol_ratio"] > 1.5
    f["vol_up"] = f["vol_trend"] > 1.0
    # 连阳
    up = pd.Series(close).diff() > 0
    grp = (~up).cumsum()
    f["consec_up"] = up.groupby(grp).cumsum()
    # OBV 方向
    obv = (np.sign(pd.Series(close).diff().fillna(0)) * pd.Series(vol)).cumsum()
    f["obv_up"] = obv > obv.shift(10)
    # 突破20期新高
    f["high_break"] = close > pd.Series(high).rolling(20, min_periods=1).max().shift(1)
    # 波动收缩
    f["volatility"] = pd.Series(close).pct_change().rolling(20, min_periods=1).std()

    # 资金流列占位（由 backtest_stock2 合并）
    f["main_net"] = 0.0
    f["super_net"] = 0.0
    f["large_net"] = 0.0
    return f


# ═══════════════════════════════════════════════
# 3. 条件池（每个条件：输入特征DataFrame → 布尔Series）
# ═══════════════════════════════════════════════

def _mk_cond(fn):
    return fn

CONDITIONS = {
    "golden":       _mk_cond(lambda f: f["golden"]),
    "ma_bull":      _mk_cond(lambda f: f["ma_bull"]),
    "rsi_lt30":     _mk_cond(lambda f: f["rsi"] < 30),
    "rsi_30_45":    _mk_cond(lambda f: (f["rsi"] >= 30) & (f["rsi"] < 45)),
    "rsi_45_60":    _mk_cond(lambda f: (f["rsi"] >= 45) & (f["rsi"] <= 60)),
    "bb_low":       _mk_cond(lambda f: f["bb_pct"] <= 0.25),
    "bb_midlow":    _mk_cond(lambda f: f["bb_pct"] <= 0.40),
    "cci_gt100":    _mk_cond(lambda f: f["cci"] > 100),
    "cci_lt_m100":  _mk_cond(lambda f: f["cci"] < -100),
    "wr_lt20":      _mk_cond(lambda f: f["wr"] < 20),
    "bias_lt5":     _mk_cond(lambda f: f["bias6"] < 5),
    "bias_pos":     _mk_cond(lambda f: f["bias6"] > 0),
    "adx_gt25":     _mk_cond(lambda f: f["adx"] > 25),
    "pdi_gt_mdi":   _mk_cond(lambda f: f["pdi"] > f["mdi"]),
    "roc_gt0":      _mk_cond(lambda f: f["roc"] > 0),
    "psy_gt50":     _mk_cond(lambda f: f["psy"] > 50),
    "macd_golden":  _mk_cond(lambda f: f["macd_golden"]),
    "kdj_golden":   _mk_cond(lambda f: f["kdj_golden"]),
    "j_lt30":       _mk_cond(lambda f: f["j"] < 30),
    "vol_break":    _mk_cond(lambda f: f["vol_break"]),
    "vol_up":       _mk_cond(lambda f: f["vol_up"]),
    "obv_up":       _mk_cond(lambda f: f["obv_up"]),
    "above_ma20":   _mk_cond(lambda f: f["above_ma20"]),
    "above_ma60":   _mk_cond(lambda f: f["above_ma60"]),
    "pos_lt40":     _mk_cond(lambda f: f["pos"] < 0.40),
    "squeeze":      _mk_cond(lambda f: f["ma_squeeze"] < 0.02),
    "high_break":   _mk_cond(lambda f: f["high_break"]),
    "consec_up2":   _mk_cond(lambda f: f["consec_up"] >= 2),
    "fund_in":      _mk_cond(lambda f: f["main_net"] > 0),
    "fund_super_in":_mk_cond(lambda f: f["super_net"] > 0),
    "fund_large_in":_mk_cond(lambda f: f["large_net"] > 0),
    "fund_strong":  _mk_cond(lambda f: f["main_net"] > f["amount_mean"] * 0.02),
}


# ═══════════════════════════════════════════════
# 4. 胜率标签（未来 hold_bars 根内最高涨幅 ≥ win_pct% 为胜）
# ═══════════════════════════════════════════════

def label_wins(high, close, hold_bars=8, win_pct=1.0):
    """返回 (win1, win2, fwd_gain)：1%胜、2%胜标签 及 未来最大涨幅%"""
    n = len(high)
    h = np.asarray(high, dtype=float)
    c = np.asarray(close, dtype=float)
    fh = np.full(n, np.nan)
    for i in range(n - hold_bars):
        fh[i] = h[i+1:i+1+hold_bars].max()
    gain = (fh / c - 1) * 100
    win1 = pd.Series(gain >= win_pct, index=range(n)).fillna(False)
    win2 = pd.Series(gain >= 2.0, index=range(n)).fillna(False)
    return win1, win2, gain


# ═══════════════════════════════════════════════
# 5. 单票回溯
# ═══════════════════════════════════════════════

def backtest_stock2(code, name="", hold_bars=8, win_pct=1.0, start="20260101"):
    """单票：拉K线+资金流 → 特征矩阵+资金流对齐 → 返回 (feature_df, win1, win2, gain)
    资金流优先东财，被拒/空则用妙想兜底（妙想个人API不受 push2his 风控）"""
    df = em_client.em_fetch_kline_15m(code, start)
    if df.empty or len(df) < 100:
        return None
    # 东财K线被拒(腾讯兜底≤120根)时跳过东财资金流；东财通(≥300根)才拉东财资金流
    em_ok = len(df) > 300
    fund = em_client.em_fund_flow_120d(code) if em_ok else []
    if not fund and name:  # 东财资金流空 → 妙想兜底（近120交易日）
        import mx_fetcher
        fund = mx_fetcher.mx_fund_flow(code, name)
    fund_by_date = {r["date"]: r for r in fund}

    f = compute_feature_matrix(df["close"].values, df["high"].values,
                               df["low"].values, df["volume"].values)
    if f.empty:
        return None
    # 资金流按交易日对齐到15min bar（bar日期取 date[:10]）
    # 资金流按交易日对齐到15min bar（腾讯date="20260807"需转"2026-08-07"）
    def _norm_date(d):
        s = str(d)[:10]
        if len(s) == 8 and "-" not in s:
            s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
        return s
    dates = [_norm_date(d) for d in df["date"].tolist()]
    main_net = np.array([fund_by_date.get(d, {}).get("main_net", 0) for d in dates])
    super_net = np.array([fund_by_date.get(d, {}).get("super_net", 0) for d in dates])
    large_net = np.array([fund_by_date.get(d, {}).get("large_net", 0) for d in dates])
    amount_mean = pd.Series(df["amount"].values).rolling(20, min_periods=1).mean().values
    f["main_net"] = main_net
    f["super_net"] = super_net
    f["large_net"] = large_net
    f["amount_mean"] = np.where(amount_mean > 0, amount_mean, 1.0)

    win1, win2, gain = label_wins(df["high"].values, df["close"].values, hold_bars, win_pct)
    return f, win1, win2, gain


# ═══════════════════════════════════════════════
# 6. 组合挖掘
# ═══════════════════════════════════════════════

def _pass(n, wr, min_n=30, wr_target=85, ci_low=80):
    """有效性门槛：样本数 + 胜率 + 95%置信下限"""
    if n < min_n or wr < wr_target:
        return False
    try:
        from scipy.stats import beta
        wins = int(round(n * wr / 100))
        low = beta.ppf(0.025, wins + 1, n - wins + 1) * 100
        return low >= ci_low
    except Exception:
        return n >= 5 * min_n


def mine(stocks, hold_bars=8, win_pct=1.0, wr_target=85, min_n=30, start="20260101", limit=None, top_k=20):
    """三级挖掘 → rules2.json 结构。stocks: {code: name}
    第一级: 单条件统计; 第二级: 两两AND精确组合; 第三级: 贪心扩展 + 资金流分离验证。"""
    t0 = time.time()
    codes = list(stocks.keys())
    if limit:
        codes = codes[:limit]

    cond_names = list(CONDITIONS.keys())

    # 第一遍: 单条件累计 + 保留每票mask（供组合精确统计）
    n_total = {c: 0 for c in cond_names}
    w_total = {c: 0 for c in cond_names}
    stock_data = []  # (masks_dict, win(bool), gain, code)

    done = 0
    for code in codes:
        res = backtest_stock2(code, stocks[code], hold_bars, win_pct, start)
        if res is None:
            continue
        f, win1, win2, gain = res
        win = win1.values.astype(bool)
        masks = {}
        for cn in cond_names:
            m = CONDITIONS[cn](f).values.astype(bool)
            masks[cn] = m
            n_total[cn] += int(m.sum())
            w_total[cn] += int((m & win).sum())
        stock_data.append((masks, win, gain, code))
        done += 1
        if done % 50 == 0:
            logger.info(f"已处理 {done}/{len(codes)} 只, 用时 {time.time()-t0:.0f}s")
    logger.info(f"有效票 {len(stock_data)}/{len(codes)}")

    # 单条件排名 → 组合搜索空间
    single = []
    for cn in cond_names:
        n = n_total[cn]
        if n < min_n:
            continue
        wr = w_total[cn] / n * 100
        single.append({"cond": cn, "n": n, "wr": round(wr, 1),
                       "pass": _pass(n, wr, min_n, wr_target)})
    single.sort(key=lambda r: -r["wr"])
    search = [s["cond"] for s in single[:top_k]]
    logger.info(f"单条件: {len(single)}个, 组合搜索空间: {search}")

    # 组合精确统计（逐票 AND）
    def combo_stats(cond_list):
        n = w = 0
        g = []
        for masks, win, gain, _ in stock_data:
            m = None
            for cn in cond_list:
                mm = masks[cn]
                m = mm if m is None else (m & mm)
            if m is None:
                continue
            if m.any():
                n += int(m.sum())
                w += int((m & win).sum())
                g.extend(gain[m])
        return n, w, g

    def build_rule(cond_list, n, w, g):
        wr = w / n * 100 if n else 0
        avg = float(np.mean(g)) if g else 0
        # 资金流分离（主力净流入 vs 净流出 胜率对比）
        in_n = in_w = out_n = out_w = 0
        for masks, win, gain, _ in stock_data:
            m = None
            for cn in cond_list:
                mm = masks[cn]
                m = mm if m is None else (m & mm)
            if m is None or not m.any():
                continue
            fi = masks.get("fund_in", np.zeros_like(m))
            in_m = m & fi
            out_m = m & ~fi
            in_n += int(in_m.sum()); in_w += int((in_m & win).sum())
            out_n += int(out_m.sum()); out_w += int((out_m & win).sum())
        in_wr = round(in_w / in_n * 100, 1) if in_n >= 10 else None
        out_wr = round(out_w / out_n * 100, 1) if out_n >= 10 else None
        fund_required = (in_wr is not None and out_wr is not None
                         and in_wr >= wr_target and out_wr < wr_target - 5)
        return {
            "conditions": cond_list, "n": n, "wr": round(wr, 1),
            "avg_gain": round(avg, 2),
            "fund_split": {"inflow": {"n": in_n, "wr": in_wr},
                           "outflow": {"n": out_n, "wr": out_wr}},
            "fund_required": fund_required,
            "sell_pct": round(avg * 0.9, 1) if avg > 0 else win_pct,
        }

    # 第二级: 两两AND组合（精确）
    combos = []
    for i in range(len(search)):
        for j in range(i + 1, len(search)):
            n, w, g = combo_stats([search[i], search[j]])
            if n >= min_n and w / n * 100 >= wr_target and _pass(n, w / n * 100, min_n, wr_target):
                combos.append((search[i], search[j], n, w, g))
    combos.sort(key=lambda x: -(x[3] / x[2]))
    logger.info(f"两两组合通过≥{wr_target}%: {len(combos)}")

    # 第三级: 贪心扩展（对最优组合加第3、4条件）
    rules = []
    seen = set()
    for a, b, n0, w0, g0 in combos[:min(15, len(combos))]:
        conds = [a, b]
        if tuple(sorted(conds)) in seen:
            continue
        seen.add(tuple(sorted(conds)))
        improved = True
        while len(conds) < 4 and improved:
            improved = False
            best = None
            cur_wr = w0 / n0 * 100
            for cn in search:
                if cn in conds:
                    continue
                n2, w2, g2 = combo_stats(conds + [cn])
                if n2 < min_n:
                    continue
                if w2 / n2 * 100 >= cur_wr + 0.5:
                    if best is None or (w2 / n2 * 100 > best[2]):
                        best = (cn, n2, w2, g2)
            if best:
                conds.append(best[0])
                n0, w0, g0 = best[1], best[2], best[3]
                improved = True
        n_f, w_f, g_f = combo_stats(conds)
        if n_f >= min_n and _pass(n_f, w_f / n_f * 100, min_n, wr_target):
            rules.append(build_rule(conds, n_f, w_f, g_f))

    # 去重 + 排序
    seen2 = set()
    final = []
    for idx, r in enumerate(sorted(rules, key=lambda x: -x["wr"])):
        k = tuple(sorted(r["conditions"]))
        if k in seen2:
            continue
        seen2.add(k)
        r["id"] = f"R{idx+1:03d}"
        final.append(r)

    return {"single": single, "rules": final[:20]}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = sys.argv[1:]
    limit = None
    if "--limit" in args:
        limit = int(args[args.index("--limit") + 1])
    pool = load_pool_window(5)
    logger.info(f"动态池 {len(pool)}只，开始挖掘")
    result = mine(pool, limit=limit)
    rules = result["rules"]
    logger.info(f"挖掘完成: {len(rules)}条规则")
    out = {
        "generated": time.strftime("%Y-%m-%d %H:%M"),
        "strategy": "2号: 动态池(成交额top150×5日) + 全量指标 + 主力资金流",
        "win_def": "买入后8根15分钟K线内最高涨幅≥1%为胜",
        "hold_bars": 8, "win_pct": 1.0, "min_n": 30, "wr_target": 85,
        "rules": rules,
    }
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rules2.json")
    with open(path, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=2)
    print(f"rules2.json 已生成: {path}, {len(rules)}条规则")
    for r in rules[:10]:
        print(f"  {r['id']} {r['conditions']} n={r['n']} wr={r['wr']}% "
              f"fund_required={r['fund_required']} 卖点{r['sell_pct']}%")


if __name__ == "__main__":
    main()
