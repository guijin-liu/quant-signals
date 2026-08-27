"""逐票独立回测 → per_stock_rules.json (2026-08-27)

理念：每只票规律不同（per-stock），全局统计会淹没个股差异。
对 2号固定池每票：T+1 胜率统计全部条件 → 选出该票的高胜率条件（n≥25 且 wr≥80% 或 95%CI≥70%）
→ 写入 data/per_stock_rules.json，实盘按票用各自规则（无则退回全局 rules2）。

用法: python per_stock_scan.py           # 全量生成
      python per_stock_scan.py --check   # 只读现有规则文件
"""
import sys, os, json, time, logging
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.stats import beta
import mx_fetcher
from fixed_pool_2 import FIXED_POOL_2
from backtest_miner2 import compute_feature_matrix, CONDITIONS, label_wins, add_fund_features

logging.basicConfig(level=logging.WARNING)
BASE = os.path.dirname(os.path.abspath(__file__))
RULES_OUT = os.path.join(BASE, "data", "per_stock_rules.json")
MIN_N = 25
MIN_WR = 80.0
MIN_CI = 70.0
TOP_K = 3


def scan_pool(pool):
    """返回 {code: {"name","rules":[{"conditions":[...], "n","wr","ci"}], "best":...}}"""
    out = {}
    for code, name in pool.items():
        try:
            df = mx_fetcher.fetch_kline(code, '15')
            if df.empty or len(df) < 70:
                continue
            fm = compute_feature_matrix(df['close'].values, df['high'].values,
                                        df['low'].values, df['volume'].values, df['open'].values)
            fm = add_fund_features(fm, df['close'].values, df['volume'].values)
            if fm.empty:
                continue
            win1, _, _ = label_wins(df['high'].values, df['close'].values, df['date'].tolist())
            w = win1.values.astype(bool)
            rows = []
            for cn, cond in CONDITIONS.items():
                try:
                    mask = cond(fm).values.astype(bool)
                except Exception:
                    continue
                n = int(mask.sum())
                if n < MIN_N:
                    continue
                wins = int((mask & w).sum())
                wr = wins / n * 100
                ci = beta.ppf(0.025, wins + 1, n - wins + 1) * 100
                rows.append({"conditions": [cn], "n": n, "wr": round(wr, 1), "ci": round(ci, 1)})
            rows.sort(key=lambda r: -r["wr"])
            good = [r for r in rows if r["wr"] >= MIN_WR or r["ci"] >= MIN_CI][:TOP_K]
            if good:
                out[code] = {"name": name, "rules": good, "best": good[0]}
        except Exception as e:
            print(f"  skip {code} {name}: {str(e)[:50]}")
    return out


def main():
    if "--check" in sys.argv:
        if os.path.exists(RULES_OUT):
            d = json.load(open(RULES_OUT, encoding="utf-8"))
            print(f"现有逐票规则: {len(d)}票")
            for code, v in list(d.items())[:10]:
                r = v["rules"][0]
                print(f"  {code} {v['name']}: {r['conditions']} n={r['n']} wr={r['wr']}% ci={r['ci']}%")
        return
    t0 = time.time()
    print(f"逐票回测开始 ({len(FIXED_POOL_2)}票, {len(CONDITIONS)}条件, T+1胜率)...")
    res = scan_pool(FIXED_POOL_2)
    os.makedirs(os.path.dirname(RULES_OUT), exist_ok=True)
    json.dump(res, open(RULES_OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    strong = [c for c, v in res.items() if v["best"]["ci"] >= MIN_CI]
    print(f"完成 {time.time()-t0:.0f}s | 有规则票 {len(res)}/{len(FIXED_POOL_2)} | 强规律(CI≥{MIN_CI}%) {len(strong)}票")
    print(f"输出: {RULES_OUT}")


if __name__ == "__main__":
    main()
