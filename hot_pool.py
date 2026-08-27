"""刘圭金3号 — 热点股池（东财人气榜前50）

每天(云端收盘后)拉当日人气榜前50 → 存 data/hot_pool/YYYYMMDD.json；
load_latest_hot_pool() 返回最新一个交易日的热点池 = 当日热点前50。
备用：东财人气榜失败 → 换手率榜前50（热度代理，东财f8→新浪turnoverratio 双降级）。
"""
import os, json, glob, logging, sys, datetime
import em_client

logger = logging.getLogger(__name__)

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "hot_pool")
HOT_N = 50


def _load_history(days=5):
    """近 days 个榜单历史 → {code: {"days": 上榜天数, "last_rank": 最近排名}}"""
    files = sorted(glob.glob(os.path.join(POOL_DIR, "*.json")))[-days:]
    hist = {}
    for path in files:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("stocks", []):
                c = s.get("code")
                if not c:
                    continue
                h = hist.setdefault(c, {"days": 0, "last_rank": 999})
                h["days"] += 1
                h["last_rank"] = min(h["last_rank"], s.get("rank", 999))
        except Exception:
            continue
    return hist


def _hotness(rank, hist_entry):
    """持续热度分（2026-08-27 借鉴概念热度策略）：基础人气 + 持续上榜 + 排名趋势。越大越热。"""
    days = hist_entry.get("days", 0) if hist_entry else 0
    prev_rank = hist_entry.get("last_rank", 999) if hist_entry else 999
    score = max(0, 100 - rank)          # 今日人气（排名越前分越高）
    score += min(days, 5) * 6           # 持续上榜加分（多日热度=资金持续关注）
    if prev_rank > rank:                # 排名上升加分（趋势向上）
        score += 5
    return score


def save_today(n=HOT_N, date=None):
    """拉当日人气前n，存 data/hot_pool/{date}.json（含持续热度分，按热度排序）。
    降级链: 东财人气榜 → 同花顺热股榜(非东财系,2026-08-27接入) → 换手率榜(东财f8→新浪)。
    返回路径或None"""
    rows = em_client.em_fetch_hot_rank(n)
    source = "hot_rank"
    if not rows:
        logger.warning("东财人气榜失败 → 同花顺热股榜兜底(非东财系)")
        rows = em_client.em_fetch_hot_rank_ths(n)
        source = "ths_hot"
    if not rows:
        logger.warning("同花顺也失败 → 换手率榜降级(热度代理)")
        rows = em_client.em_fetch_hot_rank_fallback(n)
        source = "turnover"
    if not rows:
        logger.error("人气榜+同花顺+换手率榜均失败，未保存")
        return None
    date = date or (datetime.datetime.utcnow() + datetime.timedelta(hours=8)).strftime("%Y%m%d")  # 北京时间
    hist = _load_history(5)
    for s in rows:
        s["hotness"] = _hotness(s.get("rank", 99), hist.get(s.get("code")))
    rows.sort(key=lambda x: -x.get("hotness", 0))  # 持续热度高的排前
    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "count": len(rows), "source": source, "stocks": rows},
                  f, ensure_ascii=False, indent=1)
    logger.info(f"已保存当日热点池: {date} {len(rows)}只(source={source},含热度分) → {path}")
    return path


def load_latest_hot_pool(as_of=None):
    """最新交易日热点池 → {code: name}"""
    files = sorted(glob.glob(os.path.join(POOL_DIR, "*.json")))
    if as_of:
        files = [p for p in files if os.path.basename(p)[:8] <= as_of]
    if not files:
        logger.error("无热点池文件，返回空池")
        return {}
    with open(files[-1], encoding="utf-8") as f:
        data = json.load(f)
    pool = {s["code"]: s.get("name", s["code"]) for s in data.get("stocks", []) if s.get("code")}
    logger.info(f"当日热点池(最新{os.path.basename(files[-1])[:8]}): {len(pool)}只")
    return pool


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = sys.argv[1:]
    if "--save-today" in args:
        path = save_today()
        print(f"SAVED:{path}" if path else "FAIL")
    elif "--window" in args:
        pool = load_latest_hot_pool()
        print(f"热点池: {len(pool)}只")
        for c, nm in list(pool.items())[:10]:
            print(f"  {c} {nm}")


if __name__ == "__main__":
    main()
