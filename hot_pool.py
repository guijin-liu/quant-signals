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


def save_today(n=HOT_N, date=None):
    """拉当日人气前n，存 data/hot_pool/{date}.json。东财挂→换手率榜降级。返回路径或None"""
    rows = em_client.em_fetch_hot_rank(n)
    if not rows:
        logger.warning("人气榜失败，降级换手率榜(热度代理)")
        rows = em_client.em_fetch_hot_rank_fallback(n)
    if not rows:
        logger.error("人气榜+换手率榜均失败，未保存")
        return None
    date = date or datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"{date}.json")
    source = "hot_rank" if rows[0].get("rank") else "turnover"
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "count": len(rows), "source": source, "stocks": rows},
                  f, ensure_ascii=False, indent=1)
    logger.info(f"已保存当日热点池: {date} {len(rows)}只(source={source}) → {path}")
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
