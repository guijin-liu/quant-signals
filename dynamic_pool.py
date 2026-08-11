"""刘圭金2号 — 动态股票池（成交额前150 × 5日滚动窗口）

每天(云端收盘后)拉全市场成交额前150 → 存 data/top_amount/YYYYMMDD.json；
load_pool_window(5) 返回最近5个交易日榜单并集 = 当日动态池。
冷启动: --backfill 首日取 top300 作种子，连续5个交易日填满窗口。
"""
import os, json, glob, logging, sys, datetime
import em_client

logger = logging.getLogger(__name__)

POOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "top_amount")
DAYS = 5  # 滚动窗口天数


def save_today(n=150, date=None):
    """拉当日成交额前n，存 data/top_amount/{date}.json。返回路径或None"""
    rows = em_client.em_fetch_top_amount(n)
    if not rows:
        logger.error("top150 拉取失败(空)，未保存")
        return None
    date = date or datetime.datetime.now().strftime("%Y%m%d")
    os.makedirs(POOL_DIR, exist_ok=True)
    path = os.path.join(POOL_DIR, f"{date}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"date": date, "count": len(rows), "stocks": rows}, f, ensure_ascii=False, indent=1)
    logger.info(f"已保存当日榜: {date} {len(rows)}只 → {path}")
    return path


def load_pool_window(days=DAYS, as_of=None):
    """最近 days 个交易日 top150 并集 → {code: name}（约300~450只）"""
    files = sorted(glob.glob(os.path.join(POOL_DIR, "*.json")))
    if as_of:
        files = [p for p in files if os.path.basename(p)[:8] <= as_of]
    pool = {}
    for path in files[-days:]:
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            for s in data.get("stocks", []):
                if s.get("code"):
                    pool[s["code"]] = s.get("name", s["code"])
        except Exception as e:
            logger.warning(f"读榜失败 {path}: {e}")
    return pool


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    args = sys.argv[1:]
    if "--save-today" in args:
        n = 300 if "--backfill" in args else 150
        path = save_today(n=n)
        print(f"SAVED:{path}" if path else "FAIL")
    elif "--window" in args:
        pool = load_pool_window()
        print(f"动态池: {len(pool)}只")
        for i, (c, nm) in enumerate(list(pool.items())[:10]):
            print(f"  {c} {nm}")


if __name__ == "__main__":
    main()
