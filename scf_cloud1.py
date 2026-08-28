# -*- coding: utf-8 -*-
"""1号量化程序 — 腾讯云SCF入口（评分制：广度评分+资金深化，每次触发=扫一轮+推新信号）"""
import os, json, time, logging

import cloud_function as cf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scf1")

STATE_BUCKET = os.environ.get("STATE_BUCKET", "")
STATE_KEY = "state/cloud1_state.json"
WINDOWS = [(9, 0, 11, 30), (13, 0, 15, 0)]


def _in_window(now):
    for h0, m0, h1, m1 in WINDOWS:
        t0 = now.replace(hour=h0, minute=m0, second=0, microsecond=0)
        t1 = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
        if t0 <= now <= t1:
            return True
    return False


def _cos_client():
    from qcloud_cos import CosConfig, CosS3Client
    conf = CosConfig(
        Region=os.environ.get("COS_REGION", "ap-guangzhou"),
        SecretId=os.environ.get("TX_SECRETID", ""),
        SecretKey=os.environ.get("TX_SECRETKEY", ""),
    )
    return CosS3Client(conf)


def _load_state():
    today = cf.bj_now().strftime("%Y%m%d")
    default = {"date": today, "pushed": []}
    if not STATE_BUCKET:
        return default
    try:
        resp = _cos_client().get_object(Bucket=STATE_BUCKET, Key=STATE_KEY)
        state = json.loads(resp["Body"].get_raw_stream().read().decode("utf-8"))
        return state if state.get("date") == today else default
    except Exception:
        return default


def _save_state(state):
    if not STATE_BUCKET:
        return
    try:
        _cos_client().put_object(Bucket=STATE_BUCKET, Key=STATE_KEY,
                                 Body=json.dumps(state, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        logger.warning(f"保存COS状态失败: {e}")


def main_handler(event, context):
    now = cf.bj_now()
    logger.info(f"SCF触发 @ {now.strftime('%m/%d %H:%M:%S')}")
    if not cf.is_trading_day():
        return {"ok": True, "skip": "非交易日"}
    if not _in_window(now):
        return {"ok": True, "skip": "非窗口"}

    from stock_pool import STOCK_POOL
    all_stocks = {c: STOCK_POOL[c] for c in STOCK_POOL}
    state = _load_state()
    pushed = {(p["code"], p.get("sig", "")) for p in state["pushed"]}

    results = cf.scan_once(all_stocks)
    new_n = 0
    scan_time = now.strftime("%m/%d %H:%M")
    for r in results:
        if r.get("signal") != "BUY":
            continue
        key = (r["code"], "BUY")
        if key in pushed:
            continue
        pushed.add(key)
        state["pushed"].append({"code": r["code"], "sig": "BUY"})
        try:
            cf.push_signal(r, scan_time)
            new_n += 1
        except Exception as e:
            logger.error(f"推送异常 {r['code']}: {e}")
        time.sleep(0.3)

    _save_state(state)
    logger.info(f"本轮: {len(results)}命中, {new_n}新推送")
    return {"ok": True, "hits": len(results), "pushed": new_n}
