# -*- coding: utf-8 -*-
"""2号量化程序 — 腾讯云SCF入口（serverless 定时触发器版）
每次触发 = 扫描一轮 + 推送新信号；用 COS 持久化"今日已推"状态防重复推送。
部署目录: /tmp 或函数目录内，与 cloud2_function.py 等同级。
"""
import os, json, time, logging
from datetime import timedelta

import cloud2_function as cf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("scf2")

# COS 状态桶（环境变量注入）
STATE_BUCKET = os.environ.get("STATE_BUCKET", "")
STATE_KEY = "state/cloud2_state.json"

# 扫描窗口（北京时间）：上午 9:00-11:30，下午 13:00-15:00（覆盖完整交易时段，含集合竞价）
WINDOWS = [(9, 0, 11, 30), (13, 0, 15, 0)]


def _in_window(now):
    for h0, m0, h1, m1 in WINDOWS:
        t0 = now.replace(hour=h0, minute=m0, second=0, microsecond=0)
        t1 = now.replace(hour=h1, minute=m1, second=0, microsecond=0)
        if t0 <= now <= t1:
            return True
    return False


def _cos_client():
    """惰性创建 COS 客户端（用最小权限子账号密钥，环境变量仅限该子账号）"""
    from qcloud_cos import CosConfig, CosS3Client
    conf = CosConfig(
        Region=os.environ.get("COS_REGION", "ap-guangzhou"),
        SecretId=os.environ.get("TX_SECRETID", ""),
        SecretKey=os.environ.get("TX_SECRETKEY", ""),
    )
    return CosS3Client(conf)


def _load_state():
    """从 COS 读取今日已推状态 {date, pushed:[{code,conds}]}"""
    today = cf.bj_now().strftime("%Y%m%d")
    default = {"date": today, "pushed": []}
    if not STATE_BUCKET:
        return default
    try:
        client = _cos_client()
        resp = client.get_object(Bucket=STATE_BUCKET, Key=STATE_KEY)
        raw = resp["Body"].get_raw_stream().read().decode("utf-8")
        state = json.loads(raw)
        if state.get("date") != today:  # 跨日重置
            return default
        return state
    except Exception:
        return default


def _save_state(state):
    if not STATE_BUCKET:
        return
    try:
        client = _cos_client()
        client.put_object(Bucket=STATE_BUCKET, Key=STATE_KEY,
                          Body=json.dumps(state, ensure_ascii=False).encode("utf-8"))
    except Exception as e:
        logger.warning(f"保存COS状态失败: {e}")


def main_handler(event, context):
    now = cf.bj_now()
    logger.info(f"SCF触发 @ {now.strftime('%m/%d %H:%M:%S')}")
    if not cf.is_trading_day():
        logger.info("非交易日，跳过")
        return {"ok": True, "skip": "非交易日"}
    if not _in_window(now):
        logger.info("非扫描窗口，跳过")
        return {"ok": True, "skip": "非窗口"}

    rules = cf.load_rules()
    if not rules:
        logger.error("无规则")
        return {"ok": False, "err": "无规则"}

    from fixed_pool_2 import FIXED_POOL_2
    pool = dict(FIXED_POOL_2)
    state = _load_state()
    pushed = {(p["code"], tuple(p["conds"])) for p in state["pushed"]}

    results = cf.scan_once(pool, rules)
    new_n = 0
    scan_time = now.strftime("%m/%d %H:%M")
    for r in results:
        key = (r["code"], tuple(r.get("rule", {}).get("conditions", [])))
        if key in pushed:
            continue
        pushed.add(key)
        state["pushed"].append({"code": r["code"], "conds": r.get("rule", {}).get("conditions", [])})
        try:
            cf.push_signal(r, scan_time)
            new_n += 1
        except Exception as e:
            logger.error(f"推送异常 {r['code']}: {e}")
        time.sleep(0.3)

    _save_state(state)
    logger.info(f"本轮: {len(results)}命中, {new_n}新推送")
    return {"ok": True, "hits": len(results), "pushed": new_n}
