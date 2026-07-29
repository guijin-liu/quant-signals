"""数据获取层 — 腾讯K线 + 妙想财务，替换baostock"""
import logging, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
UA = "Mozilla/5.0"
MX_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"

def _prefix(code: str) -> str:
    return "sh" if code.startswith(("6", "9")) else "sz"

# ═══════════════════════════════════════
# 腾讯 K 线（HTTP，不封IP，海外可用）
# ═══════════════════════════════════════

def fetch_kline(code: str, freq: str = "15") -> pd.DataFrame:
    """
    腾讯K线 — 格式统一: [datetime, open, close, high, low, volume, ...]
    freq: '5'=5min, '15'=15min, '60'=60min, 'day'=日线
    """
    pfx = _prefix(code)

    if freq == "day":
        url = "https://web.ifzq.gtimg.cn/appstock/app/fqkline/get"
        params = {"param": f"{pfx}{code},day,,,90,qfq"}
    else:
        freq_map = {"5": ("m5", 240), "15": ("m15", 120), "60": ("m60", 60)}
        mk, limit = freq_map.get(freq, ("m15", 120))
        url = "https://ifzq.gtimg.cn/appstock/app/kline/mkline"
        params = {"param": f"{pfx}{code},{mk},,{limit}"}

    try:
        r = requests.get(url, params=params,
                         headers={"User-Agent": UA, "Referer": "https://gu.qq.com/"}, timeout=15)
        d = r.json()

        if freq == "day":
            rows = d.get("data", {}).get(f"{pfx}{code}", {}).get("qfqday", []) or \
                   d.get("data", {}).get(f"{pfx}{code}", {}).get("day", [])
        else:
            rows = d.get("data", {}).get(f"{pfx}{code}", {}).get(freq_map[freq][0], [])

        if not rows:
            return pd.DataFrame()

        data = []
        for r in rows:
            if len(r) < 6:
                continue
            ts = str(r[0])
            data.append({
                "date": ts[:10] if freq == "day" else ts[:8],
                "time": ts if freq != "day" else "",
                "open": float(r[1]), "close": float(r[2]),
                "high": float(r[3]), "low": float(r[4]),
                "volume": float(r[5]),
            })

        df = pd.DataFrame(data)
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    except Exception as e:
        logger.error(f"fetch_kline({code}, {freq}): {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════
# 腾讯实时行情（预筛用）
# ═══════════════════════════════════════

def fetch_daily_snapshot(code: str) -> list | None:
    """腾讯实时行情 → 预筛，返回最近5天的简化数据"""
    pfx = _prefix(code)

    # 实时快照
    try:
        r = requests.get(f"https://qt.gtimg.cn/q={pfx}{code}",
                         headers={"User-Agent": UA}, timeout=10)
        r.encoding = "gbk"
        vals = r.text.split('"')[1].split("~")
        if len(vals) < 50:
            return None

        price = float(vals[3]) if vals[3] else 0
        preclose = float(vals[4]) if vals[4] else 0
        change_pct = float(vals[32]) if vals[32] else 0
        vol_ratio = float(vals[49]) if vals[49] else 0
        amount_wan = float(vals[37]) if vals[37] else 0
        volume = float(vals[6]) if vals[6] else 0

        if price <= 0:
            return None

        # 转成旧格式 [date, close, volume, turn, preclose]
        now = datetime.now().strftime("%Y-%m-%d")
        return [[now, price, volume, 0, preclose],  # 今天
                [now, preclose, 0, 0, preclose]]     # 昨收占位

    except Exception as e:
        logger.error(f"snapshot({code}): {e}")
        return None

def get_realtime_quotes(codes: list) -> dict:
    """批量实时行情 — 腾讯API"""
    if not codes:
        return {}
    prefixed = [f"{_prefix(c)}{c}" for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.encoding = "gbk"
        result = {}
        for line in r.text.strip().split(";"):
            if '="' not in line:
                continue
            key = line.split("=")[0].split("_")[-1]
            code = key[2:]
            vals = line.split('"')[1].split("~")
            if len(vals) < 50:
                continue
            result[code] = {
                "name": vals[1],
                "price": float(vals[3]) if vals[3] else 0,
                "preclose": float(vals[4]) if vals[4] else 0,
                "change_pct": float(vals[32]) if vals[32] else 0,
                "vol_ratio": float(vals[49]) if vals[49] else 0,
                "amount_wan": float(vals[37]) if vals[37] else 0,
                "volume": float(vals[6]) if vals[6] else 0,
                "pe": float(vals[39]) if vals[39] else 0,
                "pb": float(vals[46]) if vals[46] else 0,
                "mcap_yi": float(vals[44]) if vals[44] else 0,
            }
        return result
    except Exception as e:
        logger.error(f"批量行情失败: {e}")
        return {}


# ═══════════════════════════════════════
# mx-data 财务数据
# ═══════════════════════════════════════

def check_three_year_loss(code: str) -> bool:
    """mx-data查连续3年亏损 — True=亏损剔除"""
    import os
    from stock_pool import STOCK_POOL_BACKUP
    name = STOCK_POOL_BACKUP.get(code, {}).get("name", code)
    mx_key = os.environ.get("MX_APIKEY", "")

    if not mx_key:
        logger.warning("MX_APIKEY未设置，跳过财务过滤")
        return False

    try:
        r = requests.post(MX_URL,
            headers={"Content-Type": "application/json", "apikey": mx_key},
            json={"toolQuery": f"{name}{code}近三年每年净利润"}, timeout=30)
        r.raise_for_status()
        d = r.json()

        dto_list = (d.get("data", {}).get("data", {})
                     .get("searchDataResultDTO", {}).get("dataTableDTOList", []))
        if not dto_list:
            return False

        table = dto_list[0].get("table", {})
        headers = table.get("headName", [])

        # 提取净利润值
        profits = []
        for k, v in table.items():
            if k == "headName":
                continue
            for val in v:
                try:
                    p = float(val) if val else 0
                    profits.append(p)
                except:
                    pass

        if len(profits) < 3:
            return False
        recent = profits[:3]
        return all(p < 0 for p in recent)

    except Exception as e:
        logger.error(f"财务检查({code}): {e}")
        return False
