"""数据获取层 — 腾讯K线 + 妙想财务，替换baostock"""
import os, logging, requests, numpy as np, pandas as pd
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
UA = "Mozilla/5.0"
MX_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
MX_KEY = os.environ.get("MX_APIKEY", "")

_cache = {}  # 内存缓存: {key: (data, timestamp)}

def _prefix(code: str) -> str:
    return "sh" if code.startswith(("6", "9")) else "sz"

def _cached(key: str, ttl: int = 3600):
    """检查缓存是否有效，返回缓存数据或None"""
    entry = _cache.get(key)
    if entry and (datetime.now() - entry[1]).seconds < ttl:
        return entry[0]
    return None

def _cache_set(key: str, data):
    _cache[key] = (data, datetime.now())

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
# 妙想 mx-data（东方财富权威数据库）— 主力
# ═══════════════════════════════════════

def _mx_query(tool_query: str) -> dict:
    """调用mx-data API，自动节流"""
    if not MX_KEY:
        return {}
    try:
        r = requests.post(MX_URL,
            headers={"Content-Type": "application/json", "apikey": MX_KEY},
            json={"toolQuery": tool_query}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"mx_query失败: {e}")
        return {}

def _parse_kv(data: dict) -> dict:
    """解析mx-data返回的键值对 → {字段名: 值}"""
    try:
        dto = data["data"]["data"]["searchDataResultDTO"]["dataTableDTOList"][0]
        nm = dto.get("nameMap", {})
        tb = dto.get("table", {})
        result = {}
        for k, cn_name in nm.items():
            vals = tb.get(k, [])
            result[str(cn_name)] = vals[0] if vals else None
        return result
    except:
        return {}

def get_stock_valuation(code: str, name: str) -> dict:
    """个股估值快照 — PE/PB/市值/涨跌幅/换手率（日缓存1h）"""
    key = f"val_{code}"
    cached = _cached(key, 3600)
    if cached:
        return cached

    data = _mx_query(f"{name}{code} 最新价 涨跌幅 量比 换手率 PE PB 总市值")
    kv = _parse_kv(data)
    if not kv:
        return {}

    result = {
        "price": _num(kv, "最新价"), "change_pct": _num(kv, "涨跌幅"),
        "vol_ratio": _num(kv, "量比"), "turnover": _num(kv, "换手率"),
        "pe": _num(kv, "市盈率PE(TTM)"), "pb": _num(kv, "市净率"),
        "mcap": _num(kv, "总市值"),  # 单位:亿
    }
    _cache_set(key, result)
    return result

def get_financial_quality(code: str, name: str) -> dict:
    """财务质量 — ROE/营收增速/利润率/负债率（日缓存4h）"""
    key = f"fin_{code}"
    cached = _cached(key, 14400)
    if cached:
        return cached

    data = _mx_query(f"{name}{code} 净资产收益率 净利润同比增长率 营业收入同比增长率 资产负债率 毛利率")
    kv = _parse_kv(data)
    if not kv:
        return {}

    result = {
        "roe": _num(kv, "净资产收益率ROE"),  # 加权ROE %
        "profit_growth": _num(kv, "净利润同比增长率"),  # %
        "revenue_growth": _num(kv, "营业收入同比增长率"),  # %
        "debt_ratio": _num(kv, "资产负债率"),  # %
        "gross_margin": _num(kv, "毛利率"),  # %
    }
    _cache_set(key, result)
    return result

def get_market_brief() -> dict:
    """市场概况 — 指数涨跌+成交额（缓存30分钟）"""
    key = "market"
    cached = _cached(key, 1800)
    if cached:
        return cached

    data = _mx_query("上证指数 沪深300 创业板指 今日涨跌幅 成交额")
    kv = _parse_kv(data)
    if not kv:
        return {}

    result = {
        "sh_idx": _num(kv, "涨跌幅"),  # 上证涨跌幅
        "sz_amount": _num(kv, "成交额"),  # 成交额(亿)
        "update_time": datetime.now().strftime("%H:%M"),
    }
    _cache_set(key, result)
    return result

def score_financial_quality(fin: dict, val: dict) -> tuple:
    """综合财务评分 → (score:0-5, summary:str)"""
    score = 0
    tags = []

    # ROE > 10% → 优质
    roe = fin.get("roe", 0)
    if roe > 15: score += 2; tags.append(f"ROE{roe:.0f}%")
    elif roe > 5: score += 1

    # 净利润增长 > 20%
    pg = fin.get("profit_growth", 0)
    if pg > 50: score += 1; tags.append(f"净利+{pg:.0f}%")
    elif pg < -20: score -= 1; tags.append(f"净利{pg:.0f}%⚠️")

    # 负债率 < 50% → 健康
    dr = fin.get("debt_ratio", 100)
    if dr < 40: score += 1
    elif dr > 70: score -= 1; tags.append(f"负债{dr:.0f}%")

    # PE合理区间
    pe = val.get("pe", 0)
    if 10 < pe < 40: score += 1
    elif pe > 100: score -= 1; tags.append(f"PE{pe:.0f}x")

    # 毛利率
    gm = fin.get("gross_margin", 0)
    if gm > 30: score += 1

    if score >= 4: label = "优质"
    elif score >= 2: label = "良好"
    elif score >= 0: label = "一般"
    else: label = "谨慎"

    return score, f"{label}({score}分): {' '.join(tags)}" if tags else f"{label}({score}分)"

def _num(d: dict, *keys) -> float:
    """从kv中提取数值，支持部分匹配"""
    for pattern in keys:
        for k, v in d.items():
            if pattern in str(k) and v is not None:
                try:
                    s = str(v).replace(",", "").replace("%", "").replace("亿", "").replace("万", "")
                    if "万亿" in str(v):
                        return float(s) * 10000
                    return float(s)
                except:
                    pass
    return 0

# 保留旧函数兼容
check_three_year_loss = lambda code: False  # 已由财务质量评分替代
