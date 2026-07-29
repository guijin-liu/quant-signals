"""数据获取层
主力：妙想 mx-data（东方财富权威数据库）
备用：腾讯 HTTP API（mx-data 不支持的K线序列）
"""
import os, logging, requests, numpy as np, pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)
UA = "Mozilla/5.0"
MX_URL = "https://mkapi2.dfcfs.com/finskillshub/api/claw/query"
MX_KEY = os.environ.get("MX_APIKEY", "")

_cache = {}

def _prefix(code: str) -> str:
    return "sh" if code.startswith(("6", "9")) else "sz"

def _cached(key: str, ttl: int = 3600):
    entry = _cache.get(key)
    if entry and (datetime.now() - entry[1]).seconds < ttl:
        return entry[0]
    return None

def _cache_set(key: str, data):
    _cache[key] = (data, datetime.now())

def _num(d: dict, *patterns) -> float:
    for p in patterns:
        for k, v in d.items():
            if p in str(k) and v is not None:
                try:
                    s = str(v).replace(",", "").replace("%", "").replace("亿", "").replace("万", "")
                    if "万亿" in str(v): return float(s) * 10000
                    return float(s)
                except: pass
    return 0


# ═══════════════════════════════════════════════
# 主力：妙想 mx-data（东方财富权威数据库）
# ═══════════════════════════════════════════════

def _mx_query(tool_query: str) -> dict:
    if not MX_KEY: return {}
    try:
        r = requests.post(MX_URL,
            headers={"Content-Type": "application/json", "apikey": MX_KEY},
            json={"toolQuery": tool_query}, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.error(f"mx_query: {e}")
        return {}

def _parse_kv(data: dict) -> dict:
    try:
        dto = data["data"]["data"]["searchDataResultDTO"]["dataTableDTOList"][0]
        nm = dto.get("nameMap", {})
        tb = dto.get("table", {})
        result = {}
        for k, cn_name in nm.items():
            vals = tb.get(k, [])
            result[str(cn_name)] = vals[0] if vals else None
        return result
    except: return {}


def get_quotes(codes: list, stocks: dict) -> dict:
    """主力行情 — 妙想个股估值（缓存1h），失败降级腾讯批量"""
    result = {}
    need_fetch = []
    for code in codes:
        cached = _cached(f"quote_{code}", 3600)
        if cached:
            result[code] = cached
        else:
            need_fetch.append(code)

    if need_fetch:
        for code in need_fetch:
            name = stocks.get(code, {}).get("name", code) if isinstance(stocks.get(code), dict) else stocks.get(code, code)
            val = get_stock_valuation(code, name)
            if val:
                result[code] = val

    # 妙想拿不到的降级腾讯
    missing = [c for c in codes if c not in result]
    if missing:
        tencent = _tencent_batch_quotes(missing)
        result.update(tencent)

    return result


def get_stock_valuation(code: str, name: str) -> dict:
    """妙想估值快照 — PE/PB/市值/涨跌幅/换手率（缓存1h）"""
    key = f"val_{code}"
    cached = _cached(key, 3600)
    if cached: return cached

    data = _mx_query(f"{name}{code} 最新价 涨跌幅 量比 换手率 PE PB 总市值")
    kv = _parse_kv(data)
    if not kv: return {}

    result = {
        "name": name, "code": code,
        "price": _num(kv, "最新价"), "change_pct": _num(kv, "涨跌幅"),
        "vol_ratio": _num(kv, "量比"), "turnover": _num(kv, "换手率"),
        "pe": _num(kv, "市盈率"), "pb": _num(kv, "市净率"),
        "mcap": _num(kv, "总市值"),
        "source": "mx-data",
    }
    _cache_set(key, result)
    return result


def get_financial_quality(code: str, name: str) -> dict:
    """妙想财务质量 — ROE/净利增速/营收增速/负债率/毛利率（缓存4h）"""
    key = f"fin_{code}"
    cached = _cached(key, 14400)
    if cached: return cached

    data = _mx_query(f"{name}{code} 净资产收益率 净利润同比增长率 营业收入同比增长率 资产负债率 毛利率")
    kv = _parse_kv(data)
    if not kv: return {}

    result = {
        "roe": _num(kv, "净资产收益率"),
        "profit_growth": _num(kv, "净利润同比增长"),
        "revenue_growth": _num(kv, "营业收入同比增长"),
        "debt_ratio": _num(kv, "资产负债率"),
        "gross_margin": _num(kv, "毛利率"),
    }
    _cache_set(key, result)
    return result


def get_market_brief() -> dict:
    """妙想市场概况 — 指数涨跌+成交额（缓存30min）"""
    key = "market"
    cached = _cached(key, 1800)
    if cached: return cached

    data = _mx_query("上证指数 沪深300 今日涨跌幅 成交额")
    kv = _parse_kv(data)
    if not kv: return {}

    result = {
        "sh_idx": _num(kv, "涨跌幅"), "sz_amount": _num(kv, "成交额"),
        "update_time": datetime.now().strftime("%H:%M"),
    }
    _cache_set(key, result)
    return result


def score_financial_quality(fin: dict, val: dict) -> tuple:
    """妙想财务评分 0-5分"""
    score = 0; tags = []

    roe = fin.get("roe", 0)
    if roe > 15: score += 2; tags.append(f"ROE{roe:.0f}%")
    elif roe > 5: score += 1

    pg = fin.get("profit_growth", 0)
    if pg > 50: score += 1; tags.append(f"净利+{pg:.0f}%")
    elif pg < -20: score -= 1; tags.append(f"净利{pg:.0f}%")

    dr = fin.get("debt_ratio", 100)
    if dr < 40: score += 1
    elif dr > 70: score -= 1; tags.append(f"负债{dr:.0f}%")

    pe = val.get("pe", 0)
    if 10 < pe < 40: score += 1
    elif pe > 100: score -= 1; tags.append(f"PE{pe:.0f}")

    gm = fin.get("gross_margin", 0)
    if gm > 30: score += 1

    if score >= 4: label = "优质"
    elif score >= 2: label = "良好"
    elif score >= 0: label = "一般"
    else: label = "谨慎"

    return score, f"{label}({score}分): {' '.join(tags)}" if tags else f"{label}({score}分)"


# ═══════════════════════════════════════════════
# 备用：腾讯 HTTP（仅 mx-data 不支持的 K 线序列）
# ═══════════════════════════════════════════════

def fetch_kline(code: str, freq: str = "15") -> pd.DataFrame:
    """K线 — 腾讯API（mx-data不支持K线序列，此为唯一可用源）"""
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

        if not rows: return pd.DataFrame()
        data = []
        for r in rows:
            if len(r) < 6: continue
            ts = str(r[0])
            data.append({"date": ts[:10] if freq == "day" else ts[:8],
                         "time": ts if freq != "day" else "",
                         "open": float(r[1]), "close": float(r[2]),
                         "high": float(r[3]), "low": float(r[4]),
                         "volume": float(r[5])})
        df = pd.DataFrame(data)
        for c in ["open", "close", "high", "low", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df
    except Exception as e:
        logger.error(f"K线({code},{freq}): {e}")
        return pd.DataFrame()


def _tencent_batch_quotes(codes: list) -> dict:
    """腾讯批量行情 — 仅当妙想失败时降级使用"""
    if not codes: return {}
    prefixed = [f"{_prefix(c)}{c}" for c in codes]
    url = "https://qt.gtimg.cn/q=" + ",".join(prefixed)
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=15)
        r.encoding = "gbk"
        result = {}
        for line in r.text.strip().split(";"):
            if '="' not in line: continue
            key = line.split("=")[0].split("_")[-1]
            code = key[2:]
            vals = line.split('"')[1].split("~")
            if len(vals) < 50: continue
            result[code] = {
                "name": vals[1], "code": code,
                "price": float(vals[3]) if vals[3] else 0,
                "preclose": float(vals[4]) if vals[4] else 0,
                "change_pct": float(vals[32]) if vals[32] else 0,
                "vol_ratio": float(vals[49]) if vals[49] else 0,
                "volume": float(vals[6]) if vals[6] else 0,
                "pe": float(vals[39]) if vals[39] else 0,
                "pb": float(vals[46]) if vals[46] else 0,
                "mcap_yi": float(vals[44]) if vals[44] else 0,
                "source": "tencent",
            }
        return result
    except Exception as e:
        logger.error(f"腾讯行情: {e}")
        return {}
