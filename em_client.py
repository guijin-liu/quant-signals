"""刘圭金2号 — 东财统一数据客户端（数据地基）

全系统数据源：全市场成交额排行 / 历史15分钟K线 / 主力资金流(120日+分钟级)。
统一走 em_get() 串行限流防封（1s+随机抖动，UA/Referer，自动重试）。

注意：部分大陆住宅IP连东财被拒(HTTP 000)，动态池/回溯需在 GitHub Actions 云端跑。
"""
import time, random, os, logging
import requests
import pandas as pd

logger = logging.getLogger(__name__)

UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36"
EM_SESSION = requests.Session()
EM_SESSION.headers.update({"User-Agent": UA})

try:
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    _adapter = HTTPAdapter(max_retries=Retry(
        total=3, connect=3, backoff_factor=0.6,
        status_forcelist=[429, 500, 502, 503, 504], allowed_methods=["GET"]))
    EM_SESSION.mount("https://", _adapter)
    EM_SESSION.mount("http://", _adapter)
except Exception:
    pass

EM_MIN_INTERVAL = float(os.environ.get("EM_MIN_INTERVAL", "1.0"))
_em_last_call = [0.0]


def em_get(url, params=None, headers=None, timeout=15, **kwargs):
    """东财统一请求：串行限流(1s+随机抖动) + session复用 + 默认UA"""
    wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
    if wait > 0:
        time.sleep(wait + random.uniform(0.1, 0.5))
    try:
        return EM_SESSION.get(url, params=params, headers=headers, timeout=timeout, **kwargs)
    finally:
        _em_last_call[0] = time.time()


def _secid(code: str) -> str:
    """股票代码 → 东财 secid（沪6/9→1，深→0）"""
    return f"1.{code}" if code.startswith(("6", "9")) else f"0.{code}"


# ═══════════════════════════════════════════════
# 1. 全市场成交额排行（动态池数据源）
# ═══════════════════════════════════════════════

TOP_URL = "https://push2.eastmoney.com/api/qt/clist/get"
FS_ALL_A = "m:0+t:6,m:0+t:80,m:0+t:81,m:1+t:2,m:1+t:23,m:1+t:3,f:8"  # 全A主板+创业板+科创板, 排除ST


def em_fetch_top_amount(n=150) -> list:
    """全A非ST 按成交额(f5)降序取前n名。
    返回 [{code,name,amount,volume,price,chg,turnover,pe,mcap}, ...]"""
    params = {
        "pn": "1", "pz": str(n), "po": "1", "np": "1",
        "fltt": "2", "invt": "2", "dect": "1",
        "fid": "f5", "fs": FS_ALL_A,
        "fields": "f2,f3,f5,f6,f8,f9,f12,f14,f20,f21",
        "_": str(int(time.time() * 1000)),
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(TOP_URL, params=params, headers=headers, timeout=15)
        diff = (r.json().get("data") or {}).get("diff") or []
        rows = []
        for it in diff:
            rows.append({
                "code": it.get("f12", ""), "name": it.get("f14", ""),
                "amount": it.get("f5", 0), "volume": it.get("f6", 0),
                "price": it.get("f2", 0), "chg": it.get("f3", 0),
                "turnover": it.get("f8", 0), "pe": it.get("f9", 0),
                "mcap": it.get("f20", 0),
            })
        return rows
    except Exception as e:
        logger.error(f"top_amount: {e}")
        return []


# ═══════════════════════════════════════════════
# 2. 历史15分钟K线（半年回溯主数据源）
# ═══════════════════════════════════════════════

KLINE15_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def em_fetch_kline_15m(code: str, start="20260101", end="20300101") -> pd.DataFrame:
    """东财前复权15分钟K线（一条请求≈近1年）。返回 DataFrame[date,open,close,high,low,volume,amount]"""
    params = {
        "secid": _secid(code), "klt": "15", "fqt": "1",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "beg": start, "end": end,
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
    try:
        r = em_get(KLINE15_URL, params=params, headers=headers, timeout=20)
        klines = (r.json().get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            p = line.split(",")
            if len(p) < 7:
                continue
            rows.append({
                "date": p[0],
                "open": float(p[1]), "close": float(p[2]),
                "high": float(p[3]), "low": float(p[4]),
                "volume": float(p[5]), "amount": float(p[6]),
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            return df
        raise ValueError("eastmoney empty kline")
    except Exception as e:
        logger.warning(f"kline15({code}) 东财失败: {e} → 腾讯兜底")
        return _tencent_kline_fallback(code)


def _tencent_kline_fallback(code):
    """腾讯15分钟K线兜底（mx_fetcher，最多640根≈40交易日；实盘扫描够用，回溯窗口缩短）"""
    try:
        import mx_fetcher
        df = mx_fetcher.fetch_kline(code, "15")
        if df.empty:
            return pd.DataFrame()
        out = df[["date", "open", "close", "high", "low", "volume"]].copy()
        out["amount"] = 0.0
        return out
    except Exception as e:
        logger.error(f"tencent_kline_fallback({code}): {e}")
        return pd.DataFrame()


# ═══════════════════════════════════════════════
# 3. 主力资金流向
# ═══════════════════════════════════════════════

FFLOW_DAY_URL = "https://push2his.eastmoney.com/api/qt/stock/fflow/daykline/get"
FFLOW_MIN_URL = "https://push2.eastmoney.com/api/qt/stock/fflow/kline/get"


def em_fund_flow_120d(code: str) -> list:
    """日级主力资金流120个交易日。返回 [{date,main_net,small_net,mid_net,large_net,super_net}] 单位元"""
    params = {
        "secid": _secid(code), "lmt": "120",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/", "Origin": "https://quote.eastmoney.com"}
    try:
        r = em_get(FFLOW_DAY_URL, params=params, headers=headers, timeout=15)
        klines = (r.json().get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            p = line.split(",")
            if len(p) >= 6:
                rows.append({
                    "date": p[0],
                    "main_net": float(p[1]) if p[1] != "-" else 0,
                    "small_net": float(p[2]) if p[2] != "-" else 0,
                    "mid_net": float(p[3]) if p[3] != "-" else 0,
                    "large_net": float(p[4]) if p[4] != "-" else 0,
                    "super_net": float(p[5]) if p[5] != "-" else 0,
                })
        return rows
    except Exception as e:
        logger.error(f"fund_120d({code}): {e}")
        return []


def em_fund_flow_minute(code: str) -> list:
    """分钟级主力资金流（当日盘中）。返回 [{time,main_net,small_net,mid_net,large_net,super_net}] 单位元"""
    params = {
        "secid": _secid(code), "klt": "1",
        "fields1": "f1,f2,f3,f7",
        "fields2": "f51,f52,f53,f54,f55,f56,f57",
    }
    headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/", "Origin": "https://quote.eastmoney.com"}
    try:
        r = em_get(FFLOW_MIN_URL, params=params, headers=headers, timeout=10)
        klines = (r.json().get("data") or {}).get("klines") or []
        rows = []
        for line in klines:
            p = line.split(",")
            if len(p) >= 6:
                rows.append({
                    "time": p[0],
                    "main_net": float(p[1]), "small_net": float(p[2]),
                    "mid_net": float(p[3]), "large_net": float(p[4]),
                    "super_net": float(p[5]),
                })
        return rows
    except Exception as e:
        logger.error(f"fund_minute({code}): {e}")
        return []


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        code = sys.argv[2] if len(sys.argv) > 2 else "000630"
        top = em_fetch_top_amount(5)
        print("TOP5:", [(s["code"], s["name"], s["amount"]) for s in top])
        k = em_fetch_kline_15m(code, "20260701", "20260807")
        print(f"15minK线: {len(k)}根", k.iloc[-1]["date"] if len(k) else "")
        f = em_fund_flow_120d(code)
        print(f"资金流120日: {len(f)}天", (f[-1]["date"], f[-1]["main_net"]) if f else "")
