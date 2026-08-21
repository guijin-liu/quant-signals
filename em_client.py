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
        total=2, connect=1, backoff_factor=0.5,
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
    """全A非ST 按成交额(f6)降序取前n名（实测 f5=成交量手, f6=成交额元）。
    东财单页 pz 上限100，固定每页100分页 + 按 code 去重（防分页重叠）。"""
    all_rows, seen = [], set()
    pn = 1
    while len(all_rows) < n and pn <= 5:
        params = {
            "pn": str(pn), "pz": "100", "po": "1", "np": "1",
            "fltt": "2", "invt": "2", "dect": "1",
            "fid": "f6", "fs": FS_ALL_A,
            "fields": "f2,f3,f5,f6,f8,f9,f12,f14,f20,f21",
            "_": str(int(time.time() * 1000)),
        }
        headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
        try:
            r = em_get(TOP_URL, params=params, headers=headers, timeout=15)
            diff = (r.json().get("data") or {}).get("diff") or []
        except Exception as e:
            logger.error(f"top_amount p{pn}: {e}")
            break
        if not diff:
            break
        new = 0
        for it in diff:
            code = it.get("f12", "")
            if not code or code in seen:
                continue
            seen.add(code)
            all_rows.append({
                "code": code, "name": it.get("f14", ""),
                "amount": it.get("f6", 0), "volume": it.get("f5", 0),
                "price": it.get("f2", 0), "chg": it.get("f3", 0),
                "turnover": it.get("f8", 0), "pe": it.get("f9", 0),
                "mcap": it.get("f20", 0),
            })
            new += 1
        if new == 0:  # 本页全重复（分页未生效）→ 停止避免死循环
            break
        pn += 1
    if not all_rows:
        logger.warning("东财 top_amount 全失败，降级新浪排行兜底")
        return em_fetch_top_amount_sina(n)
    if len(all_rows) < n:
        logger.warning(f"东财仅拉到{len(all_rows)}/{n}，新浪补足缺口")
        sina = em_fetch_top_amount_sina(n)
        merged = {r["code"]: r for r in all_rows}
        for r in sina:
            merged.setdefault(r["code"], r)
        rows = sorted(merged.values(), key=lambda r: r.get("amount", 0), reverse=True)
        logger.info(f"补足后: {len(rows)}/{n}")
        return rows[:n]
    return all_rows[:n]


def em_fetch_top_amount_sina(n=150) -> list:
    """新浪成交额排行兜底（东财 clist 502/风控时降级）。
    Market_Center.getHQNodeData: sort=amount 按成交额降序，每页100，取前n。
    返回结构与 em_fetch_top_amount 一致 [{code,name,amount,volume,price,chg,turnover,pe,mcap}]"""
    url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    all_rows, seen = [], set()
    page = 1
    while len(all_rows) < n and page <= 3:
        params = {"page": str(page), "num": "100", "sort": "amount", "asc": "0", "node": "hs_a"}
        try:
            r = requests.get(url, params=params, timeout=12,
                             headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"})
            diff = r.json()
        except Exception as e:
            logger.error(f"top_amount_sina p{page}: {e}")
            break
        if not diff:
            break
        new = 0
        for it in diff:
            sym = str(it.get("symbol", ""))
            code = sym[-6:] if sym else ""
            if not code or code in seen:
                continue
            seen.add(code)
            all_rows.append({
                "code": code, "name": it.get("name", ""),
                "amount": it.get("amount", 0), "volume": it.get("volume", 0),
                "price": it.get("trade", 0), "chg": it.get("changepercent", 0),
                "turnover": it.get("turnoverratio", 0), "pe": it.get("per", 0),
                "mcap": it.get("mktcap", 0),
            })
            new += 1
        if new == 0:
            break
        page += 1
    logger.info(f"新浪排行兜底: {len(all_rows)}只")
    return all_rows[:n]


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


# ═══════════════════════════════════════════════
# 5. 东财人气榜（3号热点池数据源）
# ═══════════════════════════════════════════════

HOT_RANK_URL = "https://emappdata.eastmoney.com/stockrank/getAllCurrentList"


def em_fetch_hot_rank(n=50) -> list:
    """东财人气榜（股吧热度）前n → [{code,name,rank}]。
    关键：body不带sortType字段 + Origin header，否则返回code=-2。
    失败重试3次，仍失败返回[]（调用方降级换手率榜）。"""
    headers = {
        "Content-Type": "application/json",
        "User-Agent": UA,
        "Referer": "https://data.eastmoney.com/xuangu/",
        "Accept": "application/json",
        "Origin": "https://data.eastmoney.com",
    }
    body = {"appId": "appId01", "globalId": "786e4c21-70dc-435a-93bb-38",
            "marketType": "沪A", "pageNo": 1, "pageSize": 100}
    for attempt in range(3):
        try:
            wait = EM_MIN_INTERVAL - (time.time() - _em_last_call[0])
            if wait > 0:
                time.sleep(wait + random.uniform(0.1, 0.5))
            r = requests.post(HOT_RANK_URL, json=body, headers=headers, timeout=15)
            d = r.json()
            data = d.get("data") or []
            if d.get("code") == 0 and isinstance(data, list) and data:
                codes = [x.get("sc", "") for x in data]
                names = em_fetch_names_by_codes(codes)
                rows = [{"code": c[2:] if len(c) > 2 else c,
                         "name": names.get(c[2:], c[2:]), "rank": i + 1}
                        for i, c in enumerate(codes) if c]
                logger.info(f"东财人气榜: {len(rows)}只")
                return rows[:n]
            logger.warning(f"人气榜 attempt{attempt+1} code={d.get('code')}, retry")
        except Exception as e:
            logger.warning(f"人气榜 attempt{attempt+1} 异常: {e}")
        time.sleep(2 * (attempt + 1))
    return []


def em_fetch_hot_rank_fallback(n=50) -> list:
    """人气榜备用：换手率榜前n（热度代理）。东财f8 → 新浪turnoverratio 双源降级。"""
    rows = em_fetch_top_amount_turnover(n, src="em")
    if rows:
        return rows
    return em_fetch_top_amount_turnover(n, src="sina")


def em_fetch_top_amount_turnover(n=50, src="em") -> list:
    """换手率前n → [{code,name,amount,turnover}]。src=em(东财f8) / sina(新浪turnoverratio)"""
    if src == "em":
        out, seen, pn = [], set(), 1
        while len(out) < n and pn <= 2:
            params = {"pn": str(pn), "pz": "100", "po": "1", "np": "1", "fltt": "2", "invt": "2",
                      "dect": "1", "fid": "f8", "fs": FS_ALL_A,
                      "fields": "f2,f3,f6,f8,f12,f14", "_": str(int(time.time() * 1000))}
            headers = {"User-Agent": UA, "Referer": "https://quote.eastmoney.com/"}
            try:
                r = em_get(TOP_URL, params=params, headers=headers, timeout=15)
                diff = (r.json().get("data") or {}).get("diff") or []
            except Exception as e:
                logger.error(f"换手率榜em p{pn}: {e}")
                break
            if not diff:
                break
            new = 0
            for it in diff:
                code = it.get("f12", "")
                if not code or code in seen:
                    continue
                seen.add(code)
                out.append({"code": code, "name": it.get("f14", ""),
                            "amount": it.get("f6", 0), "turnover": it.get("f8", 0)})
                new += 1
            if new == 0:
                break
            pn += 1
        logger.info(f"换手率榜(东财f8): {len(out)}只")
        return out[:n]
    url = "https://vip.stock.finance.sina.com.cn/quotes_service/api/json_v2.php/Market_Center.getHQNodeData"
    out, seen, page = [], set(), 1
    while len(out) < n and page <= 2:
        params = {"page": str(page), "num": "100", "sort": "turnoverratio", "asc": "0", "node": "hs_a"}
        try:
            r = requests.get(url, params=params, headers={"User-Agent": UA,
                "Referer": "https://finance.sina.com.cn/"}, timeout=12)
            diff = r.json()
        except Exception as e:
            logger.error(f"换手率榜sina p{page}: {e}")
            break
        if not diff:
            break
        new = 0
        for it in diff:
            sym = str(it.get("symbol", ""))
            code = sym[-6:] if sym else ""
            if not code or code in seen:
                continue
            seen.add(code)
            out.append({"code": code, "name": it.get("name", ""),
                        "amount": it.get("amount", 0), "turnover": it.get("turnoverratio", 0)})
            new += 1
        if new == 0:
            break
        page += 1
    logger.info(f"换手率榜(新浪): {len(out)}只")
    return out[:n]


def em_fetch_names_by_codes(codes) -> dict:
    """新浪批量行情补名称：list=sh688836,sz002491 → {code: name}"""
    if not codes:
        return {}
    syms = [c.lower() for c in codes if c]
    url = "https://hq.sinajs.cn/list=" + ",".join(syms)
    try:
        r = requests.get(url, headers={"User-Agent": UA, "Referer": "https://finance.sina.com.cn/"}, timeout=10)
        r.encoding = "gbk"
        names = {}
        import re
        for line in r.text.strip().split("\n"):
            m = re.match(r'var hq_str_(\w+)="([^"]*)"', line.strip())
            if m:
                sym, vals = m.groups()
                nm = vals.split(",")[0] if vals else ""
                if nm:
                    names[sym[2:]] = nm
        return names
    except Exception as e:
        logger.error(f"新浪批量名称失败: {e}")
        return {}
