"""
东方财富数据适配器 — curl_cffi 绕过 TLS 指纹检测
关键发现: push2delay.eastmoney.com (CDN边缘节点) 拦截普通 Python requests
          curl_cffi 模拟 Chrome120 TLS 指纹可成功绕过

用法:
  from data.eastmoney_fetcher import get_realtime_quote, get_minute_kline, get_sector_data
"""

import logging
from datetime import datetime
from curl_cffi import requests as cffi

logger = logging.getLogger(__name__)

# ==================== 底层 HTTP ====================

def _cffi_get(url, timeout=15, retries=3):
    """curl_cffi GET — 模拟 Chrome120 TLS 指纹，自动重试"""
    last_err = None
    for attempt in range(retries):
        try:
            r = cffi.get(url, impersonate="chrome120", timeout=timeout)
            if r.status_code == 200 and len(r.content) > 0:
                return r.json()
            if r.status_code != 200:
                logger.warning(f"HTTP {r.status_code} from {url[:80]}")
                return None
        except Exception as e:
            last_err = e
            err_msg = str(e)
            # Connection closed/timeout → 重试; DNS/其他 → 不重试
            if "Connection closed" in err_msg or "timed out" in err_msg:
                if attempt < retries - 1:
                    import time
                    time.sleep(1.5)
                    continue
            break
    if last_err:
        logger.warning(f"cffi请求失败(已重试{retries}次): {str(last_err)[:100]}")
    return None


# ==================== 实时行情 ====================

def get_realtime_quote(code):
    """
    获取单只股票实时行情
    Args:
        code: 股票代码 (如 '000933' 或 '600036')
    Returns:
        dict: {price, change_pct, high, low, open, volume, amount, turnover}
    """
    market = "1" if code.startswith("6") else "0"
    secid = f"{market}.{code}"
    url = (
        f"https://push2.eastmoney.com/api/qt/stock/get"
        f"?secid={secid}"
        f"&fields=f43,f44,f45,f46,f47,f48,f50,f57,f58,f60,f170,f168,f169"
    )
    data = _cffi_get(url)
    if not data or not data.get("data"):
        return None

    d = data["data"]
    return {
        "code": code,
        "price": d.get("f43", 0) / 100 if d.get("f43") else 0,
        "change_pct": d.get("f170", 0) / 100 if d.get("f170") else 0,
        "high": d.get("f44", 0) / 100 if d.get("f44") else 0,
        "low": d.get("f45", 0) / 100 if d.get("f45") else 0,
        "open": d.get("f46", 0) / 100 if d.get("f46") else 0,
        "volume": d.get("f47", 0),  # 手
        "amount": d.get("f48", 0),  # 元
        "turnover": d.get("f168", 0) / 100 if d.get("f168") else 0,  # 换手率%
        "pe": d.get("f169", 0) / 100 if d.get("f169") else 0,  # PE
    }


def get_batch_quotes(codes):
    """
    批量获取实时行情（一次请求）
    Args:
        codes: 股票代码列表
    Returns:
        dict: {code: {price, change_pct, ...}}
    """
    secids = []
    for code in codes:
        market = "1" if code.startswith("6") else "0"
        secids.append(f"{market}.{code}")
    secid_str = ",".join(secids)
    url = (
        f"https://push2.eastmoney.com/api/qt/ulist.np/get"
        f"?secids={secid_str}"
        f"&fields=f2,f3,f4,f5,f6,f7,f15,f16,f17,f18"
    )
    data = _cffi_get(url)
    if not data or not data.get("data"):
        return {}

    result = {}
    for item in data["data"].get("diff", []):
        code = item.get("f2", "")[-6:]  # 格式: 0.000933 → 000933
        if not code:
            continue
        result[code] = {
            "code": code,
            "price": item.get("f2", 0) / 100 if item.get("f2") else 0,
            "change_pct": item.get("f3", 0) / 100 if item.get("f3") else 0,
            "volume": item.get("f5", 0),
            "amount": item.get("f6", 0),
            "high": item.get("f15", 0) / 100 if item.get("f15") else 0,
            "low": item.get("f16", 0) / 100 if item.get("f16") else 0,
            "open": item.get("f17", 0) / 100 if item.get("f17") else 0,
        }
    return result


# ==================== K线数据 ====================

def get_minute_kline(code, freq=15, count=200):
    """
    获取分钟K线（东方财富）
    Args:
        code: 股票代码
        freq: 5, 15, 30, 60
        count: 获取最近N根K线
    Returns:
        list of dict: [{datetime, open, close, high, low, volume, amount}]
    """
    market = "1" if code.startswith("6") else "0"
    secid = f"{market}.{code}"
    url = (
        f"https://push2his.eastmoney.com/api/qt/stock/kline/get"
        f"?secid={secid}"
        f"&fields1=f1,f2,f3,f4,f5,f6"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&klt={freq}&fqt=0&end=20500101&lmt={count}"
    )
    data = _cffi_get(url)
    if not data or not data.get("data") or not data["data"].get("klines"):
        return []

    result = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        if len(parts) < 7:
            continue
        result.append({
            "datetime": parts[0],
            "open": float(parts[1]),
            "close": float(parts[2]),
            "high": float(parts[3]),
            "low": float(parts[4]),
            "volume": int(parts[5]),
            "amount": float(parts[6]),
        })
    return result


def get_daily_kline(code, count=500):
    """获取日线"""
    return get_minute_kline(code, freq=101, count=count)


# ==================== 板块数据 ====================

def get_sector_list(sector_type="industry"):
    """
    获取板块列表
    Args:
        sector_type: 'industry'(行业) | 'concept'(概念)
    Returns:
        list of dict: [{code, name, change_pct}]
    """
    fs = "m:90+t2" if sector_type == "industry" else "m:90+t3"
    url = (
        f"https://push2.eastmoney.com/api/qt/clist/get"
        f"?pn=1&pz=100&po=1&np=1&fltt=2&invt=2&fid=f3&fs={fs}"
        f"&fields=f2,f3,f4,f12,f14"
    )
    data = _cffi_get(url)
    if not data or not data.get("data") or not data["data"].get("diff"):
        return []

    result = []
    for item in data["data"]["diff"]:
        result.append({
            "code": item.get("f12", ""),
            "name": item.get("f14", ""),
            "price": item.get("f2", 0) / 100 if item.get("f2") else 0,
            "change_pct": item.get("f3", 0) / 100 if item.get("f3") else 0,
        })
    return result


# ==================== 资金流向 ====================

def get_money_flow(code, days=5):
    """
    获取个股资金流向
    Returns:
        list of dict: [{date, main_net_inflow, ...}]
    """
    market = "1" if code.startswith("6") else "0"
    secid = f"{market}.{code}"
    url = (
        f"https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get"
        f"?secid={secid}"
        f"&fields1=f1,f2,f3,f7"
        f"&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61"
        f"&lmt={days}"
    )
    data = _cffi_get(url)
    if not data or not data.get("data") or not data["data"].get("klines"):
        return []

    result = []
    for line in data["data"]["klines"]:
        parts = line.split(",")
        if len(parts) < 6:
            continue
        result.append({
            "date": parts[0],
            "main_net_inflow": int(float(parts[1])),  # 主力净流入
            "super_large_net": int(float(parts[3])),   # 超大单净流入
            "large_net": int(float(parts[4])),         # 大单净流入
            "mid_net": int(float(parts[5])),           # 中单净流入
            "small_net": int(float(parts[6])),         # 小单净流入
        })
    return result


# ==================== PE / 基本面 ====================

def get_stock_basic(code):
    """获取个股基本面（PE/市值等）"""
    market = "1" if code.startswith("6") else "0"
    secid = f"{market}.{code}"
    url = (
        f"https://push2.eastmoney.com/api/qt/stock/get"
        f"?secid={secid}"
        f"&fields=f43,f57,f58,f116,f117,f162,f167,f169,f170"
    )
    data = _cffi_get(url)
    if not data or not data.get("data"):
        return None
    d = data["data"]
    return {
        "code": code,
        "pe": d.get("f169", 0) / 100 if d.get("f169") else 0,
        "total_market_cap": d.get("f116", 0),  # 总市值
        "circulating_market_cap": d.get("f117", 0),  # 流通市值
    }


# ==================== 腾讯财经备用（替代新浪） ====================

def get_tencent_realtime(codes):
    """
    腾讯财经实时行情（备用数据源，P0级可靠性）
    比新浪多 PE、换手率、市值等字段
    无频率限制，不触发反爬

    示例: https://qt.gtimg.cn/q=sz000933,sh600036
    """
    import requests

    q_codes = []
    for c in codes:
        prefix = "sh" if c.startswith("6") else "sz"
        q_codes.append(f"{prefix}{c}")

    url = f"https://qt.gtimg.cn/q={','.join(q_codes)}"
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200 or not r.text:
            return {}

        result = {}
        for line in r.text.strip().split("\n"):
            # v_sz000933="51~神火股份~000933~..."
            if "~" not in line:
                continue
            # 提取代码
            code_start = line.find("_") + 1
            code_end = line.find("=")
            if code_start <= 0 or code_end <= 0:
                continue
            raw_code = line[code_start:code_end]  # sz000933
            code = raw_code[2:]  # 000933

            # 提取数据部分
            data_start = line.find('"') + 1
            data_end = line.rfind('"')
            if data_start <= 0 or data_end <= data_start:
                continue
            fields = line[data_start:data_end].split("~")
            if len(fields) < 40:
                continue

            result[code] = {
                "code": code,
                "name": fields[1],
                "price": float(fields[3]) if fields[3] else 0,
                "prev_close": float(fields[4]) if fields[4] else 0,
                "open": float(fields[5]) if fields[5] else 0,
                "volume": int(fields[6]) if fields[6] else 0,  # 手
                "high": float(fields[33]) if fields[33] else 0,
                "low": float(fields[34]) if fields[34] else 0,
                "amount": float(fields[37]) * 10000 if fields[37] else 0,  # 万元→元
                "turnover": float(fields[38]) if fields[38] else 0,  # 换手率%
                "pe": float(fields[39]) if fields[39] else 0,  # PE
                "total_market_cap": float(fields[45]) if fields[45] else 0,  # 总市值(亿)
            }
        return result
    except Exception as e:
        logger.error(f"腾讯财经请求失败: {e}")
        return {}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 自检
    print("=== 东方财富 curl_cffi 自检 ===")

    q = get_realtime_quote("000933")
    print(f"000933 实时: {q}")

    k = get_minute_kline("000933", freq=15, count=3)
    print(f"000933 15min K线(最近3根): {k}")

    sectors = get_sector_list("industry")
    print(f"行业板块(前3): {sectors[:3]}")

    flow = get_money_flow("000933")
    print(f"000933 资金流向(最近): {flow[:1] if flow else '无数据'}")

    # 腾讯备用
    tencent = get_tencent_realtime(["000933", "600036"])
    print(f"腾讯备用: {list(tencent.keys())}")
    if "000933" in tencent:
        t = tencent["000933"]
        print(f"  神火: 价格{t['price']} PE{t['pe']} 换手率{t['turnover']}% 市值{t['total_market_cap']}亿")

    print("\n✅ 东方财富 curl_cffi 适配器就绪")
