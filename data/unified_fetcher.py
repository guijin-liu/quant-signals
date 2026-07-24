"""
统一数据源 — 多层降级，确保数据永远可用

数据源优先级:
  Tier 1: mootdx (K线主力，通达信直连，不封IP)
  Tier 2: 东方财富 curl_cffi (实时行情+资金流，模拟Chrome TLS)
  Tier 3: 腾讯财经 (实时行情+PE+市值，免费无限制)
  Tier 4: 腾讯财经 (PE+基本面)
  Tier 5: 浏览器 (Playwright+Edge，终极兜底)

cloud_function.py 接入方式:
  from data.unified_fetcher import fetch_minute_kline
  df = fetch_minute_kline(code, freq='15', days=180)
"""

import sys
from pathlib import Path
# 确保 quant_trading 根目录在 sys.path 中
_PARENT = Path(__file__).parent.parent
if str(_PARENT) not in sys.path:
    sys.path.insert(0, str(_PARENT))

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


def fetch_minute_kline(code, freq='15', days=180):
    """
    获取分钟K线 — 多层降级
    数据源优先级:
      Tier 1: mootdx (通达信直连)
      Tier 2: 腾讯财经 (ifzq.gtimg.cn — 稳定免封)
      Tier 3: 东方财富 curl_cffi (备用)

    Args:
        code: 股票代码
        freq: '5' 或 '15'
        days: 回看天数

    Returns:
        DataFrame with columns: date, time, open, high, low, close, volume
        与 cloud_function.py 的 compute_features() 完全兼容
    """
    # Tier 1: mootdx
    df = _try_mootdx(code, freq, days)
    if df is not None and not df.empty:
        logger.debug(f"{code} mootdx ✅ {len(df)}条")
        return df

    # Tier 2: 腾讯财经 (目前唯一稳定)
    df = _try_tencent_kline(code, freq, days)
    if df is not None and not df.empty:
        logger.info(f"{code} 腾讯财经 ✅ (mootdx降级) {len(df)}条")
        return df

    # Tier 3: 东方财富 curl_cffi
    df = _try_eastmoney_kline(code, freq, days)
    if df is not None and not df.empty:
        logger.info(f"{code} 东方财富 ✅ (mootdx降级) {len(df)}条")
        return df

    logger.error(f"{code} ❌ 所有数据源都失败了!")
    return pd.DataFrame()


def fetch_realtime_quote(code):
    """
    获取实时行情 — 多层降级

    Returns:
        dict: {price, change_pct, volume, amount, high, low, open, pe, turnover}
        或 None
    """
    # Tier 1: 东方财富 curl_cffi
    try:
        from data.eastmoney_fetcher import get_realtime_quote
        q = get_realtime_quote(code)
        if q and q.get("price", 0) > 0:
            return q
    except Exception as e:
        logger.warning(f"东方财富实时行情失败: {e}")

    # Tier 2: 腾讯财经
    try:
        from data.eastmoney_fetcher import get_tencent_realtime
        results = get_tencent_realtime([code])
        if code in results:
            t = results[code]
            return {
                "code": code,
                "price": t["price"],
                "change_pct": round((t["price"] / t["prev_close"] - 1) * 100, 2) if t.get("prev_close", 0) > 0 else 0,
                "high": t["high"],
                "low": t["low"],
                "open": t["open"],
                "volume": t["volume"],
                "amount": t["amount"],
                "pe": t.get("pe", 0),
                "turnover": t.get("turnover", 0),
            }
    except Exception as e:
        logger.warning(f"腾讯财经实时行情失败: {e}")

    logger.error(f"{code} 实时行情所有源都失败了!")
    return None


def fetch_batch_quotes(codes):
    """
    批量获取实时行情
    优先东方财富批量接口，降级到新浪逐个
    """
    try:
        from data.eastmoney_fetcher import get_batch_quotes
        results = get_batch_quotes(codes)
        if results:
            return results
    except Exception:
        pass

    # 降级: 腾讯逐个获取
    try:
        from data.eastmoney_fetcher import get_tencent_realtime
        return get_tencent_realtime(codes)
    except Exception:
        pass

    return {}


def fetch_money_flow(code, days=5):
    """获取资金流向"""
    try:
        from data.eastmoney_fetcher import get_money_flow
        return get_money_flow(code, days)
    except Exception as e:
        logger.warning(f"资金流向获取失败: {e}")
    return []


def fetch_sector_list(sector_type="industry"):
    """获取板块列表"""
    try:
        from data.eastmoney_fetcher import get_sector_list
        return get_sector_list(sector_type)
    except Exception as e:
        logger.warning(f"板块数据获取失败: {e}")
    return []


# ==================== 内部实现 ====================

def _try_mootdx(code, freq, days):
    """Tier 1: mootdx 通达信"""
    try:
        from data.tdx_fetcher import fetch_minute_data_cloud
        df = fetch_minute_data_cloud(code, freq=freq, days=days)
        if df is not None and not df.empty and len(df) >= 20:
            return df
    except Exception as e:
        logger.warning(f"mootdx {code} 失败: {str(e)[:80]}")
    return None


def _try_tencent_kline(code, freq, days):
    """Tier 2: 腾讯财经 ifzq.gtimg.cn — 分钟K线"""
    import requests as _req

    try:
        prefix = 'sh' if code.startswith(('6', '9')) else 'sz'
        freq_key = f'm{freq}'  # m5 or m15
        # 腾讯单次最多约320根，按需获取
        bars_per_day = 48 if freq == '5' else 16
        count = min(days * bars_per_day, 320)

        url = f'http://ifzq.gtimg.cn/appstock/app/kline/mkline?param={prefix}{code},{freq_key},,{count}'
        r = _req.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=15)
        data = r.json()

        tn_key = f'{prefix}{code}'
        raw = data.get('data', {}).get(tn_key, {}).get(freq_key, [])
        if not raw or len(raw) < 20:
            return None

        rows = []
        for bar in raw:
            dt_str = str(bar[0])
            d = dt_str[:8]
            t = dt_str[8:]
            rows.append({
                'date': f'{d[:4]}-{d[4:6]}-{d[6:8]}',
                'time': t,
                'open': float(bar[1]),
                'close': float(bar[2]),
                'high': float(bar[3]),
                'low': float(bar[4]),
                'volume': int(float(bar[5])),
            })

        df = pd.DataFrame(rows)
        # 数值类型
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df['date'] = df['date'].astype(str)
        df['time'] = df['time'].astype(str)

        return df

    except Exception as e:
        logger.warning(f"腾讯K线 {code} 失败: {str(e)[:80]}")
    return None


def _try_eastmoney_kline(code, freq, days):
    """Tier 2: 东方财富 curl_cffi"""
    try:
        from data.eastmoney_fetcher import get_minute_kline

        freq_int = int(freq)
        bars_per_day = 48 if freq_int == 5 else 16
        count = min(days * bars_per_day, 2000)  # 东方财富单次最多约2000根

        klines = get_minute_kline(code, freq=freq_int, count=count)
        if not klines or len(klines) < 20:
            return None

        # 转换为 cloud_function 期望的格式
        rows = []
        for k in klines:
            dt_str = k.get("datetime", "")
            if " " in dt_str:
                d, t = dt_str.split(" ")
                t = t.replace(":", "")[:6]
            else:
                d, t = dt_str[:10], "000000"

            rows.append({
                "date": d,
                "time": t,
                "open": float(k.get("open", 0)),
                "high": float(k.get("high", 0)),
                "low": float(k.get("low", 0)),
                "close": float(k.get("close", 0)),
                "volume": int(k.get("volume", 0)),
            })

        df = pd.DataFrame(rows)
        # 确保数值类型
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df["date"] = df["date"].astype(str)
        df["time"] = df["time"].astype(str)

        return df

    except Exception as e:
        logger.warning(f"东方财富K线 {code} 失败: {str(e)[:80]}")
    return None


# ==================== 自检 ====================

def health_check(codes=None):
    """
    数据源健康检查

    Returns:
        dict: {source_name: status}
    """
    if codes is None:
        codes = ["000933"]

    results = {}

    # mootdx
    try:
        df = _try_mootdx("000933", "15", 30)
        results["mootdx"] = "✅" if df is not None and not df.empty else "❌"
    except:
        results["mootdx"] = "❌"

    # 腾讯K线
    try:
        df = _try_tencent_kline("000933", "15", 30)
        results["tencent_kline"] = "✅" if df is not None and not df.empty else "❌"
    except:
        results["tencent_kline"] = "❌"

    # 东方财富
    try:
        from data.eastmoney_fetcher import get_realtime_quote
        q = get_realtime_quote("000933")
        results["eastmoney_curl_cffi"] = "✅" if q and q.get("price", 0) > 0 else "❌"
    except:
        results["eastmoney_curl_cffi"] = "❌"

    # 腾讯行情
    try:
        from data.eastmoney_fetcher import get_tencent_realtime
        tencent = get_tencent_realtime(["000933"])
        results["tencent"] = "✅" if "000933" in tencent else "❌"
    except:
        results["tencent"] = "❌"

    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    print("=== 统一数据源自检 ===\n")

    # 健康检查
    print("数据源状态:")
    for name, status in health_check().items():
        print(f"  {status} {name}")

    # K线获取测试
    print("\nK线获取测试 (000933 15分钟 最近30天):")
    df = fetch_minute_kline("000933", "15", 30)
    if not df.empty:
        print(f"  ✅ {len(df)}条K线")
        print(f"  日期范围: {df['date'].iloc[0]} ~ {df['date'].iloc[-1]}")
        print(f"  最近5条:")
        print(f"  {df.tail(5)[['date','time','open','close','volume']].to_string()}")
    else:
        print("  ❌ 获取失败")

    # 实时行情
    print("\n实时行情测试:")
    q = fetch_realtime_quote("000933")
    if q:
        print(f"  ✅ 000933 神火股份: {q['price']} ({q['change_pct']:+.2f}%)")
    else:
        print("  ❌ 获取失败")
