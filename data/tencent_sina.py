"""
腾讯/新浪行情数据适配器 — 替代akshare东方财富API
腾讯: qt.gtimg.cn — 实时行情、板块
新浪: hq.sinajs.cn — 实时行情(备用)
完全绕过东方财富TLS指纹拦截
"""
import logging
import pandas as pd
import requests
from datetime import datetime
from config import STOCK_CODES, STOCK_POOL

logger = logging.getLogger(__name__)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'Referer': 'https://finance.sina.com.cn/',
}


def get_realtime_quotes(codes: list = None) -> dict:
    """腾讯实时行情 — 股票+指数"""
    if codes is None:
        codes = STOCK_CODES

    # 构建代码: sh000001, sz000933...
    qt_codes = []
    for c in ['sh000001', 'sz399001', 'sz399006']:
        qt_codes.append(c)
    for c in codes:
        prefix = 'sh' if c.startswith(('6','9')) else 'sz'
        qt_codes.append(f'{prefix}{c}')

    url = f'https://qt.gtimg.cn/q={",".join(qt_codes)}'
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        r.encoding = 'gbk'
    except Exception as e:
        logger.error(f'腾讯行情失败: {e}')
        return {}

    result = {}
    for line in r.text.strip().split('\n'):
        if not line.strip() or '=' not in line:
            continue
        parts = line.split('~')
        if len(parts) < 10:
            continue
        # 格式: v_sz000933="51~神火股份~000933~23.74~24.20~..."
        code = parts[2]
        try:
            result[code] = {
                'name': parts[1],
                'price': float(parts[3]),
                'open': float(parts[5]) if parts[5] else None,
                'high': float(parts[33]) if len(parts) > 33 and parts[33] else None,
                'low': float(parts[34]) if len(parts) > 34 and parts[34] else None,
                'volume': int(parts[6]) if parts[6] else 0,
                'amount': float(parts[37]) if len(parts) > 37 and parts[37] else 0,
                'pct_change': float(parts[32]) if len(parts) > 32 and parts[32] else 0,
                'time': parts[30] if len(parts) > 30 else '',
            }
        except (ValueError, IndexError):
            continue
    return result


def get_index_realtime() -> pd.DataFrame:
    """获取大盘指数实时数据"""
    quotes = get_realtime_quotes([])
    rows = []
    for code, q in quotes.items():
        if code.startswith('000') or code.startswith('399'):
            rows.append({'code': code, **q})
    return pd.DataFrame(rows)


def get_stock_realtime() -> pd.DataFrame:
    """获取股票池实时行情"""
    quotes = get_realtime_quotes(STOCK_CODES)
    rows = []
    for code in STOCK_CODES:
        if code in quotes:
            rows.append({'code': code, **quotes[code]})
    return pd.DataFrame(rows)


def get_sina_realtime(codes: list = None) -> dict:
    """新浪行情(备用)"""
    if codes is None:
        codes = STOCK_CODES

    sina_codes = []
    for c in codes:
        prefix = 'sh' if c.startswith(('6','9')) else 'sz'
        sina_codes.append(f'{prefix}{c}')

    url = f'https://hq.sinajs.cn/list={",".join(sina_codes)}'
    try:
        r = requests.get(url, headers={
            **HEADERS,
            'Referer': 'https://finance.sina.com.cn/',
        }, timeout=10)
        r.encoding = 'gbk'
    except Exception as e:
        logger.error(f'新浪行情失败: {e}')
        return {}

    result = {}
    for line in r.text.strip().split('\n'):
        if not line.strip() or '=' not in line:
            continue
        # var hq_str_sz000933="神火股份,24.20,24.55,23.74,..."
        code_part = line.split('=')[0].replace('var hq_str_', '').strip()
        code = code_part[2:]  # sz000933 → 000933
        data = line.split('"')[1] if '"' in line else ''
        if not data:
            continue
        fields = data.split(',')
        if len(fields) < 5:
            continue
        try:
            result[code] = {
                'name': fields[0],
                'open': float(fields[1]) if fields[1] else None,
                'close_yesterday': float(fields[2]) if fields[2] else None,
                'price': float(fields[3]) if fields[3] else None,
                'high': float(fields[4]) if fields[4] else None,
                'low': float(fields[5]) if len(fields) > 5 and fields[5] else None,
                'volume': int(fields[8]) if len(fields) > 8 and fields[8] else 0,
                'amount': float(fields[9]) if len(fields) > 9 and fields[9] else 0,
                'date': fields[30] if len(fields) > 30 else '',
                'time': fields[31] if len(fields) > 31 else '',
            }
        except (ValueError, IndexError):
            continue
    return result


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    print('=== 腾讯行情 ===')
    quotes = get_realtime_quotes()
    for code, q in quotes.items():
        print(f'  {code} {q["name"]}: ¥{q["price"]} ({q["pct_change"]:+.2f}%)')

    print()
    print('=== 新浪行情 ===')
    sq = get_sina_realtime()
    for code, q in sq.items():
        print(f'  {code} {q["name"]}: ¥{q["price"]}')
