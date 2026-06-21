"""
数据获取统一调度层
- 统一缓存管理 (Parquet格式)
- 重试机制
- 各数据源协调
- IPv4强制补丁 (绕过Array VPN IPv6 TAP劫持)
"""

import logging
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from config import CACHE_DIR, CACHE_TTL, STOCK_CODES, STOCK_POOL, TIMEFRAMES

logger = logging.getLogger(__name__)

# ==================== IPv4强制补丁：绕过Array VPN TAP驱动IPv6劫持 ====================
import socket as _socket
import urllib3.util.connection as _urllib3_conn

_ORIG_GETADDRINFO = _socket.getaddrinfo
_ORIG_CREATE_CONNECTION = _urllib3_conn.create_connection


def _ipv4_getaddrinfo(host, port, family=0, type=0, proto=0, flags=0):
    """强制只用IPv4地址解析，绕过Array TAP虚拟网卡IPv6劫持"""
    return _ORIG_GETADDRINFO(host, port, _socket.AF_INET, type, proto, flags)


def _ipv4_create_connection(address, *args, **kwargs):
    """强制只建IPv4连接"""
    host, port = address
    addrs = _ORIG_GETADDRINFO(host, port, _socket.AF_INET, _socket.SOCK_STREAM)
    if addrs:
        return _ORIG_CREATE_CONNECTION((addrs[0][4][0], port), *args, **kwargs)
    return _ORIG_CREATE_CONNECTION(address, *args, **kwargs)


_socket.getaddrinfo = _ipv4_getaddrinfo
_urllib3_conn.create_connection = _ipv4_create_connection
logger.info("IPv4强制补丁已激活 (绕过Array VPN TAP IPv6劫持)")


# ==================== 补丁: 模拟浏览器请求头(绕过东方财富反爬) ====================
import requests as _requests

_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
}

_original_session_init = _requests.Session.__init__


def _patched_session_init(self, *args, **kwargs):
    _original_session_init(self, *args, **kwargs)
    self.headers.update(_BROWSER_HEADERS)


_requests.Session.__init__ = _patched_session_init
_requests.Session().headers.update(_BROWSER_HEADERS)
logger.info("已应用浏览器请求头补丁")


class DataCache:
    """Parquet文件缓存管理器"""

    def __init__(self, cache_dir: Path = CACHE_DIR):
        self.cache_dir = cache_dir

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.parquet"

    def get(self, key: str, ttl_seconds: int = 300) -> pd.DataFrame | None:
        """读取缓存，过期返回None"""
        path = self._cache_path(key)
        if not path.exists():
            return None
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        if (datetime.now() - mtime).total_seconds() > ttl_seconds:
            logger.debug(f"缓存过期: {key}")
            return None
        try:
            df = pd.read_parquet(path)
            logger.debug(f"缓存命中: {key}, shape={df.shape}")
            return df
        except Exception as e:
            logger.warning(f"缓存读取失败: {key}, {e}")
            return None

    def set(self, key: str, df: pd.DataFrame) -> None:
        """写入缓存"""
        path = self._cache_path(key)
        df.to_parquet(path, index=False)
        logger.debug(f"缓存写入: {key}, shape={df.shape}")

    def invalidate(self, key: str) -> None:
        """删除缓存"""
        path = self._cache_path(key)
        if path.exists():
            path.unlink()

    def clear_all(self) -> None:
        """清空所有缓存"""
        for f in self.cache_dir.glob("*.parquet"):
            f.unlink()
        logger.info("所有缓存已清空")


# 全局缓存实例
cache = DataCache()


def retry_on_fail(func, max_retries: int = 3, delay: float = 2.0):
    """重试装饰器"""
    import time

    def wrapper(*args, **kwargs):
        last_exc = None
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exc = e
                logger.warning(f"{func.__name__} 第{attempt+1}次失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay * (attempt + 1))
        raise last_exc

    return wrapper


def fetch_with_cache(key: str, ttl: int, fetch_fn, *args, **kwargs) -> pd.DataFrame:
    """带缓存的通用数据获取"""
    df = cache.get(key, ttl)
    if df is not None:
        return df
    df = fetch_fn(*args, **kwargs)
    if df is not None and not df.empty:
        cache.set(key, df)
    return df
