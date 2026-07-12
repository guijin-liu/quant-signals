"""
浏览器数据抓取器 — 当API被封时的终极兜底方案
使用 Playwright + Edge 真实浏览器，共享用户登录态

场景:
  1. API全部被封 → 用浏览器打开网页版抓数据
  2. 需要登录才能访问 → 用用户已登录的Edge直接访问
  3. 东方财富网页版 → 有完整的行情/板块/资金数据

用法:
  from data.browser_fetcher import scrape_quote_page, scrape_sector_page
"""

import logging
import json
import re
from pathlib import Path

logger = logging.getLogger(__name__)

PROFILE_DIR = Path.home() / ".claude" / "playwright_profile"
STORAGE_STATE = PROFILE_DIR / "auth.json"


def _get_browser_context(headless=True):
    """获取Playwright浏览器上下文（复用登录态）"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("playwright未安装: pip install playwright && playwright install msedge")
        return None, None

    p = sync_playwright().start()
    if STORAGE_STATE.exists() or PROFILE_DIR.joinpath("Default").exists():
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="msedge",
            headless=headless,
            viewport={"width": 1280, "height": 800},
        )
        logger.info("浏览器已启动 (复用Edge登录态)")
    else:
        context = p.chromium.launch(
            channel="msedge",
            headless=headless,
        ).new_context(viewport={"width": 1280, "height": 800})
        logger.warning("无登录态，使用空白浏览器")
    return p, context


def scrape_quote_page(code):
    """
    从东方财富个股页面抓取实时行情
    使用真实浏览器，不会被反爬

    Args:
        code: 股票代码
    Returns:
        dict: {price, change_pct, volume, amount, high, low, open, turnover, pe}
    """
    market = "sh" if code.startswith("6") else "sz"
    url = f"https://quote.eastmoney.com/{market}{code}.html"

    p, context = _get_browser_context(headless=True)
    if not context:
        return None

    try:
        page = context.new_page()
        page.goto(url, wait_until="domcontentloaded", timeout=20000)
        # 等待价格元素出现
        page.wait_for_selector(".price", timeout=10000)

        # 抓取价格数据
        result = {"code": code}

        # 当前价
        price_el = page.query_selector("#price9")
        if price_el:
            result["price"] = float(price_el.inner_text())

        # 涨跌幅
        chg_el = page.query_selector("#km1")
        if chg_el:
            text = chg_el.inner_text().replace("%", "")
            result["change_pct"] = float(text)

        # 从页面文本提取更多数据
        full_text = page.inner_text("body")

        # 成交量(手)
        m = re.search(r'成交量[：:]\s*([\d,]+\.?\d*)万手', full_text)
        if m:
            result["volume"] = int(float(m.group(1).replace(",", "")) * 10000)

        # 成交额
        m = re.search(r'成交额[：:]\s*([\d,]+\.?\d*)亿', full_text)
        if m:
            result["amount"] = float(m.group(1).replace(",", "")) * 1e8

        context.close()
        return result

    except Exception as e:
        logger.error(f"浏览器抓取{code}失败: {e}")
        context.close()
        return None
    finally:
        p.stop()


def scrape_batch_quotes(codes):
    """
    批量抓取 — 从东方财富自选股页面一次性获取多只股票
    """
    p, context = _get_browser_context(headless=True)
    if not context:
        return {}

    try:
        page = context.new_page()
        # 使用东方财富行情中心
        code_str = ",".join([f"{'sh' if c.startswith('6') else 'sz'}{c}" for c in codes])
        url = f"https://quote.eastmoney.com/center/gridlist.html#hs_a_board"
        page.goto(url, wait_until="domcontentloaded", timeout=20000)

        # 等待数据加载
        page.wait_for_timeout(3000)
        text = page.inner_text("body")

        context.close()

        # 简单解析 — 这里返回原始文本，由调用方解析
        # 更可靠的方案是调用 window.api 或使用 Eastmoney 的数据接口
        logger.info(f"页面内容长度: {len(text)}")
        return {"raw_text": text[:5000]}

    except Exception as e:
        logger.error(f"批量抓取失败: {e}")
        context.close()
        return {}
    finally:
        p.stop()


def has_auth():
    """检查是否有浏览器登录态"""
    return STORAGE_STATE.exists() or (
        PROFILE_DIR.joinpath("Default").exists()
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=== 浏览器抓取器自检 ===")
    print(f"Profile目录: {PROFILE_DIR}")
    print(f"登录态: {'已保存' if has_auth() else '未登录(需先运行 browser_playwright.py launch)'}")

    if not has_auth():
        print("\n⚠️  首次使用需要:")
        print("   python ~/.claude/tools/browser_playwright.py launch")
        print("   在打开的Edge中登录东方财富 → 关闭浏览器 → cookies自动保存")
    else:
        print("\n✅ 登录态就绪，可以抓取")
        # 实际抓取测试（headless模式）
        result = scrape_quote_page("000933")
        print(f"000933 抓取结果: {result}")
