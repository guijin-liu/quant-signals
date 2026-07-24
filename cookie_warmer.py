"""
Cookie 预热 + 请求节奏拟人化 — 反爬增强测试

流程:
  1. Playwright 启动 Edge（你的真实浏览器指纹）
  2. 自动浏览东方财富（模拟真人行为）→ 产生 Cookie
  3. 提取 Cookie → 喂给 curl_cffi
  4. 对比：无Cookie vs 有Cookie 的成功率

用法:
  python cookie_warmer.py          # 完整流程
  python cookie_warmer.py --test   # 仅测试已有Cookie效果
"""
import json
import sys
import time
import random
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cookie_warmer")

PROFILE_DIR = Path.home() / ".claude" / "playwright_profile"
COOKIE_FILE = PROFILE_DIR / "eastmoney_cookies.json"

# ==================== Cookie 预热 ====================

def warm_cookies():
    """用 Playwright+Edge 浏览东方财富，产生真实Cookie"""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        logger.error("playwright 未安装: pip install playwright && playwright install msedge")
        return None, None

    PROFILE_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("启动 Edge 浏览器 (可见模式，可以看到浏览器在自动操作)...")
    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            user_data_dir=str(PROFILE_DIR),
            channel="msedge",
            headless=False,  # 可见模式，让你看到
            viewport={"width": 1280, "height": 800},
        )
        page = context.new_page()
        cookies_before = len(context.cookies())

        # === 模拟真人浏览行为 ===
        pages_to_visit = [
            ("https://quote.eastmoney.com/sz000933.html", "神火股份"),
            ("https://quote.eastmoney.com/sz000960.html", "锡业股份"),
            ("https://data.eastmoney.com/zjlx/", "资金流向"),
        ]

        for i, (url, desc) in enumerate(pages_to_visit):
            logger.info(f"  [{i+1}/{len(pages_to_visit)}] 浏览: {desc}")
            try:
                # domcontentloaded = HTML解析完即可，不等图片/广告
                page.goto(url, wait_until="domcontentloaded", timeout=30000)
            except Exception as e:
                logger.warning(f"        页面加载超时，尝试继续: {str(e)[:50]}")
                try:
                    page.goto(url, timeout=30000)  # 不指定wait_until，默认load但更宽容
                except:
                    logger.warning(f"        跳过: {desc}")
                    continue

            # 模拟人：滚动页面（人不会打开页面就关）
            for _ in range(random.randint(2, 5)):
                page.evaluate(f"window.scrollBy(0, {random.randint(100, 400)})")
                time.sleep(random.uniform(0.3, 0.8))

            # 模拟人：偶尔停留久一点（"在看数据"）
            if random.random() < 0.3:
                logger.info(f"         (驻足 3-6 秒...)")
                time.sleep(random.uniform(3, 6))
            else:
                time.sleep(random.uniform(1.2, 2.5))

        # === 提取 Cookie ===
        cookies = context.cookies()
        cookies_after = len(cookies)

        # 提取东方财富相关域名的 cookie
        em_cookies = [c for c in cookies if "eastmoney" in c.get("domain", "")]
        logger.info(f"Cookie: {cookies_before}→{cookies_after} (东方财富: {len(em_cookies)})")

        # 保存为两种格式：Playwright格式 + requests字典格式
        # Playwright 格式（完整）
        json.dump(cookies, open(COOKIE_FILE, "w"), indent=2, ensure_ascii=False)

        # requests 字典格式（给 curl_cffi 用）
        requests_cookies = {c["name"]: c["value"] for c in cookies}
        requests_file = PROFILE_DIR / "eastmoney_cookies_dict.json"
        json.dump(requests_cookies, open(requests_file, "w"), indent=2, ensure_ascii=False)

        context.close()
        logger.info(f"✅ Cookie 已保存: {len(cookies)} 个总, {len(em_cookies)} 个东方财富")
        logger.info(f"   requests格式: {requests_file}")

        return cookies, requests_cookies


# ==================== 请求节奏拟人化 ====================

def human_delay():
    """拟人化延迟 — Pareto分布 + 偶尔发呆"""
    delay = random.paretovariate(2) * 0.5  # 0.1~N秒，Pareto分布
    delay = min(delay, 8)  # 上限8秒
    if random.random() < 0.05:  # 5%概率"发呆"
        delay += random.uniform(2, 5)
    time.sleep(max(0.05, delay))


# ==================== 对比测试 ====================

def test_with_vs_without(requests_cookies=None):
    """对比：无Cookie vs 有Cookie + 拟人节奏"""
    from curl_cffi import requests as cffi

    test_cases = [
        ("实时行情", "https://push2.eastmoney.com/api/qt/stock/get?secid=0.000933&fields=f43,f170,f47"),
        ("K线数据", "https://push2his.eastmoney.com/api/qt/stock/kline/get?secid=0.000933&fields1=f1,f2,f3,f4,f5,f6&fields2=f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61&klt=15&fqt=0&end=20500101&lmt=20"),
        ("资金流向", "https://push2.eastmoney.com/api/qt/stock/fflow/daykline/get?secid=0.000933&fields1=f1,f2,f3,f7&fields2=f51,f52,f53,f54,f55,f56&lmt=5"),
        ("板块数据", "https://push2.eastmoney.com/api/qt/clist/get?pn=1&pz=5&po=1&np=1&fltt=2&invt=2&fid=f3&fs=m:90+t2&fields=f2,f3,f4,f12,f14"),
    ]

    results = []

    for name, url in test_cases:
        # === 无Cookie ===
        cold_ok, cold_size, cold_err = False, 0, ""
        for attempt in range(3):
            try:
                r = cffi.get(url, impersonate="chrome120", timeout=10)
                cold_size = len(r.content)
                cold_ok = cold_size > 50 and "rc" not in r.text[:5] or cold_size > 500
                if cold_ok: break
            except Exception as e:
                cold_err = str(e)[:80]
            time.sleep(0.5)

        human_delay()

        # === 有Cookie + 拟人节奏 ===
        warm_ok, warm_size, warm_err = False, 0, ""
        if requests_cookies:
            for attempt in range(3):
                try:
                    r = cffi.get(url, impersonate="chrome120", cookies=requests_cookies, timeout=10)
                    warm_size = len(r.content)
                    warm_ok = warm_size > 50 and "rc" not in r.text[:5] or warm_size > 500
                    if warm_ok: break
                except Exception as e:
                    warm_err = str(e)[:80]
                time.sleep(0.5)

        improved = "⬆️" if warm_size > cold_size else "➡️" if warm_size == cold_size else "⬇️"

        results.append({
            "name": name,
            "cold_ok": cold_ok, "cold_size": cold_size,
            "warm_ok": warm_ok, "warm_size": warm_size,
            "improved": improved,
            "cold_err": cold_err, "warm_err": warm_err,
        })

    # === 打印结果 ===
    print("\n" + "=" * 70)
    print("Cookie 预热效果对比")
    print("=" * 70)
    print(f"{'测试项':<12} {'无Cookie':>12} {'有Cookie':>12} {'效果':>6}")
    print("-" * 48)
    for r in results:
        cold_s = f"✅ {r['cold_size']}b" if r['cold_ok'] else f"❌ {r['cold_size']}b"
        warm_s = f"✅ {r['warm_size']}b" if r['warm_ok'] else f"❌ {r['warm_size']}b"
        print(f"{r['name']:<12} {cold_s:>12} {warm_s:>12} {r['improved']:>6}")

    # 成功率
    cold_wins = sum(1 for r in results if r["cold_ok"])
    warm_wins = sum(1 for r in results if r["warm_ok"])
    print("-" * 48)
    print(f"成功率:    无Cookie={cold_wins}/{len(results)}  有Cookie={warm_wins}/{len(results)}")
    print()

    # 分析
    if warm_wins > cold_wins:
        print("🔥 Cookie 预热有效！有Cookie的成功率更高")
    elif warm_wins == cold_wins and all(r["warm_size"] >= r["cold_size"] for r in results):
        print("✅ Cookie 有正面效果（响应更大=更完整的返回）")
    else:
        print("➡️  Cookie 对当前端点无明显差异（这些API本来就没封）")

    return results


# ==================== 主流程 ====================

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="仅测试已有Cookie")
    args = parser.parse_args()

    requests_cookies = None

    if args.test:
        # 加载已有Cookie
        cookie_file = PROFILE_DIR / "eastmoney_cookies_dict.json"
        if cookie_file.exists():
            requests_cookies = json.load(open(cookie_file))
            logger.info(f"加载已有Cookie: {len(requests_cookies)} 个")
        else:
            logger.warning("没有保存的Cookie，先运行完整流程(不带--test)")
            # 降级：用Playwright直接提取
            logger.info("尝试用Playwright直接获取Cookie...")
            try:
                _, requests_cookies = warm_cookies()
            except Exception as e:
                logger.error(f"Cookie预热失败: {e}")
    else:
        # 完整流程：预热 → 对比
        _, requests_cookies = warm_cookies()

    # 测试效果
    test_with_vs_without(requests_cookies)


if __name__ == "__main__":
    main()
