"""刘圭金2号 — 实盘扫描推送（动态池 + 85%规则 + 主力资金流确认）

复用1号骨架：market_state/主循环/T+1/push_msg。
差异化：池=dynamic_pool.load_pool_window(5)，信号=命中 rules2.json 规则。
"""
import os, sys, json, logging, time, re
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import requests

import em_client
import mx_fetcher
from dynamic_pool import load_pool_window
from backtest_miner2 import compute_feature_matrix, CONDITIONS
from fund_eval import fund_eval

APP_NAME = "刘圭金2号量化程序"
PUSHPLUS_TOKEN = "f3fb5c092ba34785b6857bb45d23d4fa"
PUSHPLUS_URL = "http://www.pushplus.plus/send"
WX_WEBHOOK = os.environ.get("WX_WEBHOOK", "")
BEIJING_TZ = timezone(timedelta(hours=8))
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RULES_PATH = os.path.join(BASE_DIR, "rules2.json")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger()


def html_to_wx(text):
    text = text.replace("<br>", "\n").replace("<br/>", "\n")
    text = re.sub(r"</tr>", "\n", text)
    text = re.sub(r"<td[^>]*>", "  ", text)
    text = re.sub(r"</td>", "", text)
    text = re.sub(r"<h3[^>]*>", "**", text)
    text = re.sub(r"</h3>", "**\n", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&#x[0-9a-fA-F]+;", "", text)
    return text.strip()


def push_msg(title, content, retries=3):
    if WX_WEBHOOK:
        md = f"## {title}\n{html_to_wx(content)}"
        for attempt in range(retries):
            try:
                r = requests.post(WX_WEBHOOK,
                    json={"msgtype": "markdown", "markdown": {"content": md}}, timeout=15)
                if r.json().get("errcode") == 0:
                    logger.info(f"WX PUSH OK: {title}")
                    return True
            except Exception as e:
                logger.warning(f"WX异常({attempt+1}): {e}")
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
        return False
    for attempt in range(retries):
        try:
            r = requests.post(PUSHPLUS_URL,
                json={"token": PUSHPLUS_TOKEN, "title": title, "content": content, "template": "html"},
                timeout=15)
            if r.json().get("code") == 200:
                logger.info(f"PUSH OK: {title}")
                return True
        except Exception as e:
            logger.warning(f"PUSH异常({attempt+1}): {e}")
        if attempt < retries - 1:
            time.sleep(2 * (attempt + 1))
    return False


def bj_now():
    return datetime.now(BEIJING_TZ)


def is_trading_day():
    return bj_now().weekday() < 5


def market_state():
    now = bj_now()
    t = now.hour * 60 + now.minute
    if t < 9 * 60 + 25:    return "pre"
    if t < 11 * 60 + 30:   return "morning"
    if t < 13 * 60:        return "lunch"
    if t < 15 * 60 + 1:    return "afternoon"
    return "closed"


def load_rules():
    if not os.path.exists(RULES_PATH):
        logger.error(f"rules2.json 不存在: {RULES_PATH}")
        return []
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f).get("rules", [])


def scan_once(all_stocks, rules):
    """对动态池每票：腾讯15minK线 → 特征矩阵末行 → 命中规则 → 记录信号"""
    results = []
    for code, name in all_stocks.items():
        try:
            df = mx_fetcher.fetch_kline(code, "15")
            if df.empty or len(df) < 70:
                continue
            fm = compute_feature_matrix(df["close"].values, df["high"].values,
                                        df["low"].values, df["volume"].values)
            if fm.empty:
                continue
            for rule in rules:
                conds = rule.get("conditions", [])
                try:
                    bools = {cn: CONDITIONS[cn](fm) for cn in conds}
                    hit = all(bool(c) for cn in conds for c in [bools[cn].iloc[-1]])
                except Exception as e:
                    logger.warning(f"规则判定异常 {code}: {e}")
                    continue
                if not hit:
                    continue
                # 资金流确认（仅午盘后启用，避免半天数据误判）
                main_net = 0
                if bj_now().hour >= 13:
                    fund = mx_fetcher.mx_fund_flow(code, name)  # 妙想个人API（东财分钟级被拒时兜底，日级主力净流入）
                    main_net = fund[0]["main_net"] if fund else 0  # 妙想降序(最新在前)，[0]=最新交易日
                if rule.get("fund_required") and main_net <= 0:
                    logger.info(f"{code}{name} 命中规则但主力净流出({main_net/1e4:.0f}万)，拦截")
                    continue
                close = float(fm["close"].iloc[-1])
                results.append({
                    "code": code, "name": name, "close": round(close, 2),
                    "rule": rule, "main_net": main_net,
                    "rsi": round(float(fm["rsi"].iloc[-1]), 1),
                    "bb": round(float(fm["bb_pct"].iloc[-1]), 2),
                    "sell_pct": rule.get("sell_pct", 1.5),
                })
                break  # 一票一轮只报一条规则
        except Exception as e:
            logger.error(f"scan {code} 异常: {e}")
    return results


def push_signal(r, scan_time):
    code, name = r["code"], r["name"]
    rule = r["rule"]
    eval_txt = fund_eval(code, name)   # 主力资金+龙虎榜机构评估
    sell = round(r["close"] * (1 + r["sell_pct"] / 100), 2)
    conds_txt = "+".join(rule.get("conditions", []))
    main_str = f"主力净流入 {r['main_net']/1e4:.0f}万" if r["main_net"] else "主力资金-"
    title = f"🔴{APP_NAME}买入 {name} {r['close']}"
    content = (f'<div style="font-size:15px;padding:12px;line-height:2">'
               f'<h3 style="color:#e74c3c">🔴 买入信号 — {name}({code})</h3>'
               f'<table style="width:100%">'
               f'<tr><td>现价</td><td><b style="color:#e74c3c;font-size:20px">{r["close"]}</b></td>'
               f'<td>卖点</td><td><b style="color:#f39c12">{sell}</b> (+{r["sell_pct"]}%)</td></tr>'
               f'<tr><td>命中规则</td><td style="color:#f39c12">{conds_txt}</td></tr>'
               f'<tr><td>胜率</td><td>{rule.get("wr", "-")}% (n={rule.get("n", "-")})</td>'
               f'<td>{main_str}</td></tr>'
               f'<tr><td>RSI</td><td>{r["rsi"]}</td><td>BB%</td><td>{r["bb"]}</td></tr>'
               f'<tr><td colspan="4" style="color:#8e44ad">💰 {eval_txt}</td></tr>'
               f'</table>'
               f'<p style="color:#888;font-size:11px">{scan_time} | {APP_NAME} | 动态池+85%规则+资金流</p></div>')
    push_msg(title, content)


def main_loop():
    start_time = bj_now()
    logger.info(f"═══ {APP_NAME} 启动 @ {start_time.strftime('%m/%d %H:%M')} ═══")
    if not is_trading_day():
        logger.info("非交易日，退出")
        return

    rules = load_rules()
    if not rules:
        logger.error("无规则，退出")
        return
    logger.info(f"加载 {len(rules)} 条规则")

    pool = load_pool_window(5)
    if not pool:
        logger.error("动态池为空（data/top_amount 无榜单），退出")
        return
    logger.info(f"动态池 {len(pool)} 只")

    pushed = set()
    bought_today = set()
    all_day = []

    deadline = bj_now().replace(hour=15, minute=30, second=0, microsecond=0)
    last_state = None
    scan_count = 0

    while bj_now() < deadline:
        state = market_state()
        if state != last_state:
            logger.info(f"市场状态: {last_state} → {state}")
            last_state = state
        if state == "pre":
            time.sleep(30); continue
        if state == "lunch":
            time.sleep(300); continue
        if state == "closed":
            logger.info("已收盘，退出")
            break

        scan_count += 1
        scan_time = bj_now().strftime('%m/%d %H:%M')
        logger.info(f"═══ 第{scan_count}轮扫描 @ {scan_time} ═══")
        try:
            results = scan_once(pool, rules)
        except Exception as e:
            logger.error(f"扫描异常: {e}", exc_info=True)
            time.sleep(60); continue

        new_n = 0
        for r in results:
            code = r["code"]
            key = (code, tuple(r["rule"].get("conditions", [])))
            if key in pushed:
                continue
            if code in bought_today:  # T+1: 当日已买不重报
                continue
            pushed.add(key)
            bought_today.add(code)
            all_day.append(r)
            try:
                push_signal(r, scan_time)
                new_n += 1
            except Exception as e:
                logger.error(f"推送异常 {code}: {e}")
            time.sleep(0.5)
        logger.info(f"第{scan_count}轮: {len(results)}命中, {new_n}新推送")

        now = bj_now()
        nxt = now.replace(second=0, microsecond=0) + timedelta(minutes=5 - now.minute % 5)
        wait = max(30, (nxt - now).total_seconds())
        time.sleep(min(wait, 300))

    end_str = bj_now().strftime('%m/%d %H:%M')
    logger.info(f"═══ {APP_NAME} 全天结束 {end_str} | {scan_count}轮 | {len(all_day)}条信号 ═══")
    if all_day:
        lines = "<br>".join(
            f"🔴 {r['name']}({r['code']}) @{r['close']} 卖点+{r['sell_pct']}% | {'+'.join(r['rule']['conditions'])}"
            for r in all_day)
        push_msg(f"📊{APP_NAME}收盘汇总({len(all_day)}只) {end_str}",
                 f'<div style="font-size:14px;padding:10px">{lines}<p style="color:#888">⚠️ T+1：今日买入最快明日可卖</p></div>')
    else:
        push_msg(f"☁️{APP_NAME} {end_str} 今日无信号",
                 f'<div>动态池{len(pool)}只，{scan_count}轮扫描无规则命中</div>')


if __name__ == "__main__":
    main_loop()
