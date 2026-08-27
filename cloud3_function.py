"""刘圭金3号 — 实盘扫描推送（热点池 + 85%规则 + 主力资金流确认）

复用2号骨架：market_state/主循环/T+1/push_msg。
差异化：池=hot_pool.load_latest_hot_pool()（当日人气前50），信号=命中 rules2.json 规则。
"""
import os, sys, json, logging, time, re
import numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
import requests

import em_client
import mx_fetcher
from hot_pool import load_latest_hot_pool
from backtest_miner2 import compute_feature_matrix, CONDITIONS
from fund_eval import fund_eval

APP_NAME = "刘圭金3号量化程序"
PUSHPLUS_TOKEN = os.environ.get("PUSHPLUS_TOKEN", "f3fb5c092ba34785b6857bb45d23d4fa")
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


# A股 2026 法定节假日（简易表，按官方安排核对；周末已由 weekday 排除）
HOLIDAYS_2026 = {"0101", "0216", "0217", "0218", "0219", "0220", "0223",
                 "0406", "0501", "0504", "0505", "0506",
                 "0619", "0925", "1001", "1002", "1005", "1006", "1007"}


def is_trading_day():
    now = bj_now()
    if now.weekday() >= 5:
        return False
    return now.strftime("%m%d") not in HOLIDAYS_2026


def market_state():
    now = bj_now()
    t = now.hour * 60 + now.minute
    if t < 8 * 60 + 45:    return "pre"
    if t < 11 * 60 + 30:   return "morning"
    if t < 13 * 60:        return "lunch"
    if t < 15 * 60:        return "afternoon"
    return "closed"


def load_rules():
    if not os.path.exists(RULES_PATH):
        logger.error(f"rules2.json 不存在: {RULES_PATH}")
        return []
    with open(RULES_PATH, encoding="utf-8") as f:
        return json.load(f).get("rules", [])


def scan_once(all_stocks, rules):
    """对热点池每票：腾讯15minK线 → 特征矩阵+真实资金流覆盖 → 命中规则 → 记录信号"""
    results = []
    for code, name in all_stocks.items():
        try:
            df = mx_fetcher.fetch_kline(code, "15")
            if df.empty or len(df) < 70:
                continue
            fm = compute_feature_matrix(df["close"].values, df["high"].values,
                                        df["low"].values, df["volume"].values, df["open"].values)
            if fm.empty:
                continue
            # ── 真实资金流覆盖（妙想优先 → 新浪兜底，非东财系）──
            fund = mx_fetcher.mx_fund_flow(code, name)
            if not fund:
                fund = em_client.em_fund_flow_sina(code)
            fund_by_date = {}
            for d in fund:
                fd = str(d.get("date"))[:10]
                if len(fd) == 8 and "-" not in fd:
                    fd = f"{fd[:4]}-{fd[4:6]}-{fd[6:8]}"
                fund_by_date[fd] = d

            def _norm(d):
                s = str(d)[:10]
                if len(s) == 8 and "-" not in s:
                    s = f"{s[:4]}-{s[4:6]}-{s[6:8]}"
                return s
            dates = [_norm(d) for d in df["date"].tolist()]
            fm["main_net"] = np.array([fund_by_date.get(d, {}).get("main_net", 0) for d in dates])
            fm["super_net"] = np.array([fund_by_date.get(d, {}).get("super_net", 0) for d in dates])
            fm["large_net"] = np.array([fund_by_date.get(d, {}).get("large_net", 0) for d in dates])
            if "amount_mean" not in fm.columns:  # 腾讯K线无成交额列，用 close*volume*100(手) 估算
                est = df["close"].values * df["volume"].values * 100
                fm["amount_mean"] = pd.Series(est).rolling(20, min_periods=1).mean().values
            # 最新主力净流入（对齐最后bar日期；无则取fund最新：妙想降序[0]/新浪升序[-1]）
            main_net = 0
            last_d = dates[-1] if dates else ""
            v = fund_by_date.get(last_d)
            if v is not None:
                main_net = v.get("main_net", 0)
            elif fund:
                f0 = str(fund[0].get("date"))[:10]
                fl = str(fund[-1].get("date"))[:10]
                main_net = fund[0]["main_net"] if f0 > fl else fund[-1]["main_net"]
            for rule in rules:
                conds = rule.get("conditions", [])
                try:
                    bools = {cn: CONDITIONS[cn](fm) for cn in conds}
                    hit = all(bool(c) for cn in conds for c in [bools[cn].iloc[-1]])
                except Exception as e:
                    logger.warning(f"规则判定异常 {code} {name}: {e}")
                    continue
                if not hit:
                    continue
                # 资金流确认（仅午盘后启用，避免半天数据误判）
                if rule.get("fund_required") and bj_now().hour >= 13 and main_net <= 0:
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
               f'<p style="color:#888;font-size:11px">{scan_time} | {APP_NAME} | 当日热点池(人气前50)+85%规则+资金流</p></div>')
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

    pool = load_latest_hot_pool()
    if not pool:
        logger.error("热点池为空（data/hot_pool 无榜单），退出")
        return
    logger.info(f"当日热点池 {len(pool)} 只")

    pushed = set()
    bought_today = set()
    all_day = []

    # 截止时间可配置（多段运行跳过午休：上午段 SCAN_DEADLINE=11:30，下午段默认15:00）
    eh, em = 15, 0
    env_deadline = os.environ.get("SCAN_DEADLINE", "")
    if env_deadline and ":" in env_deadline:
        eh, em = map(int, env_deadline.split(":")[:2])
    deadline = bj_now().replace(hour=eh, minute=em, second=0, microsecond=0)

    # 延迟启动兜底：已过截止但未收盘→自动续扫到15:00；已收盘→明确告警而非"今日无信号"
    now = bj_now()
    if now >= deadline:
        if now.hour < 15:
            logger.warning(f"启动过晚({now.strftime('%H:%M')} 已过截止{eh:02d}:{em:02d})，自动续扫到15:00收盘")
            deadline = now.replace(hour=15, minute=0, second=0, microsecond=0)
        else:
            logger.warning(f"已收盘({now.strftime('%H:%M')})，本轮无法扫描")
            push_msg(f"⚠️{APP_NAME}扫描启动过晚",
                     f'<div style="font-size:14px;padding:10px"><h3 style="color:#e74c3c">⚠️ 扫描未执行</h3>'
                     f'<p>任务于 {now.strftime("%m/%d %H:%M")} 才启动，已过截止 {eh:02d}:{em:02d} 且已收盘，本轮未扫描。</p>'
                     f'<p style="color:#888">请检查GitHub Actions定时调度是否延迟。</p></div>')
            return
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
                 f'<div>当日热点池{len(pool)}只，{scan_count}轮扫描无规则命中</div>')


if __name__ == "__main__":
    main_loop()
