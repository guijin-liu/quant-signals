"""每日市场观察生成（自动化学习层）
生成 Markdown → knowledge/每日观察-YYYYMMDD.md，云端收盘后跑，commit 回仓库。
数据：妙想(市场/持仓) + 东财clist(成交额TOP) + 东财datacenter(龙虎榜)
"""
import os, sys, datetime, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mx_fetcher
import em_client

DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _dc(report_name, filter_str="", page_size=10, sort_columns="", sort_types="-1"):
    params = {"reportName": report_name, "columns": "ALL", "pageNumber": 1,
              "pageSize": page_size, "source": "WEB", "client": "WEB"}
    if filter_str:
        params["filter"] = f"({filter_str})"
    if sort_columns:
        params["sortColumns"] = sort_columns
        params["sortTypes"] = sort_types
    try:
        r = requests.get(DC_URL, params=params, timeout=15)
        return (r.json().get("result") or {}).get("data") or []
    except Exception:
        return []


def daily_observe():
    today = datetime.datetime.now().strftime("%Y-%m-%d")
    L = [f"# 📊 每日市场观察 {today}", "",
         "> 自动化学习 · 收盘后自动生成 · 数据:妙想+东财", ""]

    # 1. 市场概况
    try:
        mkt = mx_fetcher.get_market_brief()
        sh = mkt.get("sh_idx", 0)
        amt = (mkt.get("sz_amount") or 0) / 1e4
        L += ["## 📈 市场概况",
              f"- 上证指数：**{sh:+.2f}%** | 成交额：**{amt:.2f}万亿**", ""]
    except Exception:
        L += ["## 📈 市场概况", "- 获取失败", ""]

    # 2. 成交额TOP10
    L += ["## 💰 成交额TOP10", "| 排名 | 代码 | 名称 | 成交额(亿) | 涨跌% |",
          "|------|------|------|-----------|------|"]
    top = em_client.em_fetch_top_amount(10)
    if top:
        for i, s in enumerate(top):
            L.append(f"| {i+1} | {s['code']} | {s['name']} | {(s['amount'] or 0)/1e8:.1f} | {s['chg']:+.1f} |")
    else:
        L.append("| - | 获取失败 | | | |")
    L.append("")

    # 3. 龙虎榜（当日，按净买额）
    L += ["## 🐉 龙虎榜（当日）", "| 代码 | 名称 | 净买(万) | 上榜原因 |",
          "|------|------|---------|---------|"]
    try:
        rows = _dc("RPT_DAILYBILLBOARD_DETAILSNEW",
                   f"(TRADE_DATE='{today}')", 10, "BILLBOARD_NET_AMT", "-1")
        if rows:
            for row in rows:
                L.append(f"| {row.get('SECURITY_CODE','')} | {row.get('SECURITY_NAME_ABBR','')} | "
                         f"{round((row.get('BILLBOARD_NET_AMT') or 0)/1e4)} | "
                         f"{str(row.get('EXPLANATION',''))[:15]} |")
        else:
            L.append("| - | 今日无龙虎榜 | | |")
    except Exception:
        L.append("| - | 获取失败 | | |")
    L.append("")

    # 4. 持仓监控
    L += ["## 📌 持仓监控"]
    try:
        val = mx_fetcher.get_stock_valuation("002497", "雅化集团")
        fund = mx_fetcher.mx_fund_flow("002497", "雅化集团")
        price = val.get("price", 0)
        chg = val.get("change_pct", 0)
        r5 = sum(x["main_net"] for x in fund[:5]) / 1e8
        r20 = sum(x["main_net"] for x in fund[:20]) / 1e8
        L.append(f"- **雅化集团(002497)**：现价 {price} | 涨跌 {chg:+.2f}%")
        L.append(f"- 主力资金：近5日 {r5:+.2f}亿 | 近20日 {r20:+.2f}亿")
        L.append(f"- 成本 23.14，现浮盈 {(price/23.14-1)*100:+.1f}%")
    except Exception:
        L.append("- 持仓数据获取失败")
    L.append("")
    L.append("---")
    L.append(f"*自动化生成 · 刘圭金第二大脑学习流水线*")

    content = "\n".join(L)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"每日观察-{today}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成: {path}")
    return path


if __name__ == "__main__":
    daily_observe()
