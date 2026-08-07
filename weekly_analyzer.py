"""周度规律分析（智能化学习层）
汇总本周每日观察 + 本周龙虎榜净买TOP → knowledge/周度规律-YYYY-MM-DD.md
云端每周日跑，commit 回仓库。
"""
import os, sys, glob, re, datetime, requests
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

DC_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"
KNOW_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "knowledge")


def _dc(report_name, filter_str="", page_size=20, sort_columns="", sort_types="-1"):
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


def weekly():
    today = datetime.date.today()
    L = [f"# 📊 周度市场回顾 {today}", "", "> 智能化学习 · 每周自动生成", ""]

    # 1. 汇总本周每日观察
    L.append("## 📈 本周市场")
    files = sorted(glob.glob(os.path.join(KNOW_DIR, "每日观察-*.md")))
    week_files = [f for f in files if os.path.basename(f)[5:15] >= (today - datetime.timedelta(days=7)).isoformat()] or files[-5:]
    if week_files:
        for f in week_files:
            content = open(f, encoding="utf-8").read()
            m = re.search(r"上证指数：\*\*([+-][0-9.]+)%\*\* \| 成交额：\*\*([0-9.]+)万亿", content)
            d = os.path.basename(f).replace("每日观察-", "").replace(".md", "")
            if m:
                L.append(f"- **{d}**：上证 {m.group(1)}% | 成交额 {m.group(2)} 万亿")
    else:
        L.append("- 本周暂无每日观察")
    L.append("")

    # 2. 本周龙虎榜净买TOP10
    L += ["## 🐉 本周龙虎榜净买TOP10", "| 代码 | 名称 | 净买(亿) | 上榜日 |",
          "|------|------|---------|--------|"]
    try:
        start = (today - datetime.timedelta(days=7)).isoformat()
        end = today.isoformat()
        rows = _dc("RPT_DAILYBILLBOARD_DETAILSNEW",
                   f"(TRADE_DATE>='{start}')(TRADE_DATE<='{end}')", 10,
                   "BILLBOARD_NET_AMT", "-1")
        if rows:
            for row in rows:
                L.append(f"| {row.get('SECURITY_CODE','')} | {row.get('SECURITY_NAME_ABBR','')} | "
                         f"{(row.get('BILLBOARD_NET_AMT') or 0)/1e8:.2f} | "
                         f"{str(row.get('TRADE_DATE',''))[:10]} |")
        else:
            L.append("| - | 本周无龙虎榜 | | |")
    except Exception:
        L.append("| - | 获取失败 | | |")
    L.append("")

    # 3. 本周涨停统计（按上榜原因关键词）
    L.append("## 🔥 本周涨停/活跃板块")
    try:
        rows = _dc("RPT_DAILYBILLBOARD_DETAILSNEW",
                   f"(TRADE_DATE>='{start}')(TRADE_DATE<='{end}')", 50,
                   "TRADE_DATE", "-1")
        reasons = {}
        for row in rows:
            reason = str(row.get("EXPLANATION", ""))[:6]
            reasons[reason] = reasons.get(reason, 0) + 1
        for reason, cnt in sorted(reasons.items(), key=lambda x: -x[1])[:6]:
            L.append(f"- {reason}：{cnt} 次上榜")
    except Exception:
        pass
    L.append("")

    # 4. 规律观察提示
    L += ["## 🔍 规律观察（供下周关注）",
          "- 结合本周龙虎榜净买TOP + 成交额活跃股，观察资金持续流入的方向",
          "- 主力资金规律：超大单净流入 + 低位 = 高胜率（2号规则R001 100%）",
          "- 注意高低切换：成交额TOP中涨跌分化明显",
          ""]
    L.append("---")
    L.append("*智能化生成 · 刘圭金第二大脑学习流水线*")

    content = "\n".join(L)
    os.makedirs(KNOW_DIR, exist_ok=True)
    path = os.path.join(KNOW_DIR, f"周度规律-{today.isoformat()}.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"已生成: {path}")
    return path


if __name__ == "__main__":
    weekly()
