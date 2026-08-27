"""主力资金 + 龙虎榜机构评估（1号/2号推送共用）

fund_eval(code, name) → 资金面评估短文本：
  主力近5日/20日净流入（妙想 mx_fund_flow）+ 龙虎榜机构席位净额（东财 datacenter）
数据源均不受东财 push2his IP 风控影响。
"""
import requests
from datetime import datetime, timedelta
import mx_fetcher
import em_client

DATACENTER_URL = "https://datacenter-web.eastmoney.com/api/data/v1/get"


def _dc(report_name, filter_str="", page_size=10, sort_columns="", sort_types="-1"):
    params = {"reportName": report_name, "columns": "ALL", "pageNumber": 1,
              "pageSize": page_size, "source": "WEB", "client": "WEB"}
    if filter_str:
        params["filter"] = f"({filter_str})"
    if sort_columns:
        params["sortColumns"] = sort_columns
        params["sortTypes"] = sort_types
    try:
        r = requests.get(DATACENTER_URL, params=params, timeout=12)
        return (r.json().get("result") or {}).get("data") or []
    except Exception:
        return []


def fund_eval(code: str, name: str) -> str:
    """返回主力资金+机构评估短文本；异常时返回空串"""
    try:
        rows = mx_fetcher.mx_fund_flow(code, name)
        if not rows:
            rows = em_client.em_fund_flow_sina(code)  # 新浪兜底（非东财系）
        if rows:
            f0 = str(rows[0].get("date", ""))[:10]
            fl = str(rows[-1].get("date", ""))[:10]
            if f0 < fl:  # 新浪升序 → 反转成最新在前
                rows = rows[::-1]
        if not rows:
            return "主力资金-"
        r5 = sum(x["main_net"] for x in rows[:5]) / 1e8
        r20 = sum(x["main_net"] for x in rows[:20]) / 1e8

        # 龙虎榜机构专用席位（近30日最近一次上榜）
        lb_txt = ""
        try:
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            records = _dc("RPT_DAILYBILLBOARD_DETAILSNEW",
                          f"(TRADE_DATE>='{start}')(TRADE_DATE<='{end}')(SECURITY_CODE=\"{code}\")",
                          10, "TRADE_DATE", "-1")
            if records:
                latest = str(records[0].get("TRADE_DATE", ""))[:10]
                inst_net = 0
                for side in ("BUY", "SELL"):
                    report = "RPT_BILLBOARD_DAILYDETAILSBUY" if side == "BUY" else "RPT_BILLBOARD_DAILYDETAILSSELL"
                    for row in _dc(report, f"(TRADE_DATE='{latest}')(SECURITY_CODE=\"{code}\")", 20, side, "-1"):
                        if str(row.get("OPERATEDEPT_CODE", "")) == "0":  # 机构专用席位
                            amt = row.get("BUY") or row.get("SELL") or 0
                            inst_net += amt if side == "BUY" else -amt
                lb_txt = f" 龙虎榜{latest[5:]}机构{inst_net/1e4:+.0f}万"
        except Exception:
            pass

        if r5 > 0.5 and r20 > 0:
            level = "资金强"
        elif r5 > 0:
            level = "资金转好"
        else:
            level = "资金偏弱"
        return f"主力5日{r5:+.2f}亿/20日{r20:+.2f}亿{lb_txt}[{level}]"
    except Exception:
        return ""
