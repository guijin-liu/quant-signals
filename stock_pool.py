"""
股票池 — 动态拉取全部A股，过滤ST/退市，支持预筛
"""
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


def get_all_stocks():
    """拉取全部A股（非ST、正常上市），返回 {code: name}"""
    import baostock as bs
    bs.login()
    try:
        rs = bs.query_stock_basic()
        stocks = {}
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            code_name = row[0]  # sh.600000 or sz.000001
            name = row[1]
            stock_type = row[4]   # 1=股票 2=指数
            stock_status = row[5]  # 1=上市

            # 只要正常上市的股票
            if stock_type != '1' or stock_status != '1':
                continue
            # 跳过ST/退市
            if any(x in name for x in ('ST', '*ST', '退')):
                continue
            # 提取纯代码
            if '.' in code_name:
                code = code_name.split('.')[1]
            else:
                code = code_name
            # 跳过科创板(688)、北交所(8)、B股(9)
            if code.startswith(('688', '8', '9')):
                continue
            # 只保留深市主板(00/002/003)、沪市主板(60)、创业板(30)
            if not code.startswith(('00', '30', '60')):
                continue

            stocks[code] = name

        bs.logout()
        logger.info(f"股票池: {len(stocks)} 只 (已过滤ST/科创/北交所)")
        return stocks
    except Exception as e:
        try: bs.logout()
        except: pass
        logger.error(f"拉取股票列表失败: {e}")
        return {code: info["name"] for code, info in STOCK_POOL_BACKUP.items()}


# 兜底股票池（baostock挂了用）
STOCK_POOL_BACKUP = {
    "000630": {"name": "铜陵有色", "sector": "有色金属"},
    "000933": {"name": "神火股份", "sector": "有色金属"},
    "000960": {"name": "锡业股份", "sector": "有色金属"},
    "002497": {"name": "雅化集团", "sector": "锂电池"},
    "000893": {"name": "亚钾国际", "sector": "化工"},
    "600362": {"name": "江西铜业", "sector": "有色金属"},
    "601899": {"name": "紫金矿业", "sector": "有色金属"},
    "600489": {"name": "中金黄金", "sector": "黄金"},
    "600111": {"name": "北方稀土", "sector": "稀土"},
    "300750": {"name": "宁德时代", "sector": "锂电池"},
}
