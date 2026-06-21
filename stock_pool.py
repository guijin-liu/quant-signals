# 股票池配置文件
# 加股票: 加一行 "000xxx": {"name": "某某股份", "sector": "所属行业"}
# 减股票: 删一行
# 提交后自动生效

STOCK_POOL = {
    "000933": {"name": "神火股份", "sector": "有色金属"},
    "002497": {"name": "雅化集团", "sector": "化工"},
    "000960": {"name": "锡业股份", "sector": "有色金属"},
    "000893": {"name": "亚钾国际", "sector": "化工"},
}

# 下面不要改
STOCK_CODES = list(STOCK_POOL.keys())
