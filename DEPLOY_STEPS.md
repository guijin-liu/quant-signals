# 云端部署 — 最后一步

## 当前状态
✅ 代码已提交本地Git (3次commit)
✅ GitHub Actions workflow已写好
✅ 股票池可从stock_pool.py加减
✅ PushPlus推送已配置 (token: f3fb5c...ba4fa)
❌ 还没推到GitHub云端

## 你需要操作 (2步)

### 步骤1: 注册GitHub账号
打开 https://github.com → 点 Sign up → 用邮箱注册
(已有账号跳过)

### 步骤2: 告诉我的你GitHub用户名
我来创建仓库并推送代码

---

## 推上去之后的效果

```
每交易日 (周一到周五):
  09:07  GitHub机器人自动启动
  09:37  盘中扫描第1次
  10:07  盘中扫描第2次
  ...
  14:37  盘中扫描最后1次
  
  每次扫描4只股票 → 评分 → 推送到你微信
```

## 加股票
打开 GitHub上 stock_pool.py → 编辑 → 加一行 `"600xxx": {"name": "某某", "sector": "行业"}` → 提交 → 下个周期自动生效

## 删股票
打开 stock_pool.py → 删一行 → 提交 → 自动生效
