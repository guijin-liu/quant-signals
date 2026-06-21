# 量化信号 - 离线运行方案

## 当前状态
✅ PushPlus 微信推送已打通
✅ signal_pusher.py 扫描+推送脚本就绪
✅ 股票池分离 (stock_pool.py 加减即可)
❌ GitHub 被墙推不上去
❌ Gitee 需要令牌
❌ Windows计划任务语法不对

## 你浏览器操作 (30秒)
1. 打开 https://gitee.com/liuguijin
2. 登录 → 右上角头像 → 设置 → 私人令牌 → 生成新令牌
3. 描述随便填 → 勾选 projects → 提交
4. 复制 gp_ 开头的令牌 → 告诉我
5. 我推代码 + 配置 Gitee Actions 定时运行

## Gitee Push 后效果
- 代码存在国内，访问飞快
- Gitee Actions 工作日自动跑
- 信号推送到你微信
- 改股票改 stock_pool.py 提交即可

---

## 临时方案: 现在直接跑
```
python signal_pusher.py
```
信号已经在推了。上面的是让你电脑关了也能自动跑。
