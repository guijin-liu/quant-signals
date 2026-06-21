# 量化信号定时推送 - Cron 任务
# Claude离线后自动执行，信号推送到微信

## 盘前提醒 (每交易日 9:00)
name: quant-morning-alert
schedule: "0 9 * * 1-5"
command: cd /d C:\Users\Administrator\quant_trading && python -c "from push_notify import push_msg; push_msg('今日开盘提醒', '<h3>量化系统就绪</h3><p>股票池: 神火 雅化 锡业 亚钾</p><p>目标胜率>88% | 15min日内</p>')"

---

## 盘中扫描 (每15分钟, 9:30-15:00)
name: quant-intraday-scan
schedule: "*/15 9,10,11,13,14 * * 1-5"
command: cd /d C:\Users\Administrator\quant_trading && python signal_pusher.py

---

## 收盘总结 (15:05)
name: quant-eod-summary
schedule: "5 15 * * 1-5"
command: cd /d C:\Users\Administrator\quant_trading && python -c "from push_notify import push_msg; from signal_pusher import scan_and_push; scan_and_push()"
