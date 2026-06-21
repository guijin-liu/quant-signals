# 量化监控 - 自启动后台任务
# Claude Cron只能会话级，真正的离线运行靠Windows计划任务

## 方式1: Windows 计划任务(推荐 - Claude死了也能跑)

### 创建任务
按 Win+R → taskschd.msc → 创建基本任务:

| 设置 | 值 |
|------|-----|
| 名称 | QuantSignalPusher |
| 触发 | 每天 9:00，重复间隔15分钟，持续6小时 |
| 操作 | 启动程序: `python` |
| 参数 | `C:\Users\Administrator\quant_trading\signal_pusher.py` |
| 起始于 | `C:\Users\Administrator\quant_trading` |

### 或者一键命令行创建:
```
schtasks /create /tn QuantSignalPusher /tr "cmd /c cd /d C:\Users\Administrator\quant_trading && python signal_pusher.py" /sc daily /st 09:00 /ri 15 /du 06:00 /f
```

---

## 方式2: 双击启动 bat (简单但手动)
双击 `C:\Users\Administrator\quant_trading\start_watch.bat`
→ 后台持续运行，崩溃自动重启

---

## 当前Claude Cron (会话级,Claude退出就没了)
- 工作日 9:07 → 扫描信号推送
- 工作日 9:23, 10:23, 11:23, 13:23, 14:23 → 盘中扫描
