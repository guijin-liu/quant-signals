@echo off
chcp 65001 >nul
cd /d "C:\Users\Administrator\quant_trading"

echo ========================================
echo   量化信号监控 - 后台启动
echo   推送目标: PushPlus → 微信
echo   %date% %time%
echo ========================================

REM 持续运行，崩溃自动重启
:loop
echo [%time%] 启动信号监控...
python signal_pusher.py --watch --interval 15 >> watch_log.txt 2>&1
echo [%time%] 监控退出(exit=%errorlevel%), 60秒后重启...
timeout /t 60 /nobreak >nul
goto loop
