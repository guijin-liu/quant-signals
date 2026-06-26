@echo off
REM ====== A股实时买卖点推送 v10.2 ======
REM 交易时间每30分钟扫描，有信号推微信
REM 用法: 双击运行，最小化窗口，收盘后关闭

cd /d C:\Users\Administrator\quant_trading
echo ========================================
echo  A股实时买卖点推送 v10.2
echo  交易时间 9:30-15:00 每30分钟扫描
echo  有买入/卖出信号才推微信
echo  按 Ctrl+C 或关闭窗口退出
echo ========================================

:loop
REM 获取当前时间 (小时分钟)
for /f "tokens=1-3 delims=: " %%a in ('echo %time%') do (
    set /a hour=%%a
    set /a min=%%b
)

REM 判断是否在交易时段 (9:25-15:05)
set /a time_val=%hour%*100+%min%
if %time_val% geq 925 if %time_val% leq 1505 (
    echo [%time%] Trading hours - scanning...
    python cloud_function.py
) else (
    echo [%time%] Outside trading hours
    REM 如果收盘后，退出一段时间再扫
    if %time_val% gtr 1505 (
        echo Market closed. Will check again at 9:25 tomorrow.
        REM 计算到明天9:25的等待时间(秒)，大约18小时=64800s
        timeout /t 3600 /nobreak >nul
    )
)

REM 等待30分钟 (1800秒)
timeout /t 1800 /nobreak >nul
goto loop
