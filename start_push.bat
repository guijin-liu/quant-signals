@echo off
REM === 量化交易系统 — 后台自动推送 ===
REM 用法: 双击运行，最小化窗口，收盘后关闭

cd /d C:\Users\Administrator\quant_trading

echo ========================================
echo  量化买卖点推送 v12 — 40只股票
echo  交易时间: 9:25盘前 + 9:30-15:00每30分钟
echo  按 Ctrl+C 或关闭窗口退出
echo ========================================

:loop

REM 获取时间
for /f "tokens=1-3 delims=: " %%a in ('echo %time%') do (
    set /a hour=%%a
    set /a min=%%b
)
set /a time_val=%hour%*100+%min%

REM 9:25 盘前扫描
if %time_val% geq 925 if %time_val% leq 1529 (
    if %time_val% equ 925 (
        echo [%time%] Premarket scan...
        python quick_scan.py
        timeout /t 240 /nobreak >nul
    )
    echo [%time%] Trading scan...
    python cloud_function.py
) else (
    echo [%time%] Market closed
)

REM 等待30分钟
echo Waiting 30min...
timeout /t 1800 /nobreak >nul
goto loop
