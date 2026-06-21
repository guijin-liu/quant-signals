@echo off
chcp 65001 >nul 2>&1
cd /d "C:\Users\Administrator\quant_trading"

echo.
echo  ========================================
echo    量化交易系统 - 神火股份 雅化集团 锡业股份 亚钾国际
echo  ========================================
echo.
echo    [1] 模拟回测 (不需要网络，2分钟出结果)
echo    [2] 真实数据回测 (需联网)
echo    [3] 查看最新信号 (需联网)
echo    [4] 全部流程 (需联网)
echo    [0] 退出
echo.
set /p choice="  请选择 (0-4): "

if "%choice%"=="1" (
    echo.
    echo  正在运行模拟回测...
    python -c "exec(open('run_demo.py', encoding='utf-8').read())"
)
if "%choice%"=="2" python main.py backtest
if "%choice%"=="3" python main.py signal
if "%choice%"=="4" python main.py all
if "%choice%"=="0" exit

pause
