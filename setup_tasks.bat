@echo off
REM ====== Windows Task Scheduler setup for quant signal push ======
REM Run ONCE as Administrator to create all 11 tasks
REM Trading hours: 9:30-15:00, every 30min, Mon-Fri

set PYTHON=C:\Program Files\Python314\python.exe
set SCRIPT=C:\Users\Administrator\quant_trading\cloud_function.py

schtasks /create /tn "QuantSignal_0930" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 09:30 /f
schtasks /create /tn "QuantSignal_1000" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 10:00 /f
schtasks /create /tn "QuantSignal_1030" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 10:30 /f
schtasks /create /tn "QuantSignal_1100" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 11:00 /f
schtasks /create /tn "QuantSignal_1130" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 11:30 /f
schtasks /create /tn "QuantSignal_1300" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 13:00 /f
schtasks /create /tn "QuantSignal_1330" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 13:30 /f
schtasks /create /tn "QuantSignal_1400" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 14:00 /f
schtasks /create /tn "QuantSignal_1430" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 14:30 /f
schtasks /create /tn "QuantSignal_1500" /tr "\"%PYTHON%\" \"%SCRIPT%\"" /sc weekly /d MON,TUE,WED,THU,FRI /st 15:00 /f

echo === All 10 tasks created ===
schtasks /query /tn QuantSignal_0930
echo === Done ===
pause
