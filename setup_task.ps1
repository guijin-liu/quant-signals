$action = New-ScheduledTaskAction -Execute "python" -Argument "signal_pusher.py" -WorkingDirectory "C:\Users\Administrator\quant_trading"
$trigger = New-ScheduledTaskTrigger -At "09:00" -Daily
Register-ScheduledTask -TaskName "QuantPush" -Trigger $trigger -Action $action -Force
Write-Output "Task created"
