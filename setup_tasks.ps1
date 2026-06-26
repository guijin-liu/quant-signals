$python = "C:\Program Files\Python314\python.exe"
$script = "C:\Users\Administrator\quant_trading\cloud_function.py"

$times = @("09:30","10:00","10:30","11:00","11:30","13:00","13:30","14:00","14:30","15:00")
$days  = @("Monday","Tuesday","Wednesday","Thursday","Friday")

foreach ($t in $times) {
    $name = "QuantSignal_" + $t.Replace(":","")
    $trigger = New-ScheduledTaskTrigger -Weekly -DaysOfWeek $days -At $t
    $action  = New-ScheduledTaskAction -Execute $python -Argument $script
    Register-ScheduledTask -TaskName $name -Trigger $trigger -Action $action -Force
    Write-Host "OK: $name @ $t"
}
Write-Host "=== All 10 tasks registered ==="
