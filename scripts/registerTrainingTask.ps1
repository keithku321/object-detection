# Register a Windows scheduled task that trains every plan with scripts/trainAll.py, in a visible PowerShell window in the repository root.
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -StartNow                 (50 epochs, starts at once)
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1                           (50 epochs, starts in one minute)
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -At "2026-10-23 01:00" -Plans A,C
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -Remove
### Flow: find the repository root (the parent of this folder) and its .venv Python, build the command that the window runs (UTF-8 output,
### repository root, then python -m scripts.trainAll), and register a one-time task for the current user. The task runs only while that user
### is logged on, which is what lets the window appear on the desktop; it may run on battery and has no time limit, because the 50-epoch
### runs take many hours. scripts/trainAll.py keeps Windows from idle sleep while it runs, but closing a laptop lid still sends it to sleep.
param(
    # Number of epochs for every plan
    [int]$Epochs = 50,
    # Plan letters to train (A underwater, B weather, C small objects, D CamoEdge)
    [string[]]$Plans = @("A", "B", "C", "D"),
    # When the task starts
    [datetime]$At = (Get-Date).AddMinutes(1),
    # Start the task right after registering it instead of waiting for -At
    [switch]$StartNow,
    # Name of the task in Task Scheduler
    [string]$TaskName = "ObjectDetectionTraining",
    # Remove the task and exit
    [switch]$Remove
)
$ErrorActionPreference = "Stop"

# Repository root and the Python of its virtual environment
$repositoryRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"

if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Host "Removed the scheduled task $TaskName"
    return
}
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "No virtual environment at $pythonPath. Create it first (cmd.md, Setup), then run this again."
}

# With -File, "-Plans A,C" arrives as the single text "A,C", so split every item on commas
$planLetters = @($Plans | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim().ToUpper() } | Where-Object { $_ })
$unknownPlans = @($planLetters | Where-Object { $_ -notin @("A", "B", "C", "D") })
if ($unknownPlans.Count -gt 0) { throw "Unknown plan letters: $($unknownPlans -join ', '). Use A, B, C and D." }

# Single quotes inside the paths are doubled, because the paths sit inside single-quoted PowerShell text
$quotedRoot = $repositoryRoot.Replace("'", "''")
$quotedPython = $pythonPath.Replace("'", "''")
$trainingCommand = "[Console]::OutputEncoding = [Text.Encoding]::UTF8; Set-Location -LiteralPath '$quotedRoot'; " +
                   "& '$quotedPython' -m scripts.trainAll --epochs $Epochs --plans $($planLetters -join ' ')"

# -NoExit keeps the window, and with it the summary table, open after the last job
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoExit -NoProfile -ExecutionPolicy Bypass -Command `"$trainingCommand`"" -WorkingDirectory $repositoryRoot
$trigger = New-ScheduledTaskTrigger -Once -At $At
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -ExecutionTimeLimit ([TimeSpan]::Zero) -MultipleInstances IgnoreNew
$principal = New-ScheduledTaskPrincipal -UserId "$env:USERDOMAIN\$env:USERNAME" -LogonType Interactive -RunLevel Limited
$description = "Trains plans $($planLetters -join ', ') for $Epochs epochs with scripts/trainAll.py; logs in runs\epochs$Epochs\logs (object-detection repository)."
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings -Principal $principal -Description $description -Force | Out-Null

if ($StartNow) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Host "Started the scheduled task $TaskName (plans $($planLetters -join ', '), $Epochs epochs)."
} else {
    Write-Host "Registered the scheduled task $TaskName; it starts at $At (plans $($planLetters -join ', '), $Epochs epochs)."
}
Write-Host "Window command: $trainingCommand"
Write-Host "Check it:  Get-ScheduledTask -TaskName $TaskName | Get-ScheduledTaskInfo"
Write-Host "Logs and summary: $(Join-Path $repositoryRoot "runs\epochs$Epochs\logs")"
Write-Host "Keep the laptop plugged in and the lid open while it runs."
