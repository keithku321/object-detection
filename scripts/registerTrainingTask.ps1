# Register a Windows scheduled task that trains the Phase 1 baselines of all four plans with scripts/trainAllPlans.py (50 epochs each by default), and start it.
# The task opens a visible Windows PowerShell window in the repository root, so the training output can be watched while it runs (CLAUDE.md, "Do"); the window
# stays open after the last run so its summary can be read. It runs as the signed-in user, only while that user is signed in, and is not elevated.
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1                              (register and start now)
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -StartAt "2026-10-23 01:00"  (register, start at that time)
#   powershell -ExecutionPolicy Bypass -File scripts\registerTrainingTask.ps1 -Remove                      (delete the task; runs and results stay)
# Re-starting the task (Start-ScheduledTask) resumes unfinished runs and skips finished ones, because scripts/trainAllPlans.py does.
param(
    # Name of the task in Task Scheduler
    [string]$TaskName = "ObjectDetection-TrainAllPlans",
    # Epochs of every run
    [int]$Epochs = 50,
    # Folder for runs, logs and final weights, relative to the repository root
    [string]$Project = "runs",
    # Plans to train, in this order
    [string[]]$Plans = @("A", "B", "C", "D"),
    # Local start time such as "2026-10-23 01:00"; empty starts the task right after registering it
    [string]$StartAt = "",
    # Longest time the task may run before Task Scheduler stops it
    [int]$TimeLimitDays = 3,
    # Delete the task instead of registering it
    [switch]$Remove
)
$ErrorActionPreference = "Stop"

# Repository root: the folder above scripts/
$repositoryRoot = Split-Path -Parent $PSScriptRoot
# Interpreter of the project's virtual environment; the runs use the interpreter that starts the runner
$pythonPath = Join-Path $repositoryRoot ".venv\Scripts\python.exe"
# Process priority of the task: 4 is normal (Task Scheduler's default of 7 is below normal and slows the dataloaders while other programs run)
$normalPriority = 4


# Quote a value for a PowerShell command line in single quotes, doubling any single quote inside it
function ConvertTo-SingleQuoted([string]$value) {
    return "'" + ($value -replace "'", "''") + "'"
}


if ($Remove) {
    Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
    Write-Output "Removed the scheduled task '$TaskName'."
    return
}
if (-not (Test-Path -LiteralPath $pythonPath)) { throw "No virtual environment at $pythonPath; create it first (see cmd.md, Setup)." }

### The task starts Windows PowerShell, which moves to the repository root and runs the runner with the .venv interpreter; -NoExit keeps the window open.
$runnerArguments = "-m scripts.trainAllPlans --epochs $Epochs --project $(ConvertTo-SingleQuoted $Project) --plans $($Plans -join ' ')"
$windowTitle = ConvertTo-SingleQuoted "$TaskName ($Epochs epochs)"
$command = "`$Host.UI.RawUI.WindowTitle = $windowTitle; Set-Location -LiteralPath $(ConvertTo-SingleQuoted $repositoryRoot); " +
           "& $(ConvertTo-SingleQuoted $pythonPath) $runnerArguments; Write-Output 'Training finished; this window can be closed.'"
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-NoExit -NoProfile -ExecutionPolicy Bypass -Command `"$command`"" -WorkingDirectory $repositoryRoot
$startTime = if ($StartAt) { [datetime]::Parse($StartAt) } else { (Get-Date).AddMinutes(1) }
$trigger = New-ScheduledTaskTrigger -Once -At $startTime
# The signed-in account, read from .NET so a computer name in Chinese characters stays intact
$userName = [Security.Principal.WindowsIdentity]::GetCurrent().Name
$principal = New-ScheduledTaskPrincipal -UserId $userName -LogonType Interactive -RunLevel Limited
$settings = New-ScheduledTaskSettingsSet -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries -StartWhenAvailable -MultipleInstances IgnoreNew `
                                         -ExecutionTimeLimit (New-TimeSpan -Days $TimeLimitDays) -Priority $normalPriority
$description = "Trains the Phase 1 baselines of plans $($Plans -join ', ') for $Epochs epochs with scripts/trainAllPlans.py in $repositoryRoot (logs in $Project\logs)."
Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Principal $principal -Settings $settings -Description $description -Force | Out-Null
Write-Output "Registered the scheduled task '$TaskName' for $userName."
if (-not $StartAt) {
    Start-ScheduledTask -TaskName $TaskName
    Write-Output "Started it now: a PowerShell window shows the training; logs go to $(Join-Path $repositoryRoot $Project)\logs."
} else {
    Write-Output "It starts at $startTime."
}
