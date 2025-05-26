#Requires -RunAsAdministrator
param (
    # Input params
    [String]$app, # name of application to be profiled

    # Output files
    [String]$dataDir, # directory where the profiling results will be stored
    [String]$profilingResult = "workload.blg" # output file that will store the profiling results
)

$script:countersTemplate = ".\script\config\counters.txt"   # performance counter list
$script:jobRuntime = 330    # seconds
$script:samplePeriod = 5    # seconds
$script:CounterName = "HPE_ASPO_LOGMAN_COUNTER"

function New-Counter {
    <#
    This function fills the actual application name in performance counter list
    and saves it to a new file. Then, it registers the performance counters with
    the file.
    #>
    
    $outputPath = Join-Path $dataDir -ChildPath $profilingResult

    $counters = (Get-Content $script:countersTemplate) -replace "app", $app
    $countersPath = Join-Path ".\script\config\" "${app}.txt"
    Set-Content -Path $countersPath -Value $counters
    
    $message = `
        logman create counter `
        -n $script:CounterName `
        -cf $countersPath `
        -si $script:samplePeriod `
        -o $outputPath `
    | Out-String
    
    Remove-Item $countersPath

    if (($message -notmatch "completed successfully")) {
        $message = $message.trim()
        throw "[logman] $message"
    }
}

function Start-Counting {
    <#
    This function starts profiling.
    #>
    $message = logman start $script:CounterName | Out-String
    if ($message -notmatch "completed successfully") {
        $message = $message.trim()
        throw "[logman] $message"
    }
}

function Stop-Counting {
    <#
    This function stops profiling.
    #>
    logman stop $script:CounterName | Out-Null
}

function Remove-Counter {
    <#
    This function stops profiling and deletes the registered performance counters.
    #>
    Stop-Counting
    logman delete $script:CounterName | Out-Null
}

function Rename-Output {
    <#
    This function renames the output file to match the specified $profilingResult.
    #>
    $r = "workload_\d+.blg"
    $filenames = Get-ChildItem -Path $dataDir -Name
    foreach ($filename in $filenames) {
        if ($filename -match $r) {
            $filePath = Join-Path $dataDir $filename
            Rename-Item -Path $filePath -NewName $profilingResult
        }
    }
}

### Main Program ###

$ExitValue = 0

try {
    Write-Host -NoNewline "Initializing..."
    New-Counter
    Write-Host "Done"
                
    Write-Host -NoNewline "Start profiling..."
    Start-Counting
    Start-Sleep -Seconds $script:jobRuntime
    Stop-Counting
    Write-Host "Done"

    Rename-Output
}
catch {
    Write-Host "`n$_" -ForegroundColor "Red" -BackgroundColor "Black"
    Write-Host $_.ScriptStackTrace
    $ExitValue = 1
}
finally {
    Remove-Counter
}

exit $ExitValue
