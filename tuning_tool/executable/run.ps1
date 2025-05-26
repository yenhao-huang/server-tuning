#Requires -RunAsAdministrator
param (
    # The application to be optimized
    [String]$appName
)

### Tool Configuration ###

$script:dataDir = ".\data"
$script:profilingResult = "workload.blg"
$script:featureValue = "performance_log.csv"
$script:ssacli = "C:\Program Files\Smart Storage Administrator\ssacli\bin\ssacli.exe"
$script:iopatternPrediction = "prediction.txt"
$script:controllerSlot = 12
$script:logicalDiskID = 1
$script:dataDir = Join-Path $script:dataDir (Get-Date -format "yyyyMMddHHmmss")

### Main Program ###

$ExitValue = 0

$script:Usage = "Usage: `n" + `
    "$PSCommandPath " + `
    "-appName [APPLICATION NAME]"

try {
    if ([string]::IsNullOrEmpty($appName)) {
        throw "'appName' parameter is required"
    }
}
catch {
    Write-Host "Error:`n$_" -ForegroundColor "Red" -BackgroundColor "Black"
    Write-Host "`n$script:Usage`n"
    $ExitValue = 1
    exit $ExitValue
}

try {

    Get-Process -Name $appName -ErrorAction SilentlyContinue | Out-Null
    if (-Not $?) {
        throw "Cannot find a process with the name `"${appName}`"."
    }

    .\script\profiling.ps1 `
        -app $appName `
        -dataDir $script:dataDir `
        -profilingResult $script:profilingResult
    if (-Not $?) {
        throw "Error occured while profiling system."
    }

    .\script\reformat.ps1 `
        -dataDir $script:dataDir `
        -inputPath $script:profilingResult `
        -outputPath $script:featureValue
    if (-Not $?) {
        throw "Error occured while processing profiling results."
    }

    .\script\inference.exe `
        --dataDir $script:dataDir `
        --featureName $script:featureValue `
        --configDir "script/config" `
        --iopatternOutName $script:iopatternPrediction
    if (-Not $?) {
        throw "Error occured while running classification model."
    }

    .\script\setting_knob.ps1 `
        -dataDir $script:dataDir `
        -iopatternPrediction $script:iopatternPrediction `
        -ssacli $script:ssacli `
        -controllerSlot $script:controllerSlot `
        -logicalDiskID $script:logicalDiskID
    if (-Not $?) {
        throw "Error occured while looking up best system knob configuration."
    }

    Write-Host "Optimization completed successfully."
}
catch {
    Write-Host "$_ Script stopped."  -ForegroundColor "Red" -BackgroundColor "Black"
    Write-Host "Optimization failed."
    $ExitValue = 1
}

exit $ExitValue
