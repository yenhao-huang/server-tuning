#Requires -RunAsAdministrator
param (
    [String]$dataDir, # directory where the input file is located
    [String]$iopatternPrediction = "prediction.txt", # input file that contains the predicted I/O pattern
    [String]$ssacli = "C:\Program Files\Smart Storage Administrator\ssacli\bin\ssacli.exe",
    [int]$controllerSlot = 12, # slot number of the controller
    [int]$logicalDiskID = 1     # ID of the logical disk
)

$script:knobSettingPath = ".\script\config\knobMapping.json"
$script:iopattern
# Suggested configuration and corresponding commands shown to the user
$script:recommendedConfigStr = [ordered]@{}
# Commands for Windows operating system
$script:operatingSystemCommand = @()
# Commands for HPE storage controller
$script:storageControllerCli = "& `"$ssacli`" ctrl slot=$controllerSlot"
$script:storageControllerCommand = @()

function Import-IoPattern {
    <#
    This function reads the predicted I/O pattern from the file.
    #>
    $script:iopattern = Get-Content -Path (Join-Path $dataDir $iopatternPrediction)
}

function Format-RecommendedConfig ([int[]]$recommendedConfig) {
    $mc = switch ($recommendedConfig[0]) {
        0 { "Disable" }
        1 { "Enable" }
    }
    $script:operatingSystemCommand += "$mc-MMAgent -MemoryCompression"
    $script:recommendedConfigStr["$mc Memory Compression"] = $script:operatingSystemCommand[-1]

    $pc = switch ($recommendedConfig[1]) {
        0 { "Disable" }
        1 { "Enable" }
    }
    $script:operatingSystemCommand += "$pc-MMAgent -PageCombining"
    $script:recommendedConfigStr["$pc Page Combining"] = $script:operatingSystemCommand[-1]

    $dpo = switch ($recommendedConfig[2]) {
        0 { "disable" }
        1 { "enable" }
    }
    $script:storageControllerCommand += "$script:storageControllerCli modify dpo=$dpo"
    $script:recommendedConfigStr[
        (Get-Culture).TextInfo.ToTitleCase("$dpo Degraded Performance Optimization")
    ] = $script:storageControllerCommand[-1]

    $irp = switch ($recommendedConfig[3]) {
        0 { "disable" }
        1 { "enable" }
    }
    $script:storageControllerCommand += "$script:storageControllerCli modify irp=$irp"
    $script:recommendedConfigStr[
        (Get-Culture).TextInfo.ToTitleCase("$irp Inconsistency Repair Policy")
    ] = $script:storageControllerCommand[-1]

    $dwc = switch ($recommendedConfig[4]) {
        0 { "disable" }
        1 { "enable" }
    }
    $script:storageControllerCommand += "$script:storageControllerCli modify dwc=$dwc usage=configured"
    $script:recommendedConfigStr[
        (Get-Culture).TextInfo.ToTitleCase("$dwc Drive Write Cache")
    ] = $script:storageControllerCommand[-1]

    $qd = switch ($recommendedConfig[5]) {
        -1 { "automatic" }
        2 { "2" }
        32 { "32" }
    }
    $script:storageControllerCommand += "$script:storageControllerCli modify qd=$qd"
    $script:recommendedConfigStr[
    "Set Queue Depth to $qd"
    ] = $script:storageControllerCommand[-1]

    $mnpd = switch ($recommendedConfig[6]) {
        0 { "0" }
        1 { "15" }
    }
    $script:storageControllerCommand += "$script:storageControllerCli modify mnpd=$mnpd"
    $script:recommendedConfigStr[
    "Set Monitor and Performance Analysis Delay to $mnpd"
    ] = $script:storageControllerCommand[-1]

    switch ($recommendedConfig[7]) {
        0 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "disable"
        }
        1 {
            Set-ArrayAcceleratorAndCacheRatio "disable"
            Set-Ssdsmartpath "enable"
        }
        2 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "enable" "default"
        }
        3 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "enable" "100/0"
        }
        4 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "enable" "95/5"
        }
        5 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "enable" "5/95"
        }
        6 {
            Set-Ssdsmartpath "disable"
            Set-ArrayAcceleratorAndCacheRatio "enable" "0/100"
        }
    }
}

function Set-Ssdsmartpath ([String] $newSetting) {
    $currentSetting = Invoke-Expression "$script:storageControllerCli array a show" | `
        Select-String -pattern "Smart Path"

    $storageControllerCommand = "$script:storageControllerCli array a modify ssdsmartpath=$newSetting"
    $script:recommendedConfigStr[
        (Get-Culture).TextInfo.ToTitleCase("$newSetting SSD Smart Path")
    ] = $storageControllerCommand

    switch -Wildcard ($currentSetting) {
        "*disable*" {
            if ($newSetting -match "enable") {
                $script:storageControllerCommand += $storageControllerCommand
            }
        }
        "*enable*" {
            if ($newSetting -match "disable") {
                $script:storageControllerCommand += $storageControllerCommand
            }
        }
        default {
            throw "[SSACLI] ssdsmartpath setting error"
        }
    }
}

function Set-ArrayAcceleratorAndCacheRatio ([String] $aa, [String] $cr) {
    $script:storageControllerCommand += "$script:storageControllerCli ld $logicalDiskID modify aa=$aa"
    $script:recommendedConfigStr[
        (Get-Culture).TextInfo.ToTitleCase("$aa Array Accelerator")
    ] = $script:storageControllerCommand[-1]

    if ($aa -eq "enable") {
        $script:storageControllerCommand += "$script:storageControllerCli modify cr=$cr"
        $script:recommendedConfigStr[
        "Set Cache Ratio to $cr"
        ] = $script:storageControllerCommand[-1]
    }
}

function Import-RecommendedConfig {
    <#
    This function generates corresponding commands for both the operating system
    and the storage controller.
    #>
    $recommendedConfigs = Get-Content -Path $script:knobSettingPath | ConvertFrom-Json
    $recommendedConfig = $recommendedConfigs.$script:iopattern
    Format-RecommendedConfig $recommendedConfig
}

function Show-RecommendedConfig {
    Write-Host "I/O pattern: $iopattern"
    Write-Host "Recommended system knob configuration:"

    $table_heading = @{L = "Action"; E = { $_.Name } }, @{L = "Command"; E = { $_.Value } }
    $script:recommendedConfigStr |
    Format-Table -Property $table_heading -AutoSize
}

function Read-UserConfirm {
    $prompt = "Do you want to proceed? [y/n]"
    $confirmation = Read-Host $prompt
    while ($confirmation -ne "y") {
        if ($confirmation -eq "n") {
            break
        }
        $confirmation = Read-Host $prompt
    }
    return ($confirmation -eq "y")
}

function Set-SystemKnobs {
    foreach ($command in $script:operatingSystemCommand) {
        Invoke-Expression $command
    }
    foreach ($command in $script:storageControllerCommand) {
        Invoke-Expression "$command forced"
    }
}

### Main Program ###

$ExitValue = 0

try {
    Import-IoPattern
    if ($script:iopattern -eq "Unknown") {
        Write-Host "Sorry. The workload is unknown I/O pattern. Setting will not change."
        exit $ExitValue
    }

    Import-RecommendedConfig
    Show-RecommendedConfig

    $userConfirm = Read-UserConfirm
    if ($userConfirm) {
        Write-Host -NoNewline "Start setting system knobs..."
        Set-SystemKnobs
        Write-Host "Done"
    }
}
catch {
    Write-Host "`n$_" -ForegroundColor "Red" -BackgroundColor "Black"
    Write-Host $_.ScriptStackTrace
    $ExitValue = 1
}

exit $ExitValue
