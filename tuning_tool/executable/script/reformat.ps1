param (
    [String]$dataDir, # directory where the input and output files are located.
    [String]$inputFile = "workload.blg", # input file that contains the binary log data.
    [String]$outputFile = "performance_log.csv" # output file that will store the converted data in CSV format.
)

$script:Usage = "Usage: `n" + `
    "$PSCommandPath " + `
    "-dataDir [DATA FOLDER] " + `
    "-inputPath [INPUT PATH] " + `
    "-outputPath [OUTPUT PATH]"

function Convert-BlgToCsv ([String]$inputPath, [String]$outputPath) {
    <#
    This function converts the binary log data to CSV data.
    #>
    $message = relog $inputPath -f csv -o $outputPath -y | Out-String
    if ($message -notmatch $outputFile) {
        throw "[relog] $($message.trim())"
    }
}

function Remove-UselessPrefix ([String] $profilingResultCsv) {
    <#
    This function removes the unnecessary prefix from the header of the CSV file.
    #>
    $temp = Get-Content $profilingResultCsv
    $r = "\\\\WIN-\w{11}\\"
    $temp[0] = $temp[0] -replace $r, ""
    Out-File -FilePath $profilingResultCsv -Encoding "UTF8" -InputObject $temp
}

function Update-ProfilingResultCsv ([String]$filePath) {
    <#
    This function call related functions to update the profiling result.
    #>
    Remove-UselessPrefix $filePath
}

### Main Program ###

$ExitValue = 0

try {
    if ([string]::IsNullOrEmpty($dataDir)) {
        throw "'dataDir' parameter is required"
    }
}
catch {
    Write-Host "Error:`n$_" -ForegroundColor "Red" -BackgroundColor "Black"
    Write-Host "`n$script:Usage`n"
    $ExitValue = 1
    exit $ExitValue
}

try {
    $blgFile = Join-Path $dataDir $inputFile
    $csvFile = Join-Path $dataDir $outputFile
    Convert-BlgToCsv $blgFile $csvFile
    Update-ProfilingResultCsv $csvFile
}
catch {
    Write-Host "`n$_" -ForegroundColor "Red" -BackgroundColor "Black"
    $ExitValue = 1
}

exit $ExitValue
