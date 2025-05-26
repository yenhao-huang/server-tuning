Manual
===

The performance tuning tool can automatically identify the I/O pattern of a server workload with the MLP model and tweak the system knobs of the operating system and storage controller based on the optimal configuration to improve the performance of the server’s workload.

This user-friendly tool requires only one click (one command) from the user to launch the optimization. The optimization takes about 6 minutes to come into force because the tool requires 5 minutes to profile the system. If the workload does not match any pre-defined I/O pattern, the tool will stop conducting any undesired optimization and inform the user.

## Usage

The tool accepts one parameter, `$appName`, the name of the application to be optimized.

```powershell
## In PowerShell
./run.ps1 [appName]
```

> **NOTE**: This tool requires administrator privileges to run because it profiles and tunes the system. Make sure to run it in an elevated PowerShell session.

## Execution

The tool sequentially runs:

1. Profiling script (`profiling.ps1`)

    This script profiles the specified application using Windows Performance Monitor (`logman`) for five minutes.

2. Reformat script (`reformat.ps1`)

    This script formats the profiling result by converting a binary log file (.blg) to a comma-separated values file (.csv) with the `relog` command and removing some useless content.

3. Inference executable (`MLP_inference.exe`)

    This executable takes the formatted profiling result as input and identifies the I/O pattern of the specified application. 

4. Setting knob script (`setting_knob.ps1`)

    This script adjusts the settings of the operating system (Windows Server) and the storage controller (HPE SR416i-a Gen10+) based on the predicted I/O pattern.

The tool exits with an exit value of `0` if everything went well. Otherwise, it exits with an exit value of `1` and outputs informative error message.

### Tool Configuration

The script provides several variables for users to configure the script based on their needs, including:
- `$script:dataDir`: The directory where the data is stored.
- `$script:profilingResult`: The file of profiling result.
- `$script:featureValue`: The file of feature extracted from profiling result.
- `$script:ssacli`: The path to the Smart Storage Administrator Command Line Interface.
- `$script:iopatternPrediction`: The file of predicted I/O pattern.
- `$script:controllerSlot`: The slot number of the controller.
- `$script:logicalDiskID`: The ID of the logical disk.
