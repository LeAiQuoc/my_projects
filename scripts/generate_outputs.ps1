param(
    [Parameter(Position = 0, Mandatory = $true)]
    [ValidateSet("cv", "cl")]
    [string]$Command,

    [Parameter(Position = 1, Mandatory = $true)]
    [string]$JobAdSource,

    [string]$FactsFile = "data/facts.yaml",
    [string]$StyleFile = "data/style_profile.json",
    [string]$Output
)

function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }
    return "python"
}

$python = Get-PythonCommand
$cliCommand = if ($Command -eq "cv") { "generate-cv" } else { "generate-cl" }

$args = @(
    "-m",
    "src.main",
    $cliCommand,
    $JobAdSource,
    "--facts-file",
    $FactsFile,
    "--style-file",
    $StyleFile
)

if ($Output) {
    $args += @("--output", $Output)
}

& $python @args