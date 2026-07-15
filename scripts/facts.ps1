param(
    [Parameter(Position = 0)]
    [ValidateSet("add", "list", "validate")]
    [string]$Command = "list",

    [string]$FactsFile = "data/facts.yaml"
)

function Get-PythonCommand {
    $venvPython = Join-Path $PSScriptRoot "..\.venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return (Resolve-Path $venvPython).Path
    }
    return "python"
}

$python = Get-PythonCommand
$wizard = Join-Path $PSScriptRoot "facts_wizard.py"

& $python $wizard $Command --facts-file $FactsFile