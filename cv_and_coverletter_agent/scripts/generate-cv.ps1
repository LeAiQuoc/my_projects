param(
    [Parameter(Position = 0, Mandatory = $true)]
    [string]$JobAdSource,

    [string]$FactsFile = "data/facts.yaml",
    [string]$StyleFile = "data/style_profile.json",
    [string]$Output
)

$script = Join-Path $PSScriptRoot "generate_outputs.ps1"
if ($Output) {
    & $script cv $JobAdSource -FactsFile $FactsFile -StyleFile $StyleFile -Output $Output
} else {
    & $script cv $JobAdSource -FactsFile $FactsFile -StyleFile $StyleFile
}