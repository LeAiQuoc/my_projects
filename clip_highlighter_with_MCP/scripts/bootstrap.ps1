param()

$ErrorActionPreference = "Stop"

function Test-SupportedPythonVersion {
	param([string]$VersionText)

	$parts = $VersionText.Split('.')
	if ($parts.Length -lt 2) {
		return $false
	}

	$major = [int]$parts[0]
	$minor = [int]$parts[1]
	return ($major -gt 3) -or ($major -eq 3 -and $minor -ge 11)
}

function Resolve-PythonInterpreter {
	$candidates = @(
		@("py", "-3.13"),
		@("py", "-3.12"),
		@("py", "-3.11"),
		@("python")
	)

	foreach ($candidate in $candidates) {
		$exe = $candidate[0]
		$args = @()
		if ($candidate.Length -gt 1) {
			$args += $candidate[1]
		}

		if (-not (Get-Command $exe -ErrorAction SilentlyContinue)) {
			continue
		}

		$output = & $exe @args -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}'); print(sys.executable)" 2>$null
		if ($LASTEXITCODE -ne 0 -or $null -eq $output -or $output.Count -lt 2) {
			continue
		}

		$versionText = [string]$output[0]
		$pythonExe = [string]$output[1]
		if (Test-SupportedPythonVersion -VersionText $versionText) {
			return @{
				Version = $versionText
				Exe = $exe
				Args = $args
				Path = $pythonExe
			}
		}
	}

	throw "No supported Python interpreter found. Install Python 3.11+ and ensure it is on PATH."
}

Write-Host "Setting up clip highlighter environment..."

$selected = Resolve-PythonInterpreter
Write-Host "Selected Python $($selected.Version): $($selected.Path)"

$venvPython = ".\.venv\Scripts\python.exe"
$recreateVenv = $false

if (Test-Path $venvPython) {
	$venvVersion = & $venvPython -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')"
	if (-not (Test-SupportedPythonVersion -VersionText $venvVersion)) {
		Write-Host ".venv uses unsupported Python $venvVersion. Recreating with Python 3.11+..."
		Remove-Item -Recurse -Force ".venv"
		$recreateVenv = $true
	}
} else {
	$recreateVenv = $true
}

if ($recreateVenv) {
	Write-Host "Creating .venv..."
	& $selected.Exe @($selected.Args + @("-m", "venv", ".venv"))
}

Write-Host "Upgrading pip in .venv..."
& $venvPython -m pip install --upgrade pip

Write-Host "Installing dependencies from requirements.txt..."
& $venvPython -m pip install -r requirements.txt

if (-not (Test-Path ".env")) {
	Write-Host "Creating .env from .env.example..."
	Copy-Item ".env.example" ".env"
}

Write-Host "Checking FFmpeg availability..."
ffmpeg -version | Select-Object -First 1 | Out-Null

Write-Host "Running tests..."
& $venvPython -m pytest -q

Write-Host "Bootstrap complete."
