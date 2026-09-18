$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $ScriptDir

function Test-CommandExists {
    param([string]$Name)
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Install-Uv {
    if (Test-CommandExists "uv") {
        Write-Host "uv already installed: $(uv --version)"
        return
    }

    Write-Host "Installing uv..."
    Invoke-Expression (Invoke-RestMethod https://astral.sh/uv/install.ps1)

    $localBin = Join-Path $env:USERPROFILE ".local\bin"
    if ((Test-Path $localBin) -and ($env:PATH -notlike "*$localBin*")) {
        $env:PATH = "$localBin;$env:PATH"
    }

    if (-not (Test-CommandExists "uv")) {
        throw "uv installed but not found on PATH"
    }
}

function Install-Python312 {
    $installed = uv python list --only-installed 2>$null | Select-String "cpython-3\.12\."
    if ($installed) {
        Write-Host "Python 3.12 already installed via uv"
        return
    }

    Write-Host "Installing Python 3.12 via uv..."
    uv python install 3.12
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python 3.12 via uv"
    }
}

function New-VenvIfMissing {
    $cfgPath = Join-Path ".venv" "pyvenv.cfg"
    if ((Test-Path $cfgPath) -and (Select-String -Path $cfgPath -Pattern "^version(_info)? = 3\.12" -Quiet)) {
        Write-Host ".venv already exists with Python 3.12"
        return
    }

    Write-Host "Creating .venv with Python 3.12..."
    uv venv --python 3.12 .venv
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to create .venv"
    }
}

function Install-PythonPackages {
    $packages = @("pyyaml")
    $venvPython = Join-Path ".venv" "Scripts\python.exe"

    $installed = @(uv pip list --python $venvPython 2>$null | Select-Object -Skip 2 | ForEach-Object { ($_ -split "\s+")[0].ToLower() })

    $missing = $packages | Where-Object { $installed -notcontains $_.ToLower() }

    if (-not $missing) {
        Write-Host "Required Python packages already installed"
        return
    }

    Write-Host "Installing Python packages: $($missing -join ', ')"
    uv pip install --python $venvPython @missing
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to install Python packages"
    }
}

function Install-Clang {
    if (Test-CommandExists "clang") {
        $version = (clang --version) -split "`n" | Select-Object -First 1
        Write-Host "clang already installed: $version"
        return
    }

    Write-Host "clang not found, attempting install..."
    if (Test-CommandExists "winget") {
        winget install --id LLVM.LLVM -e --accept-source-agreements --accept-package-agreements
    }
    elseif (Test-CommandExists "choco") {
        choco install llvm -y
    }
    else {
        Write-Warning "No supported package manager found (winget/choco); please install clang manually."
    }
}

try {
    Install-Uv
    Install-Python312
    New-VenvIfMissing
    Install-PythonPackages
    try {
        Install-Clang
    }
    catch {
        Write-Warning "clang install step failed: $_"
    }

    Write-Host "Setup complete."
}
catch {
    Write-Error "Setup failed: $_"
    exit 1
}
finally {
    Pop-Location
}
