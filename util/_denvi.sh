#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
pushd "$SCRIPT_DIR" > /dev/null || exit 1

# bash has no try/finally; trap on EXIT is the equivalent that guarantees the popd runs.
trap 'popd > /dev/null 2>&1 || true' EXIT

have_cmd() { command -v "$1" > /dev/null 2>&1; }

install_uv() {
    if have_cmd uv; then
        echo "uv already installed: $(uv --version)"
        return 0
    fi

    echo "Installing uv..."
    if ! curl -LsSf https://astral.sh/uv/install.sh | sh; then
        echo "Failed to install uv" >&2
        return 1
    fi

    export PATH="$HOME/.local/bin:$PATH"
    if ! have_cmd uv; then
        echo "uv installed but not found on PATH" >&2
        return 1
    fi
}

install_python() {
    if uv python list --only-installed 2>/dev/null | grep -q "cpython-3\.12\."; then
        echo "Python 3.12 already installed via uv"
        return 0
    fi

    echo "Installing Python 3.12 via uv..."
    uv python install 3.12
}

create_venv() {
    if [ -f ".venv/pyvenv.cfg" ] && grep -Eq "^version(_info)? = 3\.12" ".venv/pyvenv.cfg"; then
        echo ".venv already exists with Python 3.12"
        return 0
    fi

    echo "Creating .venv with Python 3.12..."
    uv venv --python 3.12 .venv
}

install_python_packages() {
    local packages=(pyyaml)
    local venv_python=".venv/bin/python"

    local installed
    installed="$(uv pip list --python "$venv_python" 2>/dev/null | tail -n +3 | awk '{print tolower($1)}')"

    local missing=()
    local pkg pkg_lower
    for pkg in "${packages[@]}"; do
        pkg_lower="$(printf '%s' "$pkg" | tr '[:upper:]' '[:lower:]')"
        if ! grep -qx "$pkg_lower" <<< "$installed"; then
            missing+=("$pkg")
        fi
    done

    if [ "${#missing[@]}" -eq 0 ]; then
        echo "Required Python packages already installed"
        return 0
    fi

    echo "Installing Python packages: ${missing[*]}"
    uv pip install --python "$venv_python" "${missing[@]}"
}

install_clang() {
    if have_cmd clang; then
        echo "clang already installed: $(clang --version | head -n1)"
        return 0
    fi

    echo "clang not found, attempting install..."
    if have_cmd apt-get; then
        sudo apt-get update && sudo apt-get install -y clang
    elif have_cmd dnf; then
        sudo dnf install -y clang
    elif have_cmd yum; then
        sudo yum install -y clang
    elif have_cmd pacman; then
        sudo pacman -S --noconfirm clang
    elif have_cmd apk; then
        sudo apk add --no-cache clang
    elif have_cmd brew; then
        brew install llvm
    else
        echo "No supported package manager found; please install clang manually." >&2
        return 1
    fi
}

install_uv || exit 1
install_python || exit 1
create_venv || exit 1
install_python_packages || exit 1
install_clang || true

echo "Setup complete."
