#!/usr/bin/env bash
set -euo pipefail

MODEL="deepseek-r1:1.5b"
CONFIG_DIR="db/local"
CONFIG_FILE="$CONFIG_DIR/.config.json"

# --- Ollama ---
if command -v ollama >/dev/null 2>&1; then
    echo "Ollama installed!"
else
    echo "Ollama not installed; Installing..."
    OS=$(uname -s)
    case "$OS" in
        Linux|Darwin)
            curl -fsSL https://ollama.com/install.sh | sh
            ;;
        *)
            echo "Unknown operating system: $OS"
            echo "On Windows, run: irm https://ollama.com/install.ps1 | iex"
            exit 1
            ;;
    esac
fi

# --- Ollama model ---
if ollama show "$MODEL" >/dev/null 2>&1; then
    echo "Model $MODEL already present."
else
    echo "Pulling model $MODEL..."
    ollama pull "$MODEL"
fi

# --- Python ---
if command -v python3 >/dev/null 2>&1; then
    PY=python3
elif command -v python >/dev/null 2>&1; then
    PY=python
else
    echo "Python not installed; Installing..."
    OS=$(uname -s)
    case "$OS" in
        Linux)
            if command -v apt-get >/dev/null 2>&1; then
                sudo apt-get update
                sudo apt-get install -y python3 python3-pip
            elif command -v dnf >/dev/null 2>&1; then
                sudo dnf install -y python3 python3-pip
            elif command -v yum >/dev/null 2>&1; then
                sudo yum install -y python3 python3-pip
            elif command -v pacman >/dev/null 2>&1; then
                sudo pacman -Sy --noconfirm python python-pip
            else
                echo "No supported package manager found; install python3 manually."
                exit 1
            fi
            ;;
        Darwin)
            if command -v brew >/dev/null 2>&1; then
                brew install python3
            else
                echo "Homebrew not found. Install from https://brew.sh, then re-run."
                exit 1
            fi
            ;;
        *)
            echo "Unknown operating system: $OS"
            exit 1
            ;;
    esac
    PY=python3
fi

# --- Python deps ---
if ! "$PY" -c "import ollama" >/dev/null 2>&1; then
    echo "Installing python ollama package..."
    "$PY" -m pip install --user ollama
fi

# --- Config file ---
mkdir -p "$CONFIG_DIR"
if [ ! -f "$CONFIG_FILE" ]; then
    cat > "$CONFIG_FILE" <<EOF
{
    "model": "$MODEL"
}
EOF
    echo "Created $CONFIG_FILE"
else
    echo "$CONFIG_FILE already exists; leaving as-is."
fi

pip install ddgs
echo "Setup complete."
clear
python __main__.py