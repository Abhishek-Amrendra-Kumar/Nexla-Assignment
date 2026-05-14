#!/bin/bash
set -e

# =============================================================================
# Nexla CLI — Minimal setup and build commands
# =============================================================================

command -v gdown >/dev/null 2>&1 || { echo "Installing gdown..."; uv pip install gdown; }

case "$1" in
  setup)
    if [ -z "$2" ]; then
      echo "Usage: $0 setup <google-drive-link>"
      exit 1
    fi
    echo "Downloading from Google Drive..."
    gdown --Folder "$2" -O /tmp/nexla_drive --fuzzy -r 2>/dev/null || gdown "$2" -O /tmp/nexla_drive.zip --fuzzy

    if [ -d /tmp/nexla_drive ]; then
      echo "Extracting folders..."
      cp -r /tmp/nexla_drive/data . 2>/dev/null || true
      cp -r /tmp/nexla_drive/chroma_db . 2>/dev/null || true
    elif [ -f /tmp/nexla_drive.zip ]; then
      echo "Extracting zip..."
      unzip -o /tmp/nexla_drive.zip -d /tmp/nexla_drive_extract
      cp -r /tmp/nexla_drive_extract/*/data . 2>/dev/null || true
      cp -r /tmp/nexla_drive_extract/*/chroma_db . 2>/dev/null || true
    fi

    echo "Setup complete. data/ and chroma_db/ are now in $(pwd)"
    ;;

  now)
    echo "Building Docker image..."
    docker build -t nexla-mcp .
    echo "Done. Run with: docker run -i --rm -e HF_TOKEN=... -e LITELLM_API_KEY=... -e LITELLM_BASE_URL=... -e LITELLM_MODEL=... nexla-mcp"
    ;;

  config)
    echo "=== Claude Desktop config ==="
    cat << 'EOF'
{
  "mcpServers": {
    "nexla-mcp": {
      "command": "docker",
      "args": [
        "run", "-i", "--rm", "--init", "--pull=never",
        "-e", "HF_TOKEN",
        "-e", "LITELLM_API_KEY",
        "-e", "LITELLM_BASE_URL",
        "-e", "LITELLM_MODEL",
        "nexla-mcp"
      ],
      "env": {
        "HF_TOKEN": "hf_***",
        "LITELLM_API_KEY": "sk_***",
        "LITELLM_BASE_URL": "https://api.minimax.io/v1",
        "LITELLM_MODEL": "MiniMax-M2.7-highspeed"
      }
    }
  }
}
EOF
    echo ""
    echo "=== OpenCode config ==="
    cat << 'EOF'
"nexla-mcp": {
  "type": "local",
  "command": [
    "docker", "run", "-i", "--rm", "--init", "--pull=never",
    "-e", "HF_TOKEN",
    "-e", "LITELLM_API_KEY",
    "-e", "LITELLM_BASE_URL",
    "-e", "LITELLM_MODEL",
    "nexla-mcp"
  ],
  "environment": {
    "HF_TOKEN": "hf_***",
    "LITELLM_API_KEY": "sk_***",
    "LITELLM_BASE_URL": "https://api.minimax.io/v1",
    "LITELLM_MODEL": "MiniMax-M2.7-highspeed"
  }
}
EOF
    ;;

  *)
    echo "Usage: $0 {setup|now|config}"
    echo "  setup <drive-link>  — Download and extract data + chroma_db from Google Drive"
    echo "  now                 — Build Docker image"
    echo "  config              — Show MCP client configuration"
    exit 1
    ;;
esac
