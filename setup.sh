#!/bin/bash
# DSAGT Setup Script
#
# Usage:
#   export PNNL_API_KEY="your-key"
#   ./setup.sh
#   ./setup.sh --model gpt-4o-project
#
# See models.txt for available models

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOOSE_CONFIG_DIR="${HOME}/.config/goose"

# Defaults
MODEL="claude-sonnet-4-20250514-v1-project"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --list-models)
            cat "${SCRIPT_DIR}/models.txt"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: ./setup.sh [--model MODEL_NAME] [--list-models]"
            exit 1
            ;;
    esac
done

echo "DSAGT Setup"
echo "==========="
echo "Model: $MODEL"

# Check for API key
if [ -z "$PNNL_API_KEY" ]; then
    echo ""
    echo "ERROR: PNNL_API_KEY not set"
    echo ""
    echo "Run:"
    echo "  export PNNL_API_KEY='your-key-here'"
    echo "  ./setup.sh"
    exit 1
fi

# Check for Goose
if ! command -v goose &> /dev/null; then
    echo "ERROR: Goose not found."
    echo "Install with:"
    echo "  curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash"
    exit 1
fi

# Check Python deps
if ! python -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    pip install pyyaml
fi

# Create config directory
mkdir -p "$GOOSE_CONFIG_DIR"

# Write config.yaml
cat > "$GOOSE_CONFIG_DIR/config.yaml" << EOF
GOOSE_PROVIDER: openai
GOOSE_MODEL: ${MODEL}

extensions:
  developer:
    enabled: true
    name: developer
    type: builtin
  
  dsagt:
    enabled: true
    name: dsagt
    type: stdio
    cmd: python
    args:
      - ${SCRIPT_DIR}/mcp_server.py
    timeout: 300
EOF

echo "✓ Created config.yaml (model: ${MODEL})"

# Write shell profile hint
echo ""
echo "Add these to your shell profile (~/.bashrc or ~/.zshrc):"
echo ""
echo "  export PNNL_API_KEY=\"your-key\""
echo "  export OPENAI_API_KEY=\"\${PNNL_API_KEY}\""
echo "  export OPENAI_HOST=\"https://ai-incubator-api.pnnl.gov\""
echo ""
echo "Or run now:"
echo ""
echo "  export OPENAI_API_KEY=\"${PNNL_API_KEY}\""
echo "  export OPENAI_HOST=\"https://ai-incubator-api.pnnl.gov\""
echo ""
echo "Then start:"
echo "  cd ${SCRIPT_DIR}"
echo "  goose session"
