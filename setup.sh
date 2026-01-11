#!/bin/bash
# BASEDATA Setup Script
#
# Sets up Goose with the BASEDATA MCP server and PNNL API provider.
#
# Usage:
#   ./setup.sh
#   ./setup.sh --api-key YOUR_KEY

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
GOOSE_CONFIG_DIR="${HOME}/.config/goose"

# ═══════════════════════════════════════════════════════════════════════════════
# Parse arguments
# ═══════════════════════════════════════════════════════════════════════════════

API_KEY=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ═══════════════════════════════════════════════════════════════════════════════
# Check dependencies
# ═══════════════════════════════════════════════════════════════════════════════

echo "Checking dependencies..."

if ! command -v goose &> /dev/null; then
    echo "ERROR: Goose not found."
    echo "Install with: curl -fsSL https://github.com/block/goose/releases/download/stable/download_cli.sh | bash"
    exit 1
fi

if ! command -v python &> /dev/null; then
    echo "ERROR: Python not found."
    exit 1
fi

if ! python -c "import yaml" 2>/dev/null; then
    echo "Installing PyYAML..."
    pip install pyyaml
fi

if ! python -c "import mcp" 2>/dev/null; then
    echo "Installing MCP..."
    pip install mcp
fi

echo "✓ Dependencies OK"

# ═══════════════════════════════════════════════════════════════════════════════
# Setup Goose configuration
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "Setting up Goose configuration..."

mkdir -p "$GOOSE_CONFIG_DIR/custom_providers"

# Copy custom provider
cp "$SCRIPT_DIR/custom_providers/pnnl.json" "$GOOSE_CONFIG_DIR/custom_providers/"
echo "✓ Copied custom provider: pnnl"

# Create config.yaml with correct path
cat > "$GOOSE_CONFIG_DIR/config.yaml" << EOF
GOOSE_PROVIDER: pnnl
GOOSE_MODEL: claude-sonnet-4-20250514

extensions:
  developer:
    enabled: true
    name: developer
    type: builtin
  
  basedata:
    enabled: true
    name: basedata
    type: stdio
    cmd: python
    args:
      - ${SCRIPT_DIR}/mcp_server.py
    timeout: 300
EOF
echo "✓ Created config.yaml"

# ═══════════════════════════════════════════════════════════════════════════════
# Setup API key
# ═══════════════════════════════════════════════════════════════════════════════

if [ -n "$API_KEY" ]; then
    export PNNL_API_KEY="$API_KEY"
    echo "✓ API key set from argument"
elif [ -z "$PNNL_API_KEY" ]; then
    echo ""
    echo "Set your PNNL API key:"
    echo "  export PNNL_API_KEY='your-key-here'"
    echo ""
    echo "Or add to your shell profile (~/.bashrc, ~/.zshrc):"
    echo "  echo 'export PNNL_API_KEY=\"your-key-here\"' >> ~/.bashrc"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# Done
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "════════════════════════════════════════════════════════════"
echo "Setup complete!"
echo "════════════════════════════════════════════════════════════"
echo ""
echo "To start the pipeline builder:"
echo "  cd $SCRIPT_DIR"
echo "  goose session"
echo ""
echo "The .goosehints file in this directory will guide the agent."
echo ""
