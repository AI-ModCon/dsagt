#!/usr/bin/env bash
# Install fastp and megahit for the microbial-isolates walkthrough.
#
# Creates a conda environment from environment.yml under the shared DSAgt
# tools directory, using micromamba (downloaded there if neither conda nor
# micromamba is on the PATH). Idempotent: rerunning updates the environment.
#
#   bash scripts/setup_env.sh
#
# Prints the environment's bin directory; the README's <CONDA_PREFIX> is that path.
set -euo pipefail

TOOLS_DIR="${DSAGT_TOOLS_DIR:-$HOME/dsagt-projects/.tools}"
ENV_DIR="$TOOLS_DIR/microbial_isolates/env"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$TOOLS_DIR/microbial_isolates"

if command -v micromamba >/dev/null; then
    MAMBA=micromamba
elif command -v conda >/dev/null; then
    MAMBA=conda
else
    MAMBA="$TOOLS_DIR/micromamba/bin/micromamba"
    if [ ! -x "$MAMBA" ]; then
        case "$(uname -s)-$(uname -m)" in
            Darwin-arm64)  PLATFORM=osx-arm64 ;;
            Darwin-x86_64) PLATFORM=osx-64 ;;
            Linux-x86_64)  PLATFORM=linux-64 ;;
            Linux-aarch64) PLATFORM=linux-aarch64 ;;
            *) echo "no micromamba build for $(uname -s)-$(uname -m); install conda and rerun" >&2; exit 1 ;;
        esac
        mkdir -p "$TOOLS_DIR/micromamba"
        curl -Ls "https://micro.mamba.pm/api/micromamba/$PLATFORM/latest" | tar -xj -C "$TOOLS_DIR/micromamba" bin/micromamba
    fi
    export MAMBA_ROOT_PREFIX="$TOOLS_DIR/micromamba/root"
fi

if [ "$MAMBA" = conda ]; then
    if [ -d "$ENV_DIR" ]; then
        conda env update -q -p "$ENV_DIR" -f "$HERE/environment.yml"
    else
        conda env create -q -p "$ENV_DIR" -f "$HERE/environment.yml"
    fi
else
    "$MAMBA" create -y -q -p "$ENV_DIR" -f "$HERE/environment.yml"
fi

"$ENV_DIR/bin/fastp" --version
"$ENV_DIR/bin/megahit" --version
echo
echo "CONDA_PREFIX for the walkthrough: $ENV_DIR/bin"
