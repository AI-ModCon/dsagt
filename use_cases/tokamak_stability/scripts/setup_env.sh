#!/usr/bin/env bash
# Build and install fusion-io for the tokamak-stability walkthrough.
#
# Clones https://github.com/nferraro/fusion-io, builds it with CMake with the
# Python bindings on, and installs under the shared DSAgt tools directory.
# Idempotent: rerunning rebuilds from the existing clone. Prints the exports
# the walkthrough needs.
#
#   bash scripts/setup_env.sh
#
# Requires, already installed: git, cmake, pkg-config, C/C++/Fortran compilers,
# MPI, HDF5, NetCDF, and LAPACK. On macOS: brew install cmake pkgconf gcc
# open-mpi hdf5 netcdf (LAPACK comes with Accelerate). On Debian/Ubuntu:
# apt install cmake pkg-config gfortran libopenmpi-dev libhdf5-dev libnetcdf-dev
# liblapack-dev.
set -euo pipefail

TOOLS_DIR="${DSAGT_TOOLS_DIR:-$HOME/dsagt-projects/.tools}"
BASE="$TOOLS_DIR/tokamak_stability"
SRC="$BASE/fusion-io-src"
FIO_INSTALL_DIR="$BASE/fusion-io"
PYTHON="${PYTHON:-$(command -v python3)}"

missing=()
for tool in git cmake pkg-config mpicc mpicxx gfortran; do
    command -v "$tool" >/dev/null || missing+=("$tool")
done
if [ "${#missing[@]}" -gt 0 ]; then
    echo "missing build tools: ${missing[*]}" >&2
    echo "see the comment at the top of this script for the install commands" >&2
    exit 1
fi
if [ -n "${HDF5_ROOT:-}" ]; then
    :
elif command -v brew >/dev/null && brew --prefix --installed hdf5 >/dev/null 2>&1; then
    export HDF5_ROOT="$(brew --prefix hdf5)"
elif [ ! -e /usr/include/hdf5.h ] && [ ! -d /usr/include/hdf5 ]; then
    echo "HDF5 not found; set HDF5_ROOT to its install prefix" >&2
    exit 1
fi
# fusion-io locates HDF5 through pkg-config; a non-system prefix needs its .pc dir on the path.
if [ -n "${HDF5_ROOT:-}" ] && [ -d "$HDF5_ROOT/lib/pkgconfig" ]; then
    export PKG_CONFIG_PATH="$HDF5_ROOT/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
if command -v brew >/dev/null && brew --prefix --installed netcdf >/dev/null 2>&1; then
    export PKG_CONFIG_PATH="$(brew --prefix netcdf)/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
fi
pkg-config --exists netcdf || { echo "NetCDF not found by pkg-config; install it (see the comment at the top)" >&2; exit 1; }

mkdir -p "$BASE"
if [ -d "$SRC/.git" ]; then
    git -C "$SRC" pull -q --ff-only
else
    git clone -q https://github.com/nferraro/fusion-io.git "$SRC"
fi

mkdir -p "$SRC/build"
cmake -S "$SRC" -B "$SRC/build" \
    -DCMAKE_INSTALL_PREFIX="$FIO_INSTALL_DIR" \
    -DFUSIONIO_ENABLE_PYTHON=ON \
    -DPYTHON_MODULE_INSTALL_PATH="$FIO_INSTALL_DIR/lib" \
    -DPython_EXECUTABLE="$PYTHON"
cmake --build "$SRC/build" --target install -j "$(getconf _NPROCESSORS_ONLN)"

# write_neo_input (used for flux-surface quantities such as q and the Miller
# parameters) is a Makefile-only utility that the CMake build skips; compile it
# against the installed libraries. The source uses MPI without including mpi.h.
mpicxx -O2 -std=c++11 -include mpi.h "$SRC/util/write_neo_input.cpp" \
    -I"$FIO_INSTALL_DIR/include" -I"$SRC/fusion_io" -I"$SRC/m3dc1_lib" \
    $(pkg-config --cflags hdf5 netcdf) \
    -L"$FIO_INSTALL_DIR/lib" -lfusionio_fusionio -lfusionio_m3dc1 \
    $(pkg-config --libs hdf5 netcdf) \
    -o "$FIO_INSTALL_DIR/bin/write_neo_input"

case "$(uname -s)" in
    Darwin) LIBVAR=DYLD_LIBRARY_PATH ;;
    *)      LIBVAR=LD_LIBRARY_PATH ;;
esac
cat <<MSG

fusion-io installed under $FIO_INSTALL_DIR. Add to your shell before starting the session:

export FIO_INSTALL_DIR=$FIO_INSTALL_DIR
export PATH=\$FIO_INSTALL_DIR/bin:\$PATH
export PYTHONPATH=\$FIO_INSTALL_DIR/lib:\$PYTHONPATH
export $LIBVAR=\$FIO_INSTALL_DIR/lib:\${$LIBVAR:-}
MSG
