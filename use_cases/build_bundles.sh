#!/usr/bin/env bash
# Build one tarball per walkthrough: the data and every file the walkthrough's
# Setup puts into the project, laid out as the project directory, so Setup is
# one download and one `tar x` and needs no clone of this repository. The
# files stay in the repository as the source; the reference artifacts
# (`reference/`, `expected_*`) stay out of the tarballs, so an agent working in
# the project cannot read a finished answer.
#
# Usage: use_cases/build_bundles.sh <data_dir> <out_dir>
#   <data_dir> holds the large inputs that are not in the repository:
#     combustion_simulation_data.tar.gz   (data/blastnet_data, data/holdout)
#     tokamak_stability.tar.gz            (tokamak_stability/m3dc1_data)
#     microbial_isolates_reads.tar        (microbial_isolate/ reads)
#   <out_dir> receives <case>.tar.gz for each walkthrough with repository
#   assets. Upload each to the use-case Drive folder and put its file id in
#   the walkthrough README's Setup.
set -euo pipefail

DATA_DIR=$(cd "$1" && pwd)
OUT_DIR=$(mkdir -p "$2" && cd "$2" && pwd)
HERE=$(cd "$(dirname "$0")" && pwd)
STAGE=$(mktemp -d)
trap 'rm -rf "$STAGE"' EXIT

pack() {  # pack <case>: tar the staged project layout, without macOS metadata
  local case=$1
  find "$STAGE/$case" -name .DS_Store -delete
  find "$STAGE/$case" -name __pycache__ -type d -prune -exec rm -rf {} +
  COPYFILE_DISABLE=1 tar czf "$OUT_DIR/$case.tar.gz" -C "$STAGE/$case" .
  echo "$case.tar.gz  $(du -h "$OUT_DIR/$case.tar.gz" | cut -f1)"
}

# combustion_simulation: data, the two format documents, the checker. The
# holdout reference sits at the top level, outside data/, so Setup can leave it
# out of the project and step 5 can bring it in.
c=combustion_simulation; mkdir -p "$STAGE/$c/docs" "$STAGE/$c/skills/check-well-output/scripts" "$STAGE/$c/well_output"
tar xzf "$DATA_DIR/combustion_simulation_data.tar.gz" -C "$STAGE/$c"
mv "$STAGE/$c/data/holdout" "$STAGE/$c/holdout"
cp "$HERE/$c"/docs/*.md "$STAGE/$c/docs/"
cp "$HERE/$c/scripts/check_well_output.py" "$STAGE/$c/skills/check-well-output/scripts/"
pack $c

# genesis_skills: the dataset and the domain documents.
c=genesis_skills; mkdir -p "$STAGE/$c/mock_data"
cp -r "$HERE/$c/data/dataset" "$HERE/$c/data/domain" "$STAGE/$c/mock_data/"
pack $c

# microbial_isolates: the reads with their README, the two documents, the
# environment setup. The reads' README is the repository's copy, whose assembly
# note points at the best-practices document.
c=microbial_isolates; mkdir -p "$STAGE/$c/docs" "$STAGE/$c/setup"
tar xf "$DATA_DIR/microbial_isolates_reads.tar" -C "$STAGE/$c" --strip-components=1
cp "$HERE/$c/docs/reads_README.md" "$STAGE/$c/data/microbial_isolate/README.md"
cp "$HERE/$c/docs/fastp_megahit_best_practices.md" "$HERE/$c/docs/genomics.md" "$STAGE/$c/docs/"
cp "$HERE/$c/scripts/setup_env.sh" "$HERE/$c/scripts/environment.yml" "$STAGE/$c/setup/"
pack $c

# plasma_turbulence: the skill.
c=plasma_turbulence; mkdir -p "$STAGE/$c/skills"
cp -r "$HERE/$c/skills/xgc-ai-training" "$STAGE/$c/skills/"
pack $c

# tokamak_stability: the simulation output, the modules with their tests, the skill.
c=tokamak_stability; mkdir -p "$STAGE/$c/data" "$STAGE/$c/skills"
tar xzf "$DATA_DIR/tokamak_stability.tar.gz" -C "$STAGE/$c/data" --strip-components=1 tokamak_stability/m3dc1_data
cp -r "$HERE/$c/scripts" "$STAGE/$c/scripts"
cp -r "$HERE/$c/skills/m3dc1-skill" "$STAGE/$c/skills/"
pack $c

# vasp_dft: the slab and NEB calculations and the records the prompts compare against.
c=vasp_dft; mkdir -p "$STAGE/$c"
cp -r "$HERE/$c/data" "$STAGE/$c/data"
pack $c
