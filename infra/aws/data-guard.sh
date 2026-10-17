#!/usr/bin/env bash
# Data-safety guard: refuse to deploy if MIMIC-derived data could be on this
# server (CLAUDE.md hard rule 1; the PhysioNet DUA).
#
#   infra/aws/data-guard.sh [repo-dir]      exit 0 = clean, 1 = refused
#
# Only the synthetic demo files and the synthetic seed may reach the server.
# It refuses when the checkout, or the checkpoints/ folder, holds:
#   - anything matching CLAUDE.md's off-limits patterns: *_listfile.csv,
#     mimic-cxr-*.csv, discharge.csv, radiology.csv, *.pkl, the vendored
#     medpatch/mimic4extract/, medpatch/ehr_utils/mimic-cxr-ehr-split.csv,
#     valset*.csv, any file under data/;
#   - any *.csv other than the four synthetic demo vitals files;
#   - other tabular or medical-image dumps (*.parquet, *.feather, *.dcm,
#     *.nii*, *.csv.gz, *.pkl.gz, *.h5);
#   - the two demo PNGs of unknown provenance (docs/demo-examples/*.png);
#   - in checkpoints/: anything but *.pth.tar weights and their run.json
#     (the shared Drive folders also hold r3_*.csv tables and *.log files,
#     which must not be copied here).
# It prints file names only, never contents.
set -uo pipefail

REPO="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
cd "$REPO" || { echo "data-guard: no such directory: $REPO" >&2; exit 1; }

ALLOWED_CSV=(
  "docs/demo-examples/mortality_high_risk_vitals.csv"
  "docs/demo-examples/mortality_low_risk_vitals.csv"
  "docs/demo-examples/pneumonia_control_vitals.csv"
  "docs/demo-examples/pneumonia_sick_vitals.csv"
)

problems=()
add() { problems+=("$1"); }

# Every file in the working tree (tracked or not), minus git internals and
# dependency trees. Paths are relative, without the leading ./
mapfile -d '' FILES < <(find . \
  \( -path ./.git -o -path ./node_modules -o -path '*/node_modules' -o -path ./.venv \) -prune \
  -o -type f -print0 | sed -z 's#^\./##')

is_allowed_csv() {
  local f="$1" a
  for a in "${ALLOWED_CSV[@]}"; do [[ "$f" == "$a" ]] && return 0; done
  return 1
}

for f in "${FILES[@]}"; do
  base="${f##*/}"
  lower="${f,,}"
  case "$f" in
    medpatch/mimic4extract/*)            add "$f  (vendored MIMIC extraction resources)"; continue ;;
    medpatch/ehr_utils/mimic-cxr-ehr-split.csv) add "$f  (MIMIC split file)"; continue ;;
    data/.gitkeep)                       continue ;;
    data/*)                              add "$f  (data/ must stay empty on the server)"; continue ;;
    docs/demo-examples/*.png)            add "$f  (demo X-ray of unknown provenance)"; continue ;;
  esac
  case "$base" in
    *_listfile.csv|mimic-cxr-*.csv|discharge.csv|radiology.csv|valset*.csv)
      add "$f  (CLAUDE.md off-limits pattern)"; continue ;;
  esac
  case "$lower" in
    *.pkl|*.pickle|*.pkl.gz)             add "$f  (pickle: off-limits)"; continue ;;
    *.parquet|*.feather|*.dcm|*.nii|*.nii.gz|*.csv.gz|*.h5|*.hdf5)
      add "$f  (data dump)"; continue ;;
    *.csv)
      if ! is_allowed_csv "$f"; then add "$f  (only the 4 synthetic demo CSVs may be here)"; fi
      continue ;;
  esac
  if [[ "$f" == checkpoints/* ]]; then
    case "$f" in
      checkpoints/.gitkeep|*/run.json|*.pth.tar) ;;
      *) add "$f  (checkpoints/ holds only *.pth.tar and run.json)" ;;
    esac
  fi
done

# A full (non-sparse) clone would also carry the off-limits files in .git.
if [[ -d .git ]] && git rev-parse --is-inside-work-tree >/dev/null 2>&1; then
  if [[ "$(git config --get core.sparseCheckout || true)" != "true" ]]; then
    add ".git  (not a sparse checkout: clone with infra/aws/bootstrap.sh, which leaves the MIMIC-derived files out)"
  fi
fi

if ((${#problems[@]})); then
  echo "data-guard: REFUSED -- this server must hold no MIMIC-derived data:" >&2
  printf '  %s\n' "${problems[@]}" >&2
  echo "Remove these files (never copy them here), then deploy again." >&2
  exit 1
fi
echo "data-guard: OK (${#FILES[@]} files checked; only synthetic data and weights present)"
