#!/usr/bin/env bash
# Download subject reconstruction outputs from the remote server.
#
# Replaces the per-subject ~/Downloads/sub-XX/update_files.sh copies with a
# single repo-parameterized script (D-73). Reads the subject's remote
# nextflow.config to decide which output categories are enabled before
# downloading (Open Q3 / A4 in 07-RESEARCH.md).
#
# Usage: scripts/subject/update_files.sh SUBJECT_ID [REMOTE_HOST] [--dry-run]
set -euo pipefail

REMOTE_BASE="/scratch/workspace"
REMOTE_HOST="${REMOTE_HOST:-132.207.157.41}"
DRY_RUN=false
SUBJECT_ID=""

usage() {
    cat <<'EOF'
Usage: update_files.sh SUBJECT_ID [REMOTE_HOST] [--dry-run]

  SUBJECT_ID    Required. Must match ^sub-[0-9]+$ (e.g. sub-18).
  REMOTE_HOST   Optional positional override for the remote server
                (default: 132.207.157.41).
  --dry-run     Print the output categories that would be downloaded
                without contacting the remote host.
  -h, --help    Show this usage message and exit.

Environment:
  REMOTE_HOST           Overrides the default remote server.
  UPDATE_FILES_CONFIG   Path to a local nextflow.config to read instead of
                         fetching the remote config (used for --dry-run /
                         testing; never contacts a host when set).

Downloads subject reconstruction outputs from REMOTE_HOST into
~/Downloads/SUBJECT_ID/, reading REMOTE_BASE/SUBJECT_ID/nextflow.config on
the remote host to decide which output categories are enabled.
EOF
}

if [[ $# -eq 0 ]]; then
    usage >&2
    exit 1
fi

for arg in "$@"; do
    case "$arg" in
        -h|--help)
            usage
            exit 0
            ;;
    esac
done

SUBJECT_ID="$1"
shift

for arg in "$@"; do
    case "$arg" in
        --dry-run)
            DRY_RUN=true
            ;;
        *)
            REMOTE_HOST="$arg"
            ;;
    esac
done

# V5 / T-07-10: validate SUBJECT_ID before it touches any path or remote command.
if [[ ! "$SUBJECT_ID" =~ ^sub-[0-9]+$ ]]; then
    echo "Error: SUBJECT_ID must match ^sub-[0-9]+\$ (got: '${SUBJECT_ID}')" >&2
    usage >&2
    exit 1
fi

REMOTE_CONFIG_PATH="${REMOTE_BASE}/${SUBJECT_ID}/nextflow.config"
LOCAL_CONFIG="$(mktemp)"
trap 'rm -f "${LOCAL_CONFIG}"' EXIT

if [[ -n "${UPDATE_FILES_CONFIG:-}" ]]; then
    cp -- "${UPDATE_FILES_CONFIG}" "${LOCAL_CONFIG}"
elif [[ "${DRY_RUN}" == true ]]; then
    : > "${LOCAL_CONFIG}"
else
    scp -- "${REMOTE_HOST}:${REMOTE_CONFIG_PATH}" "${LOCAL_CONFIG}"
fi

config_flag_enabled() {
    local flag_name="$1"
    grep -Eq "^[[:space:]]*${flag_name}[[:space:]]*=[[:space:]]*true" "${LOCAL_CONFIG}"
}

# Category name -> remote path (relative to REMOTE_BASE/SUBJECT_ID).
# Uses case statements instead of associative arrays for bash 3.2 portability
# (macOS ships bash 3.2, which lacks `declare -A`).
category_remote_path() {
    case "$1" in
        common_space_previews) echo "output/common_space_previews" ;;
        stack) echo "output/stack" ;;
        correct_bias_field) echo "output/correct_bias_field" ;;
        register_pairwise) echo "output/register_pairwise" ;;
        align_to_ras) echo "output/align_to_ras" ;;
        detect_rehoming_events) echo "output/detect_rehoming_events" ;;
        mosaic_grids_previews) echo "mosaic-grids/previews" ;;
    esac
}

# Category name -> gating nextflow.config flag. Categories with no match here
# (empty output) are always downloaded (base outputs produced regardless of
# optional flags).
category_flag() {
    case "$1" in
        common_space_previews) echo "common_space_preview" ;;
        correct_bias_field) echo "correct_bias_field" ;;
        align_to_ras) echo "align_to_ras_enabled" ;;
        *) echo "" ;;
    esac
}

CATEGORY_ORDER=(
    common_space_previews
    stack
    correct_bias_field
    register_pairwise
    align_to_ras
    detect_rehoming_events
    mosaic_grids_previews
)

selected_categories=()
for category in "${CATEGORY_ORDER[@]}"; do
    flag="$(category_flag "${category}")"
    if [[ -z "${flag}" ]] || config_flag_enabled "${flag}"; then
        selected_categories+=("${category}")
    fi
done

if [[ "${DRY_RUN}" == true ]]; then
    echo "Would download the following output categories for ${SUBJECT_ID} from ${REMOTE_HOST}:"
    for category in "${selected_categories[@]}"; do
        echo "  - ${category} ($(category_remote_path "${category}"))"
    done
    exit 0
fi

LOCAL_DEST="${HOME}/Downloads/${SUBJECT_ID}"
mkdir -p -- "${LOCAL_DEST}"

for category in "${selected_categories[@]}"; do
    remote_rel_path="$(category_remote_path "${category}")"
    local_category_dest="${LOCAL_DEST}/${remote_rel_path}"
    mkdir -p -- "${local_category_dest}"
    echo "Downloading ${category} ..."
    rsync -avz -- "${REMOTE_HOST}:${REMOTE_BASE}/${SUBJECT_ID}/${remote_rel_path}/" "${local_category_dest}/"
done
