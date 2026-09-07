#!/usr/bin/env bash
set -euo pipefail

# Reads the pinned revision manifest and downloads snapshots serially.  Honor
# MODEL_REVISION_MANIFEST, HF_HUB_CACHE, and HF_HOME when they are supplied.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
MANIFEST="${MODEL_REVISION_MANIFEST:-${REPO_ROOT}/configs/model_revisions.yaml}"
if [[ -n "${HF_HUB_CACHE:-}" ]]; then
    CACHE_DIR="${HF_HUB_CACHE}"
elif [[ -n "${HF_HOME:-}" ]]; then
    CACHE_DIR="${HF_HOME}/hub"
else
    CACHE_DIR="${HOME}/.cache/huggingface/hub"
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "error: the 'hf' CLI is missing; activate the Ascend environment first" >&2
    exit 1
fi
if [[ ! -f "${MANIFEST}" ]]; then
    echo "error: model revision manifest not found: ${MANIFEST}" >&2
    exit 1
fi

MODEL_SPECS_OUTPUT="$(python - "${MANIFEST}" <<'PY'
import sys
from pathlib import Path

import yaml

manifest = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema_version") != 1 or not isinstance(manifest.get("models"), dict):
    raise ValueError("expected a schema_version: 1 manifest with a models mapping")
for name, spec in manifest["models"].items():
    if not spec.get("repo_id") or not spec.get("revision"):
        raise ValueError(f"model {name!r} is missing repo_id or revision")
    print(f'{name}\t{spec["repo_id"]}\t{spec["revision"]}')
PY
)"
mapfile -t MODEL_SPECS <<<"${MODEL_SPECS_OUTPUT}"
if [[ ${#MODEL_SPECS[@]} -eq 0 || -z "${MODEL_SPECS[0]}" ]]; then
    echo "error: the model revision manifest contains no models" >&2
    exit 1
fi

mkdir -p "${CACHE_DIR}"
for spec in "${MODEL_SPECS[@]}"; do
    IFS=$'\t' read -r name repo_id revision <<<"${spec}"
    echo "Downloading ${name}: ${repo_id}@${revision}"
    hf download "${repo_id}" \
        --revision "${revision}" \
        --cache-dir "${CACHE_DIR}"
done

echo "All pinned model snapshots are available in ${CACHE_DIR}."
