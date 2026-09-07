#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    # shellcheck disable=SC1091
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
fi

python scripts/preflight_ascend.py --run-hccl
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests
python -m compileall -q dllm_cache eval_model utils \
    demo_LLaDA.py demo_Dream.py benchmark_text.py evaluation_script.py

for script in scripts/*.sh; do
    bash -n "${script}"
done

echo "Ascend validation completed successfully."
