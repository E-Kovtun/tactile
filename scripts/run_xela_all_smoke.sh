#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"${SCRIPT_DIR}/run_xela_dinov2_smoke.sh"
"${SCRIPT_DIR}/run_xela_mae_smoke.sh"
