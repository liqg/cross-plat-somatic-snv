#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tests/test_smoke.py -v
