#!/usr/bin/env bash
# Convenience wrapper: set up a venv and run prototype-v0.1.
set -euo pipefail
cd "$(dirname "$0")/.."

if [ ! -d .venv ]; then
    python3 -m venv .venv
fi
source .venv/bin/activate
pip install -q -e .

echo "--------------------------------------------------------------------"
echo "Reminders (see README):"
echo " * set hf.target_repo in configs/prototype.yaml to YOUR namespace"
echo " * hf auth login  (accept bigcode dataset terms for gated sources)"
echo " * AWS credentials enable the-stack-v2 content (Software Heritage S3)"
echo "--------------------------------------------------------------------"

exec ucc run --config configs/prototype.yaml "$@"
