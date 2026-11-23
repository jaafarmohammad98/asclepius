#!/bin/sh
# Test the install on a clean machine: build the wheel, install it in a fresh Debian container
# (Dockerfile) and run both examples end to end there (e2e.py), the full paper library included.
#
#   ANTHROPIC_API_KEY=sk-ant-... sh package/clean-install.sh
#
# The library and index are kept in a Docker volume (asclepius-e2e-data), so a second run skips
# the hour-long first build; `docker volume rm asclepius-e2e-data` starts from nothing again.
# ASCLEPIUS_LLM_MODEL picks the model (default claude-haiku-5-5, the cheapest).
set -eu
here=$(cd "$(dirname "$0")" && pwd)
wheel=$(python3 "$here/build.py")
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$wheel" "$context/asclepius.whl"
cp "$here/e2e.py" "$here/Dockerfile" "$context/"
docker build --quiet --build-arg WHEEL="$(basename "$wheel")" -t asclepius-e2e "$context"
docker run --rm \
    -e ANTHROPIC_API_KEY \
    -e ASCLEPIUS_LLM_MODEL="${ASCLEPIUS_LLM_MODEL:-claude-haiku-5-5}" \
    -e NCBI_API_KEY="${NCBI_API_KEY:-}" \
    -v asclepius-e2e-data:/home/researcher/.local/share/asclepius \
    asclepius-e2e
