#!/bin/sh
# Test the install on a clean machine: build the wheel, install it in a fresh Debian container
# (Dockerfile) and run both examples end to end there (e2e.py), the full paper library included.
#
#   ANTHROPIC_API_KEY=sk-ant-... sh package/clean-install.sh
#   ASCLEPIUS_LLM_BACKEND=local ASCLEPIUS_LLM_MODEL=llama3.2:3b sh package/clean-install.sh
#
# The library and index are kept in a Docker volume (asclepius-e2e-data), so a second run skips
# the hour-long first build; `docker volume rm asclepius-e2e-data` starts from nothing again.
# ASCLEPIUS_LLM_MODEL picks the model (with the Anthropic API, default claude-haiku-5-5, the
# cheapest). With ASCLEPIUS_LLM_BACKEND=local the container reaches a model server on this computer
# (Ollama by default) at host.docker.internal; ASCLEPIUS_LLM_URL names another.
set -eu
backend=${ASCLEPIUS_LLM_BACKEND:-anthropic}
if [ "$backend" = local ]; then
    model=${ASCLEPIUS_LLM_MODEL:?name the local model, e.g. ASCLEPIUS_LLM_MODEL=llama3.2:3b}
    url=${ASCLEPIUS_LLM_URL:-http://host.docker.internal:11434/v1}
else
    model=${ASCLEPIUS_LLM_MODEL:-claude-haiku-5-5}
    url=
fi
here=$(cd "$(dirname "$0")" && pwd)
wheel=$(python3 "$here/build.py")
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$wheel" "$context/asclepius.whl"
cp "$here/e2e.py" "$here/Dockerfile" "$context/"
docker build --quiet --build-arg WHEEL="$(basename "$wheel")" -t asclepius-e2e "$context"
docker run --rm \
    -e ANTHROPIC_API_KEY \
    -e ASCLEPIUS_LLM_BACKEND="$backend" \
    -e ASCLEPIUS_LLM_MODEL="$model" \
    -e ASCLEPIUS_LLM_URL="$url" \
    -e NCBI_API_KEY="${NCBI_API_KEY:-}" \
    -v asclepius-e2e-data:/home/researcher/.local/share/asclepius \
    asclepius-e2e
