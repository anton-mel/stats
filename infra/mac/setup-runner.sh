#!/bin/bash
set -euo pipefail
TOKEN="${1:?registration token}"
PLATFORM_ID="${PLATFORM_ID:?platform id from matrix/platforms.yaml}"
REPO="${REPO:-anton-mel/stats}"
RUNNER_VERSION="${RUNNER_VERSION:-2.321.0}"
DIR="${RUNNER_DIR:-$HOME/actions-runner-stats}"
mkdir -p "$DIR" && cd "$DIR"
[ -f run.sh ] || curl -sL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz" | tar xz
[ -d "$HOME/pie-stats/.git" ] || git clone -q https://github.com/pie-project/pie "$HOME/pie-stats"
command -v ollama >/dev/null || echo "install Ollama first: https://ollama.com/download"
./config.sh --unattended --url "https://github.com/$REPO" --token "$TOKEN" --name "$(scutil --get ComputerName)-$PLATFORM_ID-stats" \
  --labels "self-hosted,macos,$PLATFORM_ID" --work _work --replace
./svc.sh install && ./svc.sh start
