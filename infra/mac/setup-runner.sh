#!/bin/bash
set -euo pipefail
TOKEN="${1:?usage: setup-runner.sh <registration-token>}"
REPO="${REPO:-anton-mel/stats}"
RUNNER_VERSION="${RUNNER_VERSION:-2.321.0}"
DIR="${RUNNER_DIR:-$HOME/actions-runner-stats}"

step() { printf '\n\033[1m[%s/5] %s\033[0m\n' "$1" "$2"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m!\033[0m %s\n' "$1"; }

[ "$(uname -s)" = Darwin ] && [ "$(uname -m)" = arm64 ] || { echo "This script sets up Apple Silicon Macs only."; exit 1; }

step 1 "Identifying this Mac"
CHIP=$(sysctl -n machdep.cpu.brand_string)
GIB=$(( $(sysctl -n hw.memsize) / 1073741824 ))
BASE=$(echo "$CHIP" | sed -E 's/^Apple //' | tr '[:upper:]' '[:lower:]' | tr ' ' '-')
if [ -n "${PLATFORM_ID:-}" ]; then :
elif [ "$BASE" = m2-max ]; then PLATFORM_ID=m2-max
else PLATFORM_ID="$BASE-${GIB}g"; fi
ok "$CHIP, ${GIB} GB unified memory -> platform $PLATFORM_ID"
if ! curl -fsSL "https://raw.githubusercontent.com/$REPO/main/matrix/platforms.yaml" | grep -q "id: $PLATFORM_ID$"; then
  warn "$PLATFORM_ID is not in matrix/platforms.yaml yet; the runner will connect but no benchmark targets it until it is added"
fi

step 2 "Checking tools"
for t in git uv cargo ollama; do
  if command -v "$t" >/dev/null; then ok "$t"; else warn "$t not found (install it before the first run)"; fi
done
pmset -g batt | grep -q "AC Power" && ok "on AC power" || warn "on battery: runs on battery are recorded as noisy"

step 3 "Fetching pie and the runner"
[ -d "$HOME/pie-stats/.git" ] && ok "pie checkout at ~/pie-stats" || { git clone -q https://github.com/pie-project/pie "$HOME/pie-stats"; ok "cloned pie to ~/pie-stats"; }
mkdir -p "$DIR" && cd "$DIR"
[ -f run.sh ] && ok "runner already in $DIR" || { curl -fsSL "https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-osx-arm64-${RUNNER_VERSION}.tar.gz" | tar xz; ok "runner $RUNNER_VERSION in $DIR"; }

step 4 "Registering with github.com/$REPO"
NAME="$(scutil --get ComputerName | tr ' ' '-')-$PLATFORM_ID"
./config.sh --unattended --url "https://github.com/$REPO" --token "$TOKEN" --name "$NAME" \
  --labels "self-hosted,macos,$PLATFORM_ID" --work _work --replace >/dev/null
ok "registered as $NAME"
SDK=$(readlink -f /Library/Developer/CommandLineTools/SDKs/MacOSX.sdk 2>/dev/null || xcrun --show-sdk-path)
{ grep -v '^SDKROOT=' .env 2>/dev/null; echo "SDKROOT=$SDK"; } > .env.new && mv .env.new .env
grep -q "$HOME/.cargo/bin" .path 2>/dev/null || { printf '%s:' "$HOME/.cargo/bin"; cat .path 2>/dev/null; } > .path.new && [ -f .path.new ] && mv .path.new .path
ok "builds link against $(basename "$SDK") with cargo from ~/.cargo/bin"

step 5 "Starting the runner service"
./svc.sh install >/dev/null 2>&1 || true
./svc.sh start >/dev/null
ok "service running; it starts again at login"
printf '\n\033[32mDone.\033[0m This Mac shows up as connected on the site in a few seconds.\n'
