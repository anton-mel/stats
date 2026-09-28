#!/bin/bash
set -euo pipefail
TOKEN="${1:?usage: remove-runner.sh <remove-token>}"
DIR="${RUNNER_DIR:-$HOME/actions-runner-stats}"

step() { printf '\n\033[1m[%s/3] %s\033[0m\n' "$1" "$2"; }
ok() { printf '  \033[32m✓\033[0m %s\n' "$1"; }

[ -f "$DIR/config.sh" ] || { echo "No stats runner in $DIR."; exit 1; }
cd "$DIR"

step 1 "Stopping the runner service"
./svc.sh stop >/dev/null 2>&1 || true
./svc.sh uninstall >/dev/null 2>&1 || true
ok "service stopped and removed"

step 2 "Unregistering from GitHub"
./config.sh remove --token "$TOKEN" >/dev/null
ok "runner unregistered"

step 3 "Cleaning up"
cd "$HOME" && rm -rf "$DIR"
ok "removed $DIR (pie at ~/pie-stats and caches under ~/.cache/pie-stats are kept)"
printf '\n\033[32mDone.\033[0m This Mac is gone from the site.\n'
