#!/usr/bin/env bash
# scripts/watch_and_restart.sh
#
# Polls the SENTRY repo for new commits on the tracked branch and restarts
# the supervised orchestrator_api service when one lands, so the API is
# always serving whatever's actually merged — without anyone needing to
# SSH in and/or restart it by hand.

set -euo pipefail

REPO_DIR="${SENTRY_REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
BRANCH="${SENTRY_DEPLOY_BRANCH:-main}"
PM2_APP_NAME="${SENTRY_API_PM2_NAME:-sentry-api}"
POLL_SECONDS="${SENTRY_WATCH_POLL_SECONDS:-30}"

cd "$REPO_DIR"

echo "[watcher] watching $REPO_DIR ($BRANCH), restarting pm2 process '$PM2_APP_NAME' on new commits, polling every ${POLL_SECONDS}s"

last_commit="$(git rev-parse "$BRANCH" 2>/dev/null || echo "")"

while true; do
    sleep "$POLL_SECONDS"

    git fetch origin "$BRANCH" --quiet || { echo "[watcher] git fetch failed, will retry"; continue; }

    remote_commit="$(git rev-parse "origin/$BRANCH")"
    if [ "$remote_commit" != "$last_commit" ]; then
        echo "[watcher] new commit on $BRANCH: ${last_commit:-none} -> $remote_commit"

        git merge --ff-only "origin/$BRANCH" --quiet || {
            echo "[watcher] fast-forward merge failed (local changes on $BRANCH?) — not restarting"
            continue
        }

        echo "[watcher] restarting $PM2_APP_NAME"
        pm2 restart "$PM2_APP_NAME"

        last_commit="$remote_commit"
    fi
done
