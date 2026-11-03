#!/bin/bash
# update-lab-code.sh
# Make this Pi's copy of the lab code exactly match GitHub main, every boot.
# Local edits to tracked files are DISCARDED on purpose. Untracked files
# (e.g. student data) are left alone unless they clash with a tracked file.

set -u

REPO_URL="https://github.com/BSphysics/Experimental-physics-control-codes.git"
BRANCH="main"
REPO_DIR="/home/dorkmaster/Desktop/PHY1035"          # one fixed path on every Pi
STATUS_FILE="/home/dorkmaster/Desktop/update-status.txt"   # outside the repo
MAX_WAIT=600                                          # seconds before giving up

HOST=$(hostname)
export GIT_TERMINAL_PROMPT=0     # never hang waiting for a username/password

log() { echo "$(date '+%F %T') [$HOST] $*"; }

write_status() {   # $1 = SUCCESSFUL / FAILED, $2 = extra detail line
    cat > "$STATUS_FILE" <<EOF
Update $1
Pi: $HOST
Date: $(date)
$2
EOF
}

# Trust the repo even if ownership got mixed up (e.g. someone once ran sudo git)
if ! git config --global --get-all safe.directory 2>/dev/null | grep -qxF "$REPO_DIR"; then
    git config --global --add safe.directory "$REPO_DIR"
fi

try_update() {
    # No valid clone -> clone fresh
    if ! git -C "$REPO_DIR" rev-parse --verify HEAD >/dev/null 2>&1; then
        log "No valid repository at $REPO_DIR - cloning"
        if [ -e "$REPO_DIR" ]; then
            mv "$REPO_DIR" "$REPO_DIR.old-$(date +%Y%m%d-%H%M%S)"
        fi
        rm -rf "$REPO_DIR.tmp"
        timeout 180 git clone --branch "$BRANCH" "$REPO_URL" "$REPO_DIR.tmp" || return 1
        mv "$REPO_DIR.tmp" "$REPO_DIR"
        return 0
    fi

    # Stale lock left by a hard power-off kills every git command until removed.
    # Safe at boot: nothing else should be running git in this repo.
    rm -f "$REPO_DIR/.git/index.lock" "$REPO_DIR/.git/shallow.lock"

    git -C "$REPO_DIR" remote set-url origin "$REPO_URL"
    timeout 120 git -C "$REPO_DIR" fetch --prune origin "$BRANCH" || return 1
    git -C "$REPO_DIR" checkout -q -B "$BRANCH" "origin/$BRANCH" || return 1
    git -C "$REPO_DIR" reset -q --hard "origin/$BRANCH" || return 1
}

network_ok() {
    timeout 30 git ls-remote --heads "$REPO_URL" "$BRANCH" >/dev/null 2>&1
}

# Stagger 25 Pis that all power on together
sleep $((RANDOM % 20))

start=$SECONDS
delay=5
while true; do
    if try_update; then
        COMMIT=$(git -C "$REPO_DIR" log -1 --format='%h (%cd)' --date=format:'%Y-%m-%d %H:%M')
        log "Update complete: $COMMIT"
        write_status "SUCCESSFUL" "Commit: $COMMIT"
        exit 0
    fi

    # GitHub reachable but update still failed -> local repo is broken
    # (corruption after power cut, permissions, etc.). Move it aside and re-clone.
    if network_ok && [ -e "$REPO_DIR" ]; then
        log "Network OK but update failed - local repo looks broken, re-cloning"
        mv "$REPO_DIR" "$REPO_DIR.broken-$(date +%Y%m%d-%H%M%S)"
        continue
    fi

    if (( SECONDS - start > MAX_WAIT )); then
        log "Gave up after ${MAX_WAIT}s"
        write_status "FAILED" "Reason: could not reach GitHub or update repo within ${MAX_WAIT}s - see: journalctl -u update-lab-code -b"
        exit 1
    fi

    log "Not ready yet (no network / clock not synced?) - retrying in ${delay}s"
    sleep "$delay"
    (( delay < 60 )) && delay=$((delay * 2))
done
