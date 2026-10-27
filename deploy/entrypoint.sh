#!/bin/sh
# Container entrypoint: checks the settings, fixes volume permissions, drops root privileges and
# starts the agent with its dashboard listening on $PORT. Extra arguments go to `btcagent run`.
set -eu

STATE_DIR="${BTCAGENT_STATE_DIR:-/data}"

if [ -z "${DASHBOARD_TOKEN:-}" ]; then
    cat >&2 <<'MSG'
ERROR: the DASHBOARD_TOKEN environment variable is not set.
It is the password that protects your dashboard, including its kill switch. Set it to a long random
value of at least 16 characters, for example the output of:  openssl rand -hex 24
MSG
    exit 1
fi

if [ "$(id -u)" = "0" ]; then
    # Hosting platforms mount volumes owned by root: hand the state directory to the app user,
    # then continue as that user by running this script again.
    mkdir -p "$STATE_DIR"
    chown -R btcagent:btcagent "$STATE_DIR"
    export HOME=/home/btcagent
    exec setpriv --reuid=btcagent --regid=btcagent --init-groups -- "$0" "$@"
fi

if [ ! -w "$STATE_DIR" ]; then
    echo "ERROR: $STATE_DIR is not writable by user $(id -un). Mount a writable volume there." >&2
    exit 1
fi

exec btcagent \
    --set "storage.data_dir=$STATE_DIR" \
    --set "storage.journal_path=$STATE_DIR/journal.db" \
    --set "model.path=$STATE_DIR/btc_model.joblib" \
    run --host 0.0.0.0 --port "${PORT:-8000}" "$@"
