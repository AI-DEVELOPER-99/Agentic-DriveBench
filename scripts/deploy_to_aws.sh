#!/usr/bin/env bash
# Deploy this project to a remote AWS host and run the quick test (test_agentic.py)
# Usage:
#   ./scripts/deploy_to_aws.sh -h 3.110.134.31 -u ec2-user -i "~/.ssh/mykey.pem" -d ~/drivebench

set -euo pipefail

HOST=""
USER="ec2-user"
KEY=""
REMOTE_DIR="~/drivebench"
LOCAL_DIR="$(pwd)"
RSYNC_EXCLUDES=(".git" "__pycache__" "results" ".env" "*.pyc" ".DS_Store")

usage() {
  cat <<EOF
Usage: $0 -h <host> -u <user> -i <identity-file> [-d <remote-dir>] [-l <local-dir>]

Examples:
  $0 -h 3.110.134.31 -u ec2-user -i "~/.ssh/my-key.pem" -d ~/drivebench

This will:
  - rsync the repository to the remote host
  - create a Python venv on the remote and install requirements
  - (optionally) run setup_agentic.sh to pull Ollama models if Ollama is available
  - run `test_agentic.py`

Note: Ollama must be installed & running on the remote for the full agentic tests to pass.
EOF
  exit 1
}

while getopts ":h:u:i:d:l:" opt; do
  case ${opt} in
    h) HOST=${OPTARG} ;;
    u) USER=${OPTARG} ;;
    i) KEY=${OPTARG} ;;
    d) REMOTE_DIR=${OPTARG} ;;
    l) LOCAL_DIR=${OPTARG} ;;
    *) usage ;;
  esac
done

if [[ -z "$HOST" || -z "$KEY" ]]; then
  usage
fi

# Expand ~ in KEY if present
KEY="$(eval echo $KEY)"

echo "🔧 Deploying to $USER@$HOST:$REMOTE_DIR"

echo "-> Checking SSH connectivity..."
ssh -i "$KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$HOST" "echo connected" || {
  echo "ERROR: SSH connection failed. Check host/user/key." >&2
  exit 2
}

echo "-> Rsyncing project (this may take a while)..."
RSYNC_EXCLUDE_ARGS=()
for e in "${RSYNC_EXCLUDES[@]}"; do
  RSYNC_EXCLUDE_ARGS+=(--exclude="$e")
done

rsync -avz --progress "${RSYNC_EXCLUDE_ARGS[@]}" -e "ssh -i '$KEY' -o StrictHostKeyChecking=no" "$LOCAL_DIR/" "$USER@$HOST:$REMOTE_DIR/"

echo "-> Running remote setup + test"
ssh -i "$KEY" -o StrictHostKeyChecking=no "$USER@$HOST" bash -lc "'
set -euo pipefail
mkdir -p $REMOTE_DIR
cd $REMOTE_DIR

# create venv if missing
if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 not found on remote — please install Python 3.8+ and re-run." >&2
  exit 3
fi
python3 -m venv .env || true
. .env/bin/activate
python -m pip install --upgrade pip wheel
python -m pip install -r requirements_agentic.txt || true

# ensure PYTHONPATH is set
export PYTHONPATH='$REMOTE_DIR':$PYTHONPATH
source env.sh || true

# Try to run setup script (will exit early if Ollama is not installed/running)
if [[ -x setup_agentic.sh ]]; then
  bash setup_agentic.sh || echo "Note: setup_agentic.sh failed — Ollama may not be installed or running."
fi

# Run quick test (test_agentic.py)
python test_agentic.py
'"

RC=$?
if [[ $RC -eq 0 ]]; then
  echo "✅ Remote test completed successfully (exit code 0)."
else
  echo "⚠️ Remote test finished with exit code $RC — check remote logs above for errors." >&2
fi

exit $RC
