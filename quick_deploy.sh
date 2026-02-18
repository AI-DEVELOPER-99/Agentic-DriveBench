#!/usr/bin/env bash
# Quick deployment helper for DriveBench to EC2
# Run this locally from your Mac
# Usage: ./quick_deploy.sh -h 3.110.134.31 -u ec2-user -i ~/.ssh/key.pem

set -euo pipefail

# Defaults
HOST=""
USER="ec2-user"
KEY=""
REMOTE_DIR="/home/ec2-user/drivebench"
LOCAL_DIR="$(pwd)"

function usage() {
  cat <<EOF
Quick DriveBench Deploy to EC2

Usage: $0 -h <host> -u <user> -i <key-file> [-d <remote-path>]

Required:
  -h, --host HOST         EC2 instance IP or hostname
  -i, --identity KEY      Path to .pem SSH key

Optional:
  -u, --user USER         SSH username (default: ec2-user)
  -d, --dir REMOTE_DIR    Remote install directory (default: /home/ec2-user/drivebench)
  -l, --local LOCAL_DIR   Local project directory (default: current dir)

Example:
  ./quick_deploy.sh \\
    -h 3.110.134.31 \\
    -u ec2-user \\
    -i "~/.ssh/my-key.pem" \\
    -d /home/ec2-user/drivebench

This will:
  1. Copy project to EC2 (excluding data, .git, __pycache__)
  2. Run setup_ec2.sh on the remote to install dependencies & Ollama
  3. Print next steps for data download & preprocessing
EOF
  exit 1
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--host) HOST=$2; shift 2 ;;
    -u|--user) USER=$2; shift 2 ;;
    -i|--identity) KEY=$2; shift 2 ;;
    -d|--dir) REMOTE_DIR=$2; shift 2 ;;
    -l|--local) LOCAL_DIR=$2; shift 2 ;;
    *) usage ;;
  esac
done

# Validate required args
if [[ -z "$HOST" || -z "$KEY" ]]; then
  echo "ERROR: Missing required arguments"
  usage
fi

# Expand ~ in KEY
KEY="$(eval echo $KEY)"

# Check key exists
if [[ ! -f "$KEY" ]]; then
  echo "ERROR: SSH key not found: $KEY"
  exit 2
fi

# Check permissions
KEY_PERMS=$(stat -f%A "$KEY" 2>/dev/null || stat -c%a "$KEY" 2>/dev/null)
if [[ "$KEY_PERMS" != "600" && "$KEY_PERMS" != "400" ]]; then
  echo "WARNING: SSH key permissions are $KEY_PERMS (should be 400 or 600)"
  echo "Fixing: chmod 400 \"$KEY\""
  chmod 400 "$KEY"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "DriveBench Quick Deploy to EC2"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Configuration:"
echo "  Host:        $USER@$HOST"
echo "  Remote Dir:  $REMOTE_DIR"
echo "  Local Dir:   $LOCAL_DIR"
echo "  SSH Key:     $KEY"
echo ""

# Step 1: Test SSH connectivity
echo "[1/3] Testing SSH connectivity..."
if ! ssh -i "$KEY" -o ConnectTimeout=10 -o StrictHostKeyChecking=no "$USER@$HOST" "echo 'SSH OK'" 2>/dev/null; then
  echo "ERROR: Cannot connect via SSH. Check:"
  echo "  - EC2 security group allows port 22"
  echo "  - Correct username ($USER)"
  echo "  - Correct host ($HOST)"
  echo "  - SSH key ($KEY)"
  exit 3
fi
echo "✓ SSH connection successful"

# Step 2: Copy project
echo ""
echo "[2/3] Copying project to remote (excluding data, .git, __pycache__)..."

RSYNC_OPTS=(
  -avz
  --progress
  --exclude='data'
  --exclude='.git'
  --exclude='.gitignore'
  --exclude='__pycache__'
  --exclude='.env'
  --exclude='*.pyc'
  --exclude='.DS_Store'
  --exclude='results'
  --exclude='*.egg-info'
  -e "ssh -i '$KEY' -o StrictHostKeyChecking=no"
)

rsync "${RSYNC_OPTS[@]}" "$LOCAL_DIR/" "$USER@$HOST:$REMOTE_DIR/" || {
  echo "ERROR: rsync failed"
  exit 4
}
echo "✓ Project copied successfully"

# Step 3: Run remote setup
echo ""
echo "[3/3] Running remote setup (install dependencies, Ollama, etc)..."
echo "      This may take 10-20 minutes on first run. Please wait..."
echo ""

ssh -i "$KEY" -o StrictHostKeyChecking=no "$USER@$HOST" bash -lc "'
  set -euo pipefail
  bash '"$REMOTE_DIR"'/setup_ec2.sh '"$REMOTE_DIR"'
'" || {
  echo ""
  echo "WARNING: Remote setup had issues. You may need to:"
  echo "  1. SSH into the instance and check /tmp/ollama.log"
  echo "  2. Run: ssh -i \"$KEY\" $USER@$HOST \"bash $REMOTE_DIR/setup_ec2.sh\""
  echo "  3. Refer to DEPLOY_TO_EC2.md for manual steps"
}

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ Deployment Complete!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Next steps:"
echo ""
echo "1. SSH into your instance:"
echo "   ssh -i \"$KEY\" $USER@$HOST"
echo ""
echo "2. Download data from (choose one or both):"
echo "   • HuggingFace: https://huggingface.co/datasets/drive-bench/arena"
echo "   • Google Drive: https://drive.google.com/file/d/1_MqbX1oXH9S55eC0r_rZvvaoAD5GVOyW"
echo ""
echo "3. Extract data into: $REMOTE_DIR/data/"
echo ""
echo "4. On remote instance, preprocess data:"
echo "   cd $REMOTE_DIR"
echo "   source .env/bin/activate"
echo "   source env.sh"
echo "   python tools/preprocess.py data/drivebench-test.json data/drivebench-test-final.json"
echo ""
echo "5. Test the setup:"
echo "   python test_agentic.py"
echo ""
echo "6. Run full comparison:"
echo "   python compare.py --max-samples 100"
echo ""
echo "For detailed instructions, see: DEPLOY_TO_EC2.md"
echo ""
