#!/usr/bin/env bash
# Remote setup script for DriveBench on EC2
# Run this ON the EC2 instance after code is copied
# Usage: bash setup_ec2.sh

set -euo pipefail

echo "🚀 DriveBench EC2 Setup"
echo "======================"

DRIVEBENCH_DIR="${1:-.}"
PYTHON_CMD="python3"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

function log_info() {
  echo -e "${GREEN}[INFO]${NC} $1"
}

function log_warn() {
  echo -e "${YELLOW}[WARN]${NC} $1"
}

function log_error() {
  echo -e "${RED}[ERROR]${NC} $1"
}

# Step 1: Detect OS and install base dependencies
log_info "Step 1: Installing system dependencies..."

if command -v yum &> /dev/null; then
  # Amazon Linux / RedHat
  log_info "Detected: Amazon Linux / RedHat"
  sudo yum update -y
  sudo yum install -y python3 python3-venv python3-pip gcc python3-devel
elif command -v apt &> /dev/null; then
  # Ubuntu / Debian
  log_info "Detected: Ubuntu / Debian"
  sudo apt update
  sudo apt install -y python3 python3-venv python3-pip build-essential python3-dev
else
  log_error "Unsupported OS. Please install Python 3.8+ manually."
  exit 1
fi

# Step 2: Create Python virtual environment
log_info "Step 2: Setting up Python virtual environment..."
cd "$DRIVEBENCH_DIR"

if [[ ! -d ".env" ]]; then
  $PYTHON_CMD -m venv .env
  log_info "Created virtual environment"
fi

source .env/bin/activate
log_info "Activated virtual environment"

# Upgrade pip
pip install --upgrade pip wheel setuptools

# Step 3: Install Python dependencies
log_info "Step 3: Installing Python packages..."
if [[ -f "requirements_agentic.txt" ]]; then
  pip install -r requirements_agentic.txt || {
    log_warn "Some pip packages failed; continuing anyway..."
  }
else
  log_error "requirements_agentic.txt not found!"
  exit 1
fi

# Step 4: Install Ollama
log_info "Step 4: Installing Ollama..."

if ! command -v ollama &> /dev/null; then
  log_info "Downloading Ollama..."
  curl -fsSL https://ollama.ai/install.sh | sh || {
    log_warn "Ollama install script failed. Attempting manual download..."
    sudo mkdir -p /usr/local/bin
    OLLAMA_VERSION="0.1.30"
    ARCH=$(uname -m)
    if [[ "$ARCH" == "x86_64" ]]; then
      ARCH="amd64"
    elif [[ "$ARCH" == "aarch64" ]]; then
      ARCH="arm64"
    fi
    curl -L "https://github.com/jmorganca/ollama/releases/download/v${OLLAMA_VERSION}/ollama-linux-${ARCH}" \
      -o /tmp/ollama
    sudo mv /tmp/ollama /usr/local/bin/ollama
    sudo chmod +x /usr/local/bin/ollama
  }
else
  log_info "Ollama already installed"
fi

# Step 5: Start Ollama server
log_info "Step 5: Starting Ollama server..."

if systemctl is-active --quiet ollama; then
  log_info "Ollama service already running"
elif command -v systemctl &> /dev/null && [[ -f /etc/systemd/system/ollama.service ]]; then
  log_info "Starting Ollama via systemctl..."
  sudo systemctl start ollama
  sudo systemctl enable ollama
  sleep 3
else
  log_info "Starting Ollama in background..."
  nohup ollama serve > /tmp/ollama.log 2>&1 &
  sleep 5
fi

# Wait for Ollama to respond
log_info "Waiting for Ollama to be ready..."
for i in {1..30}; do
  if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    log_info "Ollama is ready!"
    break
  fi
  if [[ $i -eq 30 ]]; then
    log_error "Ollama failed to start (timeout)"
    exit 1
  fi
  sleep 2
done

# Step 6: Pull models
log_info "Step 6: Pulling Ollama models (this may take 10-30 minutes)..."

log_info "Pulling llava:latest..."
ollama pull llava:latest || {
  log_warn "Failed to pull llava; trying again..."
  sleep 5
  ollama pull llava:latest || log_error "llava pull failed"
}

log_info "Pulling gpt-oss-20b..."
ollama pull gpt-oss-20b || {
  log_warn "Failed to pull gpt-oss-20b; skipping optional model"
}

# Verify models
log_info "Verifying installed models..."
curl -s http://localhost:11434/api/tags | python3 -m json.tool || true

# Step 7: Create data directories
log_info "Step 7: Creating data directories..."
mkdir -p "$DRIVEBENCH_DIR/data/nuscenes/samples"
mkdir -p "$DRIVEBENCH_DIR/data/corruption"
mkdir -p "$DRIVEBENCH_DIR/results"

# Step 8: Test basic functionality
log_info "Step 8: Testing basic functionality..."

cd "$DRIVEBENCH_DIR"
export PYTHONPATH='./':$PYTHONPATH

# Test import
if $PYTHON_CMD -c "from agentic.ollama_client import OllamaClient; print('✓ Agentic imports OK')" 2>/dev/null; then
  log_info "Python imports verified"
else
  log_warn "Python imports failed; may need manual intervention"
fi

# Test Ollama connectivity
if curl -s http://localhost:11434/api/tags > /dev/null; then
  log_info "Ollama connectivity verified"
else
  log_error "Cannot connect to Ollama at localhost:11434"
fi

# Summary
echo ""
echo "========================================"
echo -e "${GREEN}✅ EC2 Setup Complete!${NC}"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Download data from:"
echo "   - HuggingFace: https://huggingface.co/datasets/drive-bench/arena"
echo "   - Google Drive: https://drive.google.com/file/d/1_MqbX1oXH9S55eC0r_rZvvaoAD5GVOyW"
echo ""
echo "2. Extract data into:"
echo "   $DRIVEBENCH_DIR/data/"
echo ""
echo "3. Preprocess data:"
echo "   cd $DRIVEBENCH_DIR"
echo "   source .env/bin/activate"
echo "   python tools/preprocess.py data/drivebench-test.json data/drivebench-test-final.json"
echo ""
echo "4. Run quick test:"
echo "   source env.sh"
echo "   python test_agentic.py"
echo ""
echo "5. Run full comparison:"
echo "   python compare.py --max-samples 100"
echo ""
echo "Ollama models installed:"
ollama list || echo "  (check manually: ollama list)"
echo ""
