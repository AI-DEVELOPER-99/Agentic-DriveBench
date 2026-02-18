# DriveBench Deployment to AWS EC2

Complete guide to copy the DriveBench project to an EC2 instance, download data, and set up Ollama.

---

## Prerequisites

- **Local machine**: macOS or Linux with `rsync`, `ssh`, and `curl`  
- **EC2 instance**: Ubuntu 20.04+ (or compatible Linux distro)  
- **SSH key**: `.pem` file with permissions `chmod 400 key.pem`  
- **EC2 Security Group**: Allow SSH (port 22) and Ollama (port 11434) inbound  
- **EC2 IAM**: IAM role or credentials with internet access to download data

---

## Step 1: Local Setup – Prepare for Deploy

### 1.1 Make the script executable

```bash
cd /Users/arun/Desktop/Files/Thesis/Code/DriveBench
chmod +x scripts/deploy_to_aws.sh
```

### 1.2 Test SSH connectivity to your EC2 instance

```bash
ssh -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" ec2-user@3.110.134.31 "echo 'SSH works!'"
```

> If this fails, check:
> - Security group allows SSH (port 22)
> - Correct `.pem` file and permissions (`chmod 400`)
> - Correct username (`ec2-user` for Amazon Linux, `ubuntu` for Ubuntu)

---

## Step 2: Deploy Project (Exclude Data Folder)

### 2.1 Copy project to EC2 (excluding `data/` and other build artifacts)

Run from the DriveBench root directory:

```bash
rsync -avz \
  --exclude='data' \
  --exclude='.git' \
  --exclude='__pycache__' \
  --exclude='results' \
  --exclude='.env' \
  --exclude='*.pyc' \
  --exclude='.DS_Store' \
  -e "ssh -i '/Users/arun/NPlusLabs/Voice Cloner/Test1.pem'" \
  ./ \
  ec2-user@3.110.134.31:/home/ec2-user/drivebench/
```

> **Alternative**: Use the provided script (more automated):
> ```bash
> ./scripts/deploy_to_aws.sh \
>   -h 3.110.134.31 \
>   -u ec2-user \
>   -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" \
>   -d /home/ec2-user/drivebench
> ```

### 2.2 Verify deployment

```bash
ssh -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" ec2-user@3.110.134.31 \
  "ls -la /home/ec2-user/drivebench/"
```

You should see: `agentic/`, `data/`, `evaluate/`, etc. (without actual data files inside `data/`).

---

## Step 3: Remote Setup – Install Dependencies & Ollama

Connect to your EC2 instance and run these commands:

```bash
ssh -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" ec2-user@3.110.134.31
```

### 3.1 Update system and install Python

```bash
sudo yum update -y                          # Amazon Linux
sudo yum install -y python3 python3-venv python3-pip

# OR for Ubuntu:
# sudo apt update && sudo apt install -y python3 python3-venv python3-pip
```

### 3.2 Create Python virtual environment

```bash
cd /home/ec2-user/drivebench
python3 -m venv .env
source .env/bin/activate
pip install --upgrade pip wheel
```

### 3.3 Install Python dependencies

```bash
pip install -r requirements_agentic.txt
```

```bash
# Install huggingface-hub if needed
pip install huggingface-hub

# Download dataset
python - <<'PY'
from huggingface_hub import list_repo_files, hf_hub_download
import os
repo_id = "drive-bench/arena"
out_dir = "/home/ec2-user/drivebench/data"
os.makedirs(out_dir, exist_ok=True)
for fname in list_repo_files(repo_id, repo_type="dataset"):
    print("Downloading", fname)
    hf_hub_download(repo_id=repo_id, filename=fname, repo_type="dataset",
                    local_dir=out_dir, local_dir_use_symlinks=False)
print("All files downloaded to", out_dir)
PY
```

> If any package fails, try installing system deps first (e.g., `sudo yum install -y gcc python3-devel` on Amazon Linux)

### 3.4 Install Ollama

#### For Amazon Linux 2 / Ubuntu

```bash
# Download Ollama
curl -fsSL https://ollama.ai/install.sh | sh

# Or manually:
sudo mkdir -p /usr/local/bin
curl -L https://github.com/jmorganca/ollama/releases/download/v0.1.0/ollama-linux-amd64 \
  -o /usr/local/bin/ollama
sudo chmod +x /usr/local/bin/ollama
```

#### Start Ollama server (background)

```bash
# Start as a service (if installed via install.sh):
sudo systemctl start ollama
sudo systemctl enable ollama

# OR manually in tmux/screen:
ollama serve &
```

#### Verify Ollama is running

```bash
curl http://localhost:11434/api/tags
```

Should return a JSON response (initially empty model list).

### 3.5 Pull required models

```bash
ollama pull llava:latest
ollama pull gpt-oss-20b
```

> These are large (~15GB total). Monitor with:
> ```bash
> watch -n 1 'curl -s http://localhost:11434/api/tags | jq .'
> ```

### 3.6 Set environment variables

```bash
cd /home/ec2-user/drivebench
export PYTHONPATH='./':$PYTHONPATH
source env.sh
```

---

## Step 4: Download & Prepare Data

### 4.1 Create data directory

```bash
mkdir -p /home/ec2-user/drivebench/data/nuscenes/samples
mkdir -p /home/ec2-user/drivebench/data/corruption
```

### 4.2 Download data from remote sources

#### Option A: Download from Google Drive (Images)

1. Access: https://drive.google.com/file/d/1_MqbX1oXH9S55eC0r_rZvvaoAD5GVOyW/view?usp=share_link
2. Download to your local machine
3. Upload to EC2:

```bash
# From your local machine:
scp -i "/Users/arun/NPlusLabs/Voice Cloner/Test1.pem" \
  ~/Downloads/drivebench_images.zip \
  ec2-user@3.110.134.31:/home/ec2-user/drivebench/data/

# OR use S3 as intermediary (faster for large files):
# aws s3 cp ~/Downloads/drivebench_images.zip s3://your-bucket/
# Then on EC2: aws s3 cp s3://your-bucket/drivebench_images.zip /home/ec2-user/drivebench/data/
```

#### Option B: Download from HuggingFace (JSON metadata)

On the EC2 instance:

```bash
cd /home/ec2-user/drivebench/data
pip install huggingface-hub

huggingface-cli download drive-bench/arena --repo-type dataset \
  --local-dir /home/ec2-user/drivebench/data/
```

### 4.3 Extract and organize data

```bash
cd /home/ec2-user/drivebench/data

# Unzip images
unzip -q drivebench_images.zip              # or your filename

# Verify structure:
ls -la nuscenes/samples/
ls -la corruption/BitError/
ls -la *.json
```

Expected structure:
```
data/
├── nuscenes/
│   └── samples/                 (camera images)
├── corruption/
│   ├── BitError/
│   ├── Brightness/
│   ├── CameraCrash/
│   └── ...
├── drivebench-test.json         (test questions & ground truth)
├── biterror.json
├── bright.json
└── ...json files
```

### 4.4 Preprocess data

```bash
cd /home/ec2-user/drivebench
source .env/bin/activate
export PYTHONPATH='./':$PYTHONPATH

python tools/preprocess.py data/drivebench-test.json data/drivebench-test-final.json
```

---

## Step 5: Quick Test

### 5.1 Test Ollama connectivity

```bash
curl http://localhost:11434/api/tags
```

### 5.2 Run quick test (single example)

```bash
cd /home/ec2-user/drivebench
source .env/bin/activate
source env.sh

python test_agentic.py
```

Expected output: Perception analysis, scene graph, planning, and final answer for sample 10.

### 5.3 Run full comparison (100 samples)

```bash
python compare.py --max-samples 100
```

Results saved to `results/` directory.

---

## Step 6: Optional – Run with GPT Evaluation

### 6.1 Set OpenAI API key (on remote)

```bash
export OPENAI_API_KEY="sk-..."
```

### 6.2 Run comparison with GPT evaluation

```bash
python compare.py --max-samples 100 --eval-gpt
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| SSH fails | Check security group, `.pem` permissions (`chmod 400`), username |
| `python3` not found | Install: `sudo yum/apt install python3 python3-pip` |
| venv creation fails | Install `python3-venv`: `sudo yum/apt install python3-venv` |
| pip package fails | Install build deps: `sudo yum/apt install gcc python3-devel` |
| Ollama connection refused | Ensure Ollama is running: `curl http://localhost:11434/api/tags` |
| Model not found | Pull models: `ollama pull llava:latest && ollama pull gpt-oss-20b` |
| Out of memory during model pull | Use smaller models or add swap space on EC2 |
| Rsync permission denied | Check SSH key and EC2 security group allows SSH |
| Data folder issues | Verify structure matches expected layout above |

---

## Performance Tips

1. **EC2 Instance Type**: Use `t3.xlarge` or higher (4+ CPU, 16GB+ RAM) for model inference
2. **EBS Volume**: Ensure enough space for models (~20GB) and data (~50GB+)
3. **GPU (Optional)**: For faster inference, use GPU instance (e.g., `g4dn.xlarge`) with CUDA support
4. **Bandwidth**: Use S3 or direct download links for large data transfers

---

## Example: Complete Workflow

```bash
# 1. LOCAL: Copy project
rsync -avz --exclude='data' --exclude='.git' ...

# 2. REMOTE: Install + setup
ssh -i key ec2-user@host
sudo yum install -y python3 python3-venv python3-pip
cd /home/ec2-user/drivebench && python3 -m venv .env && source .env/bin/activate
pip install -r requirements_agentic.txt

# 3. REMOTE: Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
sudo systemctl start ollama
ollama pull llava:latest && ollama pull gpt-oss-20b

# 4. REMOTE: Download & prepare data
mkdir -p data/nuscenes/samples data/corruption
# (download via Google Drive or HuggingFace)
python tools/preprocess.py data/drivebench-test.json data/drivebench-test-final.json

# 5. REMOTE: Test
source env.sh && python test_agentic.py

# 6. REMOTE: Full run
python compare.py --max-samples 100
```

---

## Questions?

Refer to:
- [DriveBench GitHub](https://github.com/worldbench/DriveBench)
- [Ollama Documentation](https://ollama.ai)
- [HuggingFace Hub](https://huggingface.co/datasets/drive-bench/arena)
