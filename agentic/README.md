# Agentic Pipeline for Autonomous Driving Q&A

Multi-agent system using local Ollama models for autonomous driving scene understanding.

## Setup

1. Install Ollama and pull models:
```bash
# Install Ollama from https://ollama.ai
ollama pull llava:latest
ollama pull gpt-oss-20b
```

2. Install dependencies:
```bash
pip install requests tqdm language_evaluation
```

## Usage

### Run Comparison (Baseline vs Agentic)

```bash
# Quick test with 100 samples
python compare.py --max-samples 100

# Full test
python compare.py --test-file data/drivebench-test.json

# With GPT evaluation (requires OpenAI API key)
python compare.py --max-samples 100 --eval-gpt --api-key YOUR_API_KEY
```

### Individual Inference

```bash
# Baseline VLM
python inference/baseline_inference.py --test-file data/drivebench-test.json --output results/baseline.json

# Agentic Pipeline
python inference/agentic_inference.py --test-file data/drivebench-test.json --output results/agentic.json
```

## Architecture

The agentic pipeline consists of 5 agents:

1. **Perception Agent**: Object detection using VLM
2. **Scene Graph Agent**: Spatial relationship construction
3. **Planner Agent**: Query decomposition and reasoning plan generation
4. **Executor Agent**: Plan execution with specialized methods
5. **Verifier Agent**: Reasoning validation and confidence scoring

## Files

- `agentic/` - Agent implementations
- `inference/` - Inference scripts
- `compare.py` - Comparison script
- `results/` - Output directory
