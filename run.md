# Setup
1. Virtual environment activated
2. Dependencies installed
3. Ollama server running
4. Test successful

# 🚀 How to Run
## Quick Test (Single Example)
source .env/bin/activate
source env.sh
python test_agentic.py


## Run Comparison (100 samples)
source .env/bin/activate  
source env.sh
python compare.py --max-samples 100

## Full Benchmark
source .env/bin/activate
source env.sh  
python compare.py --test-file data/drivebench-test.json

## With GPT Evaluation (OpenAI)
export OPENAI_API_KEY=your_key_here
source .env/bin/activate
source env.sh
python compare.py --max-samples 100 --eval-gpt

## With Local GPT Evaluation (ollama)
# ensure the model is running (e.g. `ollama run gpt-oss:20b`)
source .env/bin/activate
source env.sh
python compare.py --max-samples 100 --eval-gpt --use-local-gpt \
    --ollama-model gpt-oss:20b --ollama-url http://localhost:11434

## 📊 Results
1. Outputs saved to results directory
2. Compare baseline vs agentic performance
Agentic pipeline provides more structured, driving-focused answers
The agentic pipeline uses multiple AI agents for better reasoning on autonomous driving scenarios, while the baseline uses a single VLM call.