#!/usr/bin/env python3
"""
Quick Start Guide for Agentic Pipeline
======================================

This script demonstrates the key features of the agentic pipeline.
"""

print("""
╔════════════════════════════════════════════════════════════════════════════╗
║         Agentic Pipeline for Autonomous Driving Q&A                        ║
╚════════════════════════════════════════════════════════════════════════════╝

📋 SETUP CHECKLIST:
───────────────────
1. ✓ Install Ollama: https://ollama.ai
2. ✓ Pull models:
   ollama pull llava:latest
   ollama pull gpt-oss-20b
3. ✓ Install Python dependencies:
   pip install -r requirements_agentic.txt

🎯 QUICK START:
───────────────

Test single example:
  python test_agentic.py

Run comparison (100 samples):
  python compare.py --max-samples 100

Full benchmark:
  python compare.py --test-file data/drivebench-test.json

With GPT evaluation:
  export OPENAI_API_KEY=your_key_here
  python compare.py --max-samples 100 --eval-gpt

📊 INDIVIDUAL INFERENCE:
────────────────────────

Baseline VLM only:
  python inference/baseline_inference.py \\
    --test-file data/drivebench-test.json \\
    --output results/baseline.json

Agentic Pipeline:
  python inference/agentic_inference.py \\
    --test-file data/drivebench-test.json \\
    --output results/agentic.json

Then evaluate:
  python evaluate/eval.py results/baseline.json
  python evaluate/eval.py results/agentic.json

🏗️ ARCHITECTURE:
─────────────────

Agent 1: Perception Agent
  → Uses llava:latest VLM for object detection
  → Extracts: objects, colors, positions, distances

Agent 2: Scene Graph Agent
  → Constructs spatial relationships
  → Creates nodes (objects) and edges (relations)

Agent 3: Planner Agent
  → Uses gpt-oss-20b LLM for query decomposition
  → Generates multi-step reasoning plans

Agent 4: Executor Agent
  → Executes plan steps
  → Methods: count_objects, check_spatial, get_attribute, etc.

Agent 5: Verifier Agent
  → Validates reasoning chain
  → Assigns confidence scores
  → Provides corrections if needed

📁 FILES:
─────────
agentic/
  ├── __init__.py
  ├── ollama_client.py        # Ollama API wrapper
  ├── agent1_perception.py    # Perception Agent
  ├── agent2_scene_graph.py   # Scene Graph Agent
  ├── agent3_planner.py       # Planner Agent
  ├── agent4_executor.py      # Executor Agent
  ├── agent5_verifier.py      # Verifier Agent
  ├── pipeline.py             # Main pipeline
  └── baseline.py             # Baseline VLM

inference/
  ├── agentic_inference.py    # Agentic inference script
  └── baseline_inference.py   # Baseline inference script

compare.py                    # Comparison script
test_agentic.py              # Quick test script

💡 TIPS:
────────
• Start with small --max-samples for testing (e.g., 10-50)
• The agentic pipeline is slower but more accurate
• Check results/ directory for outputs
• Use --skip-inference to re-evaluate existing predictions

🐛 TROUBLESHOOTING:
───────────────────
• "Connection refused": Make sure Ollama is running
• "Model not found": Run ollama pull <model>
• Slow inference: This is normal, each sample uses multiple LLM calls
• Out of memory: Reduce --max-samples

📧 For issues, check the code or modify agents in agentic/ directory
""")
