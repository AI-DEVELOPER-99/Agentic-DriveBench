#!/bin/bash
# Quick setup script for agentic pipeline

echo "Setting up Agentic Pipeline..."

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo "ERROR: Ollama is not installed"
    echo "Please install from: https://ollama.ai"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
    echo "ERROR: Ollama server is not running"
    echo "Please start Ollama first"
    exit 1
fi

# Pull models
echo "Pulling llava:latest..."
ollama pull llava:latest

echo "Pulling gpt-oss-20b..."
ollama pull gpt-oss-20b

# Create results directory
mkdir -p results

echo ""
echo "Setup complete!"
echo ""
echo "Run comparison:"
echo "  python compare.py --max-samples 100"
echo ""
