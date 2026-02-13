#!/usr/bin/env python3
"""Quick test of the agentic pipeline."""
import json
from agentic.pipeline import AgenticPipeline
from agentic.baseline import BaselineVLM


def test_pipeline():
    """Test the agentic pipeline with a single example."""
    # Load one test sample
    with open('data/drivebench-test.json', 'r') as f:
        test_data = json.load(f)
    
    sample = test_data[10]
    
    print("="*80)
    print("Testing Agentic Pipeline")
    print("="*80)
    print(f"\nQuestion: {sample['question']}")
    print(f"Ground Truth: {sample['answer']}")
    
    # Test baseline
    print("\n" + "-"*80)
    print("Baseline VLM:")
    print("-"*80)
    try:
        baseline = BaselineVLM()
        baseline_answer = baseline.answer_question(
            question=sample['question'],
            image_paths=sample['image_path']
        )
        print(f"Answer: {baseline_answer}")
    except Exception as e:
        print(f"Error: {e}")
    
    # Test agentic pipeline
    print("\n" + "-"*80)
    print("Agentic Pipeline:")
    print("-"*80)
    try:
        pipeline = AgenticPipeline()
        result = pipeline.process(
            question=sample['question'],
            image_paths=sample['image_path']
        )
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"\nReasoning Chain:")
        print(result['reasoning_chain'])
    except Exception as e:
        print(f"Error: {e}")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    test_pipeline()
