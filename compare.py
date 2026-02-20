#!/usr/bin/env python3
"""Compare Agentic Pipeline vs Baseline VLM."""
import os
import sys
import json
import argparse
import subprocess
from pathlib import Path


def run_inference(inference_script, test_file, output_file, max_samples=None, extra_args=None):
    """Run inference script using the same Python interpreter.

    Previously the command used the literal string ``'python'`` which could
    resolve to the system Python rather than the activated virtualenv.  This
    caused ``ModuleNotFoundError`` for packages that lived only in the env
    (e.g. ultralytics).  Using ``sys.executable`` guarantees the child
    process uses the exact interpreter executing the current script.
    """
    cmd = [sys.executable, inference_script, '--test-file', test_file, '--output', output_file]
    
    if max_samples:
        cmd.extend(['--max-samples', str(max_samples)])
    
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, check=True)
    return result.returncode == 0


def run_evaluation(pred_file, eval_gpt=False, api_key=None,
                   use_local=False, ollama_model=None, ollama_url=None):
    """Run evaluation on predictions."""
    # Import and run evaluation directly instead of subprocess
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    
    from evaluate.eval import EvaluationSuit
    from tqdm import tqdm
    
    print(f"Evaluating: {pred_file}")
    
    # Load predictions
    with open(pred_file, 'r') as f:
        data = json.load(f)
    
    # Setup evaluation
    desc_file = 'data/visual_description.json'
    log_dir = os.path.join(os.path.dirname(pred_file), "gpt_eval_logs")
    corruption = os.path.basename(pred_file).split(".")[0]
    log_path = os.path.join(log_dir, f"{corruption}_eval_log.json")
    
    api_key = api_key or os.getenv('OPENAI_API_KEY', '')
    
    evaluation = EvaluationSuit(
        api_key=api_key,
        log_file=log_path,
        desc_file=desc_file,
        eval_gpt=eval_gpt,
        temperature=0.0,
        max_tokens=100000,
        use_local_gpt=use_local,
        ollama_model=ollama_model,
        ollama_url=ollama_url
    )
    
    # Process each data item
    for data_item in tqdm(data, desc="Evaluating"):
        evaluation.forward(data_item)
    
    # Get scores
    scores = evaluation.evaluation()
    
    print("\nScores:")
    print(json.dumps(scores, indent=2))
    
    return scores


def print_comparison(baseline_scores, agentic_scores):
    """Print comparison table."""
    print("\n" + "="*80)
    print("COMPARISON: Baseline VLM vs Agentic Pipeline")
    print("="*80)
    
    if not baseline_scores or not agentic_scores:
        print("Unable to compare - missing scores")
        return
    
    # Compare each task
    for task in ['perception', 'prediction', 'planning', 'behavior']:
        if task not in baseline_scores or task not in agentic_scores:
            continue
        
        print(f"\n{task.upper()}:")
        print("-" * 80)
        
        baseline_task = baseline_scores[task]
        agentic_task = agentic_scores[task]
        
        for qtype in baseline_task.keys():
            if qtype not in agentic_task:
                continue
            
            print(f"\n  {qtype}:")
            baseline_metrics = baseline_task[qtype]
            agentic_metrics = agentic_task[qtype]
            
            for metric in baseline_metrics.keys():
                if metric not in agentic_metrics:
                    continue
                
                baseline_val = baseline_metrics[metric]
                agentic_val = agentic_metrics[metric]
                
                # Handle different metric types
                if isinstance(baseline_val, dict):
                    # Language metrics (BLEU, ROUGE, CIDEr)
                    for sub_metric, baseline_sub_val in baseline_val.items():
                        agentic_sub_val = agentic_val.get(sub_metric, 0)
                        diff = agentic_sub_val - baseline_sub_val
                        sign = "+" if diff > 0 else ""
                        print(f"    {sub_metric:15s}: Baseline={baseline_sub_val:6.2f}  Agentic={agentic_sub_val:6.2f}  ({sign}{diff:6.2f})")
                else:
                    # Single value metrics
                    diff = agentic_val - baseline_val
                    sign = "+" if diff > 0 else ""
                    print(f"    {metric:15s}: Baseline={baseline_val:6.2f}  Agentic={agentic_val:6.2f}  ({sign}{diff:6.2f})")
    
    print("\n" + "="*80)


def main():
    parser = argparse.ArgumentParser(description='Compare Agentic Pipeline vs Baseline VLM')
    parser.add_argument('--test-file', type=str, default='data/drivebench-test.json',
                        help='Path to test data')
    parser.add_argument('--max-samples', type=int, default=100,
                        help='Maximum number of samples to test')
    parser.add_argument('--eval-gpt', action='store_true',
                        help='Run GPT evaluation (requires OpenAI API key or local model)')
    parser.add_argument('--api-key', type=str, default=None,
                        help='OpenAI API key for GPT evaluation')
    parser.add_argument('--use-local-gpt', action='store_true',
                        help='Use a local Ollama model instead of OpenAI')
    parser.add_argument('--ollama-model', type=str, default='gpt-oss:20b',
                        help='Local Ollama model name to use when --use-local-gpt is set')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Base URL for local Ollama server')
    parser.add_argument('--skip-inference', action='store_true',
                        help='Skip inference and use existing predictions')
    args = parser.parse_args()
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    baseline_pred_file = 'results/baseline_predictions.json'
    agentic_pred_file = 'results/agentic_predictions.json'
    
    # Run inference if not skipped
    if not args.skip_inference:
        print("\n" + "="*80)
        print("STEP 1: Running Baseline VLM Inference")
        print("="*80)
        run_inference(
            'inference/baseline_inference.py',
            args.test_file,
            baseline_pred_file,
            max_samples=args.max_samples
        )
        
        print("\n" + "="*80)
        print("STEP 2: Running Agentic Pipeline Inference")
        print("="*80)
        run_inference(
            'inference/agentic_inference.py',
            args.test_file,
            agentic_pred_file,
            max_samples=args.max_samples
        )
    else:
        print("Skipping inference - using existing predictions")
    
    # Run evaluation
    print("\n" + "="*80)
    print("STEP 3: Evaluating Baseline VLM")
    print("="*80)
    baseline_scores = run_evaluation(
        baseline_pred_file,
        args.eval_gpt,
        args.api_key,
        use_local=args.use_local_gpt,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url
    )
    
    print("\n" + "="*80)
    print("STEP 4: Evaluating Agentic Pipeline")
    print("="*80)
    agentic_scores = run_evaluation(
        agentic_pred_file,
        args.eval_gpt,
        args.api_key,
        use_local=args.use_local_gpt,
        ollama_model=args.ollama_model,
        ollama_url=args.ollama_url
    )
    
    # Print comparison
    print_comparison(baseline_scores, agentic_scores)
    
    # Save comparison results
    comparison = {
        "baseline": baseline_scores,
        "agentic": agentic_scores
    }
    
    with open('results/comparison.json', 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\nComparison results saved to results/comparison.json")


if __name__ == '__main__':
    main()
