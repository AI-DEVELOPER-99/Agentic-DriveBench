"""Generate predictions using the Agentic Pipeline."""
import os
import sys
import json
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.pipeline import AgenticPipeline


def main():
    parser = argparse.ArgumentParser(description='Agentic Pipeline Inference')
    parser.add_argument('--test-file', type=str, default='data/drivebench-test.json',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='results/agentic_predictions.json',
                        help='Output path for predictions')
    parser.add_argument('--vlm-model', type=str, default='llava:latest',
                        help='Vision-language model')
    parser.add_argument('--llm-model', type=str, default='gpt-oss:20b',
                        help='Language model')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    args = parser.parse_args()
    
    # Load test data
    print(f"Loading test data from {args.test_file}...")
    with open(args.test_file, 'r') as f:
        test_data = json.load(f)
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    # Initialize pipeline
    print("Initializing agentic pipeline...")
    pipeline = AgenticPipeline(
        vlm_model=args.vlm_model,
        llm_model=args.llm_model
    )
    
    # Generate predictions
    print(f"Processing {len(test_data)} samples...")
    predictions = []
    
    for item in tqdm(test_data):
        try:
            answer = pipeline.answer_question(
                question=item['question'],
                image_paths=item['image_path']
            )
            
            predictions.append({
                "scene_token": item['scene_token'],
                "frame_token": item['frame_token'],
                "question": item['question'],
                "question_type": item['question_type'],
                "tag": item['tag'],
                "answer": item['answer'],
                "pred": answer,
                "image_path": item['image_path']
            })
        except Exception as e:
            print(f"Error processing item: {e}")
            # Add fallback prediction
            predictions.append({
                "scene_token": item['scene_token'],
                "frame_token": item['frame_token'],
                "question": item['question'],
                "question_type": item['question_type'],
                "tag": item['tag'],
                "answer": item['answer'],
                "pred": "Error in processing",
                "image_path": item['image_path']
            })
    
    # Save predictions
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Predictions saved to {args.output}")


if __name__ == '__main__':
    main()
