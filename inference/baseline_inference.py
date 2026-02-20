"""Generate predictions using the Baseline VLM."""
import os
import sys
import json
import argparse
import logging
import traceback
from tqdm import tqdm
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agentic.baseline import BaselineVLM

def resolve_image_paths(image_paths, base_dir):
    """Resolve relative image paths to absolute paths.
    
    Args:
        image_paths: Dict of camera -> image path
        base_dir: Base directory to resolve relative paths from
        
    Returns:
        Dict with resolved absolute paths
    """
    resolved = {}
    for camera, path in image_paths.items():
        if os.path.isabs(path):
            resolved[camera] = path
        else:
            # Try resolving relative to base_dir
            abs_path = os.path.join(base_dir, path)
            if os.path.exists(abs_path):
                resolved[camera] = abs_path
            else:
                # Keep original path and let error handling catch it
                resolved[camera] = path
    return resolved

def main():
    # Configure logger
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description='Baseline VLM Inference')
    parser.add_argument('--test-file', type=str, default='data/drivebench-test.json',
                        help='Path to test data')
    parser.add_argument('--output', type=str, default='results/baseline_predictions.json',
                        help='Output path for predictions')
    parser.add_argument('--model', type=str, default='llava:latest',
                        help='Vision-language model')
    parser.add_argument('--max-samples', type=int, default=None,
                        help='Maximum number of samples to process')
    parser.add_argument('--ollama-url', type=str, default='http://localhost:11434',
                        help='Ollama server URL')
    parser.add_argument('--base-dir', type=str, default=None,
                        help='Base directory for resolving relative image paths (default: project root)')
    args = parser.parse_args()
    
    # Determine base directory for resolving paths
    if args.base_dir:
        base_dir = args.base_dir
    else:
        # Default to project root (parent of inference directory)
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Load test data
    try:
        if not os.path.exists(args.test_file):
            logger.error(f"Test file not found: {args.test_file}")
            return
        
        with open(args.test_file, 'r') as f:
            test_data = json.load(f)
        
        if test_data:
            # Check if first sample's images exist
            first_paths = test_data[0].get('image_path', {})
            resolved_paths = resolve_image_paths(first_paths, base_dir)
            
            for camera, path in resolved_paths.items():
                if os.path.exists(path):
                    size = os.path.getsize(path)
                    logger.info(f"✓ {camera}: {path} ({size} bytes)")
                else:
                    logger.error(f"✗ {camera}: {path} NOT FOUND")
                    logger.error(f"  Working directory: {os.getcwd()}")
                    logger.error(f"  Base directory: {base_dir}")
                    logger.error(f"  Please ensure images are in the correct location")
                    return
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        logger.error(traceback.format_exc())
        return
    
    if args.max_samples:
        test_data = test_data[:args.max_samples]
    
    # Initialize baseline
    try:
        baseline = BaselineVLM(model=args.model, ollama_url=args.ollama_url)
    except Exception as e:
        logger.error(f"Failed to initialize BaselineVLM: {e}")
        logger.error(traceback.format_exc())
        return
    
    # Generate predictions
    predictions = []
    
    for idx, item in enumerate(tqdm(test_data, desc="Processing")):
        try:
            # Resolve relative image paths
            image_paths = resolve_image_paths(item['image_path'], base_dir)
            
            # Verify all images exist
            missing_images = [path for path in image_paths.values() if not os.path.exists(path)]
            if missing_images:
                logger.error(f"Missing images: {missing_images}")
                raise FileNotFoundError(f"Missing {len(missing_images)} image(s): {missing_images[0]}")
            
            answer = baseline.answer_question(
                question=item['question'],
                image_paths=image_paths
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
            success_count += 1
        except KeyError as e:
            logger.error(f"KeyError processing item {idx}: Missing key {e}")
            logger.error(f"Available keys: {item.keys()}")
            logger.error(traceback.format_exc())
            error_count += 1
            # Add fallback prediction with debug info
            predictions.append({
                "scene_token": item.get('scene_token', 'unknown'),
                "frame_token": item.get('frame_token', 'unknown'),
                "question": item.get('question', 'unknown'),
                "question_type": item.get('question_type', 'unknown'),
                "tag": item.get('tag', 'unknown'),
                "answer": item.get('answer', 'unknown'),
                "pred": f"KeyError: {e}",
                "image_path": item.get('image_path', 'unknown'),
                "error_type": "KeyError"
            })
        except Exception as e:
            logger.error(f"Error processing item {idx}: {e}")
            logger.error(f"Item keys: {item.keys()}")
            logger.error(traceback.format_exc())
            error_count += 1
            # Add fallback prediction with error details
            predictions.append({
                "scene_token": item.get('scene_token', 'unknown'),
                "frame_token": item.get('frame_token', 'unknown'),
                "question": item.get('question', 'unknown'),
                "question_type": item.get('question_type', 'unknown'),
                "tag": item.get('tag', 'unknown'),
                "answer": item.get('answer', 'unknown'),
                "pred": f"Error: {str(e)[:100]}",
                "image_path": item.get('image_path', 'unknown'),
                "error_type": type(e).__name__
            })
    
    try:
        os.makedirs(os.path.dirname(args.output) or '.', exist_ok=True)
        with open(args.output, 'w') as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        logger.error(f"Failed to save predictions: {e}")
        logger.error(traceback.format_exc())
        return
    


if __name__ == '__main__':
    main()
