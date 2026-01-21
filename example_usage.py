"""
Example: How to use the Agentic Pipeline
=========================================

This example shows how to use both the baseline and agentic pipeline programmatically.
"""

from agentic.pipeline import AgenticPipeline
from agentic.baseline import BaselineVLM


# Example driving scene question
question = "What are the important objects in the current scene?"

# Example image paths (from nuscenes dataset)
image_paths = {
    "CAM_FRONT": "data/nuscenes/samples/CAM_FRONT/example.jpg",
    "CAM_BACK": "data/nuscenes/samples/CAM_BACK/example.jpg",
    # ... more cameras
}


# ===================================================================
# METHOD 1: Baseline VLM (simple, single call)
# ===================================================================
print("Method 1: Baseline VLM")
print("-" * 60)

baseline = BaselineVLM(
    model="llava:latest",
    ollama_url="http://localhost:11434"
)

answer = baseline.answer_question(question, image_paths)
print(f"Answer: {answer}\n")


# ===================================================================
# METHOD 2: Agentic Pipeline (multi-agent reasoning)
# ===================================================================
print("Method 2: Agentic Pipeline")
print("-" * 60)

pipeline = AgenticPipeline(
    vlm_model="llava:latest",
    llm_model="gpt-oss-20b",
    ollama_url="http://localhost:11434"
)

# Get full result with metadata
result = pipeline.process(question, image_paths)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}/100")
print(f"\nReasoning Chain:")
print(result['reasoning_chain'])
print(f"\nDetailed Metadata:")
print(f"  - Detected objects: {len(result['metadata']['perception']['objects'])}")
print(f"  - Scene graph nodes: {len(result['metadata']['scene_graph']['nodes'])}")
print(f"  - Planning steps: {len(result['metadata']['plan'])}")
print(f"  - Is valid: {result['metadata']['verification']['is_valid']}")


# ===================================================================
# METHOD 3: Simple Answer Only
# ===================================================================
print("\n\nMethod 3: Quick Answer")
print("-" * 60)

# Just get the answer, no metadata
simple_answer = pipeline.answer_question(question, image_paths)
print(f"Answer: {simple_answer}\n")


# ===================================================================
# Understanding the Pipeline Flow
# ===================================================================
print("\n" + "="*60)
print("Pipeline Flow:")
print("="*60)
print("""
1. Perception Agent (Agent 1)
   Input: Images from all cameras
   Output: Detected objects with attributes
   
2. Scene Graph Agent (Agent 2)
   Input: Detected objects
   Output: Scene graph (nodes + spatial relations)
   
3. Planner Agent (Agent 3)
   Input: Question + Scene graph
   Output: Step-by-step reasoning plan
   
4. Executor Agent (Agent 4)
   Input: Reasoning plan + Scene data
   Output: Executed results for each step
   
5. Verifier Agent (Agent 5)
   Input: Question + Execution results
   Output: Verified answer + confidence
""")


# ===================================================================
# Customization Examples
# ===================================================================
print("="*60)
print("Customization Examples:")
print("="*60)
print("""
# Use different models:
pipeline = AgenticPipeline(
    vlm_model="llava:13b",  # Larger VLM
    llm_model="llama2:70b"  # Larger LLM
)

# Connect to remote Ollama:
pipeline = AgenticPipeline(
    ollama_url="http://remote-server:11434"
)

# Access individual agents:
from agentic.agent1_perception import PerceptionAgent
from agentic.ollama_client import OllamaClient

client = OllamaClient()
perception = PerceptionAgent(client)
result = perception.perceive(list(image_paths.values()))
""")
