"""Main Agentic Pipeline - Integrates all 5 agents."""
from typing import Dict, List, Any
from agentic.ollama_client import OllamaClient
from agentic.agent1_perception import PerceptionAgent
from agentic.agent2_scene_graph import SceneGraphAgent
from agentic.agent3_planner import PlannerAgent
from agentic.agent4_executor import ExecutorAgent
from agentic.agent5_verifier import VerifierAgent


class AgenticPipeline:
    """Multi-agent pipeline for autonomous driving Q&A."""
    
    def __init__(self, 
                 vlm_model: str = "llava:latest",
                 llm_model: str = "gpt-oss:20b",
                 ollama_url: str = "http://localhost:11434"):
        """Initialize the agentic pipeline.
        
        Args:
            vlm_model: Vision-language model name
            llm_model: Language model name
            ollama_url: Ollama server URL
        """
        self.client = OllamaClient(base_url=ollama_url)
        
        # Initialize all agents
        self.agent1 = PerceptionAgent(self.client, vlm_model)
        self.agent2 = SceneGraphAgent()
        self.agent3 = PlannerAgent(self.client, llm_model)
        self.agent4 = ExecutorAgent()
        self.agent5 = VerifierAgent(self.client, llm_model)
        
        self.vlm_model = vlm_model
    
    def process(self, question: str, image_paths: Dict[str, str]) -> Dict[str, Any]:
        """Process a question with multi-agent reasoning.
        
        Args:
            question: The driving question
            image_paths: Dictionary mapping camera positions to image paths
            
        Returns:
            Pipeline result with answer and metadata
        """
        # Get all available images
        images = list(image_paths.values())
        
        # Agent 1: Perception (with question context)
        perception_result = self.agent1.perceive(images, question)
        
        # Agent 2: Scene Graph Construction
        scene_graph = self.agent2.construct_graph(perception_result)
        
        # Agent 3: Planning/Reasoning
        plan = self.agent3.plan(question, scene_graph)
        
        # Agent 4: Execution
        execution_result = self.agent4.execute(plan, scene_graph, perception_result)
        
        # Agent 5: Verification and Refinement
        verification_result = self.agent5.verify(question, execution_result, scene_graph)
        
        return {
            "answer": verification_result["final_answer"],
            "confidence": verification_result["confidence"],
            "reasoning_chain": verification_result["reasoning_chain"],
            "metadata": {
                "perception": perception_result,
                "scene_graph": scene_graph,
                "plan": plan,
                "execution": execution_result,
                "verification": verification_result
            }
        }
    
    def answer_question(self, question: str, image_paths: Dict[str, str]) -> str:
        """Simplified interface - just return the answer.
        
        Args:
            question: The driving question
            image_paths: Dictionary mapping camera positions to image paths
            
        Returns:
            Answer string
        """
        result = self.process(question, image_paths)
        return result["answer"]
