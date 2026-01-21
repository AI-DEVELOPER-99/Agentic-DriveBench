"""Agent 3: Planner Agent - Decomposes questions and creates reasoning plans."""
from typing import List, Dict, Any
from agentic.ollama_client import OllamaClient


class PlannerAgent:
    """LLM-based decomposition and multi-step reasoning plan generation."""
    
    def __init__(self, client: OllamaClient, llm_model: str = "gpt-oss:20b"):
        self.client = client
        self.llm_model = llm_model
    
    def plan(self, question: str, scene_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Generate reasoning for answering the question.
        
        Args:
            question: The driving question to answer
            scene_graph: Scene graph from SceneGraphAgent
            
        Returns:
            Reasoning and approach
        """
        scene_desc = scene_graph.get("scene_description", "")
        
        prompt = f"""You are an autonomous driving assistant. Analyze the scene and answer the question.

Scene Description:
{scene_desc}

Question: {question}

Provide your reasoning and answer. Be concise and specific.

Answer:"""
        
        response = self.client.chat_llm(
            model=self.llm_model,
            prompt=prompt,
            temperature=0.1
        )
        
        return {
            "reasoning": response,
            "raw_response": response
        }
    
    def _summarize_scene(self, scene_graph: Dict[str, Any]) -> str:
        """Create text summary of scene graph."""
        nodes = scene_graph.get("nodes", [])
        edges = scene_graph.get("edges", [])
        
        summary = f"Scene contains {len(nodes)} objects: "
        obj_types = [node['type'] for node in nodes]
        summary += ", ".join(obj_types) + ". "
        
        if edges:
            summary += f"Spatial relations: "
            relations = [f"{edge['source']} {edge['relation']} {edge['target']}" for edge in edges[:3]]
            summary += "; ".join(relations) + "."
        
        return summary
    
    def _parse_plan(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured plan steps."""
        steps = []
        lines = response.strip().split('\n')
        
        for line in lines:
            if line.strip().startswith('Step'):
                # Extract method and description
                parts = line.split(':', 1)
                if len(parts) == 2:
                    rest = parts[1].strip()
                    if '-' in rest:
                        method_part, desc = rest.split('-', 1)
                        method_name = method_part.strip().split('(')[0]
                        steps.append({
                            "method": method_name,
                            "raw": method_part.strip(),
                            "description": desc.strip()
                        })
        
        # If no steps parsed, create a default direct answer step
        if not steps:
            steps.append({
                "method": "answer_direct",
                "raw": "answer_direct()",
                "description": "Answer directly from perception"
            })
        
        return steps
