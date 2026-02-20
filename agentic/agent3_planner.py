"""Agent 3: Planner Agent - Decomposes questions and creates reasoning plans."""
from typing import List, Dict, Any, Optional
from .ollama_client import OllamaClient
import re


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
        
        obj_info = ""
        match = re.search(r'<c\d+,(\w+),([\d.]+),([\d.]+)>', question)
        if match:
            cam = match.group(1)
            x = float(match.group(2))
            y = float(match.group(3))
            obj = self._find_object_at(cam, x, y, scene_graph)
            if obj:
                obj_info = f"The object at {cam} {x},{y} is a {obj['class']}."
            else:
                obj_info = f"No object detected at {cam} {x},{y}."
        
        prompt = f"""You are an autonomous driving assistant. Analyze the scene and answer the question.

Scene Description:
{scene_desc}

Object Information:
{obj_info}

Question: {question}

Provide your reasoning and answer. Be concise and specific. Focus on the specified object if mentioned.

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

    def _find_object_at(self, camera: str, x: float, y: float, scene_graph: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find the detected object closest to the given coordinates in the camera."""
        detections = scene_graph.get("raw_perception", {}).get("detections", [])
        candidates = [d for d in detections if d.get("camera_view") == camera.lower()]
        
        if not candidates:
            return None
        
        # Normalize input coordinates if they appear to be in 0-1 range
        # YOLO uses pixel coordinates, but input coords are normalized (0-1)
        img_width = 1920  # Standard image width
        img_height = 1080  # Standard image height
        
        # If x, y are small (< 2), assume they're normalized
        if x < 2 and y < 2:
            x_pixel = x * img_width
            y_pixel = y * img_height
        else:
            x_pixel = x
            y_pixel = y
        
        # Find closest by Euclidean distance
        closest = min(candidates, key=lambda d: ((d["center"][0] - x_pixel)**2 + (d["center"][1] - y_pixel)**2)**0.5)
        distance = ((closest["center"][0] - x_pixel)**2 + (closest["center"][1] - y_pixel)**2)**0.5
        
        # If within reasonable pixel distance (e.g., 100 pixels)
        if distance < 100:
            return closest
        return None