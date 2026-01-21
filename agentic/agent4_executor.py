"""Agent 4: Executor Agent - Executes reasoning plan steps."""
from typing import List, Dict, Any
import re


class ExecutorAgent:
    """Executes plan steps using defined methods."""
    
    def __init__(self):
        self.execution_trace = []
    
    def execute(self, plan: Dict[str, Any], scene_graph: Dict[str, Any], 
                perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the reasoning plan.
        
        Args:
            plan: Reasoning from PlannerAgent
            scene_graph: Scene graph from SceneGraphAgent
            perception_result: Perception results from PerceptionAgent
            
        Returns:
            Execution results
        """
        reasoning = plan.get("reasoning", "")
        
        return {
            "answer": reasoning,
            "reasoning": reasoning
        }
    
    def _execute_step(self, step: Dict[str, Any], context: Dict[str, Any]) -> Any:
        """Execute a single step."""
        method = step["method"]
        
        # Dispatch to appropriate method
        if method == "count_objects":
            return self.count_objects(step, context)
        elif method == "check_spatial":
            return self.check_spatial(step, context)
        elif method == "get_attribute":
            return self.get_attribute(step, context)
        elif method == "check_safety":
            return self.check_safety(step, context)
        elif method == "predict_behavior":
            return self.predict_behavior(step, context)
        elif method == "answer_direct":
            return self.answer_direct(step, context)
        else:
            return self.answer_direct(step, context)
    
    def count_objects(self, step: Dict[str, Any], context: Dict[str, Any]) -> int:
        """Count objects of specific type."""
        raw = step.get("raw", "")
        match = re.search(r'count_objects\(["\']?(\w+)["\']?\)', raw)
        
        if match:
            obj_type = match.group(1).lower()
            nodes = context["scene_graph"]["nodes"]
            count = sum(1 for node in nodes if obj_type in node["type"].lower())
            return count
        return 0
    
    def check_spatial(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check spatial relationship between objects."""
        edges = context["scene_graph"]["edges"]
        raw = step.get("raw", "")
        
        # Simple check - if there are spatial relations, return True
        return len(edges) > 0
    
    def get_attribute(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Get attribute of an object."""
        raw = step.get("raw", "")
        nodes = context["scene_graph"]["nodes"]
        
        if nodes:
            # Return first matching attribute
            for node in nodes:
                for attr_key, attr_val in node["attributes"].items():
                    if attr_key in raw.lower() and attr_val:
                        return attr_val
        
        return "unknown"
    
    def check_safety(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Evaluate safety of actions."""
        nodes = context["scene_graph"]["nodes"]
        
        # Simple safety heuristic
        close_objects = sum(1 for node in nodes 
                          if node["attributes"].get("distance") == "close")
        
        if close_objects > 2:
            return "High risk - multiple close objects detected"
        elif close_objects > 0:
            return "Moderate risk - maintain safe distance"
        else:
            return "Safe to proceed with caution"
    
    def predict_behavior(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Predict object behavior."""
        nodes = context["scene_graph"]["nodes"]
        
        for node in nodes:
            status = node["attributes"].get("status", "").lower()
            if "moving" in status:
                return f"{node['type']} is likely to continue moving"
        
        return "Object appears stationary"
    
    def answer_direct(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate answer directly from perception."""
        perception = context["perception"]
        raw_response = perception.get("raw_response", "")
        objects = perception.get("objects", [])
        
        if objects:
            # Create structured answer from detected objects
            answer = []
            for idx, obj in enumerate(objects):
                obj_desc = f"{obj.get('object', 'object')} ({obj.get('color', '')}) at {obj.get('position', '')}"
                answer.append(obj_desc)
            return "; ".join(answer)
        
        return raw_response
