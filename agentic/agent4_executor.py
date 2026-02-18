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
            Execution results (always a dict with answer, reasoning, trace keys)
        """
        try:
            reasoning = plan.get("reasoning", "")
            
            # Build context for step execution
            context = {
                "scene_graph": scene_graph,
                "perception": perception_result,
                "plan": plan
            }
            
            # Try to execute steps from plan
            steps = plan.get("steps", [])
            execution_trace = []
            
            if steps:
                for step in steps:
                    try:
                        result = self._execute_step(step, context)
                        execution_trace.append({
                            "step": step,
                            "result": result
                        })
                    except Exception as e:
                        execution_trace.append({
                            "step": step,
                            "result": f"Error executing step: {str(e)}"
                        })
                final_answer = execution_trace[-1]["result"] if execution_trace else reasoning
            else:
                step = {"method": "answer_direct", "raw": "answer_direct()", "description": "Answer directly"}
                final_answer = self.answer_direct(step, context)
                execution_trace.append({"step": step, "result": final_answer})
            
            self.execution_trace = execution_trace
            
            return {
                "answer": final_answer if isinstance(final_answer, str) else str(final_answer),
                "reasoning": reasoning,
                "trace": execution_trace
            }
        except Exception as e:
            # Fallback error response
            return {
                "answer": f"Error during execution: {str(e)}",
                "reasoning": plan.get("reasoning", ""),
                "trace": []
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
            nodes = context["scene_graph"].get("nodes", [])
            count = sum(1 for node in nodes if obj_type in node.get("type", "").lower())
            return count
        return 0
    
    def check_spatial(self, step: Dict[str, Any], context: Dict[str, Any]) -> bool:
        """Check spatial relationship between objects."""
        edges = context["scene_graph"].get("edges", [])
        raw = step.get("raw", "")
        
        # Simple check - if there are spatial relations, return True
        return len(edges) > 0
    
    def get_attribute(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Get attribute of an object."""
        raw = step.get("raw", "")
        nodes = context["scene_graph"].get("nodes", [])
        
        if nodes:
            # Return first matching attribute
            for node in nodes:
                for attr_key, attr_val in node.get("attributes", {}).items():
                    if attr_key in raw.lower() and attr_val:
                        return str(attr_val)
        
        return "unknown"
    
    def check_safety(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Evaluate safety of actions."""
        nodes = context["scene_graph"].get("nodes", [])
        
        # Simple safety heuristic
        close_objects = sum(1 for node in nodes 
                          if node.get("attributes", {}).get("distance") == "close")
        
        if close_objects > 2:
            return "High risk - multiple close objects detected"
        elif close_objects > 0:
            return "Moderate risk - maintain safe distance"
        else:
            return "Safe to proceed with caution"
    
    def predict_behavior(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Predict object behavior."""
        nodes = context["scene_graph"].get("nodes", [])
        
        for node in nodes:
            status = node.get("attributes", {}).get("status", "").lower()
            if "moving" in status:
                return f"{node.get('type', 'object')} is likely to continue moving"
        
        return "Object appears stationary"
    
    def answer_direct(self, step: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Generate answer directly from perception."""
        perception = context.get("perception", {})
        raw_response = perception.get("description", "")
        detections = perception.get("detections", [])
        
        if detections:
            # Create structured answer from detected objects
            answer = []
            for idx, obj in enumerate(detections):
                obj_desc = f"{obj.get('class', 'object')} at {obj.get('position', '')} ({obj.get('distance', '')})"
                answer.append(obj_desc)
            return "; ".join(answer)
        
        return raw_response if raw_response else "Unable to determine answer from perception"
