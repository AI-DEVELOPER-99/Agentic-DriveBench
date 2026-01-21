"""Agent 2: Scene Graph Agent - Constructs spatial relationships between objects."""
from typing import List, Dict, Any


class SceneGraphAgent:
    """Constructs a scene graph with spatial relations and object attributes."""
    
    def __init__(self):
        pass
    
    def construct_graph(self, perception_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build scene graph from perception results.
        
        Args:
            perception_result: Output from PerceptionAgent
            
        Returns:
            Scene graph with summary
        """
        description = perception_result.get("description", "")
        
        return {
            "scene_description": description,
            "raw_perception": perception_result
        }
    
    def _infer_relation(self, obj1: Dict, obj2: Dict) -> str:
        """Infer spatial relation between two objects."""
        pos1 = obj1["attributes"].get("position", "").lower()
        pos2 = obj2["attributes"].get("position", "").lower()
        
        # Simple spatial relations
        if "front" in pos1 and "back" in pos2:
            return "in_front_of"
        elif "back" in pos1 and "front" in pos2:
            return "behind"
        elif "left" in pos1 and "right" in pos2:
            return "left_of"
        elif "right" in pos1 and "left" in pos2:
            return "right_of"
        
        # Distance-based relations
        dist1 = obj1["attributes"].get("distance", "").lower()
        dist2 = obj2["attributes"].get("distance", "").lower()
        
        if dist1 == "close" and dist2 == "far":
            return "closer_than"
        elif dist1 == "far" and dist2 == "close":
            return "farther_than"
        
        return "near"
