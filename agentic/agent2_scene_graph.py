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
            Scene graph with nodes and edges
        """
        description = perception_result.get("description", "")
        detections = perception_result.get("detections", [])
        
        # Build nodes from detections
        nodes = []
        for idx, det in enumerate(detections):
            node = {
                "id": f"obj_{idx}",
                "type": det.get("class", "unknown"),
                "attributes": {
                    "position": det.get("position", ""),
                    "distance": det.get("distance", ""),
                    "confidence": det.get("confidence", 0),
                    "camera_view": det.get("camera_view", "")
                }
            }
            nodes.append(node)
        
        # Build edges (spatial relations)
        edges = []
        for i, obj1 in enumerate(nodes):
            for j, obj2 in enumerate(nodes):
                if i < j:
                    relation = self._infer_relation(obj1, obj2)
                    edges.append({
                        "source": obj1["id"],
                        "target": obj2["id"],
                        "relation": relation
                    })
        
        return {
            "scene_description": description,
            "nodes": nodes,
            "edges": edges,
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
