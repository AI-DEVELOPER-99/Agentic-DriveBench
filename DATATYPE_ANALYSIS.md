# Datatype Analysis: Process Implementation

## Summary
Found **6 major datatype mismatches** between method signatures and actual values passed through the agent pipeline.

---

## Issues Found

### 1. **Agent2 Scene Graph Construction - Missing "attributes" key**
**File:** [agentic/agent2_scene_graph.py](agentic/agent2_scene_graph.py#L24)

**Issue:** The `_infer_relation()` method expects objects with `attributes` key
```python
pos1 = obj1["attributes"].get("position", "").lower()  # Line 24
pos2 = obj2["attributes"].get("position", "").lower()  # Line 25
```

**But Agent1 returns objects** without this structure:
```python
detection = {
    "class": self.yolo.names[cls_id],           # Direct keys, not nested
    "confidence": round(conf, 3),
    "bbox": [int(x1), int(y1), int(x2), int(y2)],
    "center": [int(center_x), int(center_y)],
    "camera_view": camera_view,
    "position": position,                        # Not in ["attributes"]
    "distance": distance
}
```

**Impact:** `_infer_relation()` will fail with KeyError when trying to access attributes.

---

### 2. **Agent4 Executor - Expects "nodes" and "edges" keys**
**File:** [agentic/agent4_executor.py](agentic/agent4_executor.py#L58)

**Issue:** Multiple executor methods expect scene_graph structure with "nodes" and "edges":
- Line 58: `nodes = context["scene_graph"]["nodes"]` (count_objects)
- Line 65: `edges = context["scene_graph"]["edges"]` (check_spatial)
- Line 74: `nodes = context["scene_graph"]["nodes"]` (get_attribute)
- Line 87: `nodes = context["scene_graph"]["nodes"]` (check_safety)
- Line 102: `nodes = context["scene_graph"]["nodes"]` (predict_behavior)

**But Agent2 returns:**
```python
return {
    "scene_description": description,
    "raw_perception": perception_result
}
```
**No "nodes" or "edges" keys!**

**Impact:** All executor methods will crash with KeyError when executed. The `execute()` method calls `_execute_step()` which dispatches to these methods.

---

### 3. **Agent4 Executor - "context" parameter mismatch**
**File:** [agentic/agent4_executor.py](agentic/agent4_executor.py#L12-L23)

**Issue:** The `execute()` method doesn't pass context to `_execute_step()`:
```python
def execute(self, plan: Dict[str, Any], scene_graph: Dict[str, Any], 
            perception_result: Dict[str, Any]) -> Dict[str, Any]:
    reasoning = plan.get("reasoning", "")
    return {
        "answer": reasoning,
        "reasoning": reasoning
    }
```

**But internal methods expect context:**
```python
def count_objects(self, step: Dict[str, Any], context: Dict[str, Any]) -> int:
```

**Impact:** The `_execute_step()` method is never called. If it were, it would receive undefined context.

---

### 4. **Agent4 answer_direct() - Expects different structure**
**File:** [agentic/agent4_executor.py](agentic/agent4_executor.py#L110-L125)

**Issue:** The `answer_direct()` method expects:
```python
perception = context["perception"]
raw_response = perception.get("raw_response", "")
objects = perception.get("objects", [])  # Expects "objects" key
    for idx, obj in enumerate(objects):
        obj_desc = f"{obj.get('object', 'object')} ({obj.get('color', '')}) at {obj.get('position', '')}"
```

**But Agent1 returns:**
```python
result = {
    "detections": all_detections,  # Not "objects"
    "description": description,
    "detection_method": "yolo"
}
```

**Impact:** Expects `objects` key but gets `detections`. Also expects keys like `'object'` and `'color'` that aren't provided by YOLO detection.

---

### 5. **Agent3 Planner - Coordinate format mismatch**
**File:** [agentic/agent3_planner.py](agentic/agent3_planner.py#L20)

**Issue:** Uses regex pattern expecting normalized coordinates (0-1):
```python
match = re.search(r'<c\d+,(\w+),([\d.]+),([\d.]+)>', question)
if match:
    cam = match.group(1)
    x = float(match.group(2))  # Expects 0-1
    y = float(match.group(3))  # Expects 0-1
    obj = self._find_object_at(cam, x, y, scene_graph)
```

**But looks for matching in pixel coordinates:**
```python
def _find_object_at(self, camera: str, x: float, y: float, scene_graph: Dict[str, Any]) -> Dict[str, Any] | None:
    detections = scene_graph.get("raw_perception", {}).get("detections", [])
    closest = min(candidates, key=lambda d: ((d["center"][0] - x)**2 + (d["center"][1] - y)**2)**0.5)
    
    if distance < 0.1:  # Normalized distance threshold
        return closest
```

**Detection center has pixel coordinates:**
```python
"center": [int(center_x), int(center_y)],  # Pixel coords, not normalized
```

**Impact:** Distance calculation will be wrong. Pixel coordinates (e.g., 500, 300) compared to normalized (0.5, 0.3) will never match within 0.1 threshold.

---

### 6. **Pipeline - Type annotation inconsistency**
**File:** [agentic/pipeline.py](agentic/pipeline.py#L45)

**Issue:** `image_paths` parameter type is `Dict[str, str]` but contains path strings:
```python
def process(self, question: str, image_paths: Dict[str, str]) -> Dict[str, Any]:
    images = list(image_paths.values())  # Extract to list
    perception_result = self.agent1.perceive(images, question, use_vlm=True)
```

Agent1 expects `List[str]` ✓ This one is CORRECT - no issue here.

---

## Summary Table

| Issue | Component | Type | Severity | Status |
|-------|-----------|------|----------|--------|
| Missing "attributes" in object structure | Agent2 _infer_relation() | KeyError | **CRITICAL** | ❌ |
| Missing "nodes"/"edges" keys | Agent4 executor methods | KeyError | **CRITICAL** | ❌ |
| Context not passed to _execute_step() | Agent4 execute() | Logic Error | **HIGH** | ❌ |
| Expects "objects" not "detections" | Agent4 answer_direct() | KeyError | **CRITICAL** | ❌ |
| Coordinate format mismatch (pixel vs normalized) | Agent3 _find_object_at() | Logic Error | **HIGH** | ❌ |

---

## Files Affected
- ❌ [agentic/agent2_scene_graph.py](agentic/agent2_scene_graph.py) - Scene graph structure
- ❌ [agentic/agent4_executor.py](agentic/agent4_executor.py) - Expects wrong context structure
- ❌ [agentic/agent3_planner.py](agentic/agent3_planner.py) - Coordinate mismatch
- ⚠️ [agentic/pipeline.py](agentic/pipeline.py) - Data flow coordinator
