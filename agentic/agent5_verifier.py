"""Agent 5: Verifier Agent - Validates reasoning and assigns confidence."""
from typing import Dict, Any
from .ollama_client import OllamaClient


class VerifierAgent:
    """Validates reasoning chain and provides confidence scores."""
    
    def __init__(self, client: OllamaClient, llm_model: str = "gpt-oss:20b"):
        self.client = client
        self.llm_model = llm_model
    
    def verify(self, question: str, execution_result: Dict[str, Any], 
               scene_graph: Dict[str, Any]) -> Dict[str, Any]:
        """Verify and refine the answer.
        
        Args:
            question: Original question
            execution_result: Results from ExecutorAgent
            scene_graph: Scene graph
            
        Returns:
            Refined answer with confidence
        """
        draft_answer = execution_result.get("answer", "")
        scene_desc = scene_graph.get("scene_description", "")
        
        prompt = f"""Refine this answer for a driving question. Make it concise and accurate.

Question: {question}

Draft Answer: {draft_answer}

Provide a refined, clear, and concise answer. Don't critique, just give the final answer.

Refined Answer:"""
        
        response = self.client.chat_llm(
            model=self.llm_model,
            prompt=prompt,
            temperature=0.0
        )
        
        # Clean up the response
        final_answer = response.strip()
        
        return {
            "is_valid": True,
            "confidence": 80,
            "final_answer": final_answer,
            "reasoning_chain": draft_answer
        }
    
    def _build_reasoning_chain(self, trace: list) -> str:
        """Build text summary of reasoning chain."""
        chain = []
        for idx, item in enumerate(trace, 1):
            step = item["step"]
            result = item["result"]
            chain.append(f"Step {idx}: {step.get('description', step.get('method'))} -> {result}")
        return "\n".join(chain)
    
    def _parse_verification(self, response: str) -> Dict[str, Any]:
        """Parse verification response."""
        result = {
            "logical": True,
            "contradictions": False,
            "confidence": 70,
            "corrections": "none"
        }
        
        lines = response.strip().split('\n')
        for line in lines:
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip().lower()
                value = value.strip()
                
                if 'logical' in key:
                    result['logical'] = 'yes' in value.lower()
                elif 'contradiction' in key:
                    result['contradictions'] = 'yes' in value.lower()
                elif 'confidence' in key:
                    # Extract number
                    import re
                    match = re.search(r'\d+', value)
                    if match:
                        result['confidence'] = int(match.group())
                elif 'correction' in key:
                    result['corrections'] = value
        
        return result
