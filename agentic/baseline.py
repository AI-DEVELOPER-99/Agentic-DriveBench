"""Baseline VLM for comparison."""
from typing import Dict
from .ollama_client import OllamaClient


class BaselineVLM:
    """Simple baseline using VLM directly without agentic pipeline."""
    
    def __init__(self, 
                 model: str = "llava:latest",
                 ollama_url: str = "http://localhost:11434"):
        """Initialize baseline VLM.
        
        Args:
            model: Vision-language model name
            ollama_url: Ollama server URL
        """
        self.client = OllamaClient(base_url=ollama_url)
        self.model = model
    
    def answer_question(self, question: str, image_paths: Dict[str, str]) -> str:
        """Answer question using VLM directly.
        
        Args:
            question: The driving question
            image_paths: Dictionary mapping camera positions to image paths
            
        Returns:
            Answer string
        """
        images = list(image_paths.values())
        
        # Simple prompt
        prompt = f"""You are an autonomous driving assistant. Analyze the provided camera images and answer the following question accurately and concisely.

Question: {question}

Answer:"""
        
        response = self.client.chat_vlm(
            model=self.model,
            prompt=prompt,
            images=images,
            temperature=0.0
        )
        
        return response.strip()
