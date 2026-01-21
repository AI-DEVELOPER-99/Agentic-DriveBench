"""Simple Ollama client for VLM and LLM interactions."""
import requests
import json
import base64
from typing import List, Dict, Optional


class OllamaClient:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
    
    def chat_vlm(self, model: str, prompt: str, images: List[str], temperature: float = 0.0) -> str:
        """Query VLM with images and prompt.
        
        Args:
            model: Model name (e.g., 'llava:latest')
            prompt: Text prompt
            images: List of image paths
            temperature: Sampling temperature
            
        Returns:
            Model response as string
        """
        # Encode images to base64
        encoded_images = []
        for img_path in images:
            with open(img_path, 'rb') as f:
                encoded_images.append(base64.b64encode(f.read()).decode('utf-8'))
        
        payload = {
            "model": model,
            "prompt": prompt,
            "images": encoded_images,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()['response']
    
    def chat_llm(self, model: str, prompt: str, temperature: float = 0.0) -> str:
        """Query LLM with text prompt.
        
        Args:
            model: Model name (e.g., 'gpt-oss-20b')
            prompt: Text prompt
            temperature: Sampling temperature
            
        Returns:
            Model response as string
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature
            }
        }
        
        response = requests.post(f"{self.base_url}/api/generate", json=payload)
        response.raise_for_status()
        return response.json()['response']
