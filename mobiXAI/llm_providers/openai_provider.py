from .base import LLMProvider
from openai import OpenAI
from typing import List, Dict
import os

class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
        self._available_models = [
            "gpt-4-turbo-preview",
            "gpt-4",
            "gpt-4-32k",
            "gpt-3.5-turbo",
            "gpt-3.5-turbo-16k"
        ]
    
    def get_provider_name(self) -> str:
        return "openai"
    
    def get_available_models(self) -> List[str]:
        return self._available_models
    
    def validate_api_key(self) -> bool:
        try:
            # Try to list models to validate API key
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def generate_response(self, messages: List[Dict[str, str]], model: str = "gpt-4-turbo-preview", 
                        temperature: float = 0, max_tokens: int = 1000, **kwargs) -> str:
        try:
            response = self.client.chat.completions.create(
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=messages,
                **kwargs
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "gpt-4-turbo-preview" 