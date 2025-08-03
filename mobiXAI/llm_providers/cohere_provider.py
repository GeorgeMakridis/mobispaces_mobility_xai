from .base import LLMProvider
import cohere
from typing import List, Dict

class CohereProvider(LLMProvider):
    """Cohere LLM Provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = cohere.Client(api_key=api_key)
        self._available_models = [
            "command",
            "command-light",
            "command-nightly",
            "command-light-nightly"
        ]
    
    def get_provider_name(self) -> str:
        return "cohere"
    
    def get_available_models(self) -> List[str]:
        return self._available_models
    
    def validate_api_key(self) -> bool:
        try:
            # Try to list models to validate API key
            response = self.client.models.list()
            return len(response.models) > 0
        except Exception:
            return False
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to Cohere format"""
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"User: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        return prompt
    
    def generate_response(self, messages: List[Dict[str, str]], model: str = "command", 
                        temperature: float = 0, max_tokens: int = 1000, **kwargs) -> str:
        try:
            prompt = self._convert_messages_to_prompt(messages)
            response = self.client.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs
            )
            return response.generations[0].text
        except Exception as e:
            raise Exception(f"Cohere API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "command" 