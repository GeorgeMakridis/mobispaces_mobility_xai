from .base import LLMProvider
import anthropic
from typing import List, Dict

class AnthropicProvider(LLMProvider):
    """Anthropic LLM Provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = anthropic.Anthropic(api_key=api_key)
        self._available_models = [
            "claude-3-sonnet-20240229",
            "claude-3-opus-20240229", 
            "claude-3-haiku-20240307",
            "claude-2.1",
            "claude-2.0"
        ]
    
    def get_provider_name(self) -> str:
        return "anthropic"
    
    def get_available_models(self) -> List[str]:
        return self._available_models
    
    def validate_api_key(self) -> bool:
        try:
            # Try to list models to validate API key
            self.client.models.list()
            return True
        except Exception:
            return False
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to Anthropic format"""
        prompt = ""
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            
            if role == "system":
                prompt += f"System: {content}\n\n"
            elif role == "user":
                prompt += f"Human: {content}\n\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n\n"
        
        prompt += "Assistant:"
        return prompt
    
    def generate_response(self, messages: List[Dict[str, str]], model: str = "claude-3-sonnet-20240229", 
                        temperature: float = 0, max_tokens: int = 1000, **kwargs) -> str:
        try:
            prompt = self._convert_messages_to_prompt(messages)
            response = self.client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}],
                **kwargs
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "claude-3-sonnet-20240229" 