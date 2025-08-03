from .base import LLMProvider
import google.generativeai as genai
from typing import List, Dict

class GoogleProvider(LLMProvider):
    """Google Gemini LLM Provider"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self._available_models = [
            "gemini-pro",
            "gemini-pro-vision",
            "gemini-1.5-pro",
            "gemini-1.5-flash"
        ]
    
    def get_provider_name(self) -> str:
        return "google"
    
    def get_available_models(self) -> List[str]:
        return self._available_models
    
    def validate_api_key(self) -> bool:
        try:
            # Try to list models to validate API key
            models = genai.list_models()
            return len(list(models)) > 0
        except Exception:
            return False
    
    def _convert_messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert OpenAI format messages to Google format"""
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
    
    def generate_response(self, messages: List[Dict[str, str]], model: str = "gemini-pro", 
                        temperature: float = 0, max_tokens: int = 1000, **kwargs) -> str:
        try:
            prompt = self._convert_messages_to_prompt(messages)
            genai_model = genai.GenerativeModel(model)
            
            # Configure generation config
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
                **kwargs
            )
            
            response = genai_model.generate_content(
                prompt,
                generation_config=generation_config
            )
            return response.text
        except Exception as e:
            raise Exception(f"Google API error: {str(e)}")
    
    def get_default_model(self) -> str:
        return "gemini-pro" 