from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional

class LLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def get_available_models(self) -> List[str]:
        """Get list of available models for this provider"""
        pass
    
    @abstractmethod
    def get_provider_name(self) -> str:
        """Get the name of this provider"""
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        """Validate if the API key is valid"""
        pass
    
    def get_default_model(self) -> str:
        """Get the default model for this provider"""
        models = self.get_available_models()
        return models[0] if models else ""
    
    def get_default_temperature(self) -> float:
        """Get the default temperature for this provider"""
        return 0.0 