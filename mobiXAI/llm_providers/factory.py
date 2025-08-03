from typing import Dict, Type, List, Optional
from .base import LLMProvider
from .openai_provider import OpenAIProvider
from .anthropic_provider import AnthropicProvider
from .google_provider import GoogleProvider
from .cohere_provider import CohereProvider

class LLMFactory:
    """Factory class for creating LLM providers"""
    
    _providers: Dict[str, Type[LLMProvider]] = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "cohere": CohereProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str, api_key: str) -> LLMProvider:
        """Create a provider instance"""
        if provider_name not in cls._providers:
            raise ValueError(f"Unknown provider: {provider_name}")
        
        provider_class = cls._providers[provider_name]
        return provider_class(api_key)
    
    @classmethod
    def get_available_providers(cls) -> List[str]:
        """Get list of available provider names"""
        return list(cls._providers.keys())
    
    @classmethod
    def get_provider_info(cls, provider_name: str, api_key: str) -> Dict:
        """Get information about a specific provider"""
        try:
            provider = cls.create_provider(provider_name, api_key)
            return {
                "name": provider.get_provider_name(),
                "available_models": provider.get_available_models(),
                "default_model": provider.get_default_model(),
                "default_temperature": provider.get_default_temperature(),
                "api_key_valid": provider.validate_api_key()
            }
        except Exception as e:
            return {
                "name": provider_name,
                "available_models": [],
                "default_model": "",
                "default_temperature": 0.0,
                "api_key_valid": False,
                "error": str(e)
            }
    
    @classmethod
    def get_all_providers_info(cls, api_keys: Dict[str, str]) -> Dict[str, Dict]:
        """Get information about all providers"""
        providers_info = {}
        for provider_name in cls.get_available_providers():
            api_key = api_keys.get(provider_name, "")
            providers_info[provider_name] = cls.get_provider_info(provider_name, api_key)
        return providers_info 