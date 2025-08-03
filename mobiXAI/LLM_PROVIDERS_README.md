# LLM-Agnostic System for MobiSpaces XAI

This document describes the LLM-agnostic system that allows users to choose from multiple LLM providers for generating explanations in the MobiSpaces XAI system.

## Overview

The system supports multiple LLM providers through a unified interface, allowing users to:
- Choose from different LLM providers (OpenAI, Anthropic, Google, Cohere)
- Select specific models from each provider
- Configure parameters like temperature and max tokens
- Get consistent responses regardless of the underlying LLM

## Supported Providers

### 1. OpenAI
- **Models**: GPT-4 Turbo, GPT-4, GPT-4-32k, GPT-3.5 Turbo, GPT-3.5 Turbo 16k
- **API Key**: `OPENAI_API_KEY`
- **Default Model**: `gpt-4-turbo-preview`

### 2. Anthropic (Claude)
- **Models**: Claude 3 Sonnet, Claude 3 Opus, Claude 3 Haiku, Claude 2.1, Claude 2.0
- **API Key**: `ANTHROPIC_API_KEY`
- **Default Model**: `claude-3-sonnet-20240229`

### 3. Google (Gemini)
- **Models**: Gemini Pro, Gemini Pro Vision, Gemini 1.5 Pro, Gemini 1.5 Flash
- **API Key**: `GOOGLE_API_KEY`
- **Default Model**: `gemini-pro`

### 4. Cohere
- **Models**: Command, Command Light, Command Nightly, Command Light Nightly
- **API Key**: `COHERE_API_KEY`
- **Default Model**: `command`

## Architecture

### Provider Interface
All providers implement the `LLMProvider` base class:
```python
class LLMProvider(ABC):
    def generate_response(self, messages, **kwargs) -> str
    def get_available_models(self) -> List[str]
    def get_provider_name(self) -> str
    def validate_api_key(self) -> bool
```

### Factory Pattern
The `LLMFactory` manages provider creation and information:
```python
class LLMFactory:
    def create_provider(provider_name, api_key) -> LLMProvider
    def get_available_providers() -> List[str]
    def get_provider_info(provider_name, api_key) -> Dict
```

## API Endpoints

### GET `/llm_providers`
Returns information about available LLM providers and their models.

**Response:**
```json
{
  "providers": {
    "openai": {
      "name": "openai",
      "available_models": ["gpt-4-turbo-preview", "gpt-4", ...],
      "default_model": "gpt-4-turbo-preview",
      "default_temperature": 0.0,
      "api_key_valid": true
    },
    "anthropic": {
      "name": "anthropic",
      "available_models": ["claude-3-sonnet-20240229", ...],
      "default_model": "claude-3-sonnet-20240229",
      "default_temperature": 0.0,
      "api_key_valid": true
    }
  },
  "default_provider": "openai",
  "default_model": "gpt-4-turbo-preview",
  "default_temperature": 0.0
}
```

### POST `/chat_response`
Generates LLM explanations with configurable parameters.

**Request:**
```json
{
  "explanation": "LIME explanation data",
  "provider": "openai",
  "model": "gpt-4-turbo-preview",
  "temperature": 0.0,
  "max_tokens": 1000
}
```

**Response:**
```json
"Generated explanation text from the selected LLM"
```

## Configuration

### Environment Variables
Set the following environment variables in your `docker-compose.yml`:

```yaml
environment:
  - OPENAI_API_KEY="your-openai-key"
  - ANTHROPIC_API_KEY="your-anthropic-key"
  - GOOGLE_API_KEY="your-google-key"
  - COHERE_API_KEY="your-cohere-key"
  - DEFAULT_LLM_PROVIDER=openai
  - DEFAULT_LLM_MODEL=gpt-4-turbo-preview
  - DEFAULT_LLM_TEMPERATURE=0.0
```

### Dependencies
Add the following to `requirements.txt`:
```
openai
anthropic
google-generativeai
cohere
```

## Usage Examples

### Python Client
```python
import requests

# Get available providers
response = requests.get("http://localhost:8881/llm_providers")
providers = response.json()

# Generate explanation with OpenAI
payload = {
    "explanation": "Your LIME explanation data",
    "provider": "openai",
    "model": "gpt-4-turbo-preview",
    "temperature": 0.0,
    "max_tokens": 1000
}
response = requests.post("http://localhost:8881/chat_response", 
                       data=json.dumps(payload))
explanation = response.text
```

### Frontend Integration
The Streamlit frontend automatically:
1. Fetches available providers from `/llm_providers`
2. Shows dropdown for provider selection
3. Shows dropdown for model selection
4. Provides sliders for temperature and max tokens
5. Sends requests to `/chat_response` with selected parameters

## Adding New Providers

To add a new LLM provider:

1. **Create Provider Class**:
```python
# llm_providers/new_provider.py
from .base import LLMProvider

class NewProvider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize your client
    
    def get_provider_name(self) -> str:
        return "new_provider"
    
    def get_available_models(self) -> List[str]:
        return ["model1", "model2"]
    
    def validate_api_key(self) -> bool:
        # Validate API key
        pass
    
    def generate_response(self, messages, **kwargs) -> str:
        # Generate response using your API
        pass
```

2. **Register in Factory**:
```python
# llm_providers/factory.py
from .new_provider import NewProvider

class LLMFactory:
    _providers = {
        "openai": OpenAIProvider,
        "anthropic": AnthropicProvider,
        "google": GoogleProvider,
        "cohere": CohereProvider,
        "new_provider": NewProvider  # Add this line
    }
```

3. **Add Environment Variable**:
```yaml
environment:
  - NEW_PROVIDER_API_KEY="your-api-key"
```

4. **Update App Configuration**:
```python
# app.py
NEW_PROVIDER_API_KEY = os.getenv("NEW_PROVIDER_API_KEY", "")

# In chat_response_func
api_keys = {
    "openai": OPENAI_API_KEY,
    "anthropic": ANTHROPIC_API_KEY,
    "google": GOOGLE_API_KEY,
    "cohere": COHERE_API_KEY,
    "new_provider": NEW_PROVIDER_API_KEY  # Add this line
}
```

## Error Handling

The system includes comprehensive error handling:
- **Invalid API Keys**: Returns error message with provider name
- **Unavailable Models**: Falls back to default model
- **API Errors**: Returns detailed error messages
- **Network Issues**: Graceful degradation with fallback providers

## Performance Considerations

- **API Key Validation**: Cached validation to avoid repeated API calls
- **Model Lists**: Cached model lists for each provider
- **Error Recovery**: Automatic fallback to default provider on errors
- **Rate Limiting**: Respects provider-specific rate limits

## Security

- **API Key Management**: Environment variable-based configuration
- **Input Validation**: All inputs validated before processing
- **Error Sanitization**: Error messages sanitized to prevent information leakage
- **Provider Isolation**: Each provider runs in isolated context

## Testing

Test the system with:
```bash
# Test provider info endpoint
curl http://localhost:8881/llm_providers

# Test chat response endpoint
curl -X POST http://localhost:8881/chat_response \
  -H "Content-Type: application/json" \
  -d '{"explanation":"test","provider":"openai","model":"gpt-4-turbo-preview"}'
```

## Troubleshooting

### Common Issues

1. **"Missing API key for provider"**
   - Check environment variables are set correctly
   - Verify API key format and validity

2. **"API key not valid"**
   - Test API key directly with provider
   - Check account status and billing

3. **"Model not available"**
   - Check model name spelling
   - Verify model availability in your account

4. **"Error generating response"**
   - Check network connectivity
   - Verify API quotas and rate limits
   - Check provider service status

### Debug Mode
Enable debug logging by setting:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

- **Local Models**: Support for local LLM models (Llama, Mistral)
- **Model Comparison**: Side-by-side comparison of different models
- **Response Caching**: Cache responses to reduce API costs
- **Batch Processing**: Support for batch explanation generation
- **Custom Prompts**: Allow users to customize system prompts
- **Response Analysis**: Analyze response quality and consistency 