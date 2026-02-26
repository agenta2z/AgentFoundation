# AI Gateway API Clients

This package provides API clients that route LLM requests through Atlassian's AI Gateway, offering centralized authentication, rate limiting, monitoring, and cost tracking.

## Prerequisites

1. **Install AI Gateway SDK**:
   ```bash
   pip install atlassian-ai-gateway-sdk atlassian_ml_studio_sdk
   ```

2. **Install and start SLAUTH server**:
   ```bash
   # Install atlas slauth plugin
   atlas plugin install -n slauth
   
   # Start SLAUTH server (required for authentication)
   atlas slauth server --port 5000
   ```

3. **Set environment variables** (optional but recommended):
   ```bash
   export AI_GATEWAY_USER_ID="your_staff_id"  # Optional: falls back to $USER if not set
   export AI_GATEWAY_BASE_URL="https://ai-gateway.us-east-1.staging.atl-paas.net"
   export AI_GATEWAY_CLOUD_ID="local"
   export AI_GATEWAY_USE_CASE_ID="ai-gateway-eval-use-case"
   export SLAUTH_SERVER_URL="http://localhost:5000"
   ```
   
   **Note:** If `AI_GATEWAY_USER_ID` is not set, the module will automatically use your system username from `$USER`.

## Usage

### AI Gateway Claude LLM

Route Claude API requests through AI Gateway with Bedrock backend.

#### Basic Example

```python
from science_modeling_tools.apis.ai_gateway import ai_gateway_claude_llm

# Simple text generation
response = ai_gateway_claude_llm.generate_text(
    prompt_or_messages="What is the capital of France?",
    user_id="your_staff_id"  # Required if not set in environment
)
print(response)
```

#### Using Different Models

```python
from science_modeling_tools.apis.ai_gateway.ai_gateway_claude_llm import (
    generate_text,
    AIGatewayClaudeModels
)

# Use Claude Sonnet 4.5 (default)
response = generate_text(
    "Explain quantum computing in simple terms",
    model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
    user_id="your_staff_id"
)

# Use Claude Opus 4.1 for more complex reasoning
response = generate_text(
    "Write a detailed analysis of...",
    model=AIGatewayClaudeModels.CLAUDE_41_OPUS,
    user_id="your_staff_id",
    max_new_tokens=4096
)

# Use Claude Haiku 4.5 for faster, lighter tasks
response = generate_text(
    "Summarize this in one sentence",
    model=AIGatewayClaudeModels.CLAUDE_45_HAIKU,
    user_id="your_staff_id"
)
```

#### Multi-turn Conversations

```python
# As a list of alternating user/assistant messages
conversation = [
    "Hello, I need help with Python",
    "Sure! What would you like to know?",
    "How do I read a file?"
]

response = generate_text(
    conversation,
    model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
    user_id="your_staff_id"
)

# Or as a list of message dictionaries
messages = [
    {
        "role": "user",
        "content": [{"type": "text", "text": "What is machine learning?"}]
    },
    {
        "role": "assistant",
        "content": [{"type": "text", "text": "Machine learning is..."}]
    },
    {
        "role": "user",
        "content": [{"type": "text", "text": "Can you give me an example?"}]
    }
]

response = generate_text(messages, user_id="your_staff_id")
```

#### Advanced Configuration

```python
response = generate_text(
    prompt_or_messages="Write a creative story about space exploration",
    model=AIGatewayClaudeModels.CLAUDE_45_SONNET,
    user_id="your_staff_id",
    
    # Generation parameters
    max_new_tokens=2000,
    temperature=0.8,  # Higher = more creative
    top_p=0.95,
    
    # System prompt
    system="You are a creative science fiction writer with expertise in astrophysics.",
    
    # Stop sequences
    stop=["THE END", "\n\n\n"],
    
    # Timeout settings
    connect_timeout=10.0,
    response_timeout=120.0,
    
    # Debugging
    verbose=True
)
```

#### Using with Different AI Gateway Environments

```python
# Production environment
response = generate_text(
    "Your prompt here",
    user_id="your_staff_id",
    base_url="https://ai-gateway.us-east-1.prod.atl-paas.net",
    use_case_id="your-production-use-case-id"
)

# Development/staging environment
response = generate_text(
    "Your prompt here",
    user_id="your_staff_id",
    base_url="https://ai-gateway.us-east-1.staging.atl-paas.net",
    use_case_id="ai-gateway-eval-use-case"
)
```

#### Reading from File

```python
# Automatically reads file content if path exists
response = generate_text(
    prompt_or_messages="/path/to/prompt.txt",
    user_id="your_staff_id"
)
```

#### Return Raw Response

```python
# Get full API response with metadata
raw_response = generate_text(
    "Your prompt here",
    user_id="your_staff_id",
    return_raw_results=True
)

# Access response details
print(raw_response.http_status.code)
print(raw_response.body)
```

### Available Models

All models route through AI Gateway's Bedrock endpoint:

| Model | ID | Context | Use Case |
|-------|-----|---------|----------|
| Claude Sonnet 4.5 | `anthropic.claude-sonnet-4-5-20250929-v1:0` | 200K | Best overall performance (default) |
| Claude Sonnet 4.0 | `anthropic.claude-sonnet-4-20250514-v1:0` | 200K | Strong reasoning and coding |
| Claude Opus 4.1 | `anthropic.claude-opus-4-1-20250805-v1:0` | 200K | Most capable, complex tasks |
| Claude Opus 4.0 | `anthropic.claude-opus-4-20250514-v1:0` | 200K | High-quality outputs |
| Claude Sonnet 3.7 | `anthropic.claude-3-7-sonnet-20250219-v1:0` | 200K | Previous generation |
| Claude Sonnet 3.5 v2 | `anthropic.claude-3-5-sonnet-20241022-v2:0` | 200K | Enhanced 3.5 |
| Claude Sonnet 3.5 v1 | `anthropic.claude-3-5-sonnet-20240620-v1:0` | 200K | Original 3.5 |
| Claude Haiku 4.5 | `anthropic.claude-haiku-4-5-20251001-v1:0` | 200K | Fast, lightweight |
| Claude Haiku 3.5 | `anthropic.claude-3-5-haiku-20241022-v1:0` | 200K | Previous Haiku |

## Command Line Usage

Run directly as a script:

```bash
# Basic usage (requires AI_GATEWAY_USER_ID environment variable)
python -m agent_foundation.apis.ai_gateway.ai_gateway_claude_llm \
    --prompt "What is the capital of France?"

# With explicit user ID
python -m agent_foundation.apis.ai_gateway.ai_gateway_claude_llm \
    --prompt "Explain quantum computing" \
    --user_id "your_staff_id" \
    --model "anthropic.claude-sonnet-4-5-20250929-v1:0" \
    --max_new_tokens 1024 \
    --temperature 0.7

# With system prompt
python -m agent_foundation.apis.ai_gateway.ai_gateway_claude_llm \
    --prompt "Write a haiku about coding" \
    --user_id "your_staff_id" \
    --system "You are a creative poet" \
    --temperature 0.9
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AI_GATEWAY_USER_ID` | `$USER` | Your Atlassian staff ID for tracking (falls back to system username) |
| `AI_GATEWAY_BASE_URL` | `https://ai-gateway.us-east-1.staging.atl-paas.net` | AI Gateway endpoint |
| `AI_GATEWAY_CLOUD_ID` | `local` | Cloud ID for tracking |
| `AI_GATEWAY_USE_CASE_ID` | `ai-gateway-eval-use-case` | Use case ID for tracking |
| `SLAUTH_SERVER_URL` | `http://localhost:5000` | SLAUTH authentication server URL |

## Error Handling

The module provides detailed error messages including upstream errors from Bedrock/Anthropic:

```python
try:
    response = generate_text(
        "Your prompt",
        user_id="your_staff_id",
        model=AIGatewayClaudeModels.CLAUDE_45_SONNET
    )
except ValueError as e:
    print(f"Configuration error: {e}")
except Exception as e:
    print(f"API error: {e}")
    # Error includes upstream details if available
```

## Comparison with Direct Claude API

| Feature | AI Gateway | Direct Claude API |
|---------|------------|-------------------|
| Authentication | SLAUTH (SSO) | API Key |
| Cost Tracking | Centralized | Per-account |
| Rate Limiting | Shared/managed | Per-account |
| Monitoring | Built-in dashboards | Self-managed |
| Compliance | Centralized policies | Self-managed |
| Setup | Requires SLAUTH server | Just API key |

## Troubleshooting

### "user_id is required" error
- The module automatically uses your system username (`$USER`) if `AI_GATEWAY_USER_ID` is not set
- Explicitly set `AI_GATEWAY_USER_ID` environment variable if you need a different user ID, or
- Pass `user_id="your_staff_id"` parameter to `generate_text()`

### "Failed to create SLAUTH filter" error
- Ensure atlas slauth plugin is installed: `atlas plugin install -n slauth`
- Ensure SLAUTH server is running: `atlas slauth server --port 5000`
- Check that the server is accessible at the configured URL

### HTTP 400 errors
- Verify the model ID is correct and available in AI Gateway
- Check that request parameters match Anthropic's API requirements
- Enable verbose mode to see full request details: `verbose=True`

### Timeout errors
- Increase `response_timeout` for longer responses
- Consider using a faster model (e.g., Haiku) for quick tasks
- Check network connectivity to AI Gateway

## See Also

- [AI Gateway Documentation](https://hello.atlassian.net/wiki/spaces/ML/pages/3195824247/AI+Gateway)
- [Claude API Documentation](https://docs.anthropic.com/claude/reference/getting-started-with-the-api)
- [SLAUTH Documentation](https://developer.atlassian.com/platform/slauth/)
