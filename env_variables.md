# Environment Variables Guide

This document describes the environment variables used by the Browser Agent Hub (`app.py`), along with examples of how to set them.

## General Configuration

### `PUBLIC_URL`
- **Description**: The public-facing URL where the application is hosted. Used to generate absolute URLs for artifacts, such as saved screenshots returned by the agent.
- **Default**: `http://localhost:80`
- **Example**: `PUBLIC_URL=https://agent.example.com`

---

## Agent Default Settings

These variables dictate the default language model behavior when tasks are submitted to the API or UI without explicitly provided parameters.

### `DEFAULT_LLM_PROVIDER`
- **Description**: The primary provider used to instantiate the LLM. Supported values are `google`, `openai`, `openrouter`, and `z.ai`.
- **Default**: `google`
- **Example**: `DEFAULT_LLM_PROVIDER=openai`

### `DEFAULT_LLM_MODEL`
- **Description**: The primary model ID associated with the primary provider.
- **Default**: `gemini-flash-lite-latest`
- **Example**: `DEFAULT_LLM_MODEL=gpt-4.5-preview`

### `FALLBACK_LLM_PROVIDER`
- **Description**: (Optional) The backup provider to use if the primary model fails or gets rate-limited. Must be used alongside `FALLBACK_LLM_MODEL`.
- **Default**: None
- **Example**: `FALLBACK_LLM_PROVIDER=openai`

### `FALLBACK_LLM_MODEL`
- **Description**: (Optional) The backup model ID to use if the primary model encounters an error. Must be used alongside `FALLBACK_LLM_PROVIDER`.
- **Default**: None
- **Example**: `FALLBACK_LLM_MODEL=gpt-4o-mini`

---

## API Keys

Depending on the chosen providers, you must ensure the corresponding API keys are set in your environment.

### `GOOGLE_API_KEY`
- **Description**: Automatically picked up by the LangChain Google integration when `google` is set as the provider.
- **Example**: `GOOGLE_API_KEY=AIzaSy...`

### `OPENAI_API_KEY`
- **Description**: Automatically picked up by the LangChain OpenAI integration when `openai` is set as the provider.
- **Example**: `OPENAI_API_KEY=sk-proj-...`

### `OPENROUTER_API_KEY`
- **Description**: Required when setting the provider to `openrouter`.
- **Example**: `OPENROUTER_API_KEY=sk-or-v1-...`

### `ZAI_API_KEY`
- **Description**: Required when setting the provider to `z.ai`. Connects to `https://api.z.ai/v1`.
- **Example**: `ZAI_API_KEY=sk-...`

### `BROWSER_USE_API_KEY` (Optional)
- **Description**: Optional key for running the managed `ChatBrowserUse` service or using Browser Use Cloud functionalities.
- **Example**: `BROWSER_USE_API_KEY=bu_...`

---

## Example `.env` File

```env
# General
PUBLIC_URL=https://agent.example.com

# Primary Agent Defaults
DEFAULT_LLM_PROVIDER=google
DEFAULT_LLM_MODEL=gemini-2.5-flash

# Fallback Agent Options
FALLBACK_LLM_PROVIDER=openai
FALLBACK_LLM_MODEL=gpt-4o-mini

# Keys
GOOGLE_API_KEY=AIzaSyxxxxxxxxxxxxxxxxx
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxxxxxx
```
