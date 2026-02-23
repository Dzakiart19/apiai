# Api Dzeck Ai - REST API Gateway

## Overview
Api Dzeck Ai is a REST API gateway designed for personal use, providing access to multiple AI/LLM providers (GPT-4, Claude, Gemini, DeepSeek, Grok, Qwen, etc.) through the g4f library without requiring individual provider API keys. It functions as a pure REST API, complete with Swagger UI documentation, and does not include a chat interface. The project incorporates Claude AI-like agent capabilities, featuring robust server-side tool execution, and automatically generates tokens per request to mitigate rate limits. The primary goal is to offer an unlimited AI API gateway for a single user.

## User Preferences
- Language: Bahasa Indonesia
- Design: Dark theme (#1a1a2e bg, #00a896 teal accent)
- Pure REST API - no chat UI
- Single user (personal use only)
- Endpoint utama/prioritas: `/v1/agent/completions` (bukan `/v1/chat/completions`)
- Tools harus benar-benar dieksekusi server-side, bukan hanya call/prompt saja

## System Architecture

### Core Design
The project is built as a REST API gateway providing a unified interface to various LLM providers. It leverages a three-layer tool detection system with over 200 regex patterns and an advanced agent system featuring planning, loop supervision, reflection, and workspace isolation. A comprehensive multi-model auto-fallback system ensures near-zero request failures: each provider tries MULTIPLE models (e.g., PollinationsAI tries openai → gemini → claude → deepseek → grok → mistral, etc.) before moving to the next provider (PollinationsAI → DeepInfra → Groq → GeminiPro → CohereForAI → HuggingSpace → Perplexity → Yqcloud → TeachAnything → OperaAria). This dramatically increases success rates. The system prioritizes server-side execution of agent tools.

### Technical Implementation
- **Backend**: Python Flask handles all REST API functionalities, including authentication middleware, endpoint routing, and integration with AI services.
- **Frontend**: A single `swagger.html` page serves as the documentation and interactive API playground.
- **AI Integration**: The `g4f` library is central to accessing diverse LLM providers without direct API keys.
- **Authentication**: Supports session-based authentication for the Swagger UI, Bearer token authentication for API access, and an auto-token system for seamless personal use. The `X-Admin-Key` header allows for zero-setup token generation.
- **Agent Capabilities**: Features 15 built-in tools executed server-side (file_read, file_write, apply_patch, list_directory, create_directory, run_code, run_shell, install_package, debug_code, web_search, http_request, memory_write, memory_read, database_query, task_status), enabling robust agent autonomy. The agent loop is configured for a maximum of 10 iterations.
- **Deployment**: Configured for `vm` (always-on) deployment using Gunicorn with 2 workers and a 120s timeout. Production URLs are automatically detected via `REPLIT_DEPLOYMENT_URL`.
- **UI/UX**: Dark theme is applied to the Swagger UI with specific color schemes for background and accents. Health checks in Swagger UI auto-refresh every 30 seconds with visual indicators.

### Key Features
- **Auto-Token System**: Generates and manages temporary API keys per request using `X-Admin-Key` header or a dedicated endpoint, avoiding rate limits. Auto-generated keys are cached and cleaned up automatically, keeping a maximum of 3 active keys.
- **Comprehensive LLM Support**: Integrates and tests numerous models from providers like Perplexity, DeepInfra, HuggingSpace, Groq, GeminiPro, CohereForAI, TeachAnything, Yqcloud, and OperaAria, with high pass rates for both chat and tool calling.
- **Server-Side Tool Execution**: Tools such as `run_code` (Python/JS/Bash), `web_search` (DuckDuckGo), and `http_request` are genuinely executed on the server, returning real outputs.
- **OpenAI-Compatible Endpoints**: Provides `/v1/chat/completions` and `/v1/agent/completions` for compatibility with OpenAI API calls, including tool calling and response format. The `/v1/agent/completions` endpoint is the primary for advanced agent interactions.
- **Health Monitoring**: Includes health check endpoints and visual indicators in the Swagger UI.

## External Dependencies
- **LLM Providers (via g4f library)**: GPT-4, Claude, Gemini, DeepSeek, Grok, Qwen, Perplexity, PollinationsAI, DeepInfra, HuggingSpace, GeminiPro, CohereForAI, TeachAnything, Yqcloud, OperaAria.
- **Database**: Replit PostgreSQL (Neon-backed) for storing settings, user data, API keys, and conversations. Accessed via `psycopg2`.
- **Web Server**: Gunicorn for production deployment of the Flask application.
- **Search**: DuckDuckGo for `web_search` tool functionality.