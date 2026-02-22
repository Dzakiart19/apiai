# Api Dzeck Ai - REST API Gateway

## Overview
Api Dzeck Ai is a REST API gateway for accessing multiple AI/LLM providers (GPT-4, Claude, Gemini, DeepSeek, Grok, Qwen, etc.) via the g4f library without requiring provider API keys. The project serves as a pure REST API with Swagger UI documentation - no chat interface. Features Claude AI-like agent capabilities with built-in server-side tools.

**Purpose**: Personal-use unlimited AI API gateway. Auto-generates tokens per request to avoid rate limits.

## User Preferences
- Language: Bahasa Indonesia
- Design: Dark theme (#1a1a2e bg, #00a896 teal accent)
- Pure REST API - no chat UI
- Single user (personal use only)

## Recent Changes (2026-02-23, Session 6)
- **Fix Publish Error**: Deployment sekarang menggunakan gunicorn (production WSGI server) bukan Flask dev server. Health check gagal karena endpoint `/` lambat - sekarang di-cache (5 menit). Gunicorn config: 2 workers, 120s timeout.
- **Deploy command**: `cd src && gunicorn --bind 0.0.0.0:5000 --workers 2 --timeout 120 --preload FreeGPT4_Server:app`

### Previous Changes (2026-02-22, Session 5)
- **Auto Refresh Health**: Health check di Swagger UI sekarang auto-refresh setiap 30 detik dengan indikator visual (dot hijau/merah, uptime, providers, models, last check time). Bisa di-toggle on/off.
- **Production URL Detection**: Saat di-publish/deploy, backend otomatis menggunakan URL `.replit.app` (bukan preview URL). Deteksi via `REPLIT_DEPLOYMENT_URL` dan `REPLIT_DEPLOYMENT` env vars. Swagger UI juga menampilkan production URL.
- **VM Deployment Config**: Dikonfigurasi sebagai VM deployment (always-on 24/7, tidak pernah tidur).

### Previous Changes (2026-02-23, Session 4)
- **Comprehensive Testing**: Tested ALL 11 providers, 72+ models with chat & tool calling
  - Chat: 70/72 PASS (97.2%) - only 2 upstream failures (HuggingSpace/phi-4, DeepInfra/GLM-4.7-Flash)
  - Tool Calling: 25/25 PASS (100%) on agent endpoint
  - Auto-Token: VERIFIED working, no rate limit issues
  - X-Admin-Key middleware: VERIFIED working, auto-generates token per request

### Previous Changes (2026-02-22, Session 3)
- **Auto-Generate Token System**: Added `/api/auto-token` endpoint - generates fresh API key with admin password, no session needed. Old auto-generated keys auto-cleaned (keeps max 10).
- **X-Admin-Key Middleware**: Added `X-Admin-Key` header support - requests with admin password automatically get a temporary API key generated and injected. No need to pre-create tokens.
- **Auto-Cleanup**: Old auto-generated tokens are automatically removed to keep database clean.

### Previous Changes (2026-02-22, Session 2)
- Bug fixes: run_code SyntaxError, DeepInfra default model, CohereForAI contradiction, AgentLoopResult types, DDGParser type safety
- Comprehensive testing: All 11 providers tested with real API calls

### Previous Changes (2026-02-22, Session 1)
- Model Fallback System with `_get_fallback_model()`, `_is_model_compatible_with_provider()`, `PROVIDER_DEFAULT_MODELS`
- Three-Layer Tool Detection System (200+ regex patterns)
- Built-in Server-Side Tools (11 tools)
- Advanced Agent System: Planner, Loop Supervisor, Reflection, Workspace Isolation

## Auto-Token System (NEW)

### Method 1: Auto-Token Endpoint
```bash
# Generate fresh API key (no login/session needed)
curl -X POST https://YOUR_URL/api/auto-token \
  -H "Content-Type: application/json" \
  -d '{"password":"dzeckaiv1","provider":"Auto","model":"openai"}'

# Returns: {"success":true,"key":{"api_key":"sk-dzeck-xxx...","endpoints":{...}}}
# Then use the api_key as Bearer token
```

### Method 2: X-Admin-Key Header (Zero Setup)
```bash
# No need to generate token first - just add X-Admin-Key header
curl -X POST https://YOUR_URL/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "X-Admin-Key: dzeckaiv1" \
  -d '{"model":"openai","messages":[{"role":"user","content":"Hello"}]}'
```

### Method 3: Traditional Bearer Token
```bash
curl -X POST https://YOUR_URL/v1/chat/completions \
  -H "Authorization: Bearer sk-dzeck-YOUR_KEY" \
  -d '{"model":"openai","messages":[{"role":"user","content":"Hello"}]}'
```

## Provider Status (Verified 2026-02-23 00:30 UTC - COMPREHENSIVE)

### Per-Provider Summary (11/11 Providers PASS)
| Provider | Default Model | Status | Models Tested | Pass Rate | Speed |
|----------|--------------|--------|:---:|:---:|:---:|
| Auto | gpt-4 | PASS | 1/1 | 100% | 3.7s |
| PollinationsAI | openai | PASS | 2/2 | 100% | 1.1s |
| Perplexity | auto | PASS | 32/32 | 100% | 1.0-6s |
| DeepInfra | MiniMaxAI/MiniMax-M2.5 | PASS | 3/4 | 75% | 3.9s |
| HuggingSpace | command-a | PASS | 5/6 | 83% | 0.5-22s |
| Groq | llama-3.3-70b-versatile | PASS | 11/11 | 100% | 0.3-1.6s |
| GeminiPro | models/gemini-2.5-flash | PASS | 12/12 | 100% | 0.3-4.3s |
| CohereForAI_C4AI_Command | command-a-03-2025 | PASS | 6/6 | 100% | 1.1s |
| TeachAnything | gemma | PASS | 1/1 | 100% | 0.6s |
| Yqcloud | gpt-4 | PASS | 1/1 | 100% | 3.9s |
| OperaAria | default | PASS | 1/1 | 100% | 16.8s |

### Known Upstream Issues (not our API's fault)
- DeepInfra/zai-org/GLM-4.7-Flash: upstream connection timeout
- HuggingSpace/phi-4: upstream HuggingSpace endpoint unreachable
- Perplexity/r1 (DeepSeek R1): slow response (reasoning model, may timeout on short timeouts)

### Tool Calling Status (Verified 2026-02-23)
| Provider | Model | Tool Calling | Notes |
|----------|-------|:---:|-------|
| PollinationsAI | openai | PASS | Fast, reliable |
| PollinationsAI | gpt-5-nano | PASS | Fast |
| Perplexity | auto | PASS | Reliable |
| Perplexity | turbo | PASS | Fast |
| Perplexity | gpt41 | PASS | |
| Perplexity | gpt5 | PASS | |
| Perplexity | claude2 | PASS | |
| Perplexity | claude40opus | PASS | |
| Perplexity | grok | PASS | |
| Perplexity | o3 | PASS | |
| Perplexity | pplx_pro | PASS | |
| Perplexity | r1 | PASS | Slow but works |
| GeminiPro | models/gemini-2.5-flash | PASS | Fast, reliable |
| GeminiPro | models/gemma-3-27b-it | PASS | |
| GeminiPro | models/gemini-flash-latest | PASS | |
| GeminiPro | models/gemini-2.5-flash-lite | PASS | |
| GeminiPro | models/gemini-3-flash-preview | PASS | |
| Groq | llama-3.3-70b-versatile | PASS | Fastest |
| Groq | llama-4-scout-17b-16e-instruct | PASS | |
| Groq | qwen/qwen3-32b | PASS | |
| Groq | llama-3.1-8b-instant | PASS | |
| Groq | moonshotai/kimi-k2-instruct | PASS | |
| DeepInfra | MiniMaxAI/MiniMax-M2.5 | PASS | |
| HuggingSpace | command-a | PASS | Slow |
| CohereForAI | command-a-03-2025 | PASS | |

### AI Model Categories
| Category | Description | Example Models |
|----------|-------------|----------------|
| General | Chat serbaguna | GPT-4, Claude 2, Grok, Command A |
| Advanced | Model terkuat | GPT-4.1, GPT-5, Claude 4 Opus, Grok 4 |
| Thinking | Deep reasoning | o3, o3 Pro, Claude 3.7 Sonnet Thinking, DeepSeek R1 |
| Search | Web search realtime | Perplexity Auto/Turbo/Pro |
| Research | Riset ilmiah | o3 Research, Claude 4 Research |
| Labs | Coding & prototyping | o3 Labs, Claude 4 Labs |
| Fast | Respons super cepat | GPT-4o Mini, GPT-5 Nano, Gemini 2 Flash |

### Rate Limit Note
g4f free tier = ~5 requests/minute per provider. Auto-token system helps rotate tokens to minimize tracking. Provider fallback system automatically tries alternative providers when one is rate-limited.

## System Architecture

### Technical Stack
- **Backend**: Python Flask (REST API)
- **Frontend**: Single Swagger UI page (swagger.html)
- **Database**: Replit PostgreSQL (Neon-backed) via psycopg2
- **AI Integration**: g4f library for multiple LLM providers
- **Auth**: Session-based for Swagger UI, Bearer token for API, Auto-token for personal use
- **Built-in Tools**: 11 server-side tools for agent autonomy
- **Agent Engine**: Planning, loop supervision, reflection, workspace isolation

### API Endpoints
| Endpoint | Method | Auth | Description |
|----------|--------|------|-------------|
| `/` | GET | None | Swagger UI documentation |
| `/api/auth/login` | POST | Body | Login (returns session) |
| `/api/auth/check` | GET | Session | Check session status |
| `/api/auth/logout` | POST | Session | Logout |
| `/api/apikeys` | GET | Session | List API keys |
| `/api/apikeys/generate` | POST | Session | Generate new API key |
| `/api/apikeys/<id>` | DELETE | Session | Delete API key |
| `/api/apikeys/<id>/toggle` | POST | Session | Enable/disable API key |
| `/api/auto-token` | POST | Password | Auto-generate fresh API key |
| `/api/chat` | POST | Bearer | Simple chat endpoint |
| `/v1/chat/completions` | POST | Bearer/X-Admin-Key | OpenAI-compatible chat |
| `/v1/agent/completions` | POST | Bearer/X-Admin-Key | Agent API with tool calling |
| `/api/test` | POST | Session | Test single provider/model |
| `/api/test/all` | POST | Session | Test all providers |
| `/api/models` | GET | None | List models by provider |
| `/api/providers` | GET | None | List providers |
| `/api/model-info` | GET | None | Get model info |
| `/api/models/catalog` | GET | None | Full model catalog |
| `/v1/models` | GET | Bearer | OpenAI-compatible model list |
| `/health` | GET | None | Health check |
| `/ping` | GET | None | Ping/pong |

## Tool Status (Verified 2026-02-22)
| Tool | Status | Description |
|------|--------|-------------|
| run_code | PASS | Execute Python/JS/Bash code |
| web_search | PASS | DuckDuckGo search (5 results) |
| http_request | PASS | GET/POST/PUT/DELETE requests |
| debug_code | PASS | Static code analysis |
| database_query | PASS | In-memory key-value store |
| memory_write | PASS | Per-session memory storage |
| memory_read | PASS | Per-session memory retrieval |
| file_write | PASS | Sandboxed file write |
| file_read | PASS | Sandboxed file read |
| apply_patch | PASS | Code patching/editing |
| task_status | PASS | Task progress tracking |

## Important Files
- `src/FreeGPT4_Server.py` - Main Flask REST API server (auto-token, middleware, all endpoints)
- `src/templates/swagger.html` - Swagger UI documentation page
- `src/ai_service.py` - AI service layer (g4f integration, fallback system, model compatibility)
- `src/database.py` - PostgreSQL database manager (users, API keys, conversations)
- `src/config.py` - Provider/model configuration, model capabilities catalog
- `src/agent_engine.py` - Agent engine (tool calling, structured output, agent loop)
- `src/builtin_tools.py` - Built-in tools implementation (11 tools)
- `src/planner.py` - Task planning, step tracking, reflection
- `src/auth.py` - Authentication utilities
- `src/utils/provider_monitor.py` - Provider health monitoring & fallback ranking
- `src/data/db_backup.json` - Auto-backup of users and API keys
- `requirements.txt` - Python dependencies

## Key Technical Notes
- **Database**: PostgreSQL via DATABASE_URL. Tables: settings, personal, api_keys, conversations
- **API Key prefix**: `sk-dzeck-` + random hex (24 bytes)
- **Keep-alive**: Self-ping every 4 minutes
- **Deploy target**: VM (always-on)
- **Token persistence**: Auto-backup to `src/data/db_backup.json`
- **Provider fallback**: Smart model-aware fallback - uses provider's default model when original is incompatible
- **Rate limiting**: g4f free tier = ~5 requests/minute per provider
- **Auto-token cleanup**: Keeps max 10 auto-generated keys, removes oldest automatically
- **Agent loop**: Max 10 iterations, auto-executes built-in tools server-side
- **Virtual user**: dzeckyete / dzeckaiv1 (auto-created on startup)
- **Admin**: admin / dzeckaiv1
- **Production URL**: Auto-detected from `REPLIT_DEPLOYMENT_URL` env var (set by Replit saat publish)

## Future Development Ideas
- Add more g4f providers as they become available
- Implement request queuing for rate limit management
- Add usage analytics dashboard
- WebSocket support for real-time streaming
- Multi-user quota management (if needed)
- Provider auto-discovery and health checking on startup
