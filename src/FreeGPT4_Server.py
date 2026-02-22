"""Api Dzeck Ai - REST API Gateway

REST API for AI model access with Swagger UI documentation.
Supports multiple AI providers and models with API key authentication.
"""

import os
import argparse
import threading
import json
import time as _time
from pathlib import Path
from typing import Optional

from flask import Flask, request, jsonify, session, Response, render_template, redirect
from werkzeug.security import generate_password_hash
from werkzeug.middleware.proxy_fix import ProxyFix

from flask_cors import CORS

from config import config
from database import db_manager
from auth import auth_service
from ai_service import ai_service
from agent_engine import (
    build_agent_system_prompt,
    prepare_messages_for_agent,
    build_agent_response,
    parse_tool_call_from_response,
    validate_tool_call,
    AgentLoopResult,
    run_agent_loop,
)
from builtin_tools import (
    get_builtin_tool_definitions,
    BUILTIN_TOOL_NAMES,
)
from utils.logging import logger, setup_logging
from utils.exceptions import (
    FreeGPTException,
    ValidationError,
    AuthenticationError,
    AIProviderError,
)
from utils.validation import sanitize_input
from utils.helpers import generate_uuid

app = Flask(__name__)
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)
app.secret_key = config.security.secret_key
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['PERMANENT_SESSION_LIFETIME'] = 86400 * 7
app.config['SESSION_COOKIE_PATH'] = '/'
app.config['start_time'] = _time.time()

PRODUCTION_URL = "https://api-gateway--ngatwhb.replit.app"

def _get_base_url():
    deploy_url = os.environ.get('REPLIT_DEPLOYMENT_URL', '')
    if deploy_url:
        return deploy_url.rstrip('/')
    dev_domain = os.environ.get('REPLIT_DEV_DOMAIN', '')
    if dev_domain:
        return f"https://{dev_domain}"
    replit_domains = os.environ.get('REPLIT_DOMAINS', '')
    if replit_domains:
        first_domain = replit_domains.split(',')[0].strip()
        if first_domain:
            return f"https://{first_domain}"
    return PRODUCTION_URL

def _get_production_base_url():
    deploy_url = os.environ.get('REPLIT_DEPLOYMENT_URL', '')
    if deploy_url:
        return deploy_url.rstrip('/')
    return PRODUCTION_URL

def _get_request_base_url():
    """Get base URL from the current request's host header (auto-detects preview vs production)."""
    try:
        host = request.headers.get('X-Forwarded-Host') or request.headers.get('Host') or request.host
        if host:
            host = host.split(',')[0].strip()
            if ':' in host and not host.startswith('['):
                host_part = host.rsplit(':', 1)[0]
                if host_part in ('localhost', '127.0.0.1', '0.0.0.0'):
                    return _get_base_url()
            scheme = request.headers.get('X-Forwarded-Proto', 'https')
            return f"{scheme}://{host}"
    except RuntimeError:
        pass
    return _get_base_url()

def _get_allowed_origins():
    origins = [
        "https://api-dzeck.web.app",
        "https://api-dzeck.firebaseapp.com",
        PRODUCTION_URL,
        "http://localhost:5000",
    ]
    dev_domain = os.environ.get('REPLIT_DEV_DOMAIN', '')
    if dev_domain:
        origins.append(f"https://{dev_domain}")
    replit_domains = os.environ.get('REPLIT_DOMAINS', '')
    if replit_domains:
        for d in replit_domains.split(','):
            d = d.strip()
            if d:
                url = f"https://{d}"
                if url not in origins:
                    origins.append(url)
    replit_deployment = os.environ.get('REPLIT_DEPLOYMENT_URL', '')
    if replit_deployment:
        if replit_deployment not in origins:
            origins.append(replit_deployment)
    return origins

CORS(app, supports_credentials=True, origins=_get_allowed_origins())

if os.getenv('LOG_LEVEL'):
    setup_logging(level=os.getenv('LOG_LEVEL', 'INFO'))

@app.after_request
def add_cache_control(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    origin = request.headers.get('Origin', '')
    if origin in _get_allowed_origins():
        response.headers['Access-Control-Allow-Origin'] = origin
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type, Authorization, X-Requested-With'
        response.headers['Access-Control-Allow-Methods'] = 'GET, POST, PUT, DELETE, OPTIONS'
    return response

logger.info("Api Dzeck Ai REST API - Starting server...")

class ServerArgumentParser:
    def __init__(self):
        self.parser = self._create_parser()
        self.args = None

    def _create_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Api Dzeck Ai REST API Server")
        parser.add_argument("--port", action='store', type=int, help="Server port (default: 5000)")
        parser.add_argument("--password", action='store', help="Admin password")
        parser.add_argument("--enable-gui", action='store_true', help="Enable web UI")
        parser.add_argument("--enable-virtual-users", action='store_true', help="Enable virtual users")
        parser.add_argument("--enable-history", action='store_true', help="Enable message history")
        parser.add_argument("--model", action='store', type=str, help="Default model")
        parser.add_argument("--provider", action='store', type=str, help="Default provider")
        return parser

    def parse_args(self):
        self.args, _ = self.parser.parse_known_args()
        if not self.args.password and os.getenv("ADMIN_PASSWORD"):
            self.args.password = os.getenv("ADMIN_PASSWORD")
        if not self.args.port and os.getenv("PORT"):
            self.args.port = int(os.getenv("PORT", "5000"))
        if not self.args.enable_virtual_users and os.getenv("ENABLE_VIRTUAL_USERS", "").lower() in ("true", "1", "yes"):
            self.args.enable_virtual_users = True
        return self.args

class ServerManager:
    def __init__(self, args):
        self.args = args
        self._setup_working_directory()
        self._merge_settings_with_args()

    def _setup_working_directory(self):
        script_path = Path(__file__).resolve()
        os.chdir(script_path.parent)

    def _merge_settings_with_args(self):
        try:
            settings = db_manager.get_settings()
            if not self.args.port:
                self.args.port = int(settings.get("port", config.server.port))
            if not self.args.provider:
                self.args.provider = settings.get("provider", config.api.default_provider)
            if not self.args.model:
                self.args.model = settings.get("model", config.api.default_model)
        except Exception as e:
            logger.error(f"Failed to merge settings: {e}")
            self.args.port = self.args.port or config.server.port
            self.args.provider = self.args.provider or config.api.default_provider
            self.args.model = self.args.model or config.api.default_model

    def setup_password(self):
        try:
            settings = db_manager.get_settings()
            current_password = settings.get("password", "")
            if self.args.password:
                db_manager.update_settings({"password": self.args.password})
                logger.info("Admin password configured successfully")
            elif not current_password:
                logger.warning("No admin password configured")
        except Exception as e:
            logger.error(f"Failed to setup password: {e}")

    def setup_virtual_users(self):
        if not self.args.enable_virtual_users:
            return
        try:
            db_manager.update_settings({"virtual_users": True})
            existing_user = db_manager.get_user_by_username("dzeckyete")
            if not existing_user:
                db_manager.create_user("dzeckyete", "dzeckaiv1")
                logger.info("Default virtual user 'dzeckyete' created")
            else:
                logger.info("Virtual user 'dzeckyete' already exists")
        except Exception as e:
            logger.warning(f"Failed to setup virtual users: {e}")


@app.errorhandler(404)
def handle_not_found(e):
    return jsonify({"error": "Not found"}), 404

@app.errorhandler(FreeGPTException)
def handle_freegpt_exception(e):
    logger.error(f"API error: {e}")
    return jsonify({"error": str(e)}), 400

@app.errorhandler(Exception)
def handle_general_exception(e):
    from werkzeug.exceptions import NotFound
    if isinstance(e, NotFound):
        return jsonify({"error": "Not found"}), 404
    logger.error(f"Unexpected error: {e}", exc_info=True)
    return jsonify({"error": "Internal server error"}), 500


# ============================================================
# SWAGGER UI - Main Page
# ============================================================

@app.route("/", methods=["GET"])
def swagger_ui():
    current_url = _get_request_base_url()
    prod_url = _get_production_base_url()

    providers_with_models = ai_service.get_all_providers_with_models()
    return render_template(
        "swagger.html",
        base_url=current_url,
        current_url=current_url,
        production_url=prod_url,
        providers_models=providers_with_models,
        model_capabilities=config.model_capabilities,
        category_info=config.category_info,
    )


# ============================================================
# AUTH ENDPOINTS
# ============================================================

@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """Login and get session."""
    data = request.get_json(silent=True) or {}
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"success": False, "error": "Username and password required"}), 400

    is_admin = False
    if username == "admin":
        is_admin = auth_service.authenticate_admin(username, password)
        if not is_admin:
            return jsonify({"success": False, "error": "Invalid admin credentials"}), 401
    else:
        if not auth_service.authenticate_user(username, password):
            return jsonify({"success": False, "error": "Invalid credentials"}), 401

    session.permanent = True
    session['logged_in_user'] = username
    session['is_admin'] = is_admin
    return jsonify({"success": True, "username": username, "is_admin": is_admin})

@app.route("/api/auth/logout", methods=["GET", "POST"])
def api_logout():
    """Logout and clear session."""
    session.clear()
    return jsonify({"success": True})

@app.route("/api/auth/check", methods=["GET"])
def api_auth_check():
    """Check current session status."""
    username = session.get('logged_in_user')
    if username:
        return jsonify({"logged_in": True, "username": username, "is_admin": session.get('is_admin', False)})
    return jsonify({"logged_in": False})


# ============================================================
# API KEY MANAGEMENT
# ============================================================

@app.route("/api/apikeys", methods=["GET"])
@app.route("/api/keys", methods=["GET"])
def list_api_keys():
    """List all API keys for the logged-in user."""
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    is_admin = session.get('is_admin', False)
    if is_admin:
        keys = db_manager.get_api_keys()
    else:
        keys = db_manager.get_api_keys(created_by=username)
    current_base_url = _get_request_base_url()
    prod_base_url = _get_production_base_url()
    builtin_tools_list = [t["function"]["name"] for t in get_builtin_tool_definitions()]
    safe_keys = []
    for k in keys:
        safe_keys.append({
            "id": k["id"],
            "api_key": k["api_key"],
            "provider": k["provider"],
            "model": k.get("model", ""),
            "label": k.get("label", ""),
            "created_by": k["created_by"],
            "created_at": k["created_at"],
            "last_used_at": k.get("last_used_at"),
            "usage_count": k.get("usage_count", 0),
            "usage_limit": "unlimited",
            "is_active": k.get("is_active", True),
            "streaming": True,
            "base_url": current_base_url,
            "production_url": prod_base_url,
            "endpoints": {
                "chat": f"{current_base_url}/api/chat",
                "openai_compatible": f"{current_base_url}/v1/chat/completions",
                "agent": f"{current_base_url}/v1/agent/completions",
            },
            "builtin_tools": builtin_tools_list
        })
    return jsonify({
        "keys": safe_keys,
        "api_base_url": current_base_url,
        "production_url": prod_base_url,
        "endpoints": {
            "chat": f"{current_base_url}/api/chat",
            "openai_compatible": f"{current_base_url}/v1/chat/completions",
            "agent": f"{current_base_url}/v1/agent/completions",
        },
        "builtin_tools": builtin_tools_list
    })

@app.route("/api/apikeys/generate", methods=["POST"])
def generate_api_key():
    """Generate a new API key."""
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    provider = data.get("provider", "Auto")
    model = data.get("model", "")
    label = data.get("label", "")
    if not provider:
        return jsonify({"error": "Provider is required"}), 400
    current_base_url = _get_request_base_url()
    try:
        result = db_manager.create_api_key(provider, model, label, username, current_base_url)
        result["endpoints"] = {
            "chat": f"{current_base_url}/api/chat",
            "openai_compatible": f"{current_base_url}/v1/chat/completions",
            "agent": f"{current_base_url}/v1/agent/completions",
        }
        return jsonify({"success": True, "key": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auto-token", methods=["POST"])
def auto_generate_token():
    """Auto-generate a fresh API key using admin password. No session needed.
    
    Body: {"password": "admin_password", "provider": "Auto", "model": "openai"}
    Returns a fresh API key each time. Old auto-generated keys are cleaned up automatically.
    """
    data = request.get_json(silent=True) or {}
    password = data.get("password", "").strip()
    
    if not password:
        return jsonify({"error": "Password required"}), 400
    
    if not db_manager.verify_admin_password(password):
        return jsonify({"error": "Invalid password"}), 401
    
    provider = data.get("provider", "Auto")
    model = data.get("model", "openai")
    
    try:
        _cleanup_auto_tokens()
    except Exception:
        pass
    
    current_base_url = _get_request_base_url()
    try:
        result = db_manager.create_api_key(
            provider=provider,
            model=model,
            label="auto-generated",
            created_by="admin",
            base_url=current_base_url
        )
        result["endpoints"] = {
            "chat": f"{current_base_url}/api/chat",
            "openai_compatible": f"{current_base_url}/v1/chat/completions",
            "agent": f"{current_base_url}/v1/agent/completions",
        }
        logger.info(f"Auto-generated token: {result['api_key'][:20]}...")
        return jsonify({"success": True, "key": result})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _cleanup_auto_tokens(max_auto_keys=10):
    """Remove old auto-generated tokens, keeping only the most recent ones."""
    try:
        from database import db_manager as _db
        with _db.get_connection() as (conn, cursor):
            cursor.execute(
                "SELECT id FROM api_keys WHERE label = 'auto-generated' ORDER BY created_at DESC"
            )
            rows = cursor.fetchall()
            if len(rows) > max_auto_keys:
                old_ids = [r["id"] for r in rows[max_auto_keys:]]
                for old_id in old_ids:
                    cursor.execute("DELETE FROM api_keys WHERE id = %s", (old_id,))
                conn.commit()
                logger.info(f"Cleaned up {len(old_ids)} old auto-generated tokens")
    except Exception as e:
        logger.warning(f"Auto-token cleanup failed: {e}")


def _auto_auth_middleware():
    """Middleware: If X-Admin-Key header is present, auto-generate a temporary API key for this request."""
    admin_key = request.headers.get("X-Admin-Key", "").strip()
    if not admin_key:
        return
    
    if not db_manager.verify_admin_password(admin_key):
        return
    
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer ") and db_manager.get_api_key_by_key(auth_header[7:]):
        return
    
    current_base_url = _get_request_base_url()
    try:
        result = db_manager.create_api_key(
            provider="Auto",
            model="openai",
            label="auto-generated",
            created_by="admin",
            base_url=current_base_url
        )
        request.environ['HTTP_AUTHORIZATION'] = f"Bearer {result['api_key']}"
        logger.info(f"Auto-auth: generated temporary token for request")
        
        try:
            _cleanup_auto_tokens()
        except Exception:
            pass
    except Exception as e:
        logger.warning(f"Auto-auth failed: {e}")


app.before_request(_auto_auth_middleware)

@app.route("/api/apikeys/<key_id>", methods=["DELETE"])
def delete_api_key(key_id):
    """Delete an API key."""
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    try:
        db_manager.delete_api_key(key_id)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/apikeys/<key_id>/toggle", methods=["POST"])
def toggle_api_key(key_id):
    """Enable/disable an API key."""
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    data = request.get_json(silent=True) or {}
    is_active = data.get("is_active", True)
    try:
        db_manager.toggle_api_key(key_id, is_active)
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
# AI CHAT ENDPOINTS
# ============================================================

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """Simple chat endpoint. Requires Bearer API key. Supports streaming."""
    import asyncio

    data = request.get_json(silent=True) or {}
    question = data.get("text", "").strip()
    provider = data.get("provider", "")
    model = data.get("model", "")
    stream = data.get("stream", True)

    if not question:
        return jsonify({"status": "error", "message": "Field 'text' is required"}), 400

    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return jsonify({"status": "error", "message": "Missing Authorization header. Use: Bearer <your-api-key>"}), 401

    api_key = auth_header[7:]
    key_data = db_manager.get_api_key_by_key(api_key)
    if not key_data:
        return jsonify({"status": "error", "message": "Invalid or inactive API key"}), 401

    db_manager.increment_api_key_usage(api_key)
    username = key_data.get("created_by", "admin")

    if not provider:
        provider = key_data.get("provider", "Auto")
    if provider in ("g4f", "gpt4free", "auto"):
        provider = "Auto"
    if not model:
        model = key_data.get("model", "") or None

    question = sanitize_input(question, 0)

    if stream:
        return _handle_simple_chat_stream(question, username, provider, model)

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(
            ai_service.generate_response(
                message=question,
                username=username,
                provider=provider,
                model=model,
                use_history=False,
                remove_sources=True,
                use_proxies=False
            )
        )
        return jsonify({"status": "success", "data": response_text})
    except Exception as e:
        logger.error(f"API chat error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500
    finally:
        if loop:
            loop.close()


def _handle_simple_chat_stream(question, username, provider, model):
    """Handle SSE streaming for /api/chat."""
    import asyncio

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            yield f"data: {json.dumps({'type': 'start', 'status': 'streaming'})}\n\n"

            gen = ai_service.generate_response_stream(
                message=question,
                username=username,
                provider=provider,
                model=model,
                use_history=False,
                remove_sources=True,
                use_proxies=False
            )

            async def collect_chunks():
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)
                return chunks

            chunks = loop.run_until_complete(collect_chunks())
            full_text = ""

            for chunk_text in chunks:
                if chunk_text:
                    full_text += chunk_text
                    yield f"data: {json.dumps({'type': 'delta', 'text': chunk_text})}\n\n"

            yield f"data: {json.dumps({'type': 'done', 'status': 'success', 'full_text': full_text})}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )

@app.route("/v1/chat/completions", methods=["POST"])
def openai_compatible_endpoint():
    """OpenAI-compatible chat completions endpoint."""
    import asyncio

    data = request.get_json(silent=True) or {}
    auth_header = request.headers.get("Authorization", "")
    api_key = None
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    if not api_key:
        return jsonify({
            "error": {
                "message": "API key required. Use Authorization: Bearer YOUR_KEY",
                "type": "invalid_request_error",
                "code": "missing_api_key"
            }
        }), 401

    key_data = db_manager.get_api_key_by_key(api_key)
    if not key_data:
        return jsonify({
            "error": {
                "message": "Invalid API key",
                "type": "invalid_request_error",
                "code": "invalid_api_key"
            }
        }), 401

    if not key_data.get("is_active", True):
        return jsonify({
            "error": {
                "message": "API key is disabled",
                "type": "invalid_request_error",
                "code": "api_key_disabled"
            }
        }), 403

    db_manager.increment_api_key_usage(api_key)
    messages = data.get("messages", [])
    model_requested = data.get("model") or key_data.get("model") or "auto"
    stream = data.get("stream", True)
    provider = key_data.get("provider", "Auto")
    if provider in ("g4f", "gpt4free", "auto"):
        provider = "Auto"

    if not messages:
        return jsonify({
            "error": {
                "message": "Messages array is required",
                "type": "invalid_request_error",
                "code": "missing_messages"
            }
        }), 400

    last_msg = messages[-1].get("content", "") if messages else ""
    system_prompt = None
    for m in messages:
        if m.get("role") == "system":
            system_prompt = m.get("content", "")
            break

    prompt_tokens = sum(len(m.get("content", "").split()) for m in messages) * 2
    created_ts = int(_time.time())
    completion_id = f"chatcmpl-{generate_uuid().replace('-', '')[:29]}"

    if stream:
        return _handle_openai_stream(
            last_msg, system_prompt, provider, model_requested,
            key_data, prompt_tokens, completion_id, created_ts
        )

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        response_text = loop.run_until_complete(
            ai_service.generate_response(
                message=last_msg,
                username=key_data.get("created_by", "admin"),
                provider=provider,
                model=model_requested,
                system_prompt=system_prompt,
                use_history=False,
                remove_sources=True,
                use_proxies=False
            )
        )
        completion_tokens = len(response_text.split()) * 2
        return jsonify({
            "id": completion_id,
            "object": "chat.completion",
            "created": created_ts,
            "model": model_requested,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        })
    except Exception as e:
        return jsonify({
            "error": {"message": str(e), "type": "server_error", "code": "internal_error"}
        }), 500
    finally:
        if loop:
            loop.close()


def _handle_openai_stream(
    last_msg, system_prompt, provider, model_requested,
    key_data, prompt_tokens, completion_id, created_ts
):
    """Handle OpenAI-compatible SSE streaming for /v1/chat/completions."""
    import asyncio

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            first_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_requested,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant", "content": ""},
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(first_chunk)}\n\n"

            gen = ai_service.generate_response_stream(
                message=last_msg,
                username=key_data.get("created_by", "admin"),
                provider=provider,
                model=model_requested,
                system_prompt=system_prompt,
                use_history=False,
                remove_sources=True,
                use_proxies=False
            )

            async def collect_chunks():
                chunks = []
                async for chunk in gen:
                    chunks.append(chunk)
                return chunks

            chunks = loop.run_until_complete(collect_chunks())

            for chunk_text in chunks:
                if chunk_text:
                    chunk_data = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_requested,
                        "choices": [{
                            "index": 0,
                            "delta": {"content": chunk_text},
                            "finish_reason": None
                        }]
                    }
                    yield f"data: {json.dumps(chunk_data)}\n\n"

            stop_chunk = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_ts,
                "model": model_requested,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop"
                }]
            }
            yield f"data: {json.dumps(stop_chunk)}\n\n"
            yield "data: [DONE]\n\n"

        except Exception as e:
            error_data = {"error": {"message": str(e), "type": "server_error"}}
            yield f"data: {json.dumps(error_data)}\n\n"
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


# ============================================================
# AGENT COMPLETIONS - Powerful AI Agent API
# ============================================================

@app.route("/v1/agent/completions", methods=["POST"])
def agent_completions_endpoint():
    """Powerful agent API with tool calling, structured output, and agent loop.

    Similar to Claude AI Messages API. Supports:
    - tools: Array of tool definitions (function calling)
    - tool_choice: "none" | "auto" | "required" | {"name": "tool_name"}
    - response_format: {"type": "json_object"} or {"type": "json_schema", "json_schema": {...}}
    - system: System prompt override
    - messages: Conversation messages (including tool results)
    - model: Model selection
    - max_tokens: Maximum tokens
    - temperature: Creativity control
    - stream: Streaming support (boolean)
    """
    import asyncio

    data = request.get_json(silent=True) or {}

    auth_header = request.headers.get("Authorization", "")
    api_key = None
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    if not api_key:
        return jsonify({
            "error": {
                "type": "authentication_error",
                "message": "API key required. Use Authorization: Bearer YOUR_KEY"
            }
        }), 401

    key_data = db_manager.get_api_key_by_key(api_key)
    if not key_data:
        return jsonify({
            "error": {
                "type": "authentication_error",
                "message": "Invalid API key"
            }
        }), 401

    if not key_data.get("is_active", True):
        return jsonify({
            "error": {
                "type": "permission_error",
                "message": "API key is disabled"
            }
        }), 403

    db_manager.increment_api_key_usage(api_key)

    messages = data.get("messages", [])
    if not messages:
        return jsonify({
            "error": {
                "type": "invalid_request_error",
                "message": "messages array is required and must not be empty"
            }
        }), 400

    model_requested = data.get("model") or key_data.get("model") or "auto"
    provider = key_data.get("provider", "Auto")
    if provider in ("g4f", "gpt4free", "auto"):
        provider = "Auto"

    tools = data.get("tools")
    tool_choice = data.get("tool_choice", "auto")
    response_format = data.get("response_format")
    user_system = data.get("system", "")
    max_tokens = data.get("max_tokens", 4096)
    temperature = data.get("temperature")
    stream = data.get("stream", True)
    use_builtin_tools = data.get("builtin_tools", True)

    if tools and not isinstance(tools, list):
        return jsonify({
            "error": {
                "type": "invalid_request_error",
                "message": "tools must be an array of tool definitions"
            }
        }), 400

    all_tools = list(tools) if tools else []
    if use_builtin_tools:
        builtin_defs = get_builtin_tool_definitions()
        existing_names = {t.get("function", t).get("name", "") for t in all_tools}
        for bt in builtin_defs:
            if bt["function"]["name"] not in existing_names:
                all_tools.append(bt)

    prepared_messages = prepare_messages_for_agent(
        messages=messages,
        system_prompt=user_system or "",
        tools=all_tools if all_tools else None,
        tool_choice=tool_choice,
        response_format=response_format
    )

    last_user_msg = ""
    for msg in reversed(prepared_messages):
        if msg.get("role") == "user":
            last_user_msg = msg.get("content", "")
            break

    prompt_tokens = sum(len(m.get("content", "").split()) for m in prepared_messages) * 2
    from utils.helpers import generate_uuid
    completion_id = f"msg_{generate_uuid().replace('-', '')[:24]}"

    enable_planning = data.get("enable_planning", True)
    enable_reflection = data.get("enable_reflection", True)

    if stream:
        return _handle_agent_stream(
            prepared_messages, last_user_msg, provider, model_requested,
            key_data, all_tools if all_tools else None, tool_choice, response_format,
            prompt_tokens, completion_id, api_key,
            enable_planning=enable_planning,
            enable_reflection=enable_reflection,
            max_iterations=min(data.get("max_iterations", 10), 10),
        )

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        context = {
            "api_key": api_key,
            "session_id": f"session_{api_key[:16]}",
            "username": key_data.get("created_by", "admin")
        }

        loop_result = loop.run_until_complete(
            run_agent_loop(
                ai_generate_fn=ai_service.generate_response,
                messages=prepared_messages,
                tools=all_tools if all_tools else None,
                tool_choice=tool_choice,
                response_format=response_format,
                provider=provider,
                model=model_requested,
                username=key_data.get("created_by", "admin"),
                context=context,
                max_iterations=min(data.get("max_iterations", 10), 10),
                enable_planning=enable_planning,
                enable_reflection=enable_reflection,
            )
        )

        result = build_agent_response(
            response_text=loop_result.final_response,
            model=model_requested,
            tools=all_tools if all_tools else None,
            response_format=response_format,
            prompt_tokens=prompt_tokens,
            completion_id=completion_id,
            stop_reason=loop_result.stop_reason,
            plan_data=loop_result.plan_data,
            supervisor_stats=loop_result.supervisor_stats,
            reflection_data=loop_result.reflection_data,
        )

        if loop_result.tool_calls_made:
            result["agent_loop"] = loop_result.to_dict()

        return jsonify(result)

    except Exception as e:
        logger.error(f"Agent completions error: {e}", exc_info=True)
        return jsonify({
            "error": {
                "type": "server_error",
                "message": str(e)
            }
        }), 500
    finally:
        if loop:
            loop.close()


def _handle_agent_stream(
    prepared_messages, last_user_msg, provider, model_requested,
    key_data, tools, tool_choice, response_format,
    prompt_tokens, completion_id, api_key,
    enable_planning=True, enable_reflection=True, max_iterations=10
):
    """Handle streaming response for agent completions with full agent loop."""
    import asyncio

    def generate():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            context = {
                "api_key": api_key,
                "session_id": f"session_{api_key[:16]}",
                "username": key_data.get("created_by", "admin")
            }

            event_start = {
                "type": "message_start",
                "message": {
                    "id": completion_id,
                    "type": "message",
                    "role": "assistant",
                    "model": model_requested,
                    "content": [],
                    "usage": {"input_tokens": prompt_tokens, "output_tokens": 0}
                }
            }
            yield f"event: message_start\ndata: {json.dumps(event_start)}\n\n"

            loop_result = loop.run_until_complete(
                run_agent_loop(
                    ai_generate_fn=ai_service.generate_response,
                    messages=prepared_messages,
                    tools=tools,
                    tool_choice=tool_choice,
                    response_format=response_format,
                    provider=provider,
                    model=model_requested,
                    username=key_data.get("created_by", "admin"),
                    context=context,
                    max_iterations=max_iterations,
                    enable_planning=enable_planning,
                    enable_reflection=enable_reflection,
                )
            )

            final_text = loop_result.final_response or ""

            if loop_result.tool_calls_made:
                for i, tc in enumerate(loop_result.tool_calls_made):
                    tool_event = {
                        "type": "tool_call",
                        "index": i,
                        "name": tc.get("name", ""),
                        "arguments": tc.get("arguments", {}),
                        "result_preview": tc.get("result_preview", "")[:500]
                    }
                    yield f"event: tool_call\ndata: {json.dumps(tool_event)}\n\n"

            yield f"event: content_block_start\ndata: {json.dumps({'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}})}\n\n"

            chunk_size = 20
            for i in range(0, len(final_text), chunk_size):
                chunk_text = final_text[i:i + chunk_size]
                delta_event = {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "text_delta", "text": chunk_text}
                }
                yield f"event: content_block_delta\ndata: {json.dumps(delta_event)}\n\n"

            yield f"event: content_block_stop\ndata: {json.dumps({'type': 'content_block_stop', 'index': 0})}\n\n"

            completion_tokens = len(final_text.split()) * 2
            msg_stop = {
                "type": "message_stop",
                "stop_reason": loop_result.stop_reason,
                "usage": {
                    "input_tokens": prompt_tokens,
                    "output_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "supervisor": loop_result.supervisor_stats,
            }
            if loop_result.plan_data:
                msg_stop["plan_data"] = loop_result.plan_data
            if loop_result.reflection_data:
                msg_stop["reflection_data"] = loop_result.reflection_data
            if loop_result.tool_calls_made:
                msg_stop["agent_loop"] = loop_result.to_dict()

            yield f"event: message_stop\ndata: {json.dumps(msg_stop)}\n\n"
            yield "event: done\ndata: [DONE]\n\n"

        except Exception as e:
            error_event = {"type": "error", "error": {"type": "server_error", "message": str(e)}}
            yield f"event: error\ndata: {json.dumps(error_event)}\n\n"
        finally:
            loop.close()

    return Response(
        generate(),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


# ============================================================
# TEST API ENDPOINTS
# ============================================================

@app.route("/api/test", methods=["POST"])
def test_api_run():
    """Test a specific provider/model combination."""
    import asyncio

    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json(silent=True) or {}
    provider_name = data.get("provider", "Auto")
    test_message = data.get("message", "Hello, respond with one short sentence.")
    custom_model = data.get("model")

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(
            ai_service.test_provider_directly(provider_name, test_message, custom_model)
        )
        return jsonify(result)
    except Exception as e:
        return jsonify({"provider": provider_name, "status": "error", "error": str(e)}), 500
    finally:
        if loop:
            loop.close()

@app.route("/api/test/all", methods=["POST"])
def test_all_providers():
    """Test all providers and models at once."""
    import asyncio

    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401

    data = request.get_json(silent=True) or {}
    test_message = data.get("message", "Hello, respond with one short sentence.")

    providers_with_models = ai_service.get_all_providers_with_models()
    results = []

    loop = None
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        for provider_name, models in providers_with_models.items():
            for model_name in models:
                try:
                    result = loop.run_until_complete(
                        ai_service.test_provider_directly(provider_name, test_message, model_name)
                    )
                    results.append(result)
                except Exception as e:
                    results.append({
                        "provider": provider_name,
                        "model": model_name,
                        "status": "error",
                        "error": str(e)
                    })
    finally:
        if loop:
            loop.close()

    success_count = sum(1 for r in results if r.get("status") == "success")
    return jsonify({
        "total": len(results),
        "success": success_count,
        "error": len(results) - success_count,
        "results": results
    })


# ============================================================
# MODELS & PROVIDERS
# ============================================================

@app.route("/api/models", methods=["GET"])
def get_models():
    """Get all available models grouped by provider."""
    providers_with_models = ai_service.get_all_providers_with_models()
    return jsonify(providers_with_models)

@app.route("/api/providers", methods=["GET"])
def get_providers():
    """Get list of available providers."""
    return jsonify({"providers": list(config.available_providers.keys())})

@app.route("/api/model-info", methods=["GET"])
def get_model_info_api():
    """Get info about a specific model."""
    model_name = request.args.get("model", "")
    if not model_name:
        return jsonify({"error": "Model name required"}), 400
    model_info = config.get_model_info(model_name)
    cat_info = config.category_info.get(model_info.get("category", "general"), {})
    return jsonify({
        "model": model_name,
        "name": model_info.get("name", model_name),
        "category": model_info.get("category", "general"),
        "category_label": cat_info.get("label", "General"),
        "category_color": cat_info.get("color", "#034953"),
        "description": model_info.get("desc", ""),
        "tags": model_info.get("tags", [])
    })

@app.route("/api/models/catalog", methods=["GET"])
def get_model_catalog():
    """Get full model catalog with capabilities."""
    return jsonify({
        "models": config.model_capabilities,
        "categories": config.category_info,
    })


# ============================================================
# HEALTH / UTILITY
# ============================================================

@app.route("/health", methods=["GET"])
@app.route("/api/health", methods=["GET"])
def health_check():
    providers_with_models = ai_service.get_all_providers_with_models()
    total_models = sum(len(models) for models in providers_with_models.values()) if isinstance(providers_with_models, dict) else 0
    num_providers = len(providers_with_models) if isinstance(providers_with_models, dict) else 0
    return jsonify({
        "status": "ok",
        "timestamp": _time.time(),
        "available_providers": num_providers,
        "total_models": total_models,
        "uptime_seconds": _time.time() - app.config.get("start_time", _time.time())
    })

@app.route("/ping", methods=["GET"])
@app.route("/api/ping", methods=["GET"])
def ping():
    return jsonify({
        "status": "pong",
        "timestamp": _time.time(),
        "uptime_seconds": _time.time() - app.config.get("start_time", _time.time())
    })

@app.route("/v1/models", methods=["GET"])
def openai_list_models():
    api_key = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    key_record = db_manager.get_api_key_by_key(api_key) if api_key else None
    if not key_record:
        return jsonify({"error": {"message": "Invalid API key", "type": "invalid_request_error"}}), 401
    providers_with_models = ai_service.get_all_providers_with_models()
    models_list = []
    if isinstance(providers_with_models, dict):
        for provider_name, models in providers_with_models.items():
            for m in models:
                models_list.append({
                    "id": m,
                    "object": "model",
                    "created": int(_time.time()),
                    "owned_by": provider_name,
                    "permission": [],
                    "root": m,
                    "parent": None
                })
    return jsonify({"object": "list", "data": models_list})

@app.route("/docs", methods=["GET"])
def swagger_docs_redirect():
    return redirect("/")

@app.route("/provider-status", methods=["GET"])
def provider_status():
    """Get provider health status."""
    from utils.provider_monitor import provider_monitor
    username = session.get('logged_in_user')
    if not username:
        return jsonify({"error": "Not authenticated"}), 401
    return jsonify(provider_monitor.get_status_summary())

@app.route("/favicon.ico")
def favicon():
    try:
        from flask import send_from_directory
        static_folder = app.static_folder or str(Path(__file__).parent / "static")
        return send_from_directory(
            str(Path(static_folder) / "img"),
            "favicon(Nicoladipa).png",
            mimetype='image/png'
        )
    except:
        return "", 204

@app.route("/api/backup", methods=["POST"])
def api_backup():
    username = session.get('logged_in_user')
    is_admin = session.get('is_admin', False)
    if not username or not is_admin:
        return jsonify({"error": "Admin only"}), 403
    db_manager.export_data_backup()
    return jsonify({"success": True, "message": "Backup exported"})


# ============================================================
# KEEP ALIVE & SERVER INIT
# ============================================================

def _start_keep_alive():
    import urllib.request
    def keep_alive_worker():
        deploy_url = os.environ.get('REPLIT_DEPLOYMENT_URL', '')
        if deploy_url:
            url = f"{deploy_url.rstrip('/')}/ping"
        else:
            replit_domain = os.environ.get('REPLIT_DEV_DOMAIN', '') or os.environ.get('REPLIT_DOMAINS', '').split(',')[0].strip()
            if replit_domain:
                url = f"https://{replit_domain}/ping"
            else:
                url = f"{PRODUCTION_URL}/ping"
        logger.info(f"Keep-alive started, pinging {url} every 4 minutes")
        while True:
            try:
                req = urllib.request.Request(url, method='GET')
                urllib.request.urlopen(req, timeout=10)
            except Exception:
                pass
            _time.sleep(240)
    t = threading.Thread(target=keep_alive_worker, daemon=True)
    t.start()

server_manager = None

def _initialize_server():
    global server_manager
    if server_manager is not None:
        return

    arg_parser = ServerArgumentParser()
    args = arg_parser.parse_args()

    if not args.enable_virtual_users:
        args.enable_virtual_users = True

    server_manager = ServerManager(args)
    server_manager.setup_password()
    server_manager.setup_virtual_users()

    logger.info(f"Server configuration:")
    logger.info(f"  Port: {args.port}")
    logger.info(f"  Provider: {args.provider}")
    logger.info(f"  Model: {args.model}")
    logger.info(f"  Base URL: {_get_base_url()}")
    logger.info(f"  Production URL: {_get_production_base_url()}")

_initialize_server()
_start_keep_alive()

def main():
    try:
        _initialize_server()
        app.run(
            host=config.server.host,
            port=server_manager.args.port,
            debug=config.server.debug
        )
    except KeyboardInterrupt:
        logger.info("Server shutdown requested")
    except Exception as e:
        logger.error(f"Server startup failed: {e}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
