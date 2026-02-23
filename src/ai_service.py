"""AI service for handling GPT interactions."""

import json
import random
from typing import Dict, List, Any, Optional, AsyncGenerator
from pathlib import Path

import g4f

from config import config
from database import db_manager
from utils.exceptions import AIProviderError, ValidationError
from utils.logging import logger
from utils.http_utils import safe_api_call, TimeoutConfig
from utils.helpers import (
    load_json_file, 
    clean_response_sources, 
    select_random_proxy,
    create_dummy_cookies
)
from utils.provider_monitor import provider_monitor
from utils.validation import validate_provider, validate_model

class AIService:
    """Service for handling AI interactions."""
    
    def __init__(self):
        self.db = db_manager
        self.config = config
    
    async def generate_response(
        self,
        message: str,
        username: str = "admin",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        remove_sources: bool = True,
        use_proxies: bool = False,
        cookie_file: Optional[str] = None
    ) -> str:
        """Generate AI response.
        
        Args:
            message: User message
            username: Username for context
            provider: AI provider override
            model: AI model override
            system_prompt: System prompt override
            use_history: Whether to use chat history
            remove_sources: Whether to remove source references
            use_proxies: Whether to use proxies
            cookie_file: Cookie file path
            
        Returns:
            AI response text
            
        Raises:
            AIProviderError: If AI generation fails
            ValidationError: If parameters are invalid
        """
        try:
            # Get user settings
            if username == "admin":
                settings = self.db.get_settings()
                user_settings = {
                    "provider": provider or settings.get("provider", self.config.api.default_provider),
                    "model": model or settings.get("model", self.config.api.default_model),
                    "system_prompt": system_prompt or settings.get("system_prompt", ""),
                    "message_history": use_history and settings.get("message_history", False)
                }
            else:
                user_data = self.db.get_user_by_username(username)
                if not user_data:
                    raise ValidationError(f"User '{username}' not found")
                
                user_settings = {
                    "provider": provider or user_data.get("provider", self.config.api.default_provider),
                    "model": model or user_data.get("model", self.config.api.default_model),
                    "system_prompt": system_prompt or user_data.get("system_prompt", ""),
                    "message_history": use_history and user_data.get("message_history", False)
                }
            
            # Validate provider and model
            is_valid, error_msg = validate_provider(user_settings["provider"], self.config.available_providers)
            if not is_valid:
                raise ValidationError(error_msg)
            
            is_valid, error_msg = validate_model(user_settings["model"])
            if not is_valid:
                raise ValidationError(error_msg)
            
            category_prompt = self.config.get_category_system_prompt(user_settings["model"])
            effective_system_prompt = user_settings["system_prompt"]
            if category_prompt and not effective_system_prompt:
                effective_system_prompt = category_prompt
            elif category_prompt and effective_system_prompt:
                effective_system_prompt = category_prompt + "\n\n" + effective_system_prompt

            chat_history = self._prepare_chat_history(
                message=message,
                username=username,
                system_prompt=effective_system_prompt,
                use_history=user_settings["message_history"]
            )
            
            cookies = self._load_cookies(cookie_file)
            
            # Prepare proxy
            proxy = self._get_proxy() if use_proxies else None
            
            # Generate response
            response_text = await self._call_ai_api(
                chat_history=chat_history,
                provider=user_settings["provider"],
                model=user_settings["model"],
                cookies=cookies,
                proxy=proxy
            )
            
            # Clean response if needed
            if remove_sources:
                response_text = clean_response_sources(response_text)
            
            # Save chat history if enabled
            if user_settings["message_history"]:
                chat_history.append({"role": "assistant", "content": response_text})
                self.db.save_chat_history(username, json.dumps(chat_history))
            
            logger.info(f"AI response generated for user '{username}' using provider '{user_settings['provider']}'")
            return response_text
            
        except (ValidationError, AIProviderError):
            raise
        except Exception as e:
            logger.error(f"Failed to generate AI response: {e}")
            raise AIProviderError(f"AI generation failed: {e}")
    
    def _prepare_chat_history(
        self,
        message: str,
        username: str,
        system_prompt: str,
        use_history: bool
    ) -> List[Dict[str, str]]:
        """Prepare chat history for AI request.
        
        Args:
            message: Current user message
            username: Username
            system_prompt: System prompt
            use_history: Whether to load previous history
            
        Returns:
            List of chat messages
        """
        chat_history = []
        
        # Add system prompt if provided
        if system_prompt:
            chat_history.append({"role": "system", "content": system_prompt})
        
        # Load previous history if enabled
        if use_history:
            history_json = self.db.get_chat_history(username)
            if history_json:
                try:
                    previous_history = json.loads(history_json)
                    # Remove system prompt from previous history to avoid duplication
                    previous_history = [msg for msg in previous_history if msg.get("role") != "system"]
                    chat_history.extend(previous_history)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid chat history JSON for user '{username}'")
        
        # Add current message
        chat_history.append({"role": "user", "content": message})
        
        return chat_history
    
    def _load_cookies(self, cookie_file: Optional[str]) -> Dict[str, str]:
        """Load cookies from file.
        
        Args:
            cookie_file: Path to cookie file
            
        Returns:
            Dictionary of cookies
        """
        if not cookie_file:
            return create_dummy_cookies()
        
        cookie_path = Path(cookie_file)
        cookies = load_json_file(cookie_path, {})
        
        if not cookies:
            logger.warning(f"No cookies found in {cookie_file}, using dummy cookies")
            return create_dummy_cookies()
        
        logger.debug(f"Loaded {len(cookies)} cookies from {cookie_file}")
        return cookies
    
    def _get_proxy(self) -> Optional[str]:
        """Get random proxy from configuration.
        
        Returns:
            Proxy URL or None
        """
        proxies_path = Path(self.config.files.proxies_file)
        proxies = load_json_file(proxies_path, [])
        
        if not proxies:
            logger.warning("No proxies configured")
            return None
        
        proxy_url = select_random_proxy(proxies)
        if proxy_url:
            logger.debug(f"Using proxy: {proxy_url.split('@')[0]}@***")  # Mask credentials
        
        return proxy_url
    
    async def _call_ai_api(
        self,
        chat_history: List[Dict[str, str]],
        provider: str,
        model: str,
        cookies: Dict[str, str],
        proxy: Optional[str]
    ) -> str:
        """Call AI API to generate response.
        
        Args:
            chat_history: Chat message history
            provider: AI provider
            model: AI model
            cookies: Request cookies
            proxy: Proxy URL
            
        Returns:
            AI response text
            
        Raises:
            AIProviderError: If API call fails
        """
        # Check if provider is blacklisted
        if provider_monitor.is_provider_blacklisted(provider):
            logger.warning(f"Provider '{provider}' is blacklisted, using fallback")
            provider = "Auto"
        
        # Get reliable providers for fallback
        reliable_providers = provider_monitor.get_reliable_providers(self.config.available_providers)
        
        # Try original provider first
        if provider != "Auto":
            ai_provider = self.config.available_providers.get(provider)
            if ai_provider:
                logger.info(f"Attempting with provider: {provider}")
                response = await self._make_api_call(chat_history, ai_provider, model, cookies, proxy, provider)
                if response:
                    provider_monitor.record_success(provider)
                    return response
                else:
                    provider_monitor.record_failure(provider, "no_response")
        
        # Try Auto mode with compatible model
        auto_model = self._get_fallback_model(model, "Auto")
        logger.info(f"Attempting with Auto mode using model {auto_model}")
        response = await self._make_api_call(chat_history, None, auto_model, cookies, proxy, "Auto")
        if response:
            provider_monitor.record_success("Auto")
            return response
        else:
            provider_monitor.record_failure("Auto", "no_response")
        
        # Try reliable providers as fallback
        logger.warning("Auto mode failed, trying reliable providers")
        for fallback_provider in reliable_providers[:3]:  # Try top 3 reliable providers
            try:
                ai_provider = self.config.available_providers.get(fallback_provider)
                if ai_provider:
                    fallback_model = self._get_fallback_model(model, fallback_provider)
                    logger.info(f"Attempting reliable fallback: {fallback_provider} with model {fallback_model}")
                    response = await self._make_api_call(chat_history, ai_provider, fallback_model, cookies, proxy, fallback_provider)
                    if response:
                        provider_monitor.record_success(fallback_provider)
                        logger.info(f"Successfully used reliable fallback: {fallback_provider}")
                        return response
                    else:
                        provider_monitor.record_failure(fallback_provider, "no_response")
            except Exception as e:
                provider_monitor.record_failure(fallback_provider, "exception")
                logger.warning(f"Reliable fallback {fallback_provider} failed: {e}")
                continue
        
        # Last resort: try any healthy provider
        healthy_providers = provider_monitor.get_healthy_providers(self.config.available_providers)
        logger.warning("Reliable providers failed, trying any healthy provider")
        
        for fallback_provider in healthy_providers[:5]:  # Try up to 5 healthy providers
            if fallback_provider in reliable_providers:
                continue  # Already tried
            
            try:
                ai_provider = self.config.available_providers.get(fallback_provider)
                if ai_provider:
                    fallback_model = self._get_fallback_model(model, fallback_provider)
                    logger.info(f"Attempting healthy fallback: {fallback_provider} with model {fallback_model}")
                    response = await self._make_api_call(chat_history, ai_provider, fallback_model, cookies, proxy, fallback_provider)
                    if response:
                        provider_monitor.record_success(fallback_provider)
                        logger.info(f"Successfully used healthy fallback: {fallback_provider}")
                        return response
                    else:
                        provider_monitor.record_failure(fallback_provider, "no_response")
            except Exception as e:
                provider_monitor.record_failure(fallback_provider, "exception")
                logger.warning(f"Healthy fallback {fallback_provider} failed: {e}")
                continue
        
        # Log provider status summary for debugging
        status_summary = provider_monitor.get_status_summary()
        logger.error(f"All providers failed. Status summary: {status_summary}")
        
        raise AIProviderError("All providers failed to generate a response")
    
    async def generate_response_stream(
        self,
        message: str,
        username: str = "admin",
        provider: Optional[str] = None,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        use_history: bool = False,
        remove_sources: bool = True,
        use_proxies: bool = False,
        cookie_file: Optional[str] = None,
        image_data: Optional[str] = None
    ):
        """Generate AI response as an async generator yielding chunks."""
        try:
            if username == "admin":
                settings = self.db.get_settings()
                user_settings = {
                    "provider": provider or settings.get("provider", self.config.api.default_provider),
                    "model": model or settings.get("model", self.config.api.default_model),
                    "system_prompt": system_prompt or settings.get("system_prompt", ""),
                    "message_history": use_history and settings.get("message_history", False)
                }
            else:
                user_data = self.db.get_user_by_username(username)
                if not user_data:
                    raise ValidationError(f"User '{username}' not found")
                user_settings = {
                    "provider": provider or user_data.get("provider", self.config.api.default_provider),
                    "model": model or user_data.get("model", self.config.api.default_model),
                    "system_prompt": system_prompt or user_data.get("system_prompt", ""),
                    "message_history": use_history and user_data.get("message_history", False)
                }

            is_valid, error_msg = validate_provider(user_settings["provider"], self.config.available_providers)
            if not is_valid:
                raise ValidationError(error_msg)
            is_valid, error_msg = validate_model(user_settings["model"])
            if not is_valid:
                raise ValidationError(error_msg)

            category_prompt = self.config.get_category_system_prompt(user_settings["model"])
            effective_system_prompt = user_settings["system_prompt"]
            if category_prompt and not effective_system_prompt:
                effective_system_prompt = category_prompt
            elif category_prompt and effective_system_prompt:
                effective_system_prompt = category_prompt + "\n\n" + effective_system_prompt

            chat_history = self._prepare_chat_history(
                message=message,
                username=username,
                system_prompt=effective_system_prompt,
                use_history=user_settings["message_history"]
            )

            if image_data:
                last_msg = chat_history[-1] if chat_history else None
                if last_msg and last_msg.get("role") == "user":
                    original_text = last_msg["content"] if isinstance(last_msg["content"], str) else str(last_msg["content"])
                    last_msg["content"] = original_text + "\n\n[User attached an image to this message]"

            cookies = self._load_cookies(cookie_file)
            proxy = self._get_proxy() if use_proxies else None

            provider_name = user_settings["provider"]
            model_name = user_settings["model"]

            def extract_text(chunk):
                if isinstance(chunk, str):
                    return chunk
                if hasattr(chunk, 'choices'):
                    if not chunk.choices:
                        return ""
                    choice = chunk.choices[0]
                    if hasattr(choice, 'delta'):
                        delta = choice.delta
                        if hasattr(delta, 'content') and delta.content:
                            return delta.content
                        return ""
                    if hasattr(choice, 'message'):
                        msg = choice.message
                        if hasattr(msg, 'content') and msg.content:
                            return msg.content
                        return ""
                    return ""
                return ""

            async def try_stream_provider(prov_name, prov_obj, mdl):
                kwargs = {
                    "model": mdl,
                    "messages": chat_history,
                    "cookies": cookies,
                    "proxy": proxy,
                    "stream": True
                }
                if prov_obj is not None:
                    kwargs["provider"] = prov_obj
                return g4f.ChatCompletion.create_async(**kwargs)

            async def collect_stream(response):
                texts = []
                if hasattr(response, '__aiter__'):
                    async for chunk in response:
                        text = extract_text(chunk)
                        if text:
                            texts.append(text)
                elif hasattr(response, '__await__'):
                    result = await response
                    if hasattr(result, '__aiter__'):
                        async for chunk in result:
                            text = extract_text(chunk)
                            if text:
                                texts.append(text)
                    else:
                        text = extract_text(result)
                        if text:
                            texts.append(text)
                else:
                    text = extract_text(response)
                    if text:
                        texts.append(text)
                return texts

            providers_to_try = []
            if provider_name != "Auto":
                ai_provider = self.config.available_providers.get(provider_name)
                providers_to_try.append((provider_name, ai_provider, model_name))
            auto_model = self._get_fallback_model(model_name, "Auto")
            providers_to_try.append(("Auto", None, auto_model))

            from utils.provider_monitor import provider_monitor
            reliable = provider_monitor.get_reliable_providers(self.config.available_providers)
            for rp in reliable[:3]:
                if rp != provider_name:
                    rp_obj = self.config.available_providers.get(rp)
                    if rp_obj:
                        fb_model = self._get_fallback_model(model_name, rp)
                        providers_to_try.append((rp, rp_obj, fb_model))

            full_response = ""
            streamed = False

            for pname, pobj, pmodel in providers_to_try:
                try:
                    response = await try_stream_provider(pname, pobj, pmodel)
                    chunks = await collect_stream(response)
                    combined = "".join(chunks)
                    if combined and not combined.startswith("[ERROR]"):
                        for c in chunks:
                            if c and not c.startswith("[ERROR]"):
                                full_response += c
                                yield c
                        streamed = True
                        provider_monitor.record_success(pname)
                        break
                    else:
                        provider_monitor.record_failure(pname, "error_response")
                        logger.warning(f"Stream provider {pname} returned error: {combined[:100]}")
                except Exception as e:
                    provider_monitor.record_failure(pname, str(e)[:100])
                    logger.warning(f"Stream provider {pname} failed: {e}")
                    continue

            if not streamed:
                try:
                    non_stream = await self.generate_response(
                        message=chat_history[-1].get("content", "") if chat_history else "",
                        username=username,
                        provider=provider,
                        model=model,
                        system_prompt=system_prompt,
                        use_history=False,
                        remove_sources=remove_sources,
                        use_proxies=use_proxies
                    )
                    if non_stream:
                        full_response = non_stream
                        yield non_stream
                except Exception as e:
                    logger.error(f"All streaming fallbacks failed: {e}")
                    yield f"Error: All providers failed. Please try again."

            if remove_sources and full_response:
                full_response = clean_response_sources(full_response)

            if user_settings["message_history"] and full_response:
                chat_history.append({"role": "assistant", "content": full_response})
                self.db.save_chat_history(username, json.dumps(chat_history))

        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {e}"

    async def _make_api_call(
        self,
        chat_history: List[Dict[str, str]],
        ai_provider,
        model: str,
        cookies: Dict[str, str],
        proxy: Optional[str],
        provider_name: str = "Unknown"
    ) -> Optional[str]:
        
        async def make_request():
            kwargs = {
                "model": model,
                "messages": chat_history,
            }
            if proxy:
                kwargs["proxy"] = proxy
            if ai_provider is not None:
                kwargs["provider"] = ai_provider
            
            return await g4f.ChatCompletion.create_async(**kwargs)
        
        try:
            response = await safe_api_call(
                make_request,
                timeout=TimeoutConfig.DEFAULT_TIMEOUT,
                max_retries=0
            )
            
            if response is None:
                logger.warning(f"Provider {provider_name} returned no response")
                return None
            
            # Collect response
            response_text = ""
            
            # Handle both string responses and async generators
            if hasattr(response, '__aiter__'):
                # It's an async generator
                import asyncio
                try:
                    async for chunk in response:
                        response_text += str(chunk)
                        # Add small delay to prevent blocking and allow timeout handling
                        await asyncio.sleep(0.001)
                except Exception as e:
                    logger.warning(f"Error reading streaming response from {provider_name}: {e}")
                    return None
            else:
                # It's already a string
                response_text = str(response)
            
            if not response_text or response_text.strip() == "":
                logger.warning(f"Empty response from provider {provider_name}")
                return None
            
            logger.debug(f"Received response of {len(response_text)} characters from {provider_name}")
            return response_text
            
        except Exception as e:
            error_msg = str(e).lower()
            error_type = "unknown"
            
            if "401" in error_msg or "unauthorized" in error_msg:
                error_type = "unauthorized"
                logger.warning(f"Provider {provider_name} returned unauthorized error: {e}")
            elif "chrome" in error_msg or "browser" in error_msg:
                error_type = "browser_required"
                logger.warning(f"Provider {provider_name} requires browser but none found: {e}")
            elif "timeout" in error_msg or "too slow" in error_msg:
                error_type = "timeout"
                logger.warning(f"Provider {provider_name} connection timeout: {e}")
            elif "connection" in error_msg or "network" in error_msg:
                error_type = "network"
                logger.warning(f"Provider {provider_name} network error: {e}")
            else:
                logger.warning(f"Provider {provider_name} failed with error: {e}")
            
            # Record failure in monitor
            provider_monitor.record_failure(provider_name, error_type)
            return None
    
    PROVIDER_DEFAULT_MODELS = {
        "PollinationsAI": "openai",
        "TeachAnything": "gemma",
        "Yqcloud": "gpt-4",
        "Pi": "pi",
        "Copilot": "Copilot",
        "Perplexity": "auto",
        "Gemini": "gemini-2.5-flash",
        "DeepInfra": "MiniMaxAI/MiniMax-M2.5",
        "HuggingSpace": "command-a",
        "Groq": "llama-3.3-70b-versatile",
        "GeminiPro": "models/gemini-2.5-flash",
        "CohereForAI_C4AI_Command": "command-a-03-2025",
        "OperaAria": "default",
        "Auto": "gpt-4",
    }

    PROVIDER_MODEL_PREFIXES = {
        "GeminiPro": "models/",
        "DeepInfra": "/",
        "Groq": "/",
    }

    NON_CHAT_MODELS = {
        "whisper-large-v3-turbo",
        "whisper-large-v3",
        "distil-whisper-large-v3-en",
        "canopylabs/orpheus-arabic-saudi",
        "meta-llama/llama-prompt-guard-2-86m",
        "meta-llama/llama-prompt-guard-2-22m",
        "meta-llama/llama-guard-4-12b",
    }

    def _get_best_model_for_provider(self, provider_name: str) -> str:
        """Get the best working model for a specific provider."""
        if provider_name in self.PROVIDER_DEFAULT_MODELS:
            return self.PROVIDER_DEFAULT_MODELS[provider_name]
        
        ai_provider = self.config.available_providers.get(provider_name)
        if ai_provider and hasattr(ai_provider, 'default_model') and ai_provider.default_model:
            return ai_provider.default_model
        
        return "gpt-4"

    GENERIC_MODELS = {"gpt-4", "gpt-4o", "gpt-3.5-turbo", "openai"}

    def _is_model_compatible_with_provider(self, model: str, provider_name: str) -> bool:
        """Check if a model name is likely compatible with a provider."""
        default = self.PROVIDER_DEFAULT_MODELS.get(provider_name, "")
        if model == default:
            return True
        if provider_name == "Auto":
            return model in self.GENERIC_MODELS or not any(c in model for c in ['/', 'models/'])
        if provider_name == "GeminiPro" and model.startswith("models/"):
            return True
        if provider_name == "GeminiPro" and not model.startswith("models/"):
            return False
        if model.startswith("models/") and provider_name != "GeminiPro":
            return False
        if provider_name == "CohereForAI_C4AI_Command" and model.startswith("command-"):
            return True
        if provider_name == "CohereForAI_C4AI_Command" and not model.startswith("command-"):
            return False
        if provider_name == "HuggingSpace" and model.startswith("command-"):
            return True
        if provider_name == "TeachAnything" and model != "gemma":
            return False
        if provider_name == "OperaAria" and model != "default":
            return False
        if provider_name == "Yqcloud" and model != "gpt-4":
            return False
        if provider_name == "PollinationsAI":
            pollinations_models = {"openai", "gpt-5-nano", "openai-fast", "openai-large",
                "qwen-coder", "mistral", "gemini", "gemini-fast", "deepseek", "grok",
                "claude-fast", "claude", "perplexity-fast", "perplexity-reasoning",
                "kimi", "gemini-legacy", "nova-fast", "glm", "minimax", "nomnom"}
            return model in pollinations_models
        return True

    def _get_fallback_model(self, original_model: str, fallback_provider: str) -> str:
        """Get appropriate model for a fallback provider.
        
        If the original model is compatible with the fallback provider, use it.
        Otherwise, use the fallback provider's default model.
        """
        if original_model in self.NON_CHAT_MODELS:
            return self._get_best_model_for_provider(fallback_provider)
        if self._is_model_compatible_with_provider(original_model, fallback_provider):
            return original_model
        return self._get_best_model_for_provider(fallback_provider)

    async def test_provider_directly(
        self,
        provider_name: str,
        test_message: str = "Hello, respond with one short sentence.",
        custom_model: Optional[str] = None
    ) -> dict:
        """Test a specific provider directly without any fallback.
        
        Returns:
            dict with keys: provider, status ('success'/'error'), response, model
        """
        try:
            if provider_name == "Auto":
                ai_provider = None
                model = custom_model or "gpt-4"
            else:
                ai_provider = self.config.available_providers.get(provider_name)
                if ai_provider is None:
                    return {"provider": provider_name, "status": "error", "response": f"Provider '{provider_name}' not found"}
                model = custom_model or self._get_best_model_for_provider(provider_name)
            
            chat_history = [{"role": "user", "content": test_message}]
            cookies = create_dummy_cookies()
            
            response = await self._make_api_call(
                chat_history=chat_history,
                ai_provider=ai_provider,
                model=model,
                cookies=cookies,
                proxy=None,
                provider_name=provider_name
            )
            
            if response and response.strip():
                return {"provider": provider_name, "status": "success", "response": response[:500], "model": model}
            else:
                return {"provider": provider_name, "status": "error", "response": "Provider returned no response", "model": model}
        except Exception as e:
            return {"provider": provider_name, "status": "error", "response": str(e), "model": ""}

    def get_available_models(self, provider: str) -> List[str]:
        """Get available models for a provider.
        
        Args:
            provider: Provider name
            
        Returns:
            List of available models
        """
        if provider == "Auto":
            return self.config.generic_models
        
        try:
            provider_obj = self.config.available_providers.get(provider)
            if not provider_obj:
                return ["default"]
            
            models = []
            
            if hasattr(provider_obj, 'get_models'):
                try:
                    dynamic_models = provider_obj.get_models()
                    if dynamic_models:
                        models = list(dynamic_models)
                except Exception as e:
                    logger.debug(f"get_models() failed for {provider}: {e}")
            
            if not models:
                static_models = getattr(provider_obj, 'models', None)
                if static_models:
                    models = list(static_models)
            
            if not models:
                text_models_attr = getattr(provider_obj, 'text_models', None)
                if text_models_attr:
                    models = list(text_models_attr)
            
            if not models:
                default = getattr(provider_obj, 'default_model', None)
                if default:
                    models = [default]
            
            image_models = getattr(provider_obj, 'image_models', None)
            if image_models:
                image_set = set(image_models)
                models = [m for m in models if m not in image_set]
            
            models = [m for m in models if m not in self.NON_CHAT_MODELS]
            
            broken_models = {
                "PollinationsAI": {"claude-opus-4.6", "gemini-3-flash", "midijourney", "openai-audio", "qwen-3guard-gen-8b", "qwen-character", "grok-fast", "gemini-3-pro", "claude-sonnet-4.5", "gpt-4o-mini-audio-preview", "minimax-m2.1", "gpt-5.2", "qwen-3-coder", "mistral-small", "gemini-2.5-flash-lite", "deepseek-v3", "gemini-2.5-flash-search", "chickytutor", "claude-haiku-4.5", "sonar", "sonar-reasoning", "kimi-k2.5", "amazon-nova-micro", "glm-5"},
                "Perplexity": {"claude35haiku", "mistral", "claude3opus", "o1", "llama_x_large", "gemini"},
                "DeepInfra": {"deepseek-ai/DeepSeek-V3.2", "MiniMaxAI/MiniMax-M2"},
                "GeminiPro": {"models/gemini-2.0-flash", "models/gemini-2.0-flash-lite"},
                "CohereForAI_C4AI_Command": {"command-r-plus"},
            }
            exclude = broken_models.get(provider, set())
            if exclude:
                models = [m for m in models if m not in exclude]
            
            return models if models else ["default"]
        except Exception as e:
            logger.warning(f"Could not get models for provider '{provider}': {e}")
            return ["default"]
    
    def get_all_providers_with_models(self) -> Dict[str, List[str]]:
        """Get all providers with their available models."""
        result = {}
        for provider_name in self.config.available_providers:
            result[provider_name] = self.get_available_models(provider_name)
        return result

# Global AI service instance
ai_service = AIService()
