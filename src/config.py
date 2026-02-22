"""Configuration management for Api Dzeck Ai Web API."""

import os
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path

# Base configuration
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Ensure data directory exists
DATA_DIR.mkdir(exist_ok=True)

@dataclass
class DatabaseConfig:
    """Database configuration."""
    database_url: str = os.getenv("DATABASE_URL", "")
    
@dataclass
class ServerConfig:
    """Server configuration."""
    host: str = "0.0.0.0"
    port: int = 5000
    debug: bool = False
    max_content_length: int = 16 * 1024 * 1024  # 16 MB
    
@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str = os.getenv("SECRET_KEY", "freegpt4-stable-session-key-2024")
    password_min_length: int = 3
    
@dataclass
class APIConfig:
    """API configuration."""
    default_model: str = "openai"
    default_provider: str = "PollinationsAI"
    default_keyword: str = "text"
    fast_api_port: int = 1336
    
@dataclass
class FileConfig:
    """File configuration."""
    upload_folder: str = str(DATA_DIR)
    cookies_file: str = str(DATA_DIR / "cookies.json")
    proxies_file: str = str(DATA_DIR / "proxies.json")
    allowed_extensions: Optional[set] = None
    
    def __post_init__(self):
        if self.allowed_extensions is None:
            self.allowed_extensions = {'json'}

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.server = ServerConfig()
        self.security = SecurityConfig()
        self.api = APIConfig()
        self.files = FileConfig()
        
        # Load environment overrides
        self._load_env_overrides()
        
    def _load_env_overrides(self):
        """Load configuration from environment variables."""
        port_env = os.getenv("PORT")
        if port_env:
            self.server.port = int(port_env)
        debug_env = os.getenv("DEBUG")
        if debug_env:
            self.server.debug = debug_env.lower() == "true"
            
        model_env = os.getenv("DEFAULT_MODEL")
        if model_env:
            self.api.default_model = model_env
        provider_env = os.getenv("DEFAULT_PROVIDER")
        if provider_env:
            self.api.default_provider = provider_env
            
    @property
    def available_providers(self) -> Dict[str, Any]:
        """Get available providers."""
        import g4f
        return {
            "Auto": "",
            "PollinationsAI": g4f.Provider.PollinationsAI,
            "Perplexity": g4f.Provider.Perplexity,
            "DeepInfra": g4f.Provider.DeepInfra,
            "HuggingSpace": g4f.Provider.HuggingSpace,
            "Groq": g4f.Provider.Groq,
            "GeminiPro": g4f.Provider.GeminiPro,
            "CohereForAI_C4AI_Command": g4f.Provider.CohereForAI_C4AI_Command,
            "TeachAnything": g4f.Provider.TeachAnything,
            "Yqcloud": g4f.Provider.Yqcloud,
            "OperaAria": g4f.Provider.OperaAria,
        }
    
    @property
    def generic_models(self) -> list:
        """Get generic models."""
        return ["gpt-4"]

    @property
    def model_capabilities(self) -> Dict[str, Dict[str, Any]]:
        return {
            "openai": {"category": "general", "name": "GPT (OpenAI)", "icon": "brain", "desc": "Model AI serbaguna untuk chat, coding, analisis, dan tugas umum", "tags": ["chat", "code", "analysis"]},
            "gpt-4": {"category": "general", "name": "GPT-4", "icon": "brain", "desc": "Model AI serbaguna untuk chat, coding, analisis, dan tugas umum", "tags": ["chat", "code", "analysis"]},
            "gpt-4o": {"category": "general", "name": "GPT-4o", "icon": "brain", "desc": "Model multimodal cepat untuk chat dan analisis", "tags": ["chat", "code", "fast"]},
            "gpt41": {"category": "advanced", "name": "GPT-4.1", "icon": "rocket", "desc": "GPT-4.1 terbaru dengan kemampuan coding dan reasoning superior", "tags": ["chat", "code", "reasoning"]},
            "gpt45": {"category": "advanced", "name": "GPT-4.5", "icon": "rocket", "desc": "Model terbaru dengan kemampuan reasoning dan kreativitas tinggi", "tags": ["chat", "creative", "reasoning"]},
            "gpt5": {"category": "advanced", "name": "GPT-5", "icon": "rocket", "desc": "Model terkuat OpenAI dengan reasoning dan coding terbaik", "tags": ["chat", "code", "reasoning"]},
            "gpt5_thinking": {"category": "thinking", "name": "GPT-5 Thinking", "icon": "lightbulb", "desc": "GPT-5 dengan mode deep thinking - menunjukkan proses berpikir langkah demi langkah", "tags": ["reasoning", "math", "logic"]},
            "gpt-5-nano": {"category": "fast", "name": "GPT-5 Nano", "icon": "zap", "desc": "Model ringan dan super cepat untuk tugas sederhana", "tags": ["fast", "chat", "simple"]},
            "o3": {"category": "thinking", "name": "o3", "icon": "lightbulb", "desc": "Model reasoning OpenAI - analisis mendalam dan pemecahan masalah kompleks", "tags": ["reasoning", "math", "science"]},
            "o3pro": {"category": "thinking", "name": "o3 Pro", "icon": "lightbulb", "desc": "o3 Pro dengan reasoning lebih dalam untuk masalah sangat kompleks", "tags": ["reasoning", "math", "expert"]},
            "o3mini": {"category": "thinking", "name": "o3 Mini", "icon": "lightbulb", "desc": "Model reasoning ringan untuk analisis cepat", "tags": ["reasoning", "fast"]},
            "o4mini": {"category": "thinking", "name": "o4 Mini", "icon": "lightbulb", "desc": "Model reasoning terbaru generasi ke-4", "tags": ["reasoning", "fast"]},
            "claude2": {"category": "general", "name": "Claude 2", "icon": "brain", "desc": "Claude dari Anthropic - aman, membantu, dan jujur", "tags": ["chat", "writing", "analysis"]},
            "claude37sonnetthinking": {"category": "thinking", "name": "Claude 3.7 Sonnet Thinking", "icon": "lightbulb", "desc": "Claude 3.7 dengan mode thinking - reasoning mendalam dan transparan", "tags": ["reasoning", "analysis", "code"]},
            "claude40opus": {"category": "advanced", "name": "Claude 4 Opus", "icon": "rocket", "desc": "Model terkuat Anthropic untuk tugas yang paling kompleks", "tags": ["reasoning", "writing", "expert"]},
            "claude40opusthinking": {"category": "thinking", "name": "Claude 4 Opus Thinking", "icon": "lightbulb", "desc": "Claude 4 Opus dengan deep thinking mode", "tags": ["reasoning", "expert", "analysis"]},
            "claude41opusthinking": {"category": "thinking", "name": "Claude 4.1 Opus Thinking", "icon": "lightbulb", "desc": "Claude 4.1 Opus terbaru dengan thinking mode terdalam", "tags": ["reasoning", "expert", "analysis"]},
            "claude45sonnet": {"category": "advanced", "name": "Claude 4.5 Sonnet", "icon": "rocket", "desc": "Claude 4.5 Sonnet - seimbang antara kecepatan dan kualitas", "tags": ["chat", "code", "fast"]},
            "claude45sonnetthinking": {"category": "thinking", "name": "Claude 4.5 Sonnet Thinking", "icon": "lightbulb", "desc": "Claude 4.5 Sonnet dengan thinking mode", "tags": ["reasoning", "code", "analysis"]},
            "auto": {"category": "search", "name": "Perplexity Auto", "icon": "search", "desc": "Pencarian web otomatis dengan AI - jawaban real-time dari internet", "tags": ["search", "web", "realtime"]},
            "turbo": {"category": "search", "name": "Perplexity Turbo", "icon": "search", "desc": "Pencarian web cepat - respons kilat dengan data terkini", "tags": ["search", "fast", "web"]},
            "pplx_pro": {"category": "search", "name": "Perplexity Pro", "icon": "search", "desc": "Pencarian web premium dengan analisis mendalam dan sumber terpercaya", "tags": ["search", "analysis", "web"]},
            "pplx_pro_upgraded": {"category": "search", "name": "Perplexity Pro+", "icon": "search", "desc": "Perplexity Pro versi upgrade dengan akurasi lebih tinggi", "tags": ["search", "analysis", "web"]},
            "pplx_alpha": {"category": "search", "name": "Perplexity Alpha", "icon": "search", "desc": "Versi eksperimental Perplexity dengan fitur baru", "tags": ["search", "experimental"]},
            "pplx_beta": {"category": "search", "name": "Perplexity Beta", "icon": "search", "desc": "Versi beta Perplexity", "tags": ["search", "experimental"]},
            "pplx_reasoning": {"category": "thinking", "name": "Perplexity Reasoning", "icon": "lightbulb", "desc": "Perplexity dengan reasoning mode untuk analisis kompleks", "tags": ["reasoning", "search"]},
            "experimental": {"category": "search", "name": "Perplexity Experimental", "icon": "search", "desc": "Model eksperimental terbaru dari Perplexity", "tags": ["search", "experimental"]},
            "grok": {"category": "general", "name": "Grok", "icon": "brain", "desc": "Grok dari xAI - AI dengan humor dan pengetahuan real-time", "tags": ["chat", "humor", "realtime"]},
            "grok4": {"category": "advanced", "name": "Grok 4", "icon": "rocket", "desc": "Grok 4 terbaru - model terkuat dari xAI", "tags": ["chat", "reasoning", "code"]},
            "gemini2flash": {"category": "fast", "name": "Gemini 2 Flash", "icon": "zap", "desc": "Google Gemini 2 Flash - super cepat untuk tugas ringan", "tags": ["fast", "chat", "code"]},
            "r1": {"category": "thinking", "name": "DeepSeek R1", "icon": "lightbulb", "desc": "DeepSeek R1 - model reasoning open source terbaik", "tags": ["reasoning", "math", "code"]},
            "comet_max_assistant": {"category": "general", "name": "Comet Max", "icon": "brain", "desc": "Comet Max Assistant - asisten AI serbaguna", "tags": ["chat", "assistant"]},
            "o3_research": {"category": "research", "name": "o3 Research", "icon": "microscope", "desc": "o3 untuk riset mendalam - analisis paper, data, dan topik ilmiah", "tags": ["research", "science", "analysis"]},
            "o3pro_research": {"category": "research", "name": "o3 Pro Research", "icon": "microscope", "desc": "o3 Pro untuk riset tingkat lanjut", "tags": ["research", "expert", "science"]},
            "claude40sonnet_research": {"category": "research", "name": "Claude 4 Sonnet Research", "icon": "microscope", "desc": "Claude 4 Sonnet untuk riset dan analisis ilmiah", "tags": ["research", "analysis", "science"]},
            "claude40sonnetthinking_research": {"category": "research", "name": "Claude 4 Sonnet Thinking Research", "icon": "microscope", "desc": "Claude 4 Sonnet dengan thinking + research mode", "tags": ["research", "reasoning", "science"]},
            "claude40opus_research": {"category": "research", "name": "Claude 4 Opus Research", "icon": "microscope", "desc": "Claude 4 Opus untuk riset tingkat expert", "tags": ["research", "expert", "science"]},
            "claude40opusthinking_research": {"category": "research", "name": "Claude 4 Opus Thinking Research", "icon": "microscope", "desc": "Claude 4 Opus dengan deep thinking + research", "tags": ["research", "reasoning", "expert"]},
            "o3_labs": {"category": "labs", "name": "o3 Labs", "icon": "flask", "desc": "o3 Labs - model eksperimental untuk coding dan prototyping", "tags": ["code", "experimental", "prototype"]},
            "o3pro_labs": {"category": "labs", "name": "o3 Pro Labs", "icon": "flask", "desc": "o3 Pro Labs - coding dan prototyping tingkat lanjut", "tags": ["code", "experimental", "expert"]},
            "claude40sonnetthinking_labs": {"category": "labs", "name": "Claude 4 Sonnet Thinking Labs", "icon": "flask", "desc": "Claude 4 Labs - eksperimen coding dan AI", "tags": ["code", "experimental"]},
            "claude40opusthinking_labs": {"category": "labs", "name": "Claude 4 Opus Thinking Labs", "icon": "flask", "desc": "Claude 4 Opus Labs - eksperimen AI tingkat expert", "tags": ["code", "experimental", "expert"]},
            "gemma": {"category": "fast", "name": "Gemma", "icon": "zap", "desc": "Google Gemma - model ringan dan cepat", "tags": ["fast", "chat", "simple"]},
            "command-a-03-2025": {"category": "general", "name": "Command A", "icon": "brain", "desc": "Cohere Command A - model canggih untuk reasoning dan coding", "tags": ["chat", "code", "reasoning"]},
            "command-r-plus-08-2024": {"category": "general", "name": "Command R+", "icon": "brain", "desc": "Cohere Command R+ - model RAG terbaik", "tags": ["chat", "analysis"]},
            "command-r-08-2024": {"category": "fast", "name": "Command R", "icon": "zap", "desc": "Cohere Command R - respons cepat dan efisien", "tags": ["fast", "chat"]},
            "aria": {"category": "general", "name": "Opera Aria", "icon": "brain", "desc": "Opera Aria - asisten AI dari Opera Browser", "tags": ["chat", "assistant"]},
            "gpt-4o-mini": {"category": "fast", "name": "GPT-4o Mini", "icon": "zap", "desc": "GPT-4o Mini - model cepat dan hemat dari OpenAI", "tags": ["fast", "chat", "code"]},
            "qwen3-235b-a22b": {"category": "advanced", "name": "Qwen3 235B", "icon": "rocket", "desc": "Qwen3 235B - model besar dari Alibaba dengan reasoning kuat", "tags": ["reasoning", "code", "chat"]},
            "qwen3-max-preview": {"category": "advanced", "name": "Qwen3 Max Preview", "icon": "rocket", "desc": "Qwen3 Max Preview - model terbaru Alibaba", "tags": ["reasoning", "code"]},
            "qwen3-30b-a3b": {"category": "general", "name": "Qwen3 30B", "icon": "brain", "desc": "Qwen3 30B - keseimbangan kecepatan dan kualitas", "tags": ["chat", "code"]},
        }

    @property
    def category_info(self) -> Dict[str, Dict[str, str]]:
        return {
            "search": {"label": "Web Search", "color": "#2563eb", "icon": "search", "desc": "Cari info real-time dari internet"},
            "thinking": {"label": "Deep Thinking", "color": "#7c3aed", "icon": "lightbulb", "desc": "Reasoning mendalam & analisis logis"},
            "research": {"label": "Research", "color": "#059669", "icon": "microscope", "desc": "Riset ilmiah & analisis data"},
            "labs": {"label": "Labs / Code", "color": "#d97706", "icon": "flask", "desc": "Eksperimen coding & prototyping"},
            "advanced": {"label": "Advanced", "color": "#dc2626", "icon": "rocket", "desc": "Model AI terkuat & terbaru"},
            "general": {"label": "General", "color": "#034953", "icon": "brain", "desc": "Chat serbaguna untuk semua tugas"},
            "fast": {"label": "Fast", "color": "#0891b2", "icon": "zap", "desc": "Respons super cepat & ringan"},
        }

    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        caps = self.model_capabilities
        if model_name in caps:
            return caps[model_name]
        return {"category": "general", "name": model_name, "icon": "brain", "desc": "Model AI", "tags": ["chat"]}

    def get_category_system_prompt(self, model_name: str) -> str:
        info = self.get_model_info(model_name)
        cat = info.get("category", "general")
        base_prompt = "PENTING: Selalu jawab dalam Bahasa Indonesia, apapun bahasa yang digunakan user. Gunakan Bahasa Indonesia yang baik, jelas, dan mudah dipahami."
        prompts = {
            "search": base_prompt + " Kamu adalah AI search assistant. Berikan jawaban berdasarkan informasi terkini dari web. Sertakan sumber jika memungkinkan.",
            "thinking": base_prompt + " Kamu adalah AI reasoning expert. Pikirkan langkah demi langkah sebelum menjawab. Tunjukkan proses berpikirmu dengan jelas.",
            "research": base_prompt + " Kamu adalah AI research assistant. Analisis topik secara mendalam dan akademis. Berikan referensi dan data yang relevan.",
            "labs": base_prompt + " Kamu adalah AI coding assistant. Fokus pada kode yang bersih, efisien, dan best practice. Berikan contoh kode yang lengkap dan bisa langsung dijalankan.",
            "advanced": base_prompt + " Kamu adalah AI assistant tingkat expert. Berikan jawaban komprehensif dan mendalam.",
            "general": base_prompt + " Kamu adalah AI assistant yang ramah dan membantu. Berikan jawaban yang informatif dan bermanfaat.",
        }
        return prompts.get(cat, base_prompt)

# Global config instance
config = Config()
