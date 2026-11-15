"""Central config. Env-first, sane dry-run defaults for demos."""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()  # before Settings defaults are evaluated


def _get(name: str, default: str = "") -> str:
    return os.getenv(name, default)


@dataclass(frozen=True)
class Settings:
    namespace: str = _get("K8S_NAMESPACE", "streaming-demo")
    dry_run: bool = _get("K8S_DRY_RUN", "false").lower() == "true"
    mcp_host: str = _get("MCP_HOST", "127.0.0.1")
    mcp_port: int = int(_get("MCP_PORT", "8000") or 8000)
    chroma_dir: str = _get("CHROMA_DIR", "data/chroma")
    rag_top_k: int = int(_get("RAG_TOP_K", "4") or 4)
    openai_model: str = _get("OPENAI_MODEL", "gpt-4o-mini")
    ollama_model: str = _get("OLLAMA_MODEL", "")
    ollama_base_url: str = _get("OLLAMA_BASE_URL", "http://localhost:11434")
    # Generic OpenAI-compatible endpoint (preferred when set; e.g. vLLM host)
    llm_base_url: str = _get("LLM_BASE_URL", "")
    llm_api_key: str = _get("LLM_API_KEY", "")
    llm_model_name: str = _get("LLM_MODEL_NAME", "")
    # Fallback provider (Groq free tier)
    groq_api_key: str = _get("GROQ_API_KEY", "")
    groq_model: str = _get("GROQ_MODEL", "openai/gpt-oss-20b")
    groq_base_url: str = _get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    abstain_threshold: float = float(_get("ABSTAIN_THRESHOLD", "0.55") or 0.55)
    require_approval: bool = _get("REQUIRE_APPROVAL", "true").lower() == "true"

    @property
    def llm_available(self) -> bool:
        return bool(os.getenv("OPENAI_API_KEY")) or bool(self.ollama_model) or bool(
            self.llm_api_key and self.llm_base_url and self.llm_model_name
        ) or bool(self.groq_api_key)

    @property
    def groq_available(self) -> bool:
        return bool(self.groq_api_key)

    def llm_endpoint(self) -> tuple[str, str, str]:
        """Return (base_url, api_key, model) for an OpenAI-compatible endpoint."""
        if self.llm_api_key and self.llm_base_url and self.llm_model_name:
            return self.llm_base_url, self.llm_api_key, self.llm_model_name
        return "https://api.openai.com/v1", os.getenv("OPENAI_API_KEY", ""), self.openai_model

    def groq_endpoint(self) -> tuple[str, str, str]:
        """Return (base_url, api_key, model) for Groq fallback."""
        return self.groq_base_url, self.groq_api_key, self.groq_model


SETTINGS = Settings()
