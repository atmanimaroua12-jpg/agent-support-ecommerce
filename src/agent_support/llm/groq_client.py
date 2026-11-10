"""Client Groq : appel du LLM avec la liste des outils (tool calling natif)."""
import os

import groq
from dotenv import load_dotenv

from .base import LLMError, LLMResponse, ToolCall


class GroqClient:
    def __init__(self, llm_cfg: dict):
        load_dotenv()
        if not os.getenv("GROQ_API_KEY"):
            raise LLMError("GROQ_API_KEY absente : vérifie ton fichier .env")
        self.cfg = llm_cfg
        self.model = llm_cfg["model"]
        # max_retries : le SDK réessaie automatiquement en cas d'erreur réseau ou de limite de débit
        self.client = groq.Groq(timeout=llm_cfg["timeout_s"], max_retries=2)

    def chat(self, messages: list[dict], tools: list[dict]) -> LLMResponse:
        extra = {}
        if self.cfg.get("reasoning_effort"):  # réglage propre aux modèles « à raisonnement » (gpt-oss)
            extra["reasoning_effort"] = self.cfg["reasoning_effort"]
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=tools, tool_choice="auto",
                temperature=self.cfg["temperature"], max_tokens=self.cfg["max_tokens"], extra_body=extra or None,
            )
        except groq.APIError as exc:
            raise LLMError(f"{type(exc).__name__} : {str(exc)[:200]}") from exc
        message = response.choices[0].message
        calls = [ToolCall(c.id, c.function.name, c.function.arguments or "{}") for c in (message.tool_calls or [])]
        return LLMResponse(content=message.content, tool_calls=calls,
                           input_tokens=response.usage.prompt_tokens, output_tokens=response.usage.completion_tokens)