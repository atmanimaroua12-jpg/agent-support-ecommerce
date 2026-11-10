"""Socle commun des outils : schéma des arguments, validation, délai d'attente, erreurs contrôlées."""
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """Erreur métier prévue (ex. commande introuvable), renvoyée proprement à l'agent."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class ToolContext:
    """Informations fournies par l'application, JAMAIS par le LLM.

    customer_id vient de la session du client connecté : le LLM ne peut pas
    choisir le client dont il consulte les commandes.
    """

    customer_id: str | None
    cfg: dict
    services: dict = field(default_factory=dict)  # index RAG, stockage des tickets...


@dataclass
class Tool:
    name: str
    description: str               # lue par le LLM pour choisir l'outil
    args_model: type[BaseModel]    # schéma des arguments (validation + description pour le LLM)
    handler: Callable[[Any, ToolContext], dict]
    permission: str                # documentation de la permission accordée


@dataclass
class ToolResult:
    status: str                    # ok | invalid_args | unknown_tool | error | timeout
    output: dict | None = None
    error_code: str | None = None
    error: str | None = None
    duration_ms: float = 0.0

    def for_llm(self) -> dict:
        """Ce que le LLM reçoit après l'appel."""
        if self.status == "ok":
            return {"status": "ok", **self.output}
        return {"status": self.status, "error_code": self.error_code, "message": self.error}


class ToolRegistry:
    def __init__(self, tools: list[Tool]):
        self.tools = {tool.name: tool for tool in tools}

    def schemas(self) -> list[dict]:
        """Description des outils au format attendu par l'API Groq (format OpenAI)."""
        out = []
        for tool in self.tools.values():
            params = tool.args_model.model_json_schema()
            params.pop("title", None)
            out.append({"type": "function",
                        "function": {"name": tool.name, "description": tool.description, "parameters": params}})
        return out

    def execute(self, name: str, raw_args: dict, ctx: ToolContext, timeout_s: float) -> ToolResult:
        start = time.perf_counter()
        result = self._execute(name, raw_args, ctx, timeout_s)
        result.duration_ms = round((time.perf_counter() - start) * 1000, 1)
        return result

    def _execute(self, name, raw_args, ctx, timeout_s) -> ToolResult:
        # 1. L'outil existe-t-il ?
        tool = self.tools.get(name)
        if tool is None:
            return ToolResult("unknown_tool", error_code="UNKNOWN_TOOL", error=f"outil inconnu : {name}")
        # 2. Les arguments respectent-ils le schéma ? (AVANT toute exécution)
        try:
            args = tool.args_model.model_validate(raw_args)
        except ValidationError as exc:
            details = "; ".join(f"{'.'.join(map(str, e['loc'])) or 'arguments'} : {e['msg']}" for e in exc.errors())
            return ToolResult("invalid_args", error_code="INVALID_ARGS", error=details)
        # 3. Exécution avec délai d'attente
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(tool.handler, args, ctx)
        try:
            return ToolResult("ok", output=future.result(timeout=timeout_s))
        except FutureTimeout:
            return ToolResult("timeout", error_code="TIMEOUT", error=f"{name} n'a pas répondu en {timeout_s} s")
        except ToolError as exc:
            return ToolResult("error", error_code=exc.code, error=exc.message)
        except Exception as exc:  # erreur imprévue : l'agent ne doit jamais planter
            return ToolResult("error", error_code="INTERNAL", error=f"erreur interne ({type(exc).__name__})")
        finally:
            pool.shutdown(wait=False, cancel_futures=True)