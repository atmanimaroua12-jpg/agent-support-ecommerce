"""Orchestrateur : la boucle LLM <-> outils, avec ses limites et sa gestion des erreurs."""
import json
import time
from dataclasses import dataclass, field

from .config import project_path
from .llm.base import LLMError
from .prompts import get_system_prompt
from .rag import RagIndex, load_chunks
from .tools.base import ToolContext, ToolResult
from .tools.catalog import build_registry
from .tools.ticket_store import TicketStore

FALLBACK = ("Désolé, je n'ai pas pu traiter votre demande. "
            "Je peux transmettre votre question à un conseiller humain si vous le souhaitez.")


@dataclass
class ToolCallRecord:
    name: str
    args: dict | str
    status: str
    error_code: str | None
    duration_ms: float


@dataclass
class AgentResponse:
    answer: str
    outcome: str                   # completed | llm_error | max_iterations
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    draft: dict | None = None      # brouillon de ticket à faire confirmer par le client
    error: str | None = None       # message d'erreur du LLM, pour le diagnostic
    iterations: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0


class Agent:
    def __init__(self, cfg: dict, llm, registry, services: dict):
        self.cfg = cfg
        self.llm = llm
        self.registry = registry
        self.services = services
        self.system_prompt = get_system_prompt(cfg["agent"]["prompt_version"])

    def run(self, question: str, customer_id: str | None, history: list[dict] | None = None) -> AgentResponse:
        start = time.perf_counter()
        limits = self.cfg["agent"]
        ctx = ToolContext(customer_id=customer_id, cfg=self.cfg, services=self.services)
        messages = [{"role": "system", "content": self.system_prompt}, *(history or []),
                    {"role": "user", "content": question}]
        resp = AgentResponse(answer=FALLBACK, outcome="max_iterations")
        failed_calls = set()  # appels déjà échoués : on n'autorise pas le LLM à les répéter en boucle

        for _ in range(limits["max_iterations"]):
            resp.iterations += 1
            try:
                llm_resp = self.llm.chat(messages, self.registry.schemas())
            except LLMError as exc:
                resp.outcome = "llm_error"
                resp.error = str(exc)
                break
            resp.input_tokens += llm_resp.input_tokens
            resp.output_tokens += llm_resp.output_tokens

            if not llm_resp.tool_calls:  # pas d'outil demandé : c'est la réponse finale
                resp.answer = llm_resp.content or FALLBACK
                resp.outcome = "completed"
                break

            # On garde dans l'historique la demande d'outils du LLM (format attendu par l'API)
            messages.append({"role": "assistant", "content": llm_resp.content or "",
                             "tool_calls": [{"id": c.id, "type": "function",
                                             "function": {"name": c.name, "arguments": c.arguments}}
                                            for c in llm_resp.tool_calls]})
            for call in llm_resp.tool_calls:
                result, args = self._run_tool(call, ctx, len(resp.tool_calls), failed_calls)
                resp.tool_calls.append(ToolCallRecord(call.name, args, result.status, result.error_code,
                                                      result.duration_ms))
                if result.status == "ok" and call.name == "search_docs":
                    resp.sources += [f"{p['doc_id']}, {p['section']}" for p in result.output["passages"]]
                if result.status == "ok" and call.name == "create_ticket":
                    resp.draft = result.output
                # Le résultat est renvoyé au LLM comme une DONNÉE (rôle "tool")
                messages.append({"role": "tool", "tool_call_id": call.id,
                                 "content": json.dumps(result.for_llm(), ensure_ascii=False)})

        resp.latency_ms = round((time.perf_counter() - start) * 1000, 1)
        return resp

    def _run_tool(self, call, ctx, n_calls: int, failed_calls: set) -> tuple[ToolResult, dict | str]:
        if n_calls >= self.cfg["agent"]["max_tool_calls"]:
            return ToolResult("error", error_code="TOOL_LIMIT", error="trop d'appels d'outils"), call.arguments
        try:  # le LLM peut produire un JSON mal formé
            args = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            return ToolResult("invalid_args", error_code="INVALID_JSON", error="arguments JSON invalides"), \
                call.arguments
        key = (call.name, json.dumps(args, sort_keys=True))
        if key in failed_calls:
            return ToolResult("error", error_code="REPEATED_CALL", error="appel identique déjà échoué"), args
        result = self.registry.execute(call.name, args, ctx, self.cfg["tools"]["timeout_s"])
        if result.status != "ok":
            failed_calls.add(key)
        return result, args


def build_agent(cfg: dict, llm=None) -> Agent:
    """Assemble l'agent : index RAG, stockage des tickets, registre d'outils et LLM."""
    chunks, _ = load_chunks(project_path(cfg["paths"]["corpus"]))
    services = {
        "index": RagIndex(chunks),
        "tickets": TicketStore(project_path(cfg["paths"]["pending_tickets"]), project_path(cfg["paths"]["tickets"])),
    }
    if llm is None:
        from .llm.groq_client import GroqClient
        llm = GroqClient(cfg["llm"])
    return Agent(cfg, llm, build_registry(), services)