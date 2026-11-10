"""API du service : /health, /chat, confirmation des tickets, /metrics.

Lancement :  uvicorn app.api:app --reload
Documentation interactive : http://127.0.0.1:8000/docs
"""
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from fastapi import Depends, FastAPI, Header, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))  # rend agent_support importable

from agent_support.agent import Agent, build_agent  # noqa: E402
from agent_support.config import load_config, project_path  # noqa: E402
from agent_support.monitoring import compute_metrics  # noqa: E402
from agent_support.tracing import read_traces  # noqa: E402


# ------------------------------------------------------------------ schémas des requêtes / réponses
class Message(BaseModel):
    role: Literal["user", "assistant"]   # le client ne peut PAS envoyer de message "system"
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str
    history: list[Message] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    request_id: str
    answer: str
    outcome: str
    sources: list[str]
    tools_used: list[str]
    draft_id: str | None = None
    draft_category: str | None = None
    latency_ms: float


# ------------------------------------------------------------------ application
def create_app(agent: Agent | None = None) -> FastAPI:
    """Fabrique de l'application. Les tests lui passent un agent avec un faux LLM."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.agent = agent or build_agent(load_config())   # construit une seule fois au démarrage
        yield

    app = FastAPI(title="Agent de support ShopNova", version="1.0", lifespan=lifespan)

    def get_agent() -> Agent:
        return app.state.agent

    def current_customer(x_customer_id: str = Header(...)) -> str:
        """Session SIMULÉE : l'identité vient d'un en-tête HTTP, jamais du texte envoyé au LLM.
        En production, on la lirait dans un jeton d'authentification (JWT, cookie de session...)."""
        orders = json.loads(project_path(get_agent().cfg["paths"]["orders"]).read_text(encoding="utf-8"))
        if x_customer_id not in orders["customers"]:
            raise HTTPException(status_code=401, detail="client inconnu")
        return x_customer_id

    @app.get("/health")
    def health(agent: Agent = Depends(get_agent)):
        return {"status": "ok", "agent_version": agent.version.get("agent_version"),
                "model": agent.cfg["llm"]["model"], "prompt_version": agent.cfg["agent"]["prompt_version"],
                "indexed_chunks": len(agent.services["index"].chunks)}

    @app.post("/chat", response_model=ChatResponse)
    def chat(req: ChatRequest, customer_id: str = Depends(current_customer), agent: Agent = Depends(get_agent)):
        r = agent.run(req.message, customer_id, [m.model_dump() for m in req.history])
        return ChatResponse(request_id=r.request_id, answer=r.answer, outcome=r.outcome, sources=r.sources,
                            tools_used=[c.name for c in r.tool_calls],
                            draft_id=r.draft["draft_id"] if r.draft else None,
                            draft_category=r.draft["category"] if r.draft else None, latency_ms=r.latency_ms)

    @app.post("/tickets/{draft_id}/confirm")
    def confirm_ticket(draft_id: str, customer_id: str = Depends(current_customer),
                       agent: Agent = Depends(get_agent)):
        """Action HUMAINE : déclenchée par le bouton du client, jamais par le LLM."""
        try:
            ticket = agent.services["tickets"].confirm(draft_id, customer_id)
        except KeyError:
            raise HTTPException(status_code=404, detail="brouillon introuvable ou déjà traité") from None
        except PermissionError:
            raise HTTPException(status_code=403, detail="ce brouillon appartient à un autre client") from None
        return {"ticket_id": ticket["ticket_id"], "statut": ticket["statut"], "category": ticket["category"]}

    @app.post("/tickets/{draft_id}/cancel")
    def cancel_ticket(draft_id: str, customer_id: str = Depends(current_customer),
                      agent: Agent = Depends(get_agent)):
        agent.services["tickets"].cancel(draft_id, customer_id)
        return {"draft_id": draft_id, "statut": "annulé"}

    @app.get("/metrics")
    def metrics(agent: Agent = Depends(get_agent)):
        traces = read_traces(agent.tracer.path) if agent.tracer.path else []
        return {**compute_metrics(traces), "tickets_confirmed": len(agent.services["tickets"].list_tickets())}

    return app


app = create_app()