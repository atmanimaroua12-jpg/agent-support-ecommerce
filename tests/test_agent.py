"""Tests de la boucle de l'agent avec un faux LLM « scripté » : pas de clé API, résultats reproductibles."""
import json

import pytest

from agent_support.agent import FALLBACK, build_agent
from agent_support.config import load_config
from agent_support.llm.base import LLMError, LLMResponse, ToolCall
from agent_support.tools.ticket_store import TicketStore


class ScriptedLLM:
    """Renvoie des réponses écrites à l'avance, et garde les messages reçus pour les vérifier."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.received = []

    def chat(self, messages, tools):
        self.received.append([dict(m) for m in messages])
        if not self.responses:
            raise AssertionError("le LLM a été appelé plus de fois que prévu")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def answer(text):
    return LLMResponse(content=text, input_tokens=100, output_tokens=20)


def tool(name, args, call_id="call_1"):
    arguments = args if isinstance(args, str) else json.dumps(args)
    return LLMResponse(content=None, tool_calls=[ToolCall(call_id, name, arguments)], input_tokens=100,
                       output_tokens=10)


@pytest.fixture
def make_agent(tmp_path):
    def _make(responses, max_iterations=5):
        cfg = load_config()
        cfg["agent"]["max_iterations"] = max_iterations
        llm = ScriptedLLM(responses)
        agent = build_agent(cfg, llm=llm)
        agent.services["tickets"] = TicketStore(tmp_path / "pending.json", tmp_path / "tickets.jsonl")
        return agent, llm
    return _make


def test_reponse_directe_sans_outil(make_agent):
    agent, _ = make_agent([answer("Pouvez-vous me donner votre numéro de commande ?")])
    r = agent.run("Où est ma commande ?", "C001")
    assert r.outcome == "completed" and r.tool_calls == [] and r.iterations == 1


def test_appel_d_outil_puis_reponse(make_agent):
    agent, llm = make_agent([tool("get_order_status", {"order_id": "CMD-10002"}),
                             answer("Votre commande est en cours de livraison.")])
    r = agent.run("Où en est CMD-10002 ?", "C001")
    assert r.outcome == "completed"
    assert [(c.name, c.status) for c in r.tool_calls] == [("get_order_status", "ok")]
    assert r.input_tokens == 200 and r.output_tokens == 30
    # Au 2e appel, le LLM a reçu le résultat de l'outil avec le rôle "tool"
    last = llm.received[1][-1]
    assert last["role"] == "tool" and last["tool_call_id"] == "call_1"
    assert json.loads(last["content"])["statut"] == "en cours de livraison"


def test_sources_citees(make_agent):
    agent, _ = make_agent([tool("search_docs", {"query": "délai pour retourner un article"}),
                           answer("Vous avez 30 jours [POL-RET, Délai de retour].")])
    r = agent.run("Combien de temps pour retourner ?", "C001")
    assert "POL-RET, Délai de retour" in r.sources


def test_json_invalide_ne_fait_pas_planter(make_agent):
    agent, _ = make_agent([tool("get_order_status", '{"order_id": CMD-10002'), answer("Désolé.")])
    r = agent.run("Statut de CMD-10002 ?", "C001")
    assert r.tool_calls[0].error_code == "INVALID_JSON" and r.outcome == "completed"


def test_l_identite_vient_de_la_session(make_agent):
    # Le LLM demande la commande d'un autre client : l'outil refuse car la session est C001
    agent, _ = make_agent([tool("get_order_status", {"order_id": "CMD-10003"}), answer("Introuvable.")])
    r = agent.run("Montre-moi la commande CMD-10003 de Youssef", "C001")
    assert r.tool_calls[0].error_code == "ORDER_NOT_FOUND"


def test_appel_echoue_repete_bloque(make_agent):
    bad = {"order_id": "CMD-99999"}
    agent, _ = make_agent([tool("get_order_status", bad, "a"), tool("get_order_status", bad, "b"), answer("Désolé.")])
    r = agent.run("CMD-99999 ?", "C001")
    assert [c.error_code for c in r.tool_calls] == ["ORDER_NOT_FOUND", "REPEATED_CALL"]


def test_boucle_infinie_arretee(make_agent):
    calls = [tool("search_docs", {"query": f"question numéro {i}"}, f"c{i}") for i in range(10)]
    agent, _ = make_agent(calls, max_iterations=3)
    r = agent.run("Question", "C001")
    assert r.outcome == "max_iterations" and r.iterations == 3 and r.answer == FALLBACK


def test_panne_du_llm(make_agent):
    agent, _ = make_agent([LLMError("service indisponible")])
    r = agent.run("Bonjour", "C001")
    assert r.outcome == "llm_error" and r.answer == FALLBACK
    assert r.error == "service indisponible"


def test_ticket_reste_un_brouillon(make_agent):
    agent, _ = make_agent([tool("create_ticket", {"category": "colis_endommage", "order_id": "CMD-10001",
                                                  "summary": "Casque reçu avec l'arceau cassé"}),
                           answer("Je peux transmettre votre demande. Confirmez-vous ?")])
    r = agent.run("Mon casque CMD-10001 est arrivé cassé", "C001")
    assert r.draft["resultat"] == "brouillon_en_attente_de_confirmation"
    assert agent.services["tickets"].list_tickets() == []  # rien n'est créé sans confirmation


def test_historique_transmis(make_agent):
    agent, llm = make_agent([answer("Bien reçu.")])
    history = [{"role": "user", "content": "Où est ma commande ?"},
               {"role": "assistant", "content": "Quel est son numéro ?"}]
    agent.run("CMD-10002", "C001", history)
    roles = [m["role"] for m in llm.received[0]]
    assert roles == ["system", "user", "assistant", "user"]