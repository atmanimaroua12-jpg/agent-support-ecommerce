import pytest

from agent_support.config import load_config, project_path
from agent_support.rag import RagIndex, load_chunks
from agent_support.tools.base import ToolContext
from agent_support.tools.catalog import build_registry
from agent_support.tools.ticket_store import TicketStore


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def index(cfg):
    chunks, _ = load_chunks(project_path(cfg["paths"]["corpus"]))
    return RagIndex(chunks)


@pytest.fixture
def registry():
    return build_registry()


@pytest.fixture
def tickets(tmp_path):
    # Fichiers temporaires : les tests n'écrivent jamais dans les vrais fichiers
    return TicketStore(tmp_path / "pending.json", tmp_path / "tickets.jsonl")


def context(cfg, index, tickets, customer_id="C001"):
    return ToolContext(customer_id=customer_id, cfg=cfg, services={"index": index, "tickets": tickets})


def run(registry, cfg, ctx, name, args):
    return registry.execute(name, args, ctx, cfg["tools"]["timeout_s"])


# ------------------------------------------------------------------ registre
def test_trois_outils_avec_schemas(registry):
    noms = {s["function"]["name"] for s in registry.schemas()}
    assert noms == {"search_docs", "get_order_status", "create_ticket"}


def test_schema_simplifie_pour_le_llm(registry):
    ticket = next(s for s in registry.schemas() if s["function"]["name"] == "create_ticket")
    params = ticket["function"]["parameters"]
    assert "$defs" not in params and "anyOf" not in str(params)
    assert "colis_endommage" in params["properties"]["category"]["enum"]
    assert params["required"] == ["category", "summary"]


def test_outil_inconnu(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "rembourser_client", {})
    assert result.status == "unknown_tool"


# --------------------------------------------------------------- search_docs
def test_search_docs_cite_la_source(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "search_docs",
                 {"query": "délai pour retourner un article"})
    assert result.status == "ok"
    assert result.output["passages"][0]["doc_id"] == "POL-RET"


def test_search_docs_information_insuffisante(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "search_docs", {"query": "Livrez-vous en France ?"})
    assert result.output["resultat"] == "information_insuffisante"


def test_search_docs_argument_inconnu_refuse(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "search_docs", {"query": "retour", "admin": True})
    assert result.status == "invalid_args"


# ---------------------------------------------------------- get_order_status
def test_commande_du_client(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "get_order_status", {"order_id": "CMD-10002"})
    assert result.status == "ok"
    assert result.output["statut"] == "en cours de livraison"


@pytest.mark.parametrize("order_id", ["10002", "cmd-10002", "CMD-1", "CMD-10002; DROP TABLE"])
def test_format_de_commande_invalide(registry, cfg, index, tickets, order_id):
    result = run(registry, cfg, context(cfg, index, tickets), "get_order_status", {"order_id": order_id})
    assert result.status == "invalid_args"


def test_commande_d_un_autre_client_refusee(registry, cfg, index, tickets):
    # CMD-10003 appartient à C002 : C001 ne doit rien apprendre
    result = run(registry, cfg, context(cfg, index, tickets, "C001"), "get_order_status", {"order_id": "CMD-10003"})
    assert result.error_code == "ORDER_NOT_FOUND"
    inexistante = run(registry, cfg, context(cfg, index, tickets, "C001"), "get_order_status",
                      {"order_id": "CMD-99999"})
    assert result.error == inexistante.error.replace("CMD-99999", "CMD-10003")  # même message


def test_client_non_identifie(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets, None), "get_order_status", {"order_id": "CMD-10002"})
    assert result.error_code == "AUTH_REQUIRED"


def test_panne_simulee_timeout(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "get_order_status", {"order_id": "CMD-10009"})
    assert result.status == "timeout"
    assert result.duration_ms < (cfg["tools"]["timeout_s"] + 0.5) * 1000  # on n'a pas attendu la fin


# ------------------------------------------------------------- create_ticket
def test_ticket_cree_seulement_un_brouillon(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "create_ticket",
                 {"category": "colis_endommage", "summary": "Casque reçu avec l'arceau cassé",
                  "order_id": "CMD-10001", "priority": "haute"})
    assert result.output["resultat"] == "brouillon_en_attente_de_confirmation"
    assert tickets.list_tickets() == []  # aucun vrai ticket tant que le client n'a pas confirmé


def test_confirmation_par_le_client(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets), "create_ticket",
                 {"category": "garantie", "summary": "Le smartphone ne s'allume plus"})
    ticket = tickets.confirm(result.output["draft_id"], "C001")
    assert ticket["ticket_id"].startswith("TCK-")
    assert len(tickets.list_tickets()) == 1


def test_un_client_ne_confirme_pas_le_brouillon_d_un_autre(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets, "C001"), "create_ticket",
                 {"category": "autre", "summary": "Je veux parler à un conseiller"})
    with pytest.raises(PermissionError):
        tickets.confirm(result.output["draft_id"], "C002")


@pytest.mark.parametrize("args", [
    {"category": "rembourser_maintenant", "summary": "Remboursez-moi tout de suite"},  # catégorie inconnue
    {"category": "garantie", "summary": "panne"},                                     # résumé trop court
    {"summary": "Mon colis est en retard depuis une semaine"},                        # catégorie manquante
])
def test_ticket_arguments_invalides(registry, cfg, index, tickets, args):
    result = run(registry, cfg, context(cfg, index, tickets), "create_ticket", args)
    assert result.status == "invalid_args"


def test_ticket_sur_commande_d_un_autre_client(registry, cfg, index, tickets):
    result = run(registry, cfg, context(cfg, index, tickets, "C001"), "create_ticket",
                 {"category": "retard_livraison", "summary": "Commande toujours pas reçue",
                  "order_id": "CMD-10004"})
    assert result.error_code == "ORDER_NOT_FOUND"