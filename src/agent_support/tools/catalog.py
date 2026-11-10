"""Les trois outils de l'agent.

| Outil            | Permission                       | Effet                               |
|------------------|----------------------------------|-------------------------------------|
| search_docs      | lecture, corpus public           | aucun                               |
| get_order_status | lecture, commandes du client     | aucun                               |
| create_ticket    | écriture soumise à confirmation  | brouillon ; ticket après « oui »    |
"""
import json
import time
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field

from ..config import project_path
from .base import Tool, ToolContext, ToolError, ToolRegistry

ORDER_ID_PATTERN = r"^CMD-\d{5}$"


def load_orders(cfg: dict) -> dict:
    return json.loads(project_path(cfg["paths"]["orders"]).read_text(encoding="utf-8"))


def find_customer_order(cfg: dict, order_id: str, customer_id: str | None) -> dict:
    """Renvoie la commande seulement si elle appartient au client connecté."""
    if not customer_id:
        raise ToolError("AUTH_REQUIRED", "le client doit être identifié pour consulter une commande")
    order = load_orders(cfg)["orders"].get(order_id)
    # Même message si la commande n'existe pas OU appartient à un autre client : on ne révèle rien.
    if order is None or order["customer_id"] != customer_id:
        raise ToolError("ORDER_NOT_FOUND", f"aucune commande {order_id} trouvée pour ce client")
    return order


# ------------------------------------------------------------------ search_docs
class SearchDocsArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")  # tout argument inconnu est refusé
    query: str = Field(min_length=3, max_length=300,
                       description="Question reformulée avec le vocabulaire de la documentation")


def search_docs(args: SearchDocsArgs, ctx: ToolContext) -> dict:
    params = ctx.cfg["tools"]["search_docs"]
    hits = ctx.services["index"].search(args.query, top_k=params["top_k"], min_score=params["min_score"])
    if not hits:
        return {"resultat": "information_insuffisante", "passages": [],
                "consigne": "Aucun passage pertinent : ne rien inventer, proposer une escalade."}
    return {"resultat": "passages_trouves",
            "passages": [{"doc_id": c.doc_id, "section": c.section, "version": c.version,
                          "score": round(score, 3), "texte": c.text} for c, score in hits]}


# ------------------------------------------------------------- get_order_status
class OrderArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: str = Field(pattern=ORDER_ID_PATTERN, description="Numéro de commande au format CMD-12345")


def get_order_status(args: OrderArgs, ctx: ToolContext) -> dict:
    order = find_customer_order(ctx.cfg, args.order_id, ctx.customer_id)
    if order.get("_simulate") == "timeout":  # panne simulée du système de commandes
        time.sleep(ctx.cfg["tools"]["timeout_s"] + 1)
    return {"order_id": args.order_id, "statut": order["status"], "date_commande": order["order_date"],
            "livraison_estimee": order["estimated_delivery"], "livree_le": order["delivered_on"],
            "transporteur": order["carrier"], "numero_suivi": order["tracking"], "total_mad": order["total_mad"],
            "articles": [{"nom": i["name"], "quantite": i["qty"]} for i in order["items"]]}


# ---------------------------------------------------------------- create_ticket
class Category(str, Enum):
    retard_livraison = "retard_livraison"
    colis_endommage = "colis_endommage"
    garantie = "garantie"
    remboursement = "remboursement"
    litige = "litige"
    autre = "autre"


class Priority(str, Enum):
    basse = "basse"
    normale = "normale"
    haute = "haute"


class TicketArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    category: Category = Field(description="Catégorie de la demande")
    summary: str = Field(min_length=10, max_length=500, description="Résumé factuel du problème du client")
    order_id: str | None = Field(default=None, pattern=ORDER_ID_PATTERN, description="Commande concernée, si connue")
    priority: Priority = Field(default=Priority.normale, description="Priorité proposée")


def create_ticket(args: TicketArgs, ctx: ToolContext) -> dict:
    if args.order_id:  # on n'accepte pas un ticket sur la commande d'un autre client
        find_customer_order(ctx.cfg, args.order_id, ctx.customer_id)
    draft = ctx.services["tickets"].create_draft(ctx.customer_id, {
        "category": args.category.value, "summary": args.summary,
        "order_id": args.order_id, "priority": args.priority.value})
    # L'outil NE crée PAS le ticket : il prépare un brouillon que le client doit confirmer.
    return {"resultat": "brouillon_en_attente_de_confirmation", "draft_id": draft["draft_id"],
            "category": draft["category"],
            "consigne": "Demander au client de confirmer. Ne pas dire que le ticket est créé."}


def build_registry() -> ToolRegistry:
    return ToolRegistry([
        Tool("search_docs",
             "Cherche dans la documentation officielle de ShopNova (retours, remboursements, livraison, "
             "garantie, paiement, factures). À utiliser pour toute question sur une règle ou une procédure.",
             SearchDocsArgs, search_docs, permission="lecture"),
        Tool("get_order_status",
             "Donne le statut, les articles, le transporteur et la date de livraison d'une commande du client "
             "connecté. Nécessite un numéro au format CMD-12345.",
             OrderArgs, get_order_status, permission="lecture limitée au client connecté"),
        Tool("create_ticket",
             "Prépare une demande pour un conseiller humain (retard, colis endommagé, garantie, litige, "
             "remboursement non reçu). Crée seulement un BROUILLON que le client doit confirmer.",
             TicketArgs, create_ticket, permission="écriture soumise à confirmation humaine"),
    ])