"""Format commun des réponses du LLM, quel que soit le fournisseur."""
from dataclasses import dataclass, field


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: str  # texte JSON produit par le modèle : il peut être invalide !


@dataclass
class LLMResponse:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


class LLMError(Exception):
    """Le LLM n'a pas pu répondre (panne, refus, délai dépassé)."""