"""Garde-fous : contrôle de la question avant le LLM, et masquage des données personnelles dans les traces."""
import re
import unicodedata
from dataclasses import dataclass

# Motifs typiques d'injection de prompt (FR / EN). Liste simple et explicable, pas exhaustive :
# c'est une limite documentée, complétée par les règles de sécurité du prompt système.
INJECTION_PATTERNS = [
    r"ignore[sz]?\s+(toutes?\s+)?(les\s+|tes\s+|vos\s+|ces\s+)?(instructions|consignes|r[eè]gles)",
    r"oublie[sz]?\s+(toutes?\s+)?(les\s+|tes\s+|vos\s+)?(instructions|consignes|r[eè]gles)",
    r"ignore\s+(all\s+|any\s+|the\s+)?(previous|prior|above)\s+(instructions|rules)",
    r"(r[eé]v[eè]le|affiche|montre|donne|r[eé]p[eè]te)[sz]?(\s*-?\s*moi)?\s+(ton|le|votre|tes|vos)\s+"
    r"(prompt|instructions|consignes)",
    r"system\s*prompt|prompt\s+syst[eè]me",
    r"tu\s+es\s+(maintenant|d[eé]sormais)\s",
    r"you\s+are\s+now\s",
    r"mode\s+(d[eé]veloppeur|admin|administrateur|debug)|developer\s+mode|jailbreak",
    r"(je\s+suis|en\s+tant\s+qu['’e])\s*(l['’e]\s*|un\s+)?(admin|administrateur|d[eé]veloppeur)\b",
    r"<\s*/?\s*(system|instruction|admin)\s*>",
]
_INJECTION = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
PHONE_RE = re.compile(r"(?:\+212|0)[\s.-]?[5-7](?:[\s.-]?\d{2}){4}")
CARD_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
RIB_RE = re.compile(r"\b\d{24}\b")

REFUSALS = {
    "empty": "Votre message est vide. Comment puis-je vous aider ?",
    "too_long": "Votre message est trop long. Pouvez-vous résumer votre demande en quelques phrases ?",
    "prompt_injection": "Je ne peux pas traiter cette demande. Je peux vous aider pour vos commandes, les retours, "
                        "la livraison, la garantie ou le paiement.",
    "sensitive_data": "Pour votre sécurité, ne communiquez jamais de numéro de carte ni de RIB dans le chat. "
                      "ShopNova ne vous les demandera jamais. Comment puis-je vous aider ?",
}


@dataclass
class InputCheck:
    allowed: bool
    reason: str | None = None   # empty | too_long | prompt_injection | sensitive_data


def detect_injection(text: str) -> bool:
    text = re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text))
    return any(pattern.search(text) for pattern in _INJECTION)


def check_input(text: str, max_chars: int) -> InputCheck:
    """Contrôle la question AVANT de l'envoyer au LLM (aucun coût si elle est bloquée)."""
    if not text or not text.strip():
        return InputCheck(False, "empty")
    if len(text) > max_chars:
        return InputCheck(False, "too_long")
    if detect_injection(text):
        return InputCheck(False, "prompt_injection")
    if CARD_RE.search(text) or RIB_RE.search(text):
        return InputCheck(False, "sensitive_data")
    return InputCheck(True)


def mask_pii(text: str) -> str:
    """Masque e-mails, téléphones, numéros de carte et RIB avant écriture dans les traces."""
    text = RIB_RE.sub("[RIB]", text)
    text = CARD_RE.sub("[CARTE]", text)
    text = EMAIL_RE.sub(lambda m: m.group(0)[0] + "***@" + m.group(0).split("@")[1], text)
    return PHONE_RE.sub("[TEL]", text)


def mask_obj(obj):
    """Applique mask_pii à toutes les chaînes d'un dict / d'une liste."""
    if isinstance(obj, str):
        return mask_pii(obj)
    if isinstance(obj, dict):
        return {key: mask_obj(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [mask_obj(value) for value in obj]
    return obj
