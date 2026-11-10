"""Stockage simulé des tickets : brouillons en attente, puis tickets confirmés par le client."""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path


class TicketStore:
    def __init__(self, pending_path: Path, tickets_path: Path):
        self.pending_path = Path(pending_path)
        self.tickets_path = Path(tickets_path)
        self.pending_path.parent.mkdir(parents=True, exist_ok=True)

    def _read_pending(self) -> dict:
        if not self.pending_path.exists():
            return {}
        return json.loads(self.pending_path.read_text(encoding="utf-8") or "{}")

    def _write_pending(self, drafts: dict) -> None:
        self.pending_path.write_text(json.dumps(drafts, ensure_ascii=False, indent=2), encoding="utf-8")

    def create_draft(self, customer_id: str, data: dict) -> dict:
        draft = {"draft_id": f"DRAFT-{uuid.uuid4().hex[:8].upper()}", "customer_id": customer_id,
                 "created_at": datetime.now(timezone.utc).isoformat(), **data}
        drafts = self._read_pending()
        drafts[draft["draft_id"]] = draft
        self._write_pending(drafts)
        return draft

    def confirm(self, draft_id: str, customer_id: str) -> dict:
        """Appelée par l'API quand le CLIENT clique sur « Confirmer ». Le LLM n'y a pas accès."""
        drafts = self._read_pending()
        draft = drafts.get(draft_id)
        if draft is None:
            raise KeyError("brouillon introuvable ou déjà traité")
        if draft["customer_id"] != customer_id:
            raise PermissionError("ce brouillon appartient à un autre client")
        del drafts[draft_id]
        self._write_pending(drafts)
        ticket = {**draft, "ticket_id": f"TCK-{uuid.uuid4().hex[:6].upper()}", "statut": "ouvert",
                  "confirmed_at": datetime.now(timezone.utc).isoformat()}
        with open(self.tickets_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ticket, ensure_ascii=False) + "\n")
        return ticket

    def cancel(self, draft_id: str, customer_id: str) -> None:
        drafts = self._read_pending()
        if draft_id in drafts and drafts[draft_id]["customer_id"] == customer_id:
            del drafts[draft_id]
            self._write_pending(drafts)

    def list_tickets(self) -> list[dict]:
        if not self.tickets_path.exists():
            return []
        lines = self.tickets_path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]