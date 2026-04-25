"""
MAIA Agent OS — Approval Manager
Intercepta tools peligrosas y espera confirmación del usuario
antes de ejecutarlas.

Flujo:
  tool_call detectada
    → ¿requires_approval?
      → NO  → ejecuta directo
      → SÍ  → encola en pending_approvals
             → notifica al usuario via respuesta del chat
             → usuario responde "sí" / "no"
             → ejecuta o cancela
"""

import uuid
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum


class ApprovalStatus(str, Enum):
    PENDING  = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED  = "expired"


@dataclass
class PendingApproval:
    id: str
    tool_name: str
    tool_kwargs: dict
    task_id: str
    session_id: str
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved_at: Optional[str] = None

    def summary(self) -> str:
        """Texto legible para mostrarle al usuario."""
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in self.tool_kwargs.items())
        return f"`{self.tool_name}({args_str})`"


class ApprovalManager:
    """
    Gestor de aprobaciones pendientes.
    Almacenado en memoria (se pierde al reiniciar — intencional,
    no queremos aprobar acciones de sesiones anteriores).
    """

    def __init__(self):
        self._pending: dict[str, PendingApproval] = {}

    def request(
        self,
        tool_name: str,
        tool_kwargs: dict,
        task_id: str,
        session_id: str,
    ) -> PendingApproval:
        """Crea una solicitud de aprobación y la encola."""
        approval = PendingApproval(
            id=str(uuid.uuid4()),
            tool_name=tool_name,
            tool_kwargs=tool_kwargs,
            task_id=task_id,
            session_id=session_id,
        )
        self._pending[approval.id] = approval
        return approval

    def resolve(self, approval_id: str, approved: bool) -> Optional[PendingApproval]:
        """
        Resuelve una aprobación pendiente.
        Retorna el approval actualizado o None si no existe.
        """
        approval = self._pending.get(approval_id)
        if not approval:
            return None

        approval.status = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval.resolved_at = datetime.utcnow().isoformat()
        return approval

    def get(self, approval_id: str) -> Optional[PendingApproval]:
        return self._pending.get(approval_id)

    def pending_for_session(self, session_id: str) -> list[PendingApproval]:
        return [
            a for a in self._pending.values()
            if a.session_id == session_id and a.status == ApprovalStatus.PENDING
        ]

    def is_approved(self, approval_id: str) -> bool:
        a = self._pending.get(approval_id)
        return a is not None and a.status == ApprovalStatus.APPROVED

    def format_request_message(self, approval: PendingApproval) -> str:
        """Mensaje formateado para mostrarle al usuario."""
        emoji = "⚠️" if approval.tool_name == "delete_file" else "🔧"
        return (
            f"{emoji} **Acción que requiere aprobación**\n\n"
            f"La tool `{approval.tool_name}` quiere ejecutarse con:\n"
            f"```\n{self._format_kwargs(approval.tool_kwargs)}\n```\n\n"
            f"¿Aprobás? Respondé **sí** o **no**.\n"
            f"_(ID: `{approval.id}`)_"
        )

    def _format_kwargs(self, kwargs: dict) -> str:
        lines = []
        for k, v in kwargs.items():
            if isinstance(v, str) and len(v) > 100:
                v = v[:100] + "..."
            lines.append(f"  {k}: {repr(v)}")
        return "\n".join(lines)


# Instancia global
approval_manager = ApprovalManager()
