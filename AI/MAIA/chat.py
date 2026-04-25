"""
MAIA Agent OS — Rutas FastAPI

Endpoints:
  POST /api/chat          → chat principal (streaming SSE)
  GET  /api/health        → estado del sistema
  GET  /api/memory/stats  → stats de la memoria
  GET  /api/memory/search → buscar en memoria
  GET  /api/tools         → tools disponibles
  POST /api/approve       → responder a solicitud de aprobación
"""

import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.agent_kernel import agent
from app.core.memory.sqlite_memory import memory
from app.core.approval_manager import approval_manager
from app.tools.base import tool_registry
from app.config import config

router = APIRouter(prefix="/api")


# ── Modelos Pydantic ─────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None


class ApproveRequest(BaseModel):
    approval_id: str
    approved: bool
    session_id: str


# ── Chat (streaming SSE) ─────────────────────────────────────────────────────

@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Endpoint principal de chat.
    Retorna streaming Server-Sent Events para respuesta en tiempo real.
    """
    session_id = request.session_id or str(uuid.uuid4())

    async def generate():
        # Header con session_id para que el cliente lo guarde
        yield f"data: {{\"type\":\"session\",\"session_id\":\"{session_id}\"}}\n\n"

        try:
            async for chunk in agent.chat(
                user_input=request.message,
                session_id=session_id,
            ):
                # Escapar comillas para JSON seguro
                escaped = chunk.replace('"', '\\"').replace('\n', '\\n')
                yield f"data: {{\"type\":\"text\",\"content\":\"{escaped}\"}}\n\n"

        except Exception as e:
            yield f"data: {{\"type\":\"error\",\"message\":\"{str(e)}\"}}\n\n"

        yield "data: {\"type\":\"done\"}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",       # importante para nginx
            "X-Session-ID": session_id,
        }
    )


# ── Health ───────────────────────────────────────────────────────────────────

@router.get("/health")
async def health():
    """Estado del sistema y conectividad con servicios."""
    import httpx

    services = {}

    # Verificar LLM
    try:
        r = await httpx.AsyncClient(timeout=3.0).get(
            f"{config.LLM_BASE_URL.replace('/v1', '')}/health"
        )
        services["llm"] = {"status": "ok", "url": config.LLM_BASE_URL}
    except Exception as e:
        services["llm"] = {"status": "offline", "error": str(e)}

    # Verificar ComfyUI
    try:
        r = await httpx.AsyncClient(timeout=3.0).get(
            f"{config.COMFYUI_URL}/system_stats"
        )
        services["comfyui"] = {"status": "ok", "url": config.COMFYUI_URL}
    except Exception as e:
        services["comfyui"] = {"status": "offline", "error": str(e)}

    all_ok = all(s["status"] == "ok" for s in services.values())

    return {
        "status": "ok" if all_ok else "degraded",
        "services": services,
        "hardware": {
            "gpu": config.GPU_NAME,
            "vram_total_gb": config.VRAM_TOTAL_GB,
            "vram_budget_gb": config.VRAM_BUDGET_GB,
        },
        "tools": tool_registry.all_names(),
    }


# ── Memoria ───────────────────────────────────────────────────────────────────

@router.get("/memory/stats")
async def memory_stats():
    """Estadísticas de la base de datos de memoria."""
    return memory.stats()


@router.get("/memory/search")
async def memory_search(q: str, type: str = "all", limit: int = 10):
    """Busca en la memoria de MAIA."""
    if not q:
        raise HTTPException(status_code=400, detail="Parámetro 'q' requerido")

    results = {}
    if type in ("all", "episodes"):
        results["episodes"] = memory.search_episodes(q, limit=limit)
    if type in ("all", "lessons"):
        results["lessons"] = memory.search_lessons(q, limit=limit)
    if type in ("all", "messages"):
        results["messages"] = memory.search_messages(q, limit=limit)

    return {"query": q, "results": results}


@router.get("/memory/history/{session_id}")
async def session_history(session_id: str, limit: int = 50):
    """Historial de una sesión específica."""
    return {
        "session_id": session_id,
        "messages": memory.get_history(session_id, limit=limit)
    }


# ── Tools ─────────────────────────────────────────────────────────────────────

@router.get("/tools")
async def list_tools():
    """Lista todas las tools disponibles con sus schemas."""
    tools = []
    for name in tool_registry.all_names():
        tool = tool_registry.get(name)
        tools.append({
            "name": tool.name,
            "description": tool.description,
            "requires_approval": tool.requires_approval,
            "schema": tool.schema(),
        })
    return {"tools": tools, "count": len(tools)}


# ── Aprobaciones ──────────────────────────────────────────────────────────────

@router.post("/approve")
async def approve_action(request: ApproveRequest):
    """Aprueba o rechaza una acción pendiente."""
    approval = approval_manager.resolve(request.approval_id, request.approved)

    if not approval:
        raise HTTPException(status_code=404, detail=f"Aprobación no encontrada: {request.approval_id}")

    return {
        "approval_id": request.approval_id,
        "status": approval.status,
        "tool_name": approval.tool_name,
        "resolved_at": approval.resolved_at,
    }


@router.get("/approvals/pending/{session_id}")
async def pending_approvals(session_id: str):
    """Lista aprobaciones pendientes para una sesión."""
    pending = approval_manager.pending_for_session(session_id)
    return {
        "session_id": session_id,
        "pending": [
            {
                "id": a.id,
                "tool_name": a.tool_name,
                "summary": a.summary(),
                "created_at": a.created_at,
            }
            for a in pending
        ]
    }
