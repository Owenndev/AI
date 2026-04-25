"""
MAIA Agent OS — Rutas ComfyUI

Endpoints:
  GET  /api/comfyui/status       → estado y capacidades
  GET  /api/comfyui/inventory    → modelos instalados
  POST /api/comfyui/generate     → generar imagen (SSE streaming)
  GET  /api/comfyui/queue        → estado de la cola
  POST /api/comfyui/interrupt    → cancelar generación
  GET  /api/comfyui/pipelines    → pipelines disponibles con VRAM requerida
  POST /api/comfyui/inventory/refresh → forzar re-escaneo
"""

import uuid
import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.core.comfyui.client import comfyui_client
from app.core.comfyui.inventory_scanner import comfyui_inventory
from app.core.comfyui.workflow_builder import workflow_builder
from app.core.comfyui.vram_orchestrator import vram_orchestrator
from app.core.memory.sqlite_memory import memory

router = APIRouter(prefix="/api/comfyui")


class GenerateRequest(BaseModel):
    positive_prompt: str
    negative_prompt: str = ""
    format: str = "portrait"
    seed: int = -1
    use_lightning: bool = True
    session_id: Optional[str] = None


@router.get("/status")
async def comfyui_status():
    available = await comfyui_client.is_available()
    if not available:
        return {
            "online": False,
            "message": "ComfyUI no está corriendo. Ejecutá start_comfyui.bat",
            "url": comfyui_client.base_url,
        }
    inventory = await comfyui_inventory.get()
    queue = await comfyui_client.get_queue_status()
    vram_state = vram_orchestrator.get_state()
    return {
        "online": True,
        "url": comfyui_client.base_url,
        "queue": queue,
        "vram": vram_state,
        "capabilities": inventory.get("capabilities", {}),
        "models": {cat: items for cat, items in inventory.get("models", {}).items() if items},
        "nodes_count": inventory.get("nodes", {}).get("total", 0),
        "scanned_at": inventory.get("scanned_at"),
    }


@router.get("/inventory")
async def get_inventory(refresh: bool = False):
    return await comfyui_inventory.get(force_refresh=refresh)


@router.post("/generate")
async def generate_image(request: GenerateRequest):
    session_id = request.session_id or str(uuid.uuid4())

    async def stream():
        if not await comfyui_client.is_available():
            yield 'data: {"type":"error","message":"ComfyUI no disponible"}\n\n'
            return

        yield 'data: {"type":"status","message":"Construyendo workflow..."}\n\n'

        build_result = await workflow_builder.build_text_to_image(
            positive_prompt=request.positive_prompt,
            negative_prompt=request.negative_prompt,
            format=request.format,
            seed=request.seed,
            use_lightning=request.use_lightning,
        )

        if not build_result["ready"]:
            missing = build_result["missing_models"]
            hint = build_result.get("install_hint", "")[:300].replace('"', "'")
            yield f'data: {{"type":"error","message":"Faltan modelos: {missing}","hint":"{hint}"}}\n\n'
            return

        params = build_result["params_used"]
        w, h, steps = params["width"], params["height"], params["steps"]
        yield f'data: {{"type":"status","message":"Workflow listo: {w}x{h}, {steps} steps"}}\n\n'
        yield f'data: {{"type":"generating","message":"Generando en ComfyUI..."}}\n\n'

        result = await comfyui_client.generate(
            workflow=build_result["workflow"],
            timeout_seconds=300,
        )

        if not result["success"]:
            err = result.get("error", "Error desconocido").replace('"', "'")
            yield f'data: {{"type":"error","message":"{err}"}}\n\n'
            return

        images = result["images"]
        task_id = memory.create_task(session_id, f"generate_image: {request.positive_prompt[:80]}")
        memory.update_task(task_id, status="done", result=str([i.get("path") for i in images]))
        memory.save_episode(
            summary=f"Imagen generada: {request.positive_prompt[:80]}",
            outcome=f"{len(images)} imagen(s) guardada(s)",
            task_id=task_id,
            context={"format": request.format, "seed": params["seed"]},
        )

        yield f'data: {{"type":"done","images":{json.dumps(images)},"params":{json.dumps(params)}}}\n\n'

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/queue")
async def queue_status():
    if not await comfyui_client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI no disponible")
    return await comfyui_client.get_queue_status()


@router.post("/interrupt")
async def interrupt_generation():
    if not await comfyui_client.is_available():
        raise HTTPException(status_code=503, detail="ComfyUI no disponible")
    success = await comfyui_client.interrupt()
    return {"interrupted": success}


@router.get("/pipelines")
async def list_pipelines():
    pipelines = vram_orchestrator.list_pipelines()
    return {
        "hardware": {"gpu": "RTX 3090", "vram_budget_gb": vram_orchestrator.BUDGET_GB},
        "pipelines": pipelines,
        "ready_count": sum(1 for p in pipelines if p["fits_in_budget"]),
    }


@router.post("/inventory/refresh")
async def refresh_inventory():
    inventory = await comfyui_inventory.get(force_refresh=True)
    return {
        "refreshed": True,
        "scanned_at": inventory.get("scanned_at"),
        "nodes_count": inventory.get("nodes", {}).get("total", 0),
        "models_found": {
            cat: len(items)
            for cat, items in inventory.get("models", {}).items() if items
        },
    }
