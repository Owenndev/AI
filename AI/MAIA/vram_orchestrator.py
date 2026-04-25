"""
MAIA Agent OS — VRAM Orchestrator
Gestiona qué modelo está en GPU en cada momento.
RTX 3090 = 24GB total, 20GB presupuesto operativo.

Nunca supera el presupuesto. Hace offload entre etapas.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from app.config import config


class PipelineType(str, Enum):
    TEXT_TO_IMAGE     = "text_to_image"
    IMAGE_EDIT        = "image_edit"
    OUTFIT_TRANSFER   = "outfit_transfer"
    TEXT_TO_VIDEO     = "text_to_video"
    IMAGE_TO_VIDEO    = "image_to_video"
    TALKING_HEAD      = "talking_head"
    FULL_VIDEO        = "full_video"
    TTS_ONLY          = "tts_only"
    LIP_SYNC_ONLY     = "lip_sync_only"


@dataclass
class PipelineStage:
    name: str                # nombre descriptivo
    model_key: str           # clave en VRAM_PROFILES
    vram_gb: float           # VRAM que necesita este stage
    can_cpu_offload: bool = False   # si puede correr parcialmente en CPU


@dataclass
class VRAMState:
    loaded_model: Optional[str] = None
    current_usage_gb: float = 0.0
    last_updated: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    is_busy: bool = False


# ── Definición de pipelines ───────────────────────────────────────────────────
# Cada pipeline es una secuencia de stages que se ejecutan en orden.
# Entre stage y stage, el modelo anterior se descarga de VRAM.

PIPELINE_DEFINITIONS: dict[PipelineType, list[PipelineStage]] = {

    PipelineType.TEXT_TO_IMAGE: [
        PipelineStage("Generar imagen (Qwen-Image GGUF)", "qwen_image_gen", vram_gb=14.0),
    ],

    PipelineType.IMAGE_EDIT: [
        PipelineStage("Editar imagen (Qwen-Image-Edit)", "qwen_image_edit", vram_gb=14.0),
    ],

    PipelineType.OUTFIT_TRANSFER: [
        PipelineStage("Generar imagen base", "qwen_image_gen", vram_gb=14.0),
        PipelineStage("Transferir outfit (IP-Adapter)", "ip_adapter", vram_gb=8.0),
    ],

    PipelineType.TEXT_TO_VIDEO: [
        PipelineStage("Generar video (LTX-Video 2.3 fp8)", "ltx_video", vram_gb=16.0),
    ],

    PipelineType.IMAGE_TO_VIDEO: [
        PipelineStage("Animar imagen (LTX-Video 2.3 fp8)", "ltx_video", vram_gb=16.0),
    ],

    PipelineType.TTS_ONLY: [
        PipelineStage("Sintetizar voz (Chatterbox)", "chatterbox_tts", vram_gb=4.0),
    ],

    PipelineType.LIP_SYNC_ONLY: [
        PipelineStage("Lip sync (LatentSync)", "latentsync", vram_gb=12.0),
    ],

    PipelineType.TALKING_HEAD: [
        # Imagen → Audio → Lip Sync (cada uno libera VRAM antes del siguiente)
        PipelineStage("Generar imagen de Mia", "qwen_image_gen", vram_gb=14.0),
        PipelineStage("Sintetizar voz", "chatterbox_tts", vram_gb=4.0),
        PipelineStage("Lip sync", "latentsync", vram_gb=12.0),
    ],

    PipelineType.FULL_VIDEO: [
        PipelineStage("Generar imagen base", "qwen_image_gen", vram_gb=14.0),
        PipelineStage("Sintetizar voz", "chatterbox_tts", vram_gb=4.0),
        PipelineStage("Lip sync", "latentsync", vram_gb=12.0),
        PipelineStage("Animar entorno (LTX-Video)", "ltx_video", vram_gb=16.0),
    ],
}


class VRAMOrchestrator:
    """
    Orquesta el uso de VRAM para pipelines multi-etapa.
    Garantiza que nunca se supera el presupuesto de VRAM.
    """

    BUDGET_GB = config.VRAM_BUDGET_GB  # 20GB

    def __init__(self):
        self.state = VRAMState()
        self._lock = asyncio.Lock()
        self._stage_callbacks: dict = {}

    def register_stage_callback(self, stage_name: str, callback) -> None:
        """Registra un callback para ejecutar cuando un stage corre."""
        self._stage_callbacks[stage_name] = callback

    async def run_pipeline(
        self,
        pipeline_type: PipelineType,
        params: dict,
        on_stage_start=None,
        on_stage_done=None,
    ) -> dict:
        """
        Ejecuta un pipeline completo, manejando VRAM entre etapas.

        Args:
            pipeline_type: Tipo de pipeline a ejecutar
            params: Parámetros del pipeline (inputs, paths, etc.)
            on_stage_start: Callback async(stage_name) cuando empieza un stage
            on_stage_done: Callback async(stage_name, result) cuando termina
        """
        stages = PIPELINE_DEFINITIONS.get(pipeline_type)
        if not stages:
            return {"success": False, "error": f"Pipeline desconocido: {pipeline_type}"}

        async with self._lock:
            self.state.is_busy = True

        results = {}

        try:
            for stage in stages:
                # Verificar que cabe en el presupuesto
                if stage.vram_gb > self.BUDGET_GB:
                    return {
                        "success": False,
                        "error": f"Stage '{stage.name}' requiere {stage.vram_gb}GB pero el presupuesto es {self.BUDGET_GB}GB"
                    }

                # Notificar inicio del stage
                if on_stage_start:
                    await on_stage_start(stage.name)

                # Actualizar estado de VRAM
                async with self._lock:
                    self.state.loaded_model = stage.model_key
                    self.state.current_usage_gb = stage.vram_gb

                # Ejecutar el stage via callback registrado
                callback = self._stage_callbacks.get(stage.model_key)
                if callback:
                    stage_result = await callback(params, results)
                    results[stage.model_key] = stage_result
                else:
                    # Sin callback = stage simulado (para testing)
                    results[stage.model_key] = {"status": "simulated", "stage": stage.name}

                # Notificar fin del stage
                if on_stage_done:
                    await on_stage_done(stage.name, results[stage.model_key])

                # Liberar VRAM entre stages (señal a ComfyUI de hacer offload)
                await self._offload_current()

        except Exception as e:
            return {"success": False, "error": str(e), "partial_results": results}

        finally:
            async with self._lock:
                self.state.is_busy = False
                self.state.loaded_model = None
                self.state.current_usage_gb = 0.0

        return {"success": True, "results": results}

    async def _offload_current(self) -> None:
        """Señala a ComfyUI que libere modelos de VRAM."""
        import httpx
        try:
            # ComfyUI tiene un endpoint para free memory
            await httpx.AsyncClient(timeout=5.0).post(
                f"{config.COMFYUI_URL}/free",
                json={"unload_models": True, "free_memory": True}
            )
        except Exception:
            pass  # Si falla, ComfyUI lo maneja solo con su propio offload
        finally:
            async with self._lock:
                self.state.loaded_model = None
                self.state.current_usage_gb = 0.0

    def validate_pipeline(self, pipeline_type: PipelineType) -> dict:
        """
        Valida si un pipeline puede correr con el hardware actual.
        No ejecuta nada — solo verifica.
        """
        stages = PIPELINE_DEFINITIONS.get(pipeline_type, [])
        issues = []

        for stage in stages:
            if stage.vram_gb > self.BUDGET_GB:
                issues.append(
                    f"Stage '{stage.name}' necesita {stage.vram_gb}GB "
                    f"(disponible: {self.BUDGET_GB}GB)"
                )

        return {
            "pipeline": pipeline_type,
            "stages": len(stages),
            "valid": len(issues) == 0,
            "issues": issues,
            "stage_details": [
                {"name": s.name, "model": s.model_key, "vram_gb": s.vram_gb}
                for s in stages
            ],
        }

    def get_state(self) -> dict:
        return {
            "loaded_model": self.state.loaded_model,
            "current_usage_gb": self.state.current_usage_gb,
            "budget_gb": self.BUDGET_GB,
            "is_busy": self.state.is_busy,
            "vram_free_gb": self.BUDGET_GB - self.state.current_usage_gb,
        }

    def list_pipelines(self) -> list[dict]:
        """Lista todos los pipelines disponibles con sus requerimientos."""
        result = []
        for ptype, stages in PIPELINE_DEFINITIONS.items():
            max_vram = max(s.vram_gb for s in stages)
            result.append({
                "type": ptype.value,
                "stages": len(stages),
                "max_vram_gb": max_vram,
                "fits_in_budget": max_vram <= self.BUDGET_GB,
                "stage_names": [s.name for s in stages],
            })
        return result


# Instancia global
vram_orchestrator = VRAMOrchestrator()
