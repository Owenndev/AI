"""
MAIA Agent OS — ComfyUI Client
Envía workflows a ComfyUI, monitorea via WebSocket y recupera outputs.

Flujo:
  1. POST /prompt con el workflow JSON → recibe prompt_id
  2. WebSocket ws://127.0.0.1:8188/ws → monitorea progreso en tiempo real
  3. GET /history/{prompt_id} → obtiene paths de imágenes generadas
  4. GET /view → descarga la imagen
"""

import json
import uuid
import asyncio
import httpx
import websockets
from pathlib import Path
from datetime import datetime
from typing import Optional, AsyncGenerator
from app.config import config


class ComfyUIClient:
    """
    Cliente para interactuar con ComfyUI via API.
    Maneja el ciclo completo de generación con monitoreo de progreso.
    """

    def __init__(self):
        self.base_url = config.COMFYUI_URL
        self.ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://")
        self.client_id = str(uuid.uuid4())
        self._http = httpx.AsyncClient(base_url=self.base_url, timeout=30.0)

    async def is_available(self) -> bool:
        """Verifica si ComfyUI está corriendo."""
        try:
            r = await self._http.get("/system_stats")
            return r.status_code == 200
        except Exception:
            return False

    async def queue_prompt(self, workflow: dict) -> str:
        """
        Encola un workflow en ComfyUI.
        Retorna el prompt_id para monitorear la ejecución.
        """
        payload = {
            "prompt": workflow,
            "client_id": self.client_id,
        }
        r = await self._http.post("/prompt", json=payload)
        r.raise_for_status()
        data = r.json()
        prompt_id = data.get("prompt_id")
        if not prompt_id:
            raise ValueError(f"ComfyUI no retornó prompt_id: {data}")
        return prompt_id

    async def generate(
        self,
        workflow: dict,
        on_progress=None,
        timeout_seconds: int = 300,
    ) -> dict:
        """
        Ejecuta un workflow completo y espera el resultado.

        Args:
            workflow: JSON del workflow
            on_progress: callback async(node_name, value, max) para progreso
            timeout_seconds: timeout máximo (default 5 minutos)

        Returns:
            {
                "success": bool,
                "images": [{"filename": str, "path": str, "url": str}],
                "prompt_id": str,
                "error": str | None
            }
        """
        if not await self.is_available():
            return {
                "success": False,
                "images": [],
                "error": "ComfyUI no está disponible en " + self.base_url,
            }

        # 1. Encolar el prompt
        try:
            prompt_id = await self.queue_prompt(workflow)
        except Exception as e:
            return {"success": False, "images": [], "error": f"Error al encolar: {e}"}

        # 2. Monitorear via WebSocket
        try:
            completed = await asyncio.wait_for(
                self._monitor_progress(prompt_id, on_progress),
                timeout=timeout_seconds,
            )
        except asyncio.TimeoutError:
            return {
                "success": False,
                "images": [],
                "prompt_id": prompt_id,
                "error": f"Timeout después de {timeout_seconds}s",
            }
        except Exception as e:
            # Fallback: polling si WebSocket falla
            completed = await self._poll_until_done(prompt_id, timeout_seconds)

        if not completed:
            return {"success": False, "images": [], "prompt_id": prompt_id, "error": "Generación fallida"}

        # 3. Obtener outputs
        images = await self._get_output_images(prompt_id)

        # 4. Descargar imágenes a carpeta de outputs de MAIA
        saved_images = await self._save_images(images)

        return {
            "success": True,
            "images": saved_images,
            "prompt_id": prompt_id,
            "error": None,
        }

    async def _monitor_progress(
        self, prompt_id: str, on_progress=None
    ) -> bool:
        """Monitorea la generación via WebSocket."""
        ws_endpoint = f"{self.ws_url}/ws?clientId={self.client_id}"

        async with websockets.connect(ws_endpoint) as ws:
            while True:
                try:
                    msg_raw = await asyncio.wait_for(ws.recv(), timeout=60)
                    msg = json.loads(msg_raw)
                except asyncio.TimeoutError:
                    continue
                except Exception:
                    break

                msg_type = msg.get("type")
                data = msg.get("data", {})

                # Progreso de un nodo
                if msg_type == "progress" and on_progress:
                    await on_progress(
                        node=data.get("node", ""),
                        value=data.get("value", 0),
                        max_val=data.get("max", 100),
                    )

                # Ejecución de nodo completada
                elif msg_type == "executed":
                    if data.get("prompt_id") == prompt_id:
                        # Verificar si hay outputs (imagen generada)
                        if data.get("output", {}).get("images"):
                            return True

                # Error de ejecución
                elif msg_type == "execution_error":
                    if data.get("prompt_id") == prompt_id:
                        raise RuntimeError(
                            f"Error en nodo '{data.get('node_type', 'unknown')}': "
                            f"{data.get('exception_message', 'unknown error')}"
                        )

                # Completado (cola vacía)
                elif msg_type == "status":
                    queue_remaining = (
                        data.get("status", {})
                            .get("exec_info", {})
                            .get("queue_remaining", 1)
                    )
                    if queue_remaining == 0:
                        # Verificar el history para confirmar
                        history = await self._get_history(prompt_id)
                        if history and history.get(prompt_id, {}).get("outputs"):
                            return True

        return False

    async def _poll_until_done(self, prompt_id: str, timeout: int) -> bool:
        """Fallback: polling del history cuando WebSocket no funciona."""
        import asyncio
        elapsed = 0
        while elapsed < timeout:
            await asyncio.sleep(3)
            elapsed += 3
            history = await self._get_history(prompt_id)
            if history and prompt_id in history:
                entry = history[prompt_id]
                if entry.get("outputs"):
                    return True
                if entry.get("status", {}).get("status_str") == "error":
                    return False
        return False

    async def _get_history(self, prompt_id: str) -> Optional[dict]:
        try:
            r = await self._http.get(f"/history/{prompt_id}")
            return r.json()
        except Exception:
            return None

    async def _get_output_images(self, prompt_id: str) -> list[dict]:
        """Extrae la lista de imágenes del history de ComfyUI."""
        history = await self._get_history(prompt_id)
        if not history or prompt_id not in history:
            return []

        images = []
        outputs = history[prompt_id].get("outputs", {})

        for node_id, node_output in outputs.items():
            for img in node_output.get("images", []):
                images.append({
                    "filename": img["filename"],
                    "subfolder": img.get("subfolder", ""),
                    "type": img.get("type", "output"),
                })

        return images

    async def _save_images(self, images: list[dict]) -> list[dict]:
        """Descarga imágenes de ComfyUI y las guarda en outputs de MAIA."""
        saved = []
        output_dir = Path(config.OUTPUTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        for i, img in enumerate(images):
            try:
                # Descargar imagen desde ComfyUI
                params = {
                    "filename": img["filename"],
                    "subfolder": img["subfolder"],
                    "type": img["type"],
                }
                r = await self._http.get("/view", params=params)
                r.raise_for_status()

                # Guardar con nombre descriptivo
                ext = Path(img["filename"]).suffix or ".png"
                save_name = f"maia_{timestamp}_{i}{ext}"
                save_path = output_dir / save_name
                save_path.write_bytes(r.content)

                saved.append({
                    "filename": save_name,
                    "path": str(save_path),
                    "url": f"http://127.0.0.1:{config.MAIA_PORT}/outputs/{save_name}",
                    "original_comfyui_name": img["filename"],
                })

            except Exception as e:
                saved.append({
                    "filename": img["filename"],
                    "error": str(e),
                    "original_comfyui_name": img["filename"],
                })

        return saved

    async def get_queue_status(self) -> dict:
        """Estado actual de la cola de ComfyUI."""
        try:
            r = await self._http.get("/queue")
            data = r.json()
            return {
                "running": len(data.get("queue_running", [])),
                "pending": len(data.get("queue_pending", [])),
            }
        except Exception as e:
            return {"error": str(e)}

    async def interrupt(self) -> bool:
        """Interrumpe la generación actual."""
        try:
            r = await self._http.post("/interrupt")
            return r.status_code == 200
        except Exception:
            return False

    async def close(self):
        await self._http.aclose()


# Instancia global
comfyui_client = ComfyUIClient()
