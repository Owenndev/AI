"""
MAIA Agent OS — ComfyUI Workflow Builder
Construye workflows JSON válidos dinámicamente basándose en:
  - Modelos disponibles en tu instalación real
  - Hardware disponible (RTX 3090, 20GB budget)
  - Tipo de generación requerida

NUNCA hardcodea nombres de modelos. Siempre usa lo que está instalado.
"""

import json
from typing import Optional
from app.core.comfyui.inventory_scanner import comfyui_inventory


# ── Templates base de workflows ───────────────────────────────────────────────
# Cada template es un workflow mínimo funcional.
# Los nombres de modelos se reemplazan dinámicamente con lo que está instalado.

WORKFLOW_TEXT_TO_IMAGE_GGUF = {
    "1": {
        "class_type": "UnetLoaderGGUF",
        "inputs": {"unet_name": "__DIFFUSION_MODEL__"},
    },
    "2": {
        "class_type": "CLIPLoaderGGUF",
        "inputs": {
            "clip_name1": "__TEXT_ENCODER_1__",
            "clip_name2": "__TEXT_ENCODER_2__",
            "type": "wan"
        },
    },
    "3": {
        "class_type": "VAELoader",
        "inputs": {"vae_name": "__VAE__"},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "__POSITIVE_PROMPT__",
            "clip": ["2", 0]
        },
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {
            "text": "__NEGATIVE_PROMPT__",
            "clip": ["2", 0]
        },
    },
    "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {
            "width": "__WIDTH__",
            "height": "__HEIGHT__",
            "batch_size": 1
        },
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": "__SEED__",
            "steps": "__STEPS__",
            "cfg": "__CFG__",
            "sampler_name": "__SAMPLER__",
            "scheduler": "__SCHEDULER__",
            "denoise": 1.0
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {
            "samples": ["7", 0],
            "vae": ["3", 0]
        },
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {
            "images": ["8", 0],
            "filename_prefix": "__FILENAME_PREFIX__"
        },
    },
}

WORKFLOW_TEXT_TO_IMAGE_CHECKPOINT = {
    "1": {
        "class_type": "CheckpointLoaderSimple",
        "inputs": {"ckpt_name": "__CHECKPOINT__"},
    },
    "4": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__POSITIVE_PROMPT__", "clip": ["1", 1]},
    },
    "5": {
        "class_type": "CLIPTextEncode",
        "inputs": {"text": "__NEGATIVE_PROMPT__", "clip": ["1", 1]},
    },
    "6": {
        "class_type": "EmptyLatentImage",
        "inputs": {"width": "__WIDTH__", "height": "__HEIGHT__", "batch_size": 1},
    },
    "7": {
        "class_type": "KSampler",
        "inputs": {
            "model": ["1", 0],
            "positive": ["4", 0],
            "negative": ["5", 0],
            "latent_image": ["6", 0],
            "seed": "__SEED__",
            "steps": "__STEPS__",
            "cfg": "__CFG__",
            "sampler_name": "__SAMPLER__",
            "scheduler": "__SCHEDULER__",
            "denoise": 1.0
        },
    },
    "8": {
        "class_type": "VAEDecode",
        "inputs": {"samples": ["7", 0], "vae": ["1", 2]},
    },
    "9": {
        "class_type": "SaveImage",
        "inputs": {"images": ["8", 0], "filename_prefix": "__FILENAME_PREFIX__"},
    },
}


class WorkflowBuilder:
    """
    Construye workflows JSON válidos para ComfyUI.
    Usa el inventario real para seleccionar los modelos correctos.
    """

    # Parámetros óptimos para RTX 3090 por tipo de tarea
    OPTIMAL_PARAMS = {
        "text_to_image": {
            "steps": 20,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "width": 1024,
            "height": 1024,
        },
        "text_to_image_lightning": {
            # Con Lightning LoRA: 4 steps son suficientes
            "steps": 4,
            "cfg": 1.0,
            "sampler": "euler",
            "scheduler": "sgm_uniform",
            "width": 1024,
            "height": 1024,
        },
        "portrait": {
            "steps": 25,
            "cfg": 7.5,
            "sampler": "dpmpp_2m",
            "scheduler": "karras",
            "width": 832,
            "height": 1216,  # ratio 2:3 vertical, ideal para retratos
        },
        "landscape": {
            "steps": 20,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "width": 1216,
            "height": 832,  # ratio 3:2 horizontal
        },
        "square_post": {
            "steps": 20,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "width": 1024,
            "height": 1024,  # cuadrado para feed
        },
        "story_vertical": {
            "steps": 20,
            "cfg": 7.0,
            "sampler": "euler",
            "scheduler": "normal",
            "width": 576,
            "height": 1024,  # ratio 9:16 para stories/reels
        },
    }

    async def build_text_to_image(
        self,
        positive_prompt: str,
        negative_prompt: str = "",
        format: str = "portrait",
        seed: int = -1,
        filename_prefix: str = "maia_output",
        use_lightning: bool = True,
    ) -> dict:
        """
        Construye un workflow de text-to-image usando los modelos disponibles.

        Returns:
            {
                "workflow": {...},      # JSON para enviar a ComfyUI
                "ready": bool,
                "missing_models": [],  # si no está listo
                "params_used": {}      # parámetros que se usaron
            }
        """
        import random
        inventory = await comfyui_inventory.get()

        # Elegir template según lo que está disponible
        workflow, model_assignments, missing = self._select_template_and_models(
            inventory, use_lightning
        )

        if missing:
            return {
                "workflow": None,
                "ready": False,
                "missing_models": missing,
                "install_hint": self._generate_install_hint(missing),
            }

        # Elegir parámetros según formato
        param_key = "text_to_image_lightning" if (
            use_lightning and "lightning_lora" in model_assignments
        ) else format
        params = self.OPTIMAL_PARAMS.get(param_key, self.OPTIMAL_PARAMS["portrait"])

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        # Reemplazar placeholders en el template
        workflow_str = json.dumps(workflow)
        replacements = {
            "__POSITIVE_PROMPT__": positive_prompt,
            "__NEGATIVE_PROMPT__": negative_prompt or "worst quality, low quality, blurry, deformed",
            "__WIDTH__": str(params["width"]),
            "__HEIGHT__": str(params["height"]),
            "__SEED__": str(seed),
            "__STEPS__": str(params["steps"]),
            "__CFG__": str(params["cfg"]),
            "__SAMPLER__": params["sampler"],
            "__SCHEDULER__": params["scheduler"],
            "__FILENAME_PREFIX__": filename_prefix,
            **{f"__{k.upper()}__": v for k, v in model_assignments.items()},
        }
        for placeholder, value in replacements.items():
            workflow_str = workflow_str.replace(f'"{placeholder}"', f'"{value}"')
            workflow_str = workflow_str.replace(placeholder, str(value))

        return {
            "workflow": json.loads(workflow_str),
            "ready": True,
            "missing_models": [],
            "params_used": {
                "format": format,
                "seed": seed,
                "steps": params["steps"],
                "cfg": params["cfg"],
                "width": params["width"],
                "height": params["height"],
                "models": model_assignments,
            }
        }

    def _select_template_and_models(
        self, inventory: dict, use_lightning: bool
    ) -> tuple[dict, dict, list]:
        """
        Selecciona el template correcto y asigna los modelos disponibles.
        Prioriza GGUF sobre checkpoints.
        """
        models = inventory.get("models", {})
        missing = []
        assignments = {}

        # Opción 1: GGUF (Qwen-Image o similar)
        if models.get("diffusion_models"):
            diffusion = models["diffusion_models"][0]
            assignments["DIFFUSION_MODEL"] = diffusion

            # Text encoders GGUF
            if models.get("text_encoders"):
                assignments["TEXT_ENCODER_1"] = models["text_encoders"][0]
                assignments["TEXT_ENCODER_2"] = (
                    models["text_encoders"][1]
                    if len(models["text_encoders"]) > 1
                    else models["text_encoders"][0]
                )
            else:
                missing.append("text_encoder_gguf")

            # VAE
            if models.get("vae"):
                assignments["VAE"] = models["vae"][0]
            else:
                missing.append("vae")

            # Lightning LoRA opcional
            if use_lightning and models.get("loras"):
                lightning = next(
                    (l for l in models["loras"] if "lightning" in l.lower() or "Lightning" in l),
                    None
                )
                if lightning:
                    assignments["lightning_lora"] = lightning

            return WORKFLOW_TEXT_TO_IMAGE_GGUF, assignments, missing

        # Opción 2: Checkpoint clásico
        elif models.get("checkpoints"):
            assignments["CHECKPOINT"] = models["checkpoints"][0]
            return WORKFLOW_TEXT_TO_IMAGE_CHECKPOINT, assignments, missing

        # Sin modelos
        else:
            missing.extend(["diffusion_model_gguf", "text_encoder", "vae"])
            return {}, {}, missing

    def _generate_install_hint(self, missing: list) -> str:
        hints = {
            "diffusion_model_gguf": (
                "Descargá el modelo GGUF de Qwen-Image:\n"
                "curl -L -o ComfyUI/models/diffusion_models/qwen-image-2512-Q4_K_M.gguf "
                "https://huggingface.co/unsloth/Qwen-Image-2512-GGUF/resolve/main/qwen-image-2512-Q4_K_M.gguf"
            ),
            "text_encoder": (
                "Descargá el text encoder:\n"
                "curl -L -o ComfyUI/models/text_encoders/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf "
                "https://huggingface.co/unsloth/Qwen2.5-VL-7B-Instruct-GGUF/resolve/main/Qwen2.5-VL-7B-Instruct-UD-Q4_K_XL.gguf"
            ),
            "vae": (
                "Descargá el VAE:\n"
                "curl -L -o ComfyUI/models/vae/qwen_image_vae.safetensors "
                "https://huggingface.co/Comfy-Org/Qwen-Image_ComfyUI/resolve/main/split_files/vae/qwen_image_vae.safetensors"
            ),
        }
        lines = ["📥 Modelos faltantes — ejecutá estos comandos desde C:\\AI\\ComfyUI:"]
        for m in missing:
            if m in hints:
                lines.append(f"\n### {m}\n{hints[m]}")
            else:
                lines.append(f"\n### {m}\n_(buscá el modelo en HuggingFace o CivitAI)_")
        return "\n".join(lines)

    def estimate_time(self, steps: int, width: int, height: int) -> str:
        """Estimación aproximada de tiempo para RTX 3090."""
        # Aproximado basado en benchmarks reales con GGUF Q4 en 3090
        pixels = width * height
        base_seconds = (pixels / (1024 * 1024)) * steps * 1.2
        if base_seconds < 30:
            return "~20-30 segundos"
        elif base_seconds < 90:
            return "~1-2 minutos"
        else:
            return f"~{int(base_seconds / 60)} minutos"


# Instancia global
workflow_builder = WorkflowBuilder()
