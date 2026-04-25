"""
MAIA Agent OS — Configuración Central
Carga variables de entorno y expone configuración tipada.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Cargar .env desde la raíz del proyecto
_ROOT = Path(__file__).parent.parent.parent
load_dotenv(_ROOT / ".env")


class Config:
    # ── Paths ────────────────────────────────────────────────
    MAIA_ROOT: Path = Path(os.getenv("MAIA_ROOT", str(_ROOT)))
    DATA_DIR: Path = Path(os.getenv("DATA_DIR", str(_ROOT / "data")))
    OUTPUTS_DIR: Path = Path(os.getenv("OUTPUTS_DIR", str(_ROOT / "data" / "outputs")))
    LOGS_DIR: Path = Path(os.getenv("LOGS_DIR", str(_ROOT / "data" / "logs")))

    # ── Base de datos ────────────────────────────────────────
    DATABASE_PATH: str = os.getenv(
        "DATABASE_PATH",
        str(_ROOT / "data" / "memory" / "maia.db")
    )

    # ── LLM ─────────────────────────────────────────────────
    LLM_BASE_URL: str = os.getenv("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "qwen3.6-35b-a3b")
    LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "4096"))
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))

    # ── ComfyUI ──────────────────────────────────────────────
    COMFYUI_URL: str = os.getenv("COMFYUI_URL", "http://127.0.0.1:8188")
    COMFYUI_OUTPUT_DIR: str = os.getenv("COMFYUI_OUTPUT_DIR", "C:/AI/ComfyUI/output")

    # ── Servidor ─────────────────────────────────────────────
    MAIA_HOST: str = os.getenv("MAIA_HOST", "127.0.0.1")
    MAIA_PORT: int = int(os.getenv("MAIA_PORT", "8080"))
    DEBUG: bool = os.getenv("DEBUG", "true").lower() == "true"

    # ── Hardware (fijo para RTX 3090) ────────────────────────
    GPU_NAME: str = "RTX 3090"
    VRAM_TOTAL_GB: int = 24
    VRAM_BUDGET_GB: int = 20       # reserva 4GB para OS y overhead
    RAM_GB: int = 24

    @classmethod
    def ensure_dirs(cls) -> None:
        """Crea directorios necesarios si no existen."""
        dirs = [
            cls.DATA_DIR,
            cls.OUTPUTS_DIR,
            cls.LOGS_DIR,
            cls.DATA_DIR / "memory",
            cls.DATA_DIR / "skills",
        ]
        for d in dirs:
            Path(d).mkdir(parents=True, exist_ok=True)


config = Config()
