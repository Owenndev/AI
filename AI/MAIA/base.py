"""
MAIA Agent OS — Base de Tools y Registry
Todas las tools heredan de BaseTool.
El registry las registra y el executor las llama por nombre.
"""

import time
from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field


# ── Resultado estándar de cualquier tool ────────────────────────────────────

@dataclass
class ToolResult:
    success: bool
    output: Any
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: dict = field(default_factory=dict)

    def to_str(self) -> str:
        if self.success:
            return str(self.output)
        return f"ERROR: {self.error}"


# ── Base de toda tool ────────────────────────────────────────────────────────

class BaseTool(ABC):
    """
    Clase base para todas las tools de MAIA.
    Cada tool tiene nombre, descripción y método execute().
    """

    name: str = "base_tool"
    description: str = "Herramienta base"
    requires_approval: bool = False    # True → pide confirmación al usuario

    @abstractmethod
    async def execute(self, **kwargs) -> ToolResult:
        """Lógica principal de la tool."""
        pass

    async def run(self, **kwargs) -> ToolResult:
        """Wrapper que mide tiempo y captura errores."""
        start = time.monotonic()
        try:
            result = await self.execute(**kwargs)
            result.duration_ms = int((time.monotonic() - start) * 1000)
            return result
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e),
                duration_ms=int((time.monotonic() - start) * 1000),
            )

    def schema(self) -> dict:
        """Schema de la tool para el LLM (formato OpenAI function calling)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._parameters(),
            }
        }

    def _parameters(self) -> dict:
        """Override en subclases para definir parámetros."""
        return {"type": "object", "properties": {}, "required": []}


# ── Registry ─────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Registro central de todas las tools disponibles.
    El agente consulta este registry para saber qué puede hacer.
    """

    def __init__(self):
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def all_names(self) -> list[str]:
        return list(self._tools.keys())

    def all_schemas(self) -> list[dict]:
        """Retorna todos los schemas para pasarle al LLM."""
        return [t.schema() for t in self._tools.values()]

    def requires_approval(self, name: str) -> bool:
        tool = self._tools.get(name)
        return tool.requires_approval if tool else False

    def __repr__(self) -> str:
        return f"ToolRegistry({list(self._tools.keys())})"


# Instancia global
tool_registry = ToolRegistry()
