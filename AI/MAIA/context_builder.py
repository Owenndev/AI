"""
MAIA Agent OS — Context Builder
Ensambla todo el contexto relevante antes de llamar al LLM.

Orden de construcción:
  1. System prompt base
  2. Memoria relevante (episodios + lecciones)
  3. Historial de conversación
  4. Skills disponibles
  5. Tools disponibles
  6. Contexto de MAIA (Mia, clima, locación — cuando esté disponible)
"""

from typing import Optional

from app.core.memory.sqlite_memory import memory
from app.tools.base import tool_registry
from app.config import config


SYSTEM_PROMPT_BASE = """Sos MAIA, un agente de inteligencia artificial local y autónomo.

## Tu forma de operar

- Usás tools reales para ejecutar tareas: archivos, búsquedas, generación de imágenes, video, audio.
- Tenés memoria persistente: recordás conversaciones anteriores, episodios y lecciones aprendidas.
- Aprendés de cada tarea y mejorás continuamente.
- Cuando una tool requiere aprobación, la solicitás explícitamente.
- Nunca inventás resultados: si no podés hacer algo, lo decís claramente.
- Respondés siempre en el idioma del usuario.

## Reglas de uso de tools

- Antes de ejecutar una tarea, buscás en memoria si ya la resolviste antes.
- Usás la tool correcta para cada subtarea.
- Si una tool falla, analizás el error y buscás una solución alternativa.
- Registrás lo aprendido para no repetir errores.

## Hardware disponible

- GPU: RTX 3090 (24GB VRAM, presupuesto operativo: 20GB)
- RAM: 24GB
- ComfyUI disponible en: {comfyui_url}
- LLM local: {llm_model}

## Capacidades actuales

{tools_description}
"""


class ContextBuilder:
    """
    Construye el contexto completo para cada llamada al LLM.
    Centraliza toda la lógica de qué se inyecta y en qué orden.
    """

    def build_system_prompt(self) -> str:
        """Construye el system prompt con estado actual del sistema."""
        tools_desc = self._build_tools_description()
        return SYSTEM_PROMPT_BASE.format(
            comfyui_url=config.COMFYUI_URL,
            llm_model=config.LLM_MODEL,
            tools_description=tools_desc,
        )

    def build_messages(
        self,
        session_id: str,
        user_input: str,
        relevant_context: Optional[dict] = None,
    ) -> list[dict]:
        """
        Construye la lista de mensajes para la API del LLM.

        Estructura:
          [system] → [memory_context] → [history] → [user]
        """
        messages = []

        # 1. System prompt
        messages.append({
            "role": "system",
            "content": self.build_system_prompt()
        })

        # 2. Contexto de memoria relevante (si hay algo)
        if relevant_context is None:
            relevant_context = memory.get_relevant_context(user_input)

        memory_block = self._format_memory_context(relevant_context)
        if memory_block:
            messages.append({
                "role": "system",
                "content": f"## Contexto relevante de memoria\n\n{memory_block}"
            })

        # 3. Historial de conversación
        history = memory.get_history(session_id, limit=20)
        for msg in history:
            messages.append({
                "role": msg["role"],
                "content": msg["content"]
            })

        # 4. Input actual del usuario
        messages.append({
            "role": "user",
            "content": user_input
        })

        return messages

    def _build_tools_description(self) -> str:
        names = tool_registry.all_names()
        if not names:
            return "_(ninguna tool registrada aún)_"

        lines = []
        for name in names:
            tool = tool_registry.get(name)
            approval_note = " ⚠️ (requiere aprobación)" if tool.requires_approval else ""
            lines.append(f"- **{name}**{approval_note}: {tool.description}")
        return "\n".join(lines)

    def _format_memory_context(self, ctx: dict) -> str:
        """Formatea el contexto de memoria en texto legible para el LLM."""
        sections = []

        episodes = ctx.get("relevant_episodes", [])
        if episodes:
            lines = ["### Episodios relevantes"]
            for ep in episodes:
                lines.append(f"- {ep['summary']} → {ep.get('outcome', 'sin resultado registrado')}")
            sections.append("\n".join(lines))

        lessons = ctx.get("relevant_lessons", [])
        if lessons:
            lines = ["### Lecciones aprendidas"]
            for lesson in lessons:
                lines.append(f"- **Problema**: {lesson['problem']}")
                lines.append(f"  **Solución**: {lesson['solution']}")
            sections.append("\n".join(lines))

        return "\n\n".join(sections)


# Instancia global
context_builder = ContextBuilder()
