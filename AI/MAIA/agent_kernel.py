"""
MAIA Agent OS — Agent Kernel
Orquestador principal. Conecta todos los módulos.

Flujo por mensaje:
  input
    → buscar aprobaciones pendientes (responde "sí/no")
    → guardar mensaje en memoria
    → construir contexto
    → llamar LLM
    → parsear tool calls
    → ejecutar tools (con approval si aplica)
    → respuesta final
    → guardar en memoria
    → learning loop
"""

import json
import time
import uuid
from typing import AsyncGenerator, Optional

import httpx

from app.config import config
from app.core.memory.sqlite_memory import memory
from app.core.context_builder import context_builder
from app.core.approval_manager import approval_manager, ApprovalStatus
from app.tools.base import tool_registry, ToolResult


class AgentKernel:
    """
    Núcleo del agente. Una instancia por proceso.
    Maneja el loop completo de input → output con memoria y tools.
    """

    def __init__(self):
        self.llm_client = httpx.AsyncClient(
            base_url=config.LLM_BASE_URL,
            timeout=120.0,
        )

    async def chat(
        self,
        user_input: str,
        session_id: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """
        Punto de entrada principal.
        Genera la respuesta de forma streaming.
        """
        session_id = session_id or str(uuid.uuid4())
        task_id = memory.create_task(session_id, user_input)
        start_time = time.monotonic()

        # ── 1. Verificar si el usuario está respondiendo a una aprobación ──────
        approval_response = self._check_approval_response(user_input, session_id)
        if approval_response:
            memory.save_message("user", user_input, session_id)
            yield approval_response
            memory.save_message("assistant", approval_response, session_id)

            # Si fue aprobado, ejecutar la tool pendiente
            pending = approval_manager.pending_for_session(session_id)
            for approval in pending:
                if approval.status == ApprovalStatus.APPROVED:
                    async for chunk in self._execute_approved_tool(approval, task_id, session_id):
                        yield chunk
            return

        # ── 2. Guardar mensaje del usuario ────────────────────────────────────
        memory.save_message("user", user_input, session_id)
        memory.update_task(task_id, status="running")

        # ── 3. Construir contexto y llamar al LLM ─────────────────────────────
        messages = context_builder.build_messages(session_id, user_input)
        tools = tool_registry.all_schemas()

        full_response = ""
        tool_calls_buffer = []

        try:
            async for chunk in self._stream_llm(messages, tools):
                if chunk["type"] == "text":
                    full_response += chunk["content"]
                    yield chunk["content"]
                elif chunk["type"] == "tool_call":
                    tool_calls_buffer.append(chunk["data"])

        except Exception as e:
            error_msg = f"\n\n❌ Error al llamar al LLM: {str(e)}"
            yield error_msg
            memory.update_task(task_id, status="failed", error=str(e))
            return

        # ── 4. Ejecutar tool calls si las hay ────────────────────────────────
        if tool_calls_buffer:
            tools_used = []
            for tc in tool_calls_buffer:
                tool_name = tc.get("name", "")
                tool_kwargs = tc.get("arguments", {})
                if isinstance(tool_kwargs, str):
                    try:
                        tool_kwargs = json.loads(tool_kwargs)
                    except Exception:
                        tool_kwargs = {}

                tools_used.append(tool_name)

                # ¿Requiere aprobación?
                if tool_registry.requires_approval(tool_name):
                    approval = approval_manager.request(
                        tool_name=tool_name,
                        tool_kwargs=tool_kwargs,
                        task_id=task_id,
                        session_id=session_id,
                    )
                    msg = "\n\n" + approval_manager.format_request_message(approval)
                    yield msg
                    full_response += msg
                else:
                    # Ejecutar directo
                    async for chunk in self._run_tool(tool_name, tool_kwargs, task_id):
                        yield chunk
                        full_response += chunk

            memory.update_task(task_id, tools_used=tools_used)

        # ── 5. Guardar respuesta en memoria ───────────────────────────────────
        if full_response.strip():
            memory.save_message("assistant", full_response, session_id)

        duration_ms = int((time.monotonic() - start_time) * 1000)
        memory.update_task(task_id, status="done", duration_ms=duration_ms)

        # ── 6. Learning loop básico ───────────────────────────────────────────
        await self._learning_loop(task_id, user_input, full_response)

    async def _stream_llm(
        self,
        messages: list[dict],
        tools: list[dict],
    ) -> AsyncGenerator[dict, None]:
        """
        Llama al LLM via streaming y parsea la respuesta.
        Compatible con llama-server (OpenAI API format).
        """
        payload = {
            "model": config.LLM_MODEL,
            "messages": messages,
            "max_tokens": config.LLM_MAX_TOKENS,
            "temperature": config.LLM_TEMPERATURE,
            "stream": True,
        }

        # Agregar tools solo si hay alguna registrada
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"

        text_buffer = ""
        tool_buffer = {}
        in_tool_call = False

        async with self.llm_client.stream(
            "POST",
            "/chat/completions",
            json=payload,
        ) as response:
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break

                try:
                    data = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                choice = data.get("choices", [{}])[0]
                delta = choice.get("delta", {})

                # Texto normal
                if delta.get("content"):
                    yield {"type": "text", "content": delta["content"]}

                # Tool call
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        if idx not in tool_buffer:
                            tool_buffer[idx] = {"name": "", "arguments": ""}

                        if tc_delta.get("function", {}).get("name"):
                            tool_buffer[idx]["name"] += tc_delta["function"]["name"]
                        if tc_delta.get("function", {}).get("arguments"):
                            tool_buffer[idx]["arguments"] += tc_delta["function"]["arguments"]

                # Fin del stream para este choice
                if choice.get("finish_reason") in ("tool_calls", "stop"):
                    for tc in tool_buffer.values():
                        if tc["name"]:
                            try:
                                tc["arguments"] = json.loads(tc["arguments"])
                            except Exception:
                                tc["arguments"] = {}
                            yield {"type": "tool_call", "data": tc}
                    tool_buffer = {}

    async def _run_tool(
        self,
        tool_name: str,
        tool_kwargs: dict,
        task_id: str,
    ) -> AsyncGenerator[str, None]:
        """Ejecuta una tool y formatea el output."""
        tool = tool_registry.get(tool_name)
        if not tool:
            yield f"\n\n❌ Tool no encontrada: `{tool_name}`"
            return

        yield f"\n\n🔧 Ejecutando `{tool_name}`..."

        result: ToolResult = await tool.run(**tool_kwargs)

        # Log en memoria
        memory.log_tool_call(
            tool_name=tool_name,
            input_data=tool_kwargs,
            output=str(result.output) if result.success else None,
            status="ok" if result.success else "error",
            task_id=task_id,
            duration_ms=result.duration_ms,
        )

        if result.success:
            output_str = self._format_tool_output(tool_name, result)
            yield f"\n\n✅ **{tool_name}** ({result.duration_ms}ms):\n{output_str}"
        else:
            yield f"\n\n❌ **{tool_name}** falló: {result.error}"

            # Guardar como lección si es un error nuevo
            memory.save_lesson(
                problem=f"Tool {tool_name} falló: {result.error}",
                solution="Verificar parámetros y permisos antes de ejecutar",
                task_id=task_id,
            )

    async def _execute_approved_tool(
        self,
        approval,
        task_id: str,
        session_id: str,
    ) -> AsyncGenerator[str, None]:
        """Ejecuta una tool que acaba de ser aprobada."""
        async for chunk in self._run_tool(
            approval.tool_name,
            approval.tool_kwargs,
            task_id,
        ):
            yield chunk

    def _check_approval_response(
        self, user_input: str, session_id: str
    ) -> Optional[str]:
        """
        Detecta si el usuario está respondiendo a una solicitud de aprobación.
        Retorna mensaje de confirmación o None.
        """
        pending = approval_manager.pending_for_session(session_id)
        if not pending:
            return None

        text = user_input.lower().strip()
        is_yes = text in ("sí", "si", "yes", "s", "y", "dale", "ok", "aprobado", "✅")
        is_no = text in ("no", "n", "cancelar", "cancel", "rechazado", "❌")

        if not (is_yes or is_no):
            return None

        # Resolver la primera aprobación pendiente
        approval = pending[0]
        approval_manager.resolve(approval.id, approved=is_yes)

        if is_yes:
            return f"✅ Aprobado. Ejecutando `{approval.tool_name}`..."
        else:
            return f"❌ Cancelado. La tool `{approval.tool_name}` no se ejecutó."

    def _format_tool_output(self, tool_name: str, result: ToolResult) -> str:
        """Formatea el output de una tool de forma legible."""
        output = result.output

        if isinstance(output, dict):
            # list_files → mostrar lista
            if "items" in output:
                items = output["items"]
                header = f"📁 `{output.get('path', '')}` — {output['count']} items"
                if len(items) <= 20:
                    return header + "\n```\n" + "\n".join(items) + "\n```"
                else:
                    preview = "\n".join(items[:20])
                    return header + f"\n```\n{preview}\n... ({len(items) - 20} más)\n```"

            # memory_search → mostrar resultados
            if "episodes" in output or "lessons" in output:
                sections = []
                for key, items in output.items():
                    if items:
                        sections.append(f"**{key}**: {len(items)} resultado(s)")
                return "\n".join(sections) if sections else "_(sin resultados)_"

            return f"```json\n{json.dumps(output, indent=2, ensure_ascii=False)}\n```"

        return str(output)

    async def _learning_loop(
        self, task_id: str, user_input: str, response: str
    ) -> None:
        """
        Loop de aprendizaje básico.
        Guarda un episodio por cada tarea completada.
        Más adelante: análisis de errores y generación de skills.
        """
        task = memory.get_task(task_id)
        if not task:
            return

        # Guardar episodio
        memory.save_episode(
            summary=f"Usuario pidió: {user_input[:100]}",
            outcome="completado" if task["status"] == "done" else "fallido",
            task_id=task_id,
            context={
                "tools_used": json.loads(task.get("tools_used") or "[]"),
                "duration_ms": task.get("duration_ms"),
            }
        )

    async def close(self):
        await self.llm_client.aclose()


# Instancia global
agent = AgentKernel()
