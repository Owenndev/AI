"""
MAIA Agent OS — Filesystem Tools
list_files, read_file, write_file, delete_file

delete_file y write_file (overwrite) requieren aprobación.
"""

import os
import aiofiles
from pathlib import Path
from app.tools.base import BaseTool, ToolResult


class ListFilesTool(BaseTool):
    name = "list_files"
    description = "Lista archivos y carpetas en un directorio. Input: path (str), recursive (bool, opcional)."
    requires_approval = False

    async def execute(self, path: str = ".", recursive: bool = False) -> ToolResult:
        target = Path(path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, output=None, error=f"Path no existe: {path}")
        if not target.is_dir():
            return ToolResult(success=False, output=None, error=f"No es un directorio: {path}")

        try:
            if recursive:
                items = []
                for root, dirs, files in os.walk(target):
                    # Ignorar carpetas ocultas y __pycache__
                    dirs[:] = [d for d in dirs if not d.startswith('.') and d != '__pycache__']
                    rel_root = Path(root).relative_to(target)
                    for f in files:
                        items.append(str(rel_root / f))
            else:
                items = sorted([
                    f"{item.name}{'/' if item.is_dir() else ''}"
                    for item in target.iterdir()
                    if not item.name.startswith('.')
                ])

            output = {
                "path": str(target),
                "count": len(items),
                "items": items
            }
            return ToolResult(success=True, output=output)

        except PermissionError:
            return ToolResult(success=False, output=None, error=f"Sin permisos para leer: {path}")


class ReadFileTool(BaseTool):
    name = "read_file"
    description = "Lee el contenido de un archivo de texto. Input: path (str), encoding (str, opcional, default utf-8)."
    requires_approval = False

    async def execute(self, path: str, encoding: str = "utf-8") -> ToolResult:
        target = Path(path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, output=None, error=f"Archivo no encontrado: {path}")
        if not target.is_file():
            return ToolResult(success=False, output=None, error=f"No es un archivo: {path}")

        # Límite de seguridad: no leer archivos mayores a 10MB
        size_mb = target.stat().st_size / (1024 * 1024)
        if size_mb > 10:
            return ToolResult(
                success=False, output=None,
                error=f"Archivo demasiado grande ({size_mb:.1f}MB). Máximo 10MB."
            )

        try:
            async with aiofiles.open(target, mode='r', encoding=encoding) as f:
                content = await f.read()

            return ToolResult(
                success=True,
                output=content,
                metadata={"path": str(target), "size_bytes": target.stat().st_size, "lines": content.count('\n')}
            )
        except UnicodeDecodeError:
            return ToolResult(success=False, output=None, error=f"No se puede leer como texto (archivo binario?): {path}")


class WriteFileTool(BaseTool):
    name = "write_file"
    description = "Escribe contenido en un archivo. Crea el archivo si no existe. Input: path (str), content (str), overwrite (bool, default False)."
    requires_approval = True   # ⚠️ requiere aprobación si overwrite=True

    async def execute(self, path: str, content: str, overwrite: bool = False) -> ToolResult:
        target = Path(path).expanduser().resolve()

        if target.exists() and not overwrite:
            return ToolResult(
                success=False, output=None,
                error=f"El archivo ya existe. Usá overwrite=True para sobreescribir: {path}"
            )

        # Crear directorios padre si no existen
        target.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with aiofiles.open(target, mode='w', encoding='utf-8') as f:
                await f.write(content)

            return ToolResult(
                success=True,
                output=f"Archivo guardado: {path}",
                metadata={"path": str(target), "bytes_written": len(content.encode('utf-8'))}
            )
        except PermissionError:
            return ToolResult(success=False, output=None, error=f"Sin permisos de escritura: {path}")

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Ruta del archivo a escribir"},
                "content": {"type": "string", "description": "Contenido a escribir"},
                "overwrite": {"type": "boolean", "description": "Sobreescribir si existe (default: false)"},
            },
            "required": ["path", "content"]
        }


class DeleteFileTool(BaseTool):
    name = "delete_file"
    description = "Elimina un archivo. ⚠️ Acción irreversible. Input: path (str)."
    requires_approval = True   # ⚠️ SIEMPRE requiere aprobación

    async def execute(self, path: str) -> ToolResult:
        target = Path(path).expanduser().resolve()

        if not target.exists():
            return ToolResult(success=False, output=None, error=f"Archivo no encontrado: {path}")
        if not target.is_file():
            return ToolResult(success=False, output=None, error=f"No es un archivo (usá rmdir para carpetas): {path}")

        try:
            target.unlink()
            return ToolResult(
                success=True,
                output=f"Archivo eliminado: {path}",
                metadata={"path": str(target)}
            )
        except PermissionError:
            return ToolResult(success=False, output=None, error=f"Sin permisos para eliminar: {path}")

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Ruta del archivo a eliminar"}
            },
            "required": ["path"]
        }


class ShellTool(BaseTool):
    name = "shell"
    description = "Ejecuta un comando de shell. ⚠️ Peligroso. Solo para operaciones necesarias. Input: command (str), timeout (int, segundos, default 30)."
    requires_approval = True   # ⚠️ SIEMPRE requiere aprobación

    # Comandos que NUNCA se permiten aunque haya aprobación
    BLOCKED = ["rm -rf /", "format", "del /f /s /q C:\\", "shutdown", "mkfs"]

    async def execute(self, command: str, timeout: int = 30) -> ToolResult:
        import asyncio

        # Bloquear comandos peligrosos
        cmd_lower = command.lower()
        for blocked in self.BLOCKED:
            if blocked in cmd_lower:
                return ToolResult(
                    success=False, output=None,
                    error=f"Comando bloqueado por seguridad: contiene '{blocked}'"
                )

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)

            output = stdout.decode('utf-8', errors='replace')
            error_output = stderr.decode('utf-8', errors='replace')

            if proc.returncode == 0:
                return ToolResult(
                    success=True,
                    output=output or "(sin output)",
                    metadata={"returncode": 0, "stderr": error_output}
                )
            else:
                return ToolResult(
                    success=False,
                    output=output,
                    error=f"Exit code {proc.returncode}: {error_output}",
                    metadata={"returncode": proc.returncode}
                )

        except asyncio.TimeoutError:
            return ToolResult(success=False, output=None, error=f"Timeout después de {timeout}s")

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Comando de shell a ejecutar"},
                "timeout": {"type": "integer", "description": "Timeout en segundos (default: 30)"}
            },
            "required": ["command"]
        }


class MemorySearchTool(BaseTool):
    name = "memory_search"
    description = "Busca en la memoria de MAIA: episodios pasados, lecciones aprendidas y mensajes previos. Input: query (str), type (str: 'all'|'episodes'|'lessons'|'messages')."
    requires_approval = False

    async def execute(self, query: str, type: str = "all") -> ToolResult:
        from app.core.memory.sqlite_memory import memory

        results = {}

        if type in ("all", "episodes"):
            results["episodes"] = memory.search_episodes(query, limit=3)
        if type in ("all", "lessons"):
            results["lessons"] = memory.search_lessons(query, limit=3)
        if type in ("all", "messages"):
            results["messages"] = memory.search_messages(query, limit=3)

        total = sum(len(v) for v in results.values())
        return ToolResult(
            success=True,
            output=results,
            metadata={"total_results": total, "query": query}
        )

    def _parameters(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Texto a buscar en la memoria"},
                "type": {
                    "type": "string",
                    "enum": ["all", "episodes", "lessons", "messages"],
                    "description": "Tipo de memoria a buscar (default: all)"
                }
            },
            "required": ["query"]
        }
