"""
MAIA Agent OS — Test de Fase 1
Verifica que todos los componentes estén funcionando correctamente.
Ejecutar con: python test_fase1.py
"""

import asyncio
import sys
import os

# Agregar el directorio raíz al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_all():
    results = []

    def ok(name):
        print(f"  ✅ {name}")
        results.append((name, True, None))

    def fail(name, error):
        print(f"  ❌ {name}: {error}")
        results.append((name, False, str(error)))

    print("\n" + "="*50)
    print("  MAIA Fase 1 — Test de componentes")
    print("="*50 + "\n")

    # ── 1. Config ─────────────────────────────────────────────
    print("📋 Config...")
    try:
        from app.config import config
        config.ensure_dirs()
        ok("Config cargada")
        ok(f"MAIA_ROOT: {config.MAIA_ROOT}")
        ok(f"LLM_BASE_URL: {config.LLM_BASE_URL}")
    except Exception as e:
        fail("Config", e)

    # ── 2. Memoria SQLite ──────────────────────────────────────
    print("\n🧠 Memoria SQLite...")
    try:
        from app.core.memory.sqlite_memory import memory

        # Test guardar y recuperar mensaje
        import uuid as _uuid
        sid = f"test-session-{_uuid.uuid4().hex[:8]}"
        memory.save_message("user", "Hola MAIA, esto es un test", sid)
        memory.save_message("assistant", "Hola! Test recibido correctamente.", sid)
        history = memory.get_history(sid, limit=5)
        assert len(history) == 2, f"Esperaba 2 mensajes, got {len(history)}"
        ok("save_message + get_history")

        # Test búsqueda FTS5
        memory.save_message("user", "quiero generar una imagen de Miami", sid)
        results = memory.search_messages("Miami imagen", limit=5)
        assert len(results) >= 1, "FTS5 no encontró resultados"
        ok("FTS5 search mensajes")

        # Test tareas
        task_id = memory.create_task(sid, "tarea de prueba")
        memory.update_task(task_id, status="done", result="ok")
        task = memory.get_task(task_id)
        assert task["status"] == "done"
        ok("create_task + update_task + get_task")

        # Test episodios
        memory.save_episode("Se generó una imagen de Miami", "exitoso", task_id)
        episodes = memory.search_episodes("imagen Miami", limit=3)
        assert len(episodes) >= 1
        ok("save_episode + search_episodes")

        # Test lecciones
        memory.save_lesson(
            "La tool write_file falla si el directorio no existe",
            "Crear el directorio padre con mkdir -p antes de escribir",
            task_id
        )
        lessons = memory.search_lessons("write_file directorio", limit=3)
        assert len(lessons) >= 1
        ok("save_lesson + search_lessons")

        # Stats
        stats = memory.stats()
        ok(f"Stats: {stats}")

    except Exception as e:
        fail("Memoria SQLite", e)

    # ── 3. Tool Registry ───────────────────────────────────────
    print("\n🔧 Tool Registry...")
    try:
        from app.tools.base import tool_registry, ToolResult
        from app.tools.filesystem import (
            ListFilesTool, ReadFileTool, WriteFileTool,
            DeleteFileTool, ShellTool, MemorySearchTool
        )

        tools = [ListFilesTool(), ReadFileTool(), WriteFileTool(),
                 DeleteFileTool(), ShellTool(), MemorySearchTool()]
        for tool in tools:
            tool_registry.register(tool)

        assert "list_files" in tool_registry.all_names()
        assert "read_file" in tool_registry.all_names()
        assert "write_file" in tool_registry.all_names()
        assert "delete_file" in tool_registry.all_names()
        assert "shell" in tool_registry.all_names()
        assert "memory_search" in tool_registry.all_names()
        ok(f"Registry: {tool_registry.all_names()}")

        # Verificar que delete y write requieren aprobación
        assert tool_registry.requires_approval("delete_file")
        assert tool_registry.requires_approval("write_file")
        assert not tool_registry.requires_approval("read_file")
        ok("Approval flags correctos")

    except Exception as e:
        fail("Tool Registry", e)

    # ── 4. Tools individuales ──────────────────────────────────
    print("\n🛠️ Tools individuales...")
    try:
        from app.tools.filesystem import ListFilesTool, WriteFileTool, ReadFileTool, DeleteFileTool
        import tempfile, os

        # list_files
        tool = ListFilesTool()
        result = await tool.run(path=".")
        assert result.success
        ok(f"list_files: {result.output['count']} items en '.'")

        # write_file + read_file + delete_file
        with tempfile.TemporaryDirectory() as tmp:
            test_file = os.path.join(tmp, "test_maia.txt")

            # write
            wt = WriteFileTool()
            r = await wt.run(path=test_file, content="MAIA test content 🚀")
            assert r.success, f"write_file falló: {r.error}"
            ok("write_file")

            # read
            rt = ReadFileTool()
            r = await rt.run(path=test_file)
            assert r.success and "MAIA test content" in r.output
            ok("read_file")

            # delete
            dt = DeleteFileTool()
            r = await dt.run(path=test_file)
            assert r.success
            ok("delete_file")

    except Exception as e:
        fail("Tools individuales", e)

    # ── 5. Approval Manager ────────────────────────────────────
    print("\n🔐 Approval Manager...")
    try:
        from app.core.approval_manager import approval_manager

        approval = approval_manager.request(
            tool_name="delete_file",
            tool_kwargs={"path": "/test/archivo.txt"},
            task_id="test-task",
            session_id="test-session",
        )
        assert approval.status.value == "pending"
        ok("request approval creado")

        # Aprobar
        resolved = approval_manager.resolve(approval.id, approved=True)
        assert resolved.status.value == "approved"
        ok("resolve approved")

        # Rechazar
        approval2 = approval_manager.request("shell", {"command": "ls"}, "t2", "s2")
        approval_manager.resolve(approval2.id, approved=False)
        assert approval_manager.get(approval2.id).status.value == "rejected"
        ok("resolve rejected")

    except Exception as e:
        fail("Approval Manager", e)

    # ── 6. Context Builder ─────────────────────────────────────
    print("\n📝 Context Builder...")
    try:
        from app.core.context_builder import context_builder

        prompt = context_builder.build_system_prompt()
        assert "MAIA" in prompt
        assert "RTX 3090" in prompt
        ok("System prompt generado")

        messages = context_builder.build_messages(
            session_id="test-context-session",
            user_input="que podes hacer"
        )
        assert messages[0]["role"] == "system"
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "¿Qué podés hacer?"
        ok(f"build_messages: {len(messages)} mensajes")

    except Exception as e:
        fail("Context Builder", e)

    # ── 7. Conectividad LLM (opcional) ────────────────────────
    print("\n🤖 LLM (llama-server)...")
    try:
        import httpx
        from app.config import config
        r = await asyncio.wait_for(
            httpx.AsyncClient().get(f"{config.LLM_BASE_URL}/models"),
            timeout=3.0
        )
        if r.status_code == 200:
            ok("LLM respondiendo en :8000")
        else:
            print(f"  ⚠️  LLM responde con status {r.status_code} (normal si está cargando)")
    except Exception:
        print("  ⚠️  LLM no disponible (normal si no arrancaste start_llm.bat)")

    # ── 8. Conectividad ComfyUI (opcional) ────────────────────
    print("\n🎨 ComfyUI...")
    try:
        import httpx
        r = await asyncio.wait_for(
            httpx.AsyncClient().get(f"{config.COMFYUI_URL}/system_stats"),
            timeout=3.0
        )
        if r.status_code == 200:
            ok("ComfyUI respondiendo en :8188")
        else:
            print(f"  ⚠️  ComfyUI responde con status {r.status_code}")
    except Exception:
        print("  ⚠️  ComfyUI no disponible (normal si no arrancaste start_comfyui.bat)")

    # ── Resumen ───────────────────────────────────────────────
    print("\n" + "="*50)
    passed = sum(1 for _, status, _ in results if status)
    total = len(results)
    print(f"  Resultado: {passed}/{total} tests pasaron")

    if passed == total:
        print("  🎉 Fase 1 lista para arrancar!")
        print(f"\n  Ejecutá: python main.py")
        print(f"  O doble click en: start_maia.bat")
    else:
        failed = [name for name, status, _ in results if not status]
        print(f"  ❌ Fallaron: {failed}")
        print("  Revisá los errores arriba antes de continuar.")

    print("="*50 + "\n")


if __name__ == "__main__":
    asyncio.run(test_all())
