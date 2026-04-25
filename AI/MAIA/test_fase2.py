"""
MAIA Agent OS — Test de Fase 2 (ComfyUI Bridge)
Ejecutar con: python test_fase2.py
No requiere ComfyUI corriendo — testea la lógica, no la conectividad.
"""

import asyncio, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


async def test_all():
    results = []

    def ok(name):
        print(f"  ✅ {name}")
        results.append((name, True))

    def warn(name, msg):
        print(f"  ⚠️  {name}: {msg}")
        results.append((name, True))   # warning no falla el test

    def fail(name, error):
        print(f"  ❌ {name}: {error}")
        results.append((name, False))

    print("\n" + "="*55)
    print("  MAIA Fase 2 — Test ComfyUI Bridge")
    print("="*55 + "\n")

    # ── 1. Imports ────────────────────────────────────────────
    print("📦 Imports...")
    try:
        from app.core.comfyui.inventory_scanner import ComfyUIInventory
        from app.core.comfyui.workflow_builder import WorkflowBuilder
        from app.core.comfyui.vram_orchestrator import VRAMOrchestrator, PipelineType
        from app.core.comfyui.client import ComfyUIClient
        from app.tools.comfyui_tool import GenerateImageTool, ComfyUIStatusTool
        ok("Todos los módulos importan correctamente")
    except Exception as e:
        fail("Imports", e)
        return

    # ── 2. VRAM Orchestrator ──────────────────────────────────
    print("\n🎮 VRAM Orchestrator...")
    try:
        orch = VRAMOrchestrator()

        state = orch.get_state()
        assert state["budget_gb"] == 20
        assert state["is_busy"] == False
        ok(f"Estado inicial: budget={state['budget_gb']}GB, busy={state['is_busy']}")

        pipelines = orch.list_pipelines()
        assert len(pipelines) > 0
        ok(f"Pipelines definidos: {len(pipelines)}")

        # Verificar que todos caben en 20GB (RTX 3090 budget)
        for p in pipelines:
            assert p["fits_in_budget"], f"Pipeline {p['type']} no cabe: {p['max_vram_gb']}GB"
        ok(f"Todos los pipelines caben en 20GB budget")

        # Validar pipeline específico
        val = orch.validate_pipeline(PipelineType.TEXT_TO_IMAGE)
        assert val["valid"], f"Pipeline text_to_image inválido: {val['issues']}"
        ok(f"text_to_image: {val['stages']} stages, válido")

        val2 = orch.validate_pipeline(PipelineType.TALKING_HEAD)
        assert val2["valid"]
        ok(f"talking_head: {val2['stages']} stages, válido")

    except Exception as e:
        import traceback; traceback.print_exc()
        fail("VRAM Orchestrator", e)

    # ── 3. Workflow Builder (sin modelos) ─────────────────────
    print("\n🔧 Workflow Builder...")
    try:
        builder = WorkflowBuilder()

        # Verificar parámetros por formato
        assert "portrait" in builder.OPTIMAL_PARAMS
        assert "story_vertical" in builder.OPTIMAL_PARAMS
        assert "square_post" in builder.OPTIMAL_PARAMS
        ok("Formatos definidos: portrait, landscape, square_post, story_vertical")

        # story_vertical debe ser 9:16
        sv = builder.OPTIMAL_PARAMS["story_vertical"]
        ratio = sv["width"] / sv["height"]
        assert abs(ratio - (9/16)) < 0.1, f"story_vertical no es 9:16: {sv['width']}x{sv['height']}"
        ok(f"story_vertical ratio correcto: {sv['width']}x{sv['height']}")

        # portrait debe ser vertical
        portrait = builder.OPTIMAL_PARAMS["portrait"]
        assert portrait["height"] > portrait["width"], "portrait debe ser más alto que ancho"
        ok(f"portrait correcto: {portrait['width']}x{portrait['height']}")

        # Estimación de tiempo
        time_est = builder.estimate_time(20, 1024, 1024)
        assert isinstance(time_est, str) and len(time_est) > 0
        ok(f"estimate_time: {time_est} para 1024x1024 20steps")

        # Build sin inventario real (retorna missing_models)
        result = await builder.build_text_to_image(
            positive_prompt="test prompt",
            negative_prompt="bad quality",
            format="portrait",
        )
        # Sin ComfyUI corriendo va a retornar missing_models o ready
        assert "ready" in result
        if result["ready"]:
            assert "workflow" in result
            assert result["workflow"] is not None
            ok("Workflow construido con modelos del inventario")
        else:
            assert "missing_models" in result
            assert "install_hint" in result
            warn("Workflow Builder", f"ComfyUI no corriendo — missing: {result['missing_models']}")

    except Exception as e:
        import traceback; traceback.print_exc()
        fail("Workflow Builder", e)

    # ── 4. Inventory Scanner ──────────────────────────────────
    print("\n🔍 Inventory Scanner...")
    try:
        scanner = ComfyUIInventory()

        # Test inventario vacío (ComfyUI no corriendo)
        inv = await scanner.get()
        assert "models" in inv
        assert "hardware" in inv
        assert inv["hardware"]["gpu"] == "RTX 3090"
        assert inv["hardware"]["vram_budget_gb"] == 20

        if inv.get("error"):
            warn("Inventory Scanner", f"ComfyUI offline — inventario vacío: {inv['error']}")
        else:
            caps = inv.get("capabilities", {})
            nodes = inv.get("nodes", {}).get("total", 0)
            ok(f"Inventario: {nodes} nodos, capacidades: {[k for k,v in caps.items() if v]}")

        # Verificar check_model_available
        missing = scanner.get_missing_models(
            ["modelo_que_no_existe.gguf"],
            inv
        )
        assert "modelo_que_no_existe.gguf" in missing
        ok("get_missing_models funciona correctamente")

        await scanner.close()

    except Exception as e:
        import traceback; traceback.print_exc()
        fail("Inventory Scanner", e)

    # ── 5. ComfyUI Client ─────────────────────────────────────
    print("\n🌐 ComfyUI Client...")
    try:
        client = ComfyUIClient()

        available = await client.is_available()
        if available:
            ok("ComfyUI online y respondiendo")
            queue = await client.get_queue_status()
            ok(f"Queue status: {queue}")
        else:
            warn("ComfyUI Client", "ComfyUI offline (normal si no arrancaste start_comfyui.bat)")

        await client.close()

    except Exception as e:
        fail("ComfyUI Client", e)

    # ── 6. Tools registradas ──────────────────────────────────
    print("\n🛠️ Tools ComfyUI...")
    try:
        gen_tool = GenerateImageTool()
        status_tool = ComfyUIStatusTool()

        assert gen_tool.name == "generate_image"
        assert gen_tool.requires_approval == False
        ok(f"generate_image: name OK, approval={gen_tool.requires_approval}")

        assert status_tool.name == "comfyui_status"
        ok(f"comfyui_status: name OK")

        # Verificar schema de generate_image
        schema = gen_tool.schema()
        props = schema["function"]["parameters"]["properties"]
        assert "positive_prompt" in props
        assert "format" in props
        assert "seed" in props
        ok("Schema de generate_image correcto")

    except Exception as e:
        fail("Tools ComfyUI", e)

    # ── 7. Routes importan correctamente ─────────────────────
    print("\n🚦 Routes...")
    try:
        from app.routes.comfyui import router
        routes = [r.path for r in router.routes]
        assert "/api/comfyui/status" in routes
        assert "/api/comfyui/generate" in routes
        assert "/api/comfyui/inventory" in routes
        assert "/api/comfyui/pipelines" in routes
        ok(f"Routes registradas: {routes}")
    except Exception as e:
        fail("Routes", e)

    # ── Resumen ───────────────────────────────────────────────
    print("\n" + "="*55)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    print(f"  Resultado: {passed}/{total} tests pasaron")

    if passed == total:
        print("  🎉 Fase 2 lista!")
        print("\n  Próximo paso:")
        print("  1. Arrancá start_comfyui.bat")
        print("  2. Arrancá start_maia.bat")
        print("  3. Probá: curl -X POST http://127.0.0.1:8080/api/comfyui/generate \\")
        print('       -H "Content-Type: application/json" \\')
        print('       -d \'{"positive_prompt": "beautiful woman in Miami beach sunset"}\'')
    else:
        failed = [n for n, s in results if not s]
        print(f"  ❌ Fallaron: {failed}")

    print("="*55 + "\n")


if __name__ == "__main__":
    asyncio.run(test_all())
