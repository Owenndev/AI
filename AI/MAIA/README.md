# MAIA Agent OS — Fase 1

Agente de IA local con memoria persistente, tools reales y aprendizaje continuo.

## Estructura

```
MAIA/
├── main.py                          ← Punto de entrada
├── requirements.txt                 ← Dependencias Python
├── .env.example                     ← Copiar a .env y configurar
├── start_maia.bat                   ← Arrancar en Windows
├── test_fase1.py                    ← Verificar que todo funciona
│
├── app/
│   ├── config.py                    ← Configuración central
│   │
│   ├── core/
│   │   ├── agent_kernel.py          ← Orquestador principal
│   │   ├── context_builder.py       ← Construye contexto para el LLM
│   │   ├── approval_manager.py      ← Maneja aprobaciones de tools peligrosas
│   │   └── memory/
│   │       └── sqlite_memory.py     ← Memoria persistente SQLite + FTS5
│   │
│   ├── tools/
│   │   ├── base.py                  ← BaseTool + ToolRegistry
│   │   └── filesystem.py            ← list_files, read_file, write_file, delete_file, shell, memory_search
│   │
│   └── routes/
│       └── chat.py                  ← Endpoints FastAPI
│
└── data/
    ├── memory/maia.db               ← Base de datos (se crea automáticamente)
    ├── outputs/                     ← Archivos generados
    └── logs/                        ← Logs del sistema
```

## Setup inicial

```powershell
# 1. Activar entorno virtual
cd C:\AI\MAIA
.\venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Copiar y configurar .env
copy .env.example .env
# Editar .env si necesitás cambiar algún path

# 4. Verificar que todo funciona
python test_fase1.py

# 5. Arrancar MAIA
python main.py
# o doble click en start_maia.bat
```

## Endpoints disponibles

| Método | URL | Descripción |
|--------|-----|-------------|
| POST | `/api/chat` | Chat principal (streaming SSE) |
| GET | `/api/health` | Estado del sistema |
| GET | `/api/memory/stats` | Estadísticas de memoria |
| GET | `/api/memory/search?q=...` | Buscar en memoria |
| GET | `/api/memory/history/{session_id}` | Historial de sesión |
| GET | `/api/tools` | Tools disponibles |
| POST | `/api/approve` | Aprobar/rechazar acción |
| GET | `/docs` | Documentación automática FastAPI |

## Uso básico desde terminal

```bash
# Chat simple
curl -X POST http://127.0.0.1:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "listá los archivos en C:/AI"}'

# Health check
curl http://127.0.0.1:8080/api/health

# Stats de memoria
curl http://127.0.0.1:8080/api/memory/stats

# Buscar en memoria
curl "http://127.0.0.1:8080/api/memory/search?q=imagen+miami"
```

## Tools disponibles en Fase 1

| Tool | Aprobación | Descripción |
|------|-----------|-------------|
| `list_files` | No | Lista archivos en un directorio |
| `read_file` | No | Lee contenido de un archivo |
| `write_file` | Sí (si overwrite) | Escribe/crea archivos |
| `delete_file` | Siempre | Elimina archivos |
| `shell` | Siempre | Ejecuta comandos de shell |
| `memory_search` | No | Busca en la memoria de MAIA |

## Flujo de aprobación

Cuando MAIA quiere ejecutar `delete_file` o `shell`:

1. MAIA muestra: _"⚠️ Acción que requiere aprobación: delete_file(path='...'). ¿Aprobás?"_
2. El usuario responde: `sí` / `no`
3. MAIA ejecuta o cancela

## Próximas fases

- **Fase 2**: ComfyUI bridge, generación de imágenes con Qwen-Image
- **Fase 3**: Weather tool, Location resolver, Fashion researcher
- **Fase 4**: TTS (Chatterbox), Lip sync (LatentSync), LTX-Video
- **Fase 5**: Frontend React + Tauri
- **Fase 6**: MAIA persona completa (Mia Álvarez)
