@echo off
title MAIA Agent OS

echo.
echo ============================================
echo   MAIA Agent OS — Iniciando servidor...
echo ============================================
echo.

REM Activar entorno virtual
call C:\AI\MAIA\venv\Scripts\activate

REM Ir al directorio del proyecto
cd /d C:\AI\MAIA

REM Verificar que el LLM está corriendo
curl -s http://127.0.0.1:8000/health > nul 2>&1
if %errorlevel% neq 0 (
    echo [ADVERTENCIA] El LLM no está corriendo en :8000
    echo Arranca start_llm.bat primero si querés capacidad de razonamiento completa.
    echo.
)

REM Verificar que ComfyUI está corriendo
curl -s http://127.0.0.1:8188/system_stats > nul 2>&1
if %errorlevel% neq 0 (
    echo [ADVERTENCIA] ComfyUI no está corriendo en :8188
    echo Arranca start_comfyui.bat primero si querés generación de imágenes.
    echo.
)

echo Iniciando MAIA en http://127.0.0.1:8080
echo Documentación en http://127.0.0.1:8080/docs
echo.
echo Ctrl+C para detener.
echo.

python main.py

pause
