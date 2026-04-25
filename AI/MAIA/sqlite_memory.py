"""
MAIA Agent OS — Sistema de Memoria SQLite + FTS5
Nunca empieza de cero. Todo queda guardado y es buscable.

Tablas:
  messages  → historial de conversación
  tasks     → registro de tareas ejecutadas
  episodes  → memoria episódica (qué pasó en cada sesión)
  lessons   → aprendizajes de errores y soluciones
  skills    → capacidades generadas automáticamente
  tool_calls → registro de cada tool ejecutada
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional
from app.config import config


# ── Esquema completo ─────────────────────────────────────────────────────────

SCHEMA = """
-- Mensajes de conversación
CREATE TABLE IF NOT EXISTS messages (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    role        TEXT NOT NULL,          -- user | assistant | system | tool
    content     TEXT NOT NULL,
    metadata    TEXT DEFAULT '{}',      -- JSON extra
    created_at  TEXT NOT NULL
);

-- FTS5 sobre mensajes para búsqueda semántica rápida
CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts USING fts5(
    content,
    content='messages',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS messages_ai AFTER INSERT ON messages BEGIN
    INSERT INTO messages_fts(rowid, content) VALUES (new.rowid, new.content);
END;

-- Tareas ejecutadas
CREATE TABLE IF NOT EXISTS tasks (
    id          TEXT PRIMARY KEY,
    session_id  TEXT NOT NULL,
    input       TEXT NOT NULL,          -- lo que pidió el usuario
    plan        TEXT DEFAULT '[]',      -- JSON: lista de pasos planificados
    status      TEXT DEFAULT 'pending', -- pending | running | done | failed
    result      TEXT,
    error       TEXT,
    tools_used  TEXT DEFAULT '[]',      -- JSON: lista de tools usadas
    duration_ms INTEGER,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Memoria episódica
CREATE TABLE IF NOT EXISTS episodes (
    id          TEXT PRIMARY KEY,
    task_id     TEXT,
    summary     TEXT NOT NULL,          -- resumen del episodio
    outcome     TEXT,                   -- qué resultó
    context     TEXT DEFAULT '{}',      -- JSON: contexto relevante
    created_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS episodes_fts USING fts5(
    summary,
    content='episodes',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS episodes_ai AFTER INSERT ON episodes BEGIN
    INSERT INTO episodes_fts(rowid, summary) VALUES (new.rowid, new.summary);
END;

-- Lecciones aprendidas
CREATE TABLE IF NOT EXISTS lessons (
    id          TEXT PRIMARY KEY,
    task_id     TEXT,
    problem     TEXT NOT NULL,          -- qué falló o qué era difícil
    solution    TEXT NOT NULL,          -- cómo se resolvió
    reuse_when  TEXT DEFAULT '[]',      -- JSON: keywords de cuándo aplicar
    confidence  REAL DEFAULT 1.0,       -- 0.0-1.0
    used_count  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL
);

CREATE VIRTUAL TABLE IF NOT EXISTS lessons_fts USING fts5(
    problem, solution,
    content='lessons',
    content_rowid='rowid'
);

CREATE TRIGGER IF NOT EXISTS lessons_ai AFTER INSERT ON lessons BEGIN
    INSERT INTO lessons_fts(rowid, problem, solution)
    VALUES (new.rowid, new.problem, new.solution);
END;

-- Skills generadas automáticamente
CREATE TABLE IF NOT EXISTS skills (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    code        TEXT,                   -- Python si aplica
    prompt      TEXT,                   -- prompt si es un skill de LLM
    tags        TEXT DEFAULT '[]',      -- JSON
    version     INTEGER DEFAULT 1,
    used_count  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

-- Registro de tool calls
CREATE TABLE IF NOT EXISTS tool_calls (
    id          TEXT PRIMARY KEY,
    task_id     TEXT,
    tool_name   TEXT NOT NULL,
    input       TEXT DEFAULT '{}',      -- JSON
    output      TEXT,
    status      TEXT DEFAULT 'ok',      -- ok | error | pending_approval
    duration_ms INTEGER,
    created_at  TEXT NOT NULL
);

-- SOPs (procedimientos estándar)
CREATE TABLE IF NOT EXISTS sops (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    description TEXT NOT NULL,
    steps       TEXT DEFAULT '[]',      -- JSON: lista de pasos
    triggers    TEXT DEFAULT '[]',      -- JSON: cuándo activar
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
"""


# ── Clase principal de memoria ───────────────────────────────────────────────

class MAIAMemory:
    """
    Sistema central de memoria persistente de MAIA.
    Una sola instancia por proceso (singleton-like via módulo).
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or config.DATABASE_PATH
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")    # mejor concurrencia
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def _init_db(self) -> None:
        """Inicializa el esquema si no existe."""
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    def _now(self) -> str:
        return datetime.utcnow().isoformat()

    def _uid(self) -> str:
        return str(uuid.uuid4())

    # ── Mensajes ──────────────────────────────────────────────────────────────

    def save_message(
        self,
        role: str,
        content: str,
        session_id: str,
        metadata: Optional[dict] = None,
    ) -> str:
        mid = self._uid()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO messages (id, session_id, role, content, metadata, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (mid, session_id, role, content,
                 json.dumps(metadata or {}), self._now()),
            )
        return mid

    def get_history(self, session_id: str, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT role, content, created_at FROM messages
                   WHERE session_id = ?
                   ORDER BY created_at DESC LIMIT ?""",
                (session_id, limit),
            ).fetchall()
        return [dict(r) for r in reversed(rows)]

    def _sanitize_fts_query(self, query: str) -> str:
        """Limpia la query para FTS5 — elimina caracteres especiales que rompen la sintaxis."""
        import re
        # Remover caracteres especiales de FTS5: " ' ^ * ( ) - +
        clean = re.sub(r'["\'^*()\-+?¿¡!@#$%&=<>{}\\|]', ' ', query)
        # Colapsar espacios múltiples
        clean = re.sub(r'\s+', ' ', clean).strip()
        return clean if clean else "maia"

    def search_messages(self, query: str, limit: int = 5) -> list[dict]:
        query = self._sanitize_fts_query(query)
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT m.role, m.content, m.created_at
                   FROM messages_fts fts
                   JOIN messages m ON fts.rowid = m.rowid
                   WHERE messages_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Tareas ────────────────────────────────────────────────────────────────

    def create_task(self, session_id: str, input_text: str) -> str:
        tid = self._uid()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tasks
                   (id, session_id, input, status, created_at, updated_at)
                   VALUES (?, ?, ?, 'pending', ?, ?)""",
                (tid, session_id, input_text, now, now),
            )
        return tid

    def update_task(self, task_id: str, **kwargs) -> None:
        allowed = {"plan", "status", "result", "error", "tools_used", "duration_ms"}
        fields = {k: v for k, v in kwargs.items() if k in allowed}
        if not fields:
            return
        fields["updated_at"] = self._now()

        # Serializar listas a JSON
        for key in ("plan", "tools_used"):
            if key in fields and isinstance(fields[key], list):
                fields[key] = json.dumps(fields[key])

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [task_id]
        with self._connect() as conn:
            conn.execute(
                f"UPDATE tasks SET {set_clause} WHERE id = ?", values
            )

    def get_task(self, task_id: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM tasks WHERE id = ?", (task_id,)
            ).fetchone()
        return dict(row) if row else None

    # ── Episodios ─────────────────────────────────────────────────────────────

    def save_episode(
        self,
        summary: str,
        outcome: str,
        task_id: Optional[str] = None,
        context: Optional[dict] = None,
    ) -> str:
        eid = self._uid()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO episodes (id, task_id, summary, outcome, context, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (eid, task_id, summary, outcome,
                 json.dumps(context or {}), self._now()),
            )
        return eid

    def search_episodes(self, query: str, limit: int = 5) -> list[dict]:
        query = self._sanitize_fts_query(query)
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT e.summary, e.outcome, e.created_at
                   FROM episodes_fts fts
                   JOIN episodes e ON fts.rowid = e.rowid
                   WHERE episodes_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Lecciones ─────────────────────────────────────────────────────────────

    def save_lesson(
        self,
        problem: str,
        solution: str,
        task_id: Optional[str] = None,
        reuse_when: Optional[list] = None,
        confidence: float = 1.0,
    ) -> str:
        lid = self._uid()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO lessons
                   (id, task_id, problem, solution, reuse_when, confidence, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (lid, task_id, problem, solution,
                 json.dumps(reuse_when or []), confidence, self._now()),
            )
        return lid

    def search_lessons(self, query: str, limit: int = 5) -> list[dict]:
        query = self._sanitize_fts_query(query)
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT l.problem, l.solution, l.reuse_when, l.confidence
                   FROM lessons_fts fts
                   JOIN lessons l ON fts.rowid = l.rowid
                   WHERE lessons_fts MATCH ?
                   ORDER BY rank LIMIT ?""",
                (query, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    # ── Skills ────────────────────────────────────────────────────────────────

    def save_skill(
        self,
        name: str,
        description: str,
        code: Optional[str] = None,
        prompt: Optional[str] = None,
        tags: Optional[list] = None,
    ) -> str:
        sid = self._uid()
        now = self._now()
        with self._connect() as conn:
            conn.execute(
                """INSERT OR REPLACE INTO skills
                   (id, name, description, code, prompt, tags, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (sid, name, description, code, prompt,
                 json.dumps(tags or []), now, now),
            )
        return sid

    def get_skill(self, name: str) -> Optional[dict]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM skills WHERE name = ?", (name,)
            ).fetchone()
        return dict(row) if row else None

    # ── Tool calls ────────────────────────────────────────────────────────────

    def log_tool_call(
        self,
        tool_name: str,
        input_data: dict,
        output: Optional[str] = None,
        status: str = "ok",
        task_id: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> str:
        tcid = self._uid()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO tool_calls
                   (id, task_id, tool_name, input, output, status, duration_ms, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (tcid, task_id, tool_name, json.dumps(input_data),
                 output, status, duration_ms, self._now()),
            )
        return tcid

    # ── Contexto recuperado (para inyectar al LLM) ────────────────────────────

    def get_relevant_context(self, query: str) -> dict:
        """
        Recupera el contexto más relevante para una query.
        Esto es lo que se inyecta al LLM antes de responder.
        """
        return {
            "relevant_episodes": self.search_episodes(query, limit=3),
            "relevant_lessons": self.search_lessons(query, limit=3),
            "recent_messages": [],  # se inyecta por session_id por separado
        }

    # ── Stats ─────────────────────────────────────────────────────────────────

    def stats(self) -> dict:
        with self._connect() as conn:
            return {
                "messages": conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0],
                "tasks": conn.execute("SELECT COUNT(*) FROM tasks").fetchone()[0],
                "episodes": conn.execute("SELECT COUNT(*) FROM episodes").fetchone()[0],
                "lessons": conn.execute("SELECT COUNT(*) FROM lessons").fetchone()[0],
                "skills": conn.execute("SELECT COUNT(*) FROM skills").fetchone()[0],
                "tool_calls": conn.execute("SELECT COUNT(*) FROM tool_calls").fetchone()[0],
            }


# Instancia global — importar desde cualquier módulo
memory = MAIAMemory()
