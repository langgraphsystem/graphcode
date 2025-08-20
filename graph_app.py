# graph_app.py
from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json
from pathlib import Path
from typing import TypedDict, Optional, Iterable

from langgraph.graph import StateGraph, END

# Чекпойнтер: сначала пробуем SQLite (требует отдельный пакет),
# если модуль недоступен — используем MemorySaver (без персистентности).
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # из langgraph-checkpoint-sqlite
    _CHECKPOINTER_KIND = "sqlite"
except Exception:
    from langgraph.checkpoint.memory import MemorySaver
    _CHECKPOINTER_KIND = "memory"

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI

# ---------- ПУТИ/НАСТРОЙКИ ----------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./out")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gpt-5")
REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "300"))

# Параметры PROMPT-ADAPTER (можно переопределять в Railway Variables)
ADAPTER_MODEL = os.getenv("ADAPTER_MODEL", DEFAULT_MODEL)
CODEGEN_MODEL = os.getenv("CODEGEN_MODEL", DEFAULT_MODEL)
ADAPTER_TARGETS = os.getenv("ADAPTER_TARGETS", "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict")
ADAPTER_CONSTRAINTS = os.getenv("ADAPTER_CONSTRAINTS", "No secrets; reasonable perf; minimal deps")
ADAPTER_TEST_POLICY = os.getenv("ADAPTER_TEST_POLICY", "NO_TESTS")
ADAPTER_OUTPUT_LANG = os.getenv("ADAPTER_OUTPUT_LANG", "EN")
ADAPTER_OUTPUT_PREF = os.getenv("ADAPTER_OUTPUT_PREF", "FILES_JSON")  # FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ---------- ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ----------
EXT2LANG = {
    ".py":"python", ".js":"javascript", ".ts":"typescript", ".html":"html",
    ".css":"css", ".json":"json", ".yml":"yaml", ".yaml":"yaml",
    ".sh":"bash", ".sql":"sql", ".txt":"text",
}
def detect_language(filename: str) -> str:
    return EXT2LANG.get(Path(filename).suffix.lower(), "text")

def chat_dir(chat_id: int) -> Path:
    p = OUTPUT_DIR / str(chat_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def latest_path(chat_id: int, filename: str) -> Path:
    return chat_dir(chat_id) / f"latest-{filename}"

def ensure_latest_placeholder(chat_id: int, filename: str, language: str) -> Path:
    lp = latest_path(chat_id, filename)
    if lp.exists():
        return lp
    stubs = {
        'python':      "# -*- coding: utf-8 -*-\n# created via /create\n",
        'javascript':  "// created via /create\n",
        'typescript':  "// created via /create\n",
        'html':        "<!-- created via /create -->\n",
        'css':         "/* created via /create */\n",
        'json':        "{\n}\n",
        'yaml':        "# created via /create\n",
        'bash':        "#!/usr/bin/env bash\n# created via /create\n",
        'sql':         "-- created via /create\n",
        'text':        "",
    }
    try:
        lp.write_text(stubs.get(language, ""), encoding="utf-8")
    except Exception:
        lp.touch()
    return lp

def list_files(chat_id: int) -> list[str]:
    base = chat_dir(chat_id)
    return sorted([p.name for p in base.iterdir() if p.is_file()])

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())

def version_current_file(chat_id: int, filename: str, new_content: str) -> Path:
    lp = latest_path(chat_id, filename)
    old = lp.read_text(encoding="utf-8") if lp.exists() else ""
    if hashlib.sha256(old.encode()).hexdigest() == hashlib.sha256(new_content.encode()).hexdigest():
        return lp
    ts = time.strftime("%Y%m%d-%H%M%S")
    ver = chat_dir(chat_id) / f"{ts}-{filename}"
    ver.write_text(new_content, encoding="utf-8")
    lp.write_text(new_content, encoding="utf-8")
    return lp

# --- парсинг кода/диффа ---
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)
DIFF_BLOCK_RE = re.compile(r"```(diff|patch)\n(.*?)```", re.DOTALL | re.IGNORECASE)
UNIFIED_DIFF_HINT_RE = re.compile(r"(?m)^(--- |\+\+\+ |@@ )")
GIT_DIFF_HINT_RE = re.compile(r"(?m)^diff --git ")

def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return text.strip()
    return m.group(2).strip()

def extract_diff_and_spec(text: str) -> tuple[str, str]:
    diff_parts: list[str] = []
    def _grab(m: re.Match) -> str:
        diff_parts.append(m.group(2).strip())
        return ""
    text_wo = DIFF_BLOCK_RE.sub(_grab, text)
    diff_text = "\n\n".join(diff_parts).strip()
    if not diff_text and (GIT_DIFF_HINT_RE.search(text_wo) or UNIFIED_DIFF_HINT_RE.search(text_wo)):
        return "", text_wo.strip()
    return text_wo.strip(), diff_text

PLACEHOLDER_HINT = "created via /create"
def _is_placeholder_or_empty(content: str) -> bool:
    if not content.strip(): return True
    if PLACEHOLDER_HINT in content: return True
    return len(content.strip()) < 8

# ---------- PROMPT-ADAPTER V3 шаблон ----------
PROMPT_ADAPTER_V3 = r"""[PROMPT-ADAPTER v3 — EN-adapt, API-ready]

[STATIC RULES — cacheable]
You are a PromptAdapter for code generation via OpenAI API (GPT-5). Your job: take RAW_TASK (any language) + CONTEXT and return an API-ready package with:
- clean English developer instructions,
- user message containing both original content and an English adaptation of the *instructions/specs only*,
- a strict response contract (FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS).

Principles:
1) Role separation: put rules in developer; data/context in user. 
2) Output must follow the selected mode exactly (no extra prose).
3) Minimal necessary context: do not invent files not provided.
4) Short plan (3–6 steps), no chain-of-thought.
5) If inputs are incomplete, state careful assumptions explicitly.
6) For multi-file changes use TOOLS_CALLS with atomic tool calls.
7) Limit “verbal text” to ≤200 lines outside code/DIFF (code/DIFF not limited, but keep within model output limits).

**English Adaptation Policy (very important):**
- Translate *instructions/specs/requirements* to English concisely.
- DO NOT translate or alter: code blocks, stack traces, file paths, API names, JSON/YAML/TOML, unified diffs, quoted UI strings, or domain terms when translation could change semantics.
- If OUTPUT_LANG is specified (e.g., RU for UI text), keep user-facing strings in that language; keep identifiers/comments as requested.

[OUTPUT SCHEMA — return ONE JSON object]
{
  "messages": [
    {"role": "developer", "content": "string"},
    {"role": "user", "content": "string"}
  ],
  "response_contract": {
    "mode": "FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS",
    "structured_outputs_schema": {
      "type":"object",
      "properties":{
        "files":{
          "type":"array",
          "items":{
            "type":"object",
            "properties":{
              "path":{"type":"string"},
              "content":{"type":"string"}
            },
            "required":["path","content"],
            "additionalProperties":false
          }
        },
        "notes":{"type":"string"}
      },
      "required":["files"],
      "additionalProperties":false
    }
  },
  "runbook":{
    "plan":["step 1","step 2","step 3"],
    "commands":["<install/build/test cmds>"],
    "tests_hint":"what to cover if TEST_POLICY=TDD"
  },
  "assumptions":["list of assumptions"],
  "risks":["edge cases & risks"],
  "notes":"short CI/CD hints"
}

[HOW TO FILL]
1) messages.developer (EN): set strict code rules—language/version, style, formatter/linter, security/perf constraints, error/log policy, and the exact output format per response_contract.mode. 
2) messages.user: pack inputs with clear delimiters, include bilingual content:
   - RAW (original, any language)
   - EN_ADAPT (concise English adaptation of *instructions/specs only*)
   Use delimiters:
     <<<RAW_TASK>>>
     …original text…
     <<<END>>>
     <<<EN_ADAPT>>>
     …English adaptation of requirements/specs…
     <<<END>>>
     <<<CONTEXT:FILE path/to/file.ext>>>
     …code/logs (do NOT translate)…
     <<<END>>>
     <<<LOGS>>>
     …stack traces (do NOT translate)…
     <<<END>>>
     <<<SPEC>>>
     …requirements; translate to English inside EN_ADAPT above…
     <<<END>>>
3) response_contract.mode:
   - FILES_JSON — for new files/large rewrites (return {files:[{path,content}],notes}).
   - UNIFIED_DIFF — for minimal patches (return a valid unified diff only).
   - TOOLS_CALLS — for multi-file edits; the model must call tools.
4) runbook: 3–6 steps for build/run/test, brief.
5) assumptions/risks: explicit and minimal.

[STYLE]
- Developer message: English.
- User message: includes RAW + EN_ADAPT.
- Keep non-code prose concise (≤200 lines).
- No internal reasoning; only final results and a short plan.

[DYNAMIC INPUT — fill at call-time]
RAW_TASK: <<<RAW_TASK>>>
{RAW_TASK}
<<<END>>>
CONTEXT (optional): <<<CONTEXT>>>
{CONTEXT}
<<<END>>>
MODE: {MODE}
TARGETS: {TARGETS}
CONSTRAINTS: {CONSTRAINTS}
TEST_POLICY: {TEST_POLICY}
OUTPUT_PREF: {OUTPUT_PREF}
OUTPUT_LANG: {OUTPUT_LANG}

[NOW DO]
Construct and return ONE JSON object strictly matching OUTPUT SCHEMA, with developer in English and user containing both RAW and EN_ADAPT, following the English Adaptation Policy.
"""

def _build_context_block(chat_id: int, filename: str) -> str:
    lp = latest_path(chat_id, filename)
    if not lp.exists():
        return ""
    lang = detect_language(filename)
    code = lp.read_text(encoding="utf-8")
    return f"<<<CONTEXT:FILE {filename}>>>\n```{lang}\n{code}\n```\n<<<END>>>"

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _openai_create(model: str, input_payload):
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY")).with_options(timeout=REQUEST_TIMEOUT).responses.create(
        model=model,
        input=input_payload,
        max_output_tokens=4096,
        temperature=0.2,
    )

def _call_adapter(raw_task: str, context_block: str, mode_tag: str, output_pref: str) -> dict:
    adapter_prompt = PROMPT_ADAPTER_V3.format(
        RAW_TASK=raw_task,
        CONTEXT=context_block or "(none)",
        MODE=mode_tag,
        TARGETS=ADAPTER_TARGETS,
        CONSTRAINTS=ADAPTER_CONSTRAINTS,
        TEST_POLICY=ADAPTER_TEST_POLICY,
        OUTPUT_PREF=output_pref,
        OUTPUT_LANG=ADAPTER_OUTPUT_LANG,
    )
    resp = _openai_create(ADAPTER_MODEL, adapter_prompt)
    text = getattr(resp, "output_text", None) or extract_code(str(resp))
    try:
        return json.loads(text)
    except Exception:
        inner = extract_code(text)
        return json.loads(inner)

def _call_codegen_from_messages(messages: list[dict]) -> str:
    resp = _openai_create(CODEGEN_MODEL, messages)
    return getattr(resp, "output_text", None) or extract_code(str(resp))

def _apply_files_json(chat_id: int, active_filename: str, files_obj: list[dict]) -> Path:
    active_written = None
    for item in files_obj:
        path = (item.get("path") or "").strip().lstrip("/\\")
        content = item.get("content", "")
        if not path:
            continue
        out_path = chat_dir(chat_id) / path
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(content, encoding="utf-8")
        if Path(path).name == active_filename:
            active_written = version_current_file(chat_id, active_filename, content)
    if active_written is None and files_obj:
        first = files_obj[0]
        active_written = version_current_file(chat_id, active_filename, first.get("content", ""))
    return active_written or latest_path(chat_id, active_filename)

# ---------- AUDIT LOG ----------
AUDIT_DB = OUTPUT_DIR / "audit.db"

def _audit_connect():
    conn = sqlite3.connect(AUDIT_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            chat_id INTEGER NOT NULL,
            event_type TEXT NOT NULL,
            active_file TEXT,
            model TEXT,
            prompt TEXT,
            output_path TEXT,
            output_sha256 TEXT,
            output_bytes INTEGER,
            meta TEXT
        )
    """)
    return conn

def _truncate(s: Optional[str], limit: int = 4000) -> Optional[str]:
    if s is None: return None
    if len(s) <= limit: return s
    return s[:limit]

def _file_meta(path: Optional[Path]) -> tuple[Optional[str], Optional[int]]:
    if not path or not path.exists():
        return None, None
    b = path.stat().st_size
    h = _sha256_file(path)
    return h, b

def audit_event(chat_id: int, event_type: str, active_file: Optional[str] = None,
                model: Optional[str] = None, prompt: Optional[str] = None,
                output_path: Optional[Path] = None, meta: Optional[dict] = None):
    conn = _audit_connect()
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sha, size = _file_meta(output_path)
        conn.execute(
            "INSERT INTO events (ts, chat_id, event_type, active_file, model, prompt, output_path, output_sha256, output_bytes, meta)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, chat_id, event_type, active_file, model, _truncate(prompt), str(output_path) if output_path else None, sha, size, json.dumps(meta or {}))
        )
        conn.commit()
    finally:
        conn.close()

# ---------- СОСТОЯНИЕ ----------
class InMsg(BaseModel):
    chat_id: int
    text: str

class GraphState(TypedDict, total=False):
    chat_id: int
    input_text: str
    command: str               # CREATE | SWITCH | FILES | MODEL | RESET | GENERATE | DOWNLOAD
    arg: Optional[str]
    active_file: Optional[str]
    model: str
    reply_text: str
    file_to_send: Optional[str]

# ---------- УЗЛЫ ----------
def parse_message(state: GraphState) -> GraphState:
    text = state["input_text"].strip()
    state["command"] = "GENERATE"
    state["arg"] = None
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None
        mapping = {
            "/create":"CREATE", "/switch":"SWITCH", "/files":"FILES",
            "/model":"MODEL", "/reset":"RESET", "/download":"DOWNLOAD"
        }
        state["command"] = mapping.get(cmd, "GENERATE")
        state["arg"] = arg
    return state

def node_create(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    filename = (state.get("arg") or "main.py").strip()
    language = detect_language(filename)
    ensure_latest_placeholder(chat_id, filename, language)
    state["active_file"] = filename
    state["reply_text"] = f"✅ Файл создан/активирован: `{filename}`\n🔤 Язык: {language}"
    audit_event(chat_id, "CREATE", active_file=filename, model=state.get("model"))
    return state

def node_switch(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    filename = (state.get("arg") or "").strip()
    if not filename:
        state["reply_text"] = "Укажи имя: `/switch app.py`"
        return state
    if not latest_path(chat_id, filename).exists():
        state["reply_text"] = f"Файл `{filename}` ещё не создан. Используй `/create {filename}`."
        return state
    state["active_file"] = filename
    state["reply_text"] = f"🔀 Переключился на `{filename}`."
    audit_event(chat_id, "SWITCH", active_file=filename, model=state.get("model"))
    return state

def node_files(state: GraphState) -> GraphState:
    files = list_files(state["chat_id"])
    if not files:
        state["reply_text"] = "Файлов пока нет. Начни с `/create app.py`."
    else:
        state["reply_text"] = "🗂 Файлы:\n" + "\n".join(f"- {f}" for f in files)
    audit_event(state["chat_id"], "FILES", active_file=state.get("active_file"), model=state.get("model"))
    return state

def node_model(state: GraphState) -> GraphState:
    arg = (state.get("arg") or "").strip()
    if not arg:
        state["reply_text"] = f"Текущая модель: {state.get('model', DEFAULT_MODEL)}\nСмена: `/model gpt-5`"
        audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model=state.get("model"))
        return state
    state["model"] = arg
    state["reply_text"] = f"🧠 Модель установлена: `{arg}`"
    audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model=arg)
    return state

def node_reset(state: GraphState) -> GraphState:
    state["active_file"] = None
    state["model"] = DEFAULT_MODEL
    state["reply_text"] = "♻️ Сбросил состояние чата. Начни с `/create <filename>`."
    audit_event(state["chat_id"], "RESET")
    return state

def _infer_output_pref(raw_text: str, has_context: bool) -> str:
    if has_context and (DIFF_BLOCK_RE.search(raw_text) or UNIFIED_DIFF_HINT_RE.search(raw_text) or GIT_DIFF_HINT_RE.search(raw_text)):
        return "UNIFIED_DIFF"
    return ADAPTER_OUTPUT_PREF

@retry(stop=stop_after_attempt(3), wait=wait_exponential(1, 1, 8), retry=retry_if_exception_type(Exception), reraise=True)
def _openai_create_simple(model: str, prompt: str) -> str:
    resp = client.with_options(timeout=REQUEST_TIMEOUT).responses.create(model=model, input=prompt, max_output_tokens=4096, temperature=0.2)
    return getattr(resp, "output_text", None) or extract_code(str(resp))

def node_generate(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    active = state.get("active_file")
    if not active:
        active = "main.py"
        ensure_latest_placeholder(chat_id, active, detect_language(active))
        state["active_file"] = active

    model = state.get("model") or DEFAULT_MODEL
    raw_user_text = state["input_text"]

    lp = latest_path(chat_id, active)
    existed_before = lp.exists()
    ensure_latest_placeholder(chat_id, active, detect_language(active))
    current_text = latest_path(chat_id, active).read_text(encoding="utf-8") if latest_path(chat_id, active).exists() else ""

    language = detect_language(active)
    has_context = existed_before and not _is_placeholder_or_empty(current_text)
    context_block = _build_context_block(chat_id, active) if has_context else ""
    mode_tag = "DIFF_PATCH" if has_context else "NEW_FILE"
    output_pref = _infer_output_pref(raw_user_text, has_context)

    # Шаг 1: PROMPT-ADAPTER -> JSON с messages+контрактом
    adapter_obj = _call_adapter(raw_user_text, context_block, mode_tag, output_pref)
    messages = adapter_obj.get("messages") or []
    if not messages:
        raise RuntimeError("Adapter returned empty messages")

    # Шаг 2: Codegen
    codegen_text = _call_codegen_from_messages(messages)

    # Применение результата
    mode = (adapter_obj.get("response_contract") or {}).get("mode", output_pref)
    updated_path = None

    if (mode or "").upper() == "FILES_JSON":
        try:
            obj = json.loads(codegen_text)
        except Exception:
            obj = json.loads(extract_code(codegen_text))
        files = obj.get("files") or []
        updated_path = _apply_files_json(chat_id, active, files)

    elif (mode or "").upper() == "UNIFIED_DIFF":
        # TODO: можно подключить настоящий патч-апплайер; сейчас фоллбэк на полный файл
        code = extract_code(codegen_text)
        updated_path = version_current_file(chat_id, active, code)

    else:
        code = extract_code(codegen_text)
        updated_path = version_current_file(chat_id, active, code)

    rel = latest_path(chat_id, active).relative_to(OUTPUT_DIR)
    state["reply_text"] = (
        f"🧩 Обновил `{active}` через PROMPT-ADAPTER v3\n"
        f"Контракт: `{(mode or output_pref)}` → latest: `{rel}`\n"
        f"Отправь следующий промпт/дифф или `/files`."
    )
    audit_event(
        chat_id, "GENERATE",
        active_file=active, model=model,
        prompt=json.dumps(adapter_obj, ensure_ascii=False)[:4000],
        output_path=updated_path,
        meta={"adapter_mode": mode_tag, "contract_mode": mode or output_pref}
    )
    return state

# --------- DOWNLOAD ---------
def _iter_selected_files(base: Path, arg: Optional[str]) -> Iterable[Path]:
    files = [p for p in base.iterdir() if p.is_file()]
    if not arg:
        return sorted(files)
    arg = arg.strip().lower()
    if arg == "latest":
        return sorted([p for p in files if p.name.startswith("latest-")])
    if arg == "versions":
        return sorted([p for p in files if not p.name.startswith("latest-")])
    return sorted([p for p in files if p.name == f"latest-{arg}" or p.name.endswith(f"-{arg}")])

def _make_zip(chat_id: int, arg: Optional[str]) -> Path:
    base = chat_dir(chat_id)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = base / f"export-{ts}.zip"
    to_pack = list(_iter_selected_files(base, arg))
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in to_pack:
            z.write(p, arcname=p.name)
    return out

def node_download(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    arg = state.get("arg")
    z = _make_zip(chat_id, arg)
    state["file_to_send"] = str(z)
    sel = arg or "all"
    state["reply_text"] = f"📦 Подготовил архив `{z.name}` ({sel})."
    audit_event(chat_id, "DOWNLOAD", active_file=state.get("active_file"), model=state.get("model"), output_path=z)
    return state

# ---------- РОУТЕР/СБОРКА ----------
def router(state: GraphState) -> str:
    return state["command"]

def build_app():
    sg = StateGraph(GraphState)
    sg.add_node("parse", parse_message)
    sg.add_node("CREATE", node_create)
    sg.add_node("SWITCH", node_switch)
    sg.add_node("FILES", node_files)
    sg.add_node("MODEL", node_model)
    sg.add_node("RESET", node_reset)
    sg.add_node("GENERATE", node_generate)
    sg.add_node("DOWNLOAD", node_download)

    sg.set_entry_point("parse")
    sg.add_conditional_edges("parse", router, {
        "CREATE": "CREATE",
        "SWITCH": "SWITCH",
        "FILES": "FILES",
        "MODEL": "MODEL",
        "RESET": "RESET",
        "GENERATE": "GENERATE",
        "DOWNLOAD": "DOWNLOAD",
    })
    for node in ("CREATE","SWITCH","FILES","MODEL","RESET","GENERATE","DOWNLOAD"):
        sg.add_edge(node, END)

    if _CHECKPOINTER_KIND == "sqlite":
        checkpointer = SqliteSaver.from_file(OUTPUT_DIR / "langgraph.db")
    else:
        checkpointer = MemorySaver()
    return sg.compile(checkpointer=checkpointer)

APP = build_app()
