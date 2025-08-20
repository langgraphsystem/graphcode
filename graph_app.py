```python
from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json, logging
from pathlib import Path
from typing import TypedDict, Optional, Iterable
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI, APIError

# ---------- НАСТРОЙКА ЛОГИРОВАНИЯ ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- ПУТИ/НАСТРОЙКИ ----------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./out")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_MODEL = "gpt-5"
VALID_MODELS = {"gpt-5"}
REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "300"))

ADAPTER_MODEL = os.getenv("ADAPTER_MODEL", DEFAULT_MODEL)
CODEGEN_MODEL = os.getenv("CODEGEN_MODEL", DEFAULT_MODEL)
ADAPTER_TARGETS = os.getenv("ADAPTER_TARGETS", "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict")
ADAPTER_CONSTRAINTS = os.getenv("ADAPTER_CONSTRAINTS", "No secrets; reasonable perf; minimal deps")
ADAPTER_TEST_POLICY = os.getenv("ADAPTER_TEST_POLICY", "NO_TESTS")
ADAPTER_OUTPUT_LANG = os.getenv("ADAPTER_OUTPUT_LANG", "EN")
ADAPTER_OUTPUT_PREF = os.getenv("ADAPTER_OUTPUT_PREF", "FILES_JSON")

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=REQUEST_TIMEOUT
)

# ---------- ВСПОМОГАТЕЛЬНЫЕ УТИЛИТЫ ----------
EXT2LANG = {
    ".py":"python", ".js":"javascript", ".ts":"typescript", ".html":"html",
    ".css":"css", ".json":"json", ".yml":"yaml", ".yaml":"yaml",
    ".sh":"bash", ".sql":"sql", ".txt":"text", ".rs":"rust",
    ".go":"go", ".java":"java", ".cpp":"cpp", ".c":"c",
}

def detect_language(filename: str) -> str:
    return EXT2LANG.get(Path(filename).suffix.lower(), "text")

def sanitize_filename(filename: str) -> str:
    unsafe_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    clean_name = filename
    for char in unsafe_chars:
        clean_name = clean_name.replace(char, '_')
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length - len(ext)] + ext
    if not clean_name or clean_name.strip() in ['.', '..']:
        clean_name = 'unnamed_file'
    return clean_name.strip()

def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    try:
        clean_path = relative_path.strip().lstrip('/\\')
        if '..' in clean_path or clean_path.startswith('/'):
            logger.warning(f"Potentially unsafe path rejected: {relative_path}")
            return None
        full_path = base_dir / clean_path
        try:
            full_path.resolve().relative_to(base_dir.resolve())
        except ValueError:
            logger.warning(f"Path outside base directory rejected: {relative_path}")
            return None
        return full_path
    except Exception as e:
        logger.error(f"Error processing path: {e}")
        return None

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
        'rust':        "// created via /create\n",
        'go':          "// created via /create\npackage main\n",
        'java':        "// created via /create\npublic class Main {}\n",
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
    logger.info(f"Created version: {ver.name}")
    return lp

# --- Парсинг кода/диффа ---
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
    if not content.strip(): 
        return True
    if PLACEHOLDER_HINT in content: 
        return True
    return len(content.strip()) < 8

# ---------- OpenAI API ВЫЗОВЫ ----------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=2, min=2, max=16),
    retry=retry_if_exception_type((APIError, ConnectionError)),
    reraise=True
)
def _openai_create(model: str, input_payload):
    model = "gpt-5"
    if isinstance(input_payload, list):
        messages = input_payload
    elif isinstance(input_payload, dict) and "messages" in input_payload:
        messages = input_payload["messages"]
    else:
        messages = [{"role": "user", "content": str(input_payload)}]
    
    try:
        logger.info(f"Calling OpenAI API with model {model}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=2048,
            temperature=0.2,
        )
        return response
    except APIError as e:
        logger.error(f"OpenAI API error: {e}, HTTP status: {e.http_status}, Headers: {e.headers}")
        if e.http_status == 429:
            logger.warning("Rate limit error detected")
        raise
    except ConnectionError as e:
        logger.error(f"Network error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in OpenAI call: {e}")
        raise

# ---------- PROMPT-ADAPTER V3 ШАБЛОН ----------
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
    },
    "tools":[
      {
        "name":"apply_patch",
        "description":"Apply a unified diff to a file",
        "parameters":{
          "type":"object",
          "properties":{
            "file_path":{"type":"string"},
            "patch":{"type":"string"}
          },
          "required":["file_path","patch"],
          "additionalProperties":false
        }
      },
      {
        "name":"write_file",
        "description":"Write or overwrite a file",
        "parameters":{
          "type":"object",
          "properties":{
            "path":{"type":"string"},
            "content":{"type":"string"}
          },
          "required":["path","content"],
          "additionalProperties":false
        }
      }
    ]
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
CONTEXT (optional): <<<CONTEXT>>>
MODE: <<<MODE>>>                          // NEW_FILE | DIFF_PATCH | MULTIFILE_TOOLS
TARGETS: <<<TARGETS>>>                    // lang versions/linters/deps
CONSTRAINTS: <<<CONSTRAINTS>>>            // perf/security/licenses
TEST_POLICY: <<<TEST_POLICY>>>            // TDD | NO_TESTS
OUTPUT_PREF: <<<OUTPUT_PREF>>>            // FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS
OUTPUT_LANG: <<<OUTPUT_LANG>>>            // e.g., RU for UI strings

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

def _render_adapter_prompt(raw_task: str, context_block: str, mode_tag: str, targets: str, constraints: str, test_policy: str, output_pref: str, output_lang: str) -> str:
    prompt = PROMPT_ADAPTER_V3
    prompt = prompt.replace("<<<RAW_TASK>>>", raw_task)
    prompt = prompt.replace("<<<CONTEXT>>>", context_block)
    prompt = prompt.replace("<<<MODE>>>", mode_tag)
    prompt = prompt.replace("<<<TARGETS>>>", targets)
    prompt = prompt.replace("<<<CONSTRAINTS>>>", constraints)
    prompt = prompt.replace("<<<TEST_POLICY>>>", test_policy)
    prompt = prompt.replace("<<<OUTPUT_PREF>>>", output_pref)
    prompt = prompt.replace("<<<OUTPUT_LANG>>>", output_lang)
    return prompt

def _call_adapter_and_codegen(raw_task: str, context_block: str, mode_tag: str, output_pref: str) -> tuple[str, dict]:
    adapter_prompt = _render_adapter_prompt(
        raw_task, context_block, mode_tag, ADAPTER_TARGETS, ADAPTER_CONSTRAINTS,
        ADAPTER_TEST_POLICY, output_pref, ADAPTER_OUTPUT_LANG
    )
    messages = [
        {"role": "system", "content": "You are a code generation assistant using GPT-5. Follow the PROMPT-ADAPTER v3 rules to generate code based on the provided task and context."},
        {"role": "user", "content": adapter_prompt}
    ]
    try:
        response = _openai_create("gpt-5", messages)
        if not hasattr(response, 'choices') or not response.choices:
            raise ValueError("Invalid response format from OpenAI")
        text = response.choices[0].message.content
        try:
            adapter_obj = json.loads(text)
        except json.JSONDecodeError:
            code_block_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
            if code_block_match:
                adapter_obj = json.loads(code_block_match.group(1))
            else:
                cleaned = extract_code(text)
                adapter_obj = json.loads(cleaned)
        return text, adapter_obj
    except Exception as e:
        logger.error(f"Combined adapter/codegen call failed: {e}")
        return "# Error generating code", {
            "messages": [
                {"role": "system", "content": "Generate code based on user request using GPT-5"},
                {"role": "user", "content": raw_task}
            ],
            "response_contract": {"mode": output_pref}
        }

def _apply_files_json(chat_id: int, active_filename: str, files_obj: list[dict]) -> Path:
    active_written = None
    base_dir = chat_dir(chat_id)
    
    for item in files_obj:
        raw_path = item.get("path", "").strip()
        content = item.get("content", "")
        
        if not raw_path:
            continue
        
        safe_output_path = safe_path_join(base_dir, raw_path)
        if safe_output_path is None:
            logger.warning(f"Skipping unsafe path: {raw_path}")
            continue
        
        try:
            safe_output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_output_path.write_text(content, encoding="utf-8")
            logger.info(f"Written file: {safe_output_path}")
            if Path(raw_path).name == active_filename:
                active_written = version_current_file(chat_id, active_filename, content)
        except Exception as e:
            logger.error(f"Failed to write file {raw_path}: {e}")
            continue
    
    if active_written is None and files_obj:
        first = files_obj[0]
        content = first.get("content", "")
        active_written = version_current_file(chat_id, active_filename, content)
    
    return active_written or latest_path(chat_id, active_filename)

def _infer_output_pref(raw_text: str, has_context: bool) -> str:
    if has_context and (DIFF_BLOCK_RE.search(raw_text) or UNIFIED_DIFF_HINT_RE.search(raw_text) or GIT_DIFF_HINT_RE.search(raw_text)):
        return "UNIFIED_DIFF"
    return ADAPTER_OUTPUT_PREF

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
            "INSERT INTO events (ts, chat_id, event_type, event_type, active_file, model, prompt, output_path, output_sha256, output_bytes, meta)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (ts, chat_id, event_type, active_file, model, _truncate(prompt), str(output_path) if output_path else None, sha, size, json.dumps(meta or {}))
        )
        conn.commit()
    except sqlite3.Error as e:
        logger.error(f"Failed to write audit event: {e}")
    finally:
        conn.close()

# ---------- СОСТОЯНИЕ ----------
class InMsg(BaseModel):
    chat_id: int
    text: str

class GraphState(TypedDict, total=False):
    chat_id: int
    input_text: str
    command: str
    arg: Optional[str]
    active_file: Optional[str]
    model: str
    reply_text: str
    file_to_send: Optional[str]

# ---------- ДЕКОРАТОР ДЛЯ БЕЗОПАСНОСТИ УЗЛОВ ----------
def safe_node(func):
    def wrapper(state: GraphState) -> GraphState:
        try:
            return func(state)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            error_msg = f"❌ Ошибка в {func.__name__}: "
            if isinstance(e, APIError) and e.http_status == 429:
                error_msg += "Превышен лимит запросов к API"
            elif "api_key" in str(e).lower():
                error_msg += "Проблема с API ключом OpenAI"
            elif "timeout" in str(e).lower():
                error_msg += "Превышено время ожидания ответа"
            elif "json" in str(e).lower():
                error_msg += "Ошибка парсинга ответа от AI"
            else:
                error_msg += str(e)[:200]
            state["reply_text"] = error_msg
            return state
    wrapper.__name__ = func.__name__
    return wrapper

# ---------- УЗЛЫ ГРАФА ----------
def parse_message(state: GraphState) -> GraphState:
    text = state["input_text"].strip()
    state["command"] = "GENERATE"
    state["arg"] = None
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None
        mapping = {
            "/create": "CREATE",
            "/switch": "SWITCH", 
            "/files": "FILES",
            "/model": "MODEL",
            "/reset": "RESET",
            "/download": "DOWNLOAD"
        }
        state["command"] = mapping.get(cmd, "GENERATE")
        state["arg"] = arg
    return state

@safe_node
def node_create(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    raw_filename = (state.get("arg") or "main.py").strip()
    filename = sanitize_filename(raw_filename)
    language = detect_language(filename)
    ensure_latest_placeholder(chat_id, filename, language)
    state["active_file"] = filename
    state["reply_text"] = f"✅ Файл создан/активирован: {filename}\n🔤 Язык: {language}"
    if filename != raw_filename:
        state["reply_text"] += f"\n⚠️ Имя файла было очищено от небезопасных символов"
    audit_event(chat_id, "CREATE", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_switch(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    filename = (state.get("arg") or "").strip()
    if not filename:
        state["reply_text"] = "Укажи имя: /switch app.py"
        return state
    filename = sanitize_filename(filename)
    if not latest_path(chat_id, filename).exists():
        state["reply_text"] = f"Файл {filename} ещё не создан. Используй /create {filename}."
        return state
    state["active_file"] = filename
    state["reply_text"] = f"🔀 Переключился на {filename}."
    audit_event(chat_id, "SWITCH", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_files(state: GraphState) -> GraphState:
    files = list_files(state["chat_id"])
    if not files:
        state["reply_text"] = "Файлов пока нет. Начни с /create app.py."
    else:
        state["reply_text"] = "🗂 Файлы:\n" + "\n".join(f"- {f}" for f in files)
    audit_event(state["chat_id"], "FILES", active_file=state.get("active_file"), model=state.get("model"))
    return state

@safe_node
def node_model(state: GraphState) -> GraphState:
    state["reply_text"] = (
        f"🧠 Используется модель: GPT-5\n\n"
        f"ℹ️ Это единственная поддерживаемая модель.\n"
        f"Все запросы автоматически направляются на GPT-5."
    )
    audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model="gpt-5")
    return state

@safe_node
def node_reset(state: GraphState) -> GraphState:
    state["active_file"] = None
    state["model"] = DEFAULT_MODEL
    state["reply_text"] = "♻️ Сбросил состояние чата. Начни с /create <filename>."
    audit_event(state["chat_id"], "RESET")
    return state

@safe_node
def node_generate(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    active = state.get("active_file")
    
    if not active:
        active = "main.py"
        ensure_latest_placeholder(chat_id, active, detect_language(active))
        state["active_file"] = active
        logger.info(f"Auto-created file: {active}")

    model = "gpt-5"
    state["model"] = model
    raw_user_text = state["input_text"][:10000]  # Cap input
    state["reply_text"] = f"📥 Получен промпт: {raw_user_text[:100]}"  # Step 1: Prompt received
    
    lp = latest_path(chat_id, active)
    existed_before = lp.exists()
    ensure_latest_placeholder(chat_id, active, detect_language(active))
    current_text = lp.read_text(encoding="utf-8")[:10000] if lp.exists() else ""
    language = detect_language(active)
    has_context = existed_before and not _is_placeholder_or_empty(current_text)
    context_block = _build_context_block(chat_id, active) if has_context else ""
    mode_tag = "DIFF_PATCH" if has_context else "NEW_FILE"
    output_pref = _infer_output_pref(raw_user_text, has_context)

    logger.info(f"Generating for {active} with GPT-5, mode {mode_tag}")

    try:
        # Step 2: Form universal prompt and show it
        codegen_text, adapter_obj = _call_adapter_and_codegen(raw_user_text, context_block, mode_tag, output_pref)
        user_message = (adapter_obj.get("messages") or [{}])[-1].get("content", "No user message")
        state["reply_text"] += f"\n📝 Сформирован универсальный промпт:\nИсходный: {raw_user_text[:100]}\nАдаптированный:\n```{user_message[:200]}```"

        # Step 3: API call sent and response received
        state["reply_text"] += f"\n🚀 Отправлен запрос в API GPT-5, ответ получен"

        if not codegen_text or codegen_text == "# Error generating code":
            raise ValueError("Failed to generate code")

        # Step 4: Process response for code generation
        state["reply_text"] += f"\n🔍 Получен готовый промпт, обработка для генерации кода:\n```{codegen_text[:100]}```"
        
        mode = (adapter_obj.get("response_contract") or {}).get("mode", output_pref)
        updated_path = None

        if mode.upper() == "FILES_JSON":
            try:
                obj = json.loads(codegen_text)
            except json.JSONDecodeError:
                obj = json.loads(extract_code(codegen_text))
            files = obj.get("files") or []
            if not files:
                raise ValueError("No files in response")
            updated_path = _apply_files_json(chat_id, active, files)
        elif mode.upper() == "UNIFIED_DIFF":
            logger.info("UNIFIED_DIFF mode - using full file replacement")
            code = extract_code(codegen_text)
            updated_path = version_current_file(chat_id, active, code)
        else:
            code = extract_code(codegen_text)
            updated_path = version_current_file(chat_id, active, code)

        rel = latest_path(chat_id, active).relative_to(OUTPUT_DIR)
        state["reply_text"] += (
            f"\n✅ Обновил {active} через PROMPT-ADAPTER v3\n"
            f"🧠 Модель: GPT-5\n"
            f"📁 Контракт: {mode}\n"
            f"💾 Сохранено: {rel}\n\n"
            f"Отправь следующий промпт или используй команды."
        )
        
        audit_event(
            chat_id, "GENERATE",
            active_file=active, 
            model="gpt-5",
            prompt=raw_user_text[:4000],
            output_path=updated_path,
            meta={"adapter_mode": mode_tag, "contract_mode": mode, "success": True}
        )
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        audit_event(
            chat_id, "GENERATE_ERROR",
            active_file=active,
            model="gpt-5",
            meta={"error": str(e)[:500]}
        )
        raise

    return state

def _make_zip(chat_id: int, selector: Optional[str] = None) -> Path:
    base = chat_dir(chat_id)
    ts = time.strftime("%Y%m%d_%H%M%S")
    zip_path = OUTPUT_DIR / f"chat_{chat_id}_{ts}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        if selector:
            safe_selector = sanitize_filename(selector)
            p = base / safe_selector
            if p.exists() and p.is_file():
                zf.write(p, arcname=p.name)
            else:
                for fp in base.glob(f"*{safe_selector}*"):
                    if fp.is_file():
                        zf.write(fp, arcname=fp.name)
        else:
            for fp in base.iterdir():
                if fp.is_file():
                    zf.write(fp, arcname=fp.name)
    return zip_path

@safe_node
def node_download(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    arg = state.get("arg")
    
    try:
        z = _make_zip(chat_id, arg)
        state["file_to_send"] = str(z)
        sel = arg or "all"
        state["reply_text"] = f"📦 Подготовил архив {z.name} ({sel})."
        audit_event(chat_id, "DOWNLOAD", active_file=state.get("active_file"), model="gpt-5", output_path=z)
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        state["reply_text"] = f"❌ Не удалось создать архив: {str(e)[:200]}"
    
    return state

# ---------- РОУТЕР И СБОРКА ПРИЛОЖЕНИЯ ----------
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
    for node in ("CREATE", "SWITCH", "FILES", "MODEL", "RESET", "GENERATE", "DOWNLOAD"):
        sg.add_edge(node, END)
    checkpointer = MemorySaver()
    logger.info("Using MemorySaver (non-persistent)")
    compiled_app = sg.compile(checkpointer=checkpointer)
    logger.info("LangGraph application compiled successfully")
    return compiled_app

# ---------- ИНИЦИАЛИЗАЦИЯ ----------
APP = build_app()
__all__ = ['APP', 'DEFAULT_MODEL', 'VALID_MODELS']
logger.info(f"Graph app initialized. Model: GPT-5 only. Output dir: {OUTPUT_DIR}")
```
