# graph_app.py
from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json, logging
from pathlib import Path
from typing import TypedDict, Optional, Iterable

from langgraph.graph import StateGraph, END

# Чекпойнтер: сначала пробуем SQLite, если недоступен — используем MemorySaver
try:
    from langgraph.checkpoint.sqlite import SqliteSaver
    _CHECKPOINTER_KIND = "sqlite"
except Exception:
    from langgraph.checkpoint.memory import MemorySaver
    _CHECKPOINTER_KIND = "memory"

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI

# ---------- НАСТРОЙКА ЛОГИРОВАНИЯ ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- ПУТИ/НАСТРОЙКИ ----------
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "./out")).resolve()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ВАЖНО: Используем только GPT-5 как основную модель
DEFAULT_MODEL = "gpt-5"
VALID_MODELS = {"gpt-5"}  # Только GPT-5 поддерживается

REQUEST_TIMEOUT = int(os.getenv("OPENAI_REQUEST_TIMEOUT", "300"))

# Параметры PROMPT-ADAPTER (можно переопределять в Railway Variables)
ADAPTER_MODEL = os.getenv("ADAPTER_MODEL", DEFAULT_MODEL)
CODEGEN_MODEL = os.getenv("CODEGEN_MODEL", DEFAULT_MODEL)
ADAPTER_TARGETS = os.getenv("ADAPTER_TARGETS", "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict")
ADAPTER_CONSTRAINTS = os.getenv("ADAPTER_CONSTRAINTS", "No secrets; reasonable perf; minimal deps")
ADAPTER_TEST_POLICY = os.getenv("ADAPTER_TEST_POLICY", "NO_TESTS")
ADAPTER_OUTPUT_LANG = os.getenv("ADAPTER_OUTPUT_LANG", "EN")
ADAPTER_OUTPUT_PREF = os.getenv("ADAPTER_OUTPUT_PREF", "FILES_JSON")  # FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS

# Глобальный клиент OpenAI
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
    """Определение языка по расширению файла"""
    return EXT2LANG.get(Path(filename).suffix.lower(), "text")

def sanitize_filename(filename: str) -> str:
    """Очистка имени файла от небезопасных символов"""
    unsafe_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|', '\x00']
    clean_name = filename
    for char in unsafe_chars:
        clean_name = clean_name.replace(char, '_')
    
    # Ограничиваем длину
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length - len(ext)] + ext
    
    # Если имя пустое после очистки
    if not clean_name or clean_name.strip() in ['.', '..']:
        clean_name = 'unnamed_file'
    
    return clean_name.strip()

def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    """Безопасное объединение путей с проверкой выхода за пределы базовой директории"""
    try:
        clean_path = relative_path.strip().lstrip('/\\')
        
        # Проверяем на path traversal
        if '..' in clean_path or clean_path.startswith('/'):
            logger.warning(f"Potentially unsafe path rejected: {relative_path}")
            return None
        
        full_path = base_dir / clean_path
        
        # Проверяем, что путь находится внутри базовой директории
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
    """Получение директории чата"""
    p = OUTPUT_DIR / str(chat_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def latest_path(chat_id: int, filename: str) -> Path:
    """Путь к последней версии файла"""
    return chat_dir(chat_id) / f"latest-{filename}"

def ensure_latest_placeholder(chat_id: int, filename: str, language: str) -> Path:
    """Создание файла-заглушки если его нет"""
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
    """Список файлов в директории чата"""
    base = chat_dir(chat_id)
    return sorted([p.name for p in base.iterdir() if p.is_file()])

def _sha256_bytes(data: bytes) -> str:
    """SHA256 хеш байтов"""
    return hashlib.sha256(data).hexdigest()

def _sha256_file(path: Path) -> str:
    """SHA256 хеш файла"""
    return _sha256_bytes(path.read_bytes())

def version_current_file(chat_id: int, filename: str, new_content: str) -> Path:
    """Версионирование файла с временной меткой"""
    lp = latest_path(chat_id, filename)
    old = lp.read_text(encoding="utf-8") if lp.exists() else ""
    
    # Если содержимое не изменилось, не создаем новую версию
    if hashlib.sha256(old.encode()).hexdigest() == hashlib.sha256(new_content.encode()).hexdigest():
        return lp
    
    # Создаем версию с временной меткой
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
    """Извлечение кода из markdown блока"""
    m = CODE_BLOCK_RE.search(text)
    if not m:
        return text.strip()
    return m.group(2).strip()

def extract_diff_and_spec(text: str) -> tuple[str, str]:
    """Извлечение diff и спецификации из текста"""
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
    """Проверка, является ли файл заглушкой"""
    if not content.strip(): 
        return True
    if PLACEHOLDER_HINT in content: 
        return True
    return len(content.strip()) < 8

# ---------- OpenAI API ВЫЗОВЫ ----------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def _openai_create(model: str, input_payload):
    """
    Универсальная функция для вызова OpenAI API с GPT-5
    """
    # Всегда используем GPT-5
    model = "gpt-5"
    
    # Если это список сообщений или словарь с messages
    if isinstance(input_payload, list):
        messages = input_payload
    elif isinstance(input_payload, dict) and "messages" in input_payload:
        messages = input_payload["messages"]
    else:
        # Если это строка, оборачиваем в формат chat completion
        messages = [{"role": "user", "content": str(input_payload)}]
    
    try:
        logger.info(f"Calling OpenAI API with model {model}")
        response = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=4096,
            temperature=0.2,
        )
        
        return response
        
    except Exception as e:
        logger.error(f"OpenAI API error: {e}")
        raise

# ---------- PROMPT-ADAPTER V3 ШАБЛОН ----------
PROMPT_ADAPTER_V3 = r"""[PROMPT-ADAPTER v3 — EN-adapt, API-ready for GPT-5]
[STATIC RULES — cacheable]
You are a PromptAdapter for code generation via OpenAI API (GPT-5). Your job: take RAW_TASK (any language) + CONTEXT and return an API-ready package with:
- clean English developer instructions,
- user message containing both original content and an English adaptation,
- a strict response contract (FILES_JSON | UNIFIED_DIFF | TOOLS_CALLS).

Principles:
1) Role separation: put rules in developer; data/context in user. 
2) Output must follow the selected mode exactly (no extra prose).
3) Minimal necessary context: do not invent files not provided.
4) Short plan (3–6 steps), no chain-of-thought.
5) If inputs are incomplete, state careful assumptions explicitly.
6) For multi-file changes use TOOLS_CALLS with atomic tool calls.
7) Limit "verbal text" to ≤200 lines outside code/DIFF.

**English Adaptation Policy:**
- Translate instructions/specs/requirements to English concisely.
- DO NOT translate: code blocks, stack traces, file paths, API names, JSON/YAML, diffs, UI strings, domain terms.
- Keep user-facing strings in OUTPUT_LANG if specified.

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

[DYNAMIC INPUT]
RAW_TASK: <<<RAW_TASK>>>
{RAW_TASK}
<<<END>>>
CONTEXT: <<<CONTEXT>>>
{CONTEXT}
<<<END>>>
MODE: {MODE}
TARGETS: {TARGETS}
CONSTRAINTS: {CONSTRAINTS}
TEST_POLICY: {TEST_POLICY}
OUTPUT_PREF: {OUTPUT_PREF}
OUTPUT_LANG: {OUTPUT_LANG}

[NOW DO]
Construct and return ONE JSON object strictly matching OUTPUT SCHEMA.
"""

def _build_context_block(chat_id: int, filename: str) -> str:
    """Построение блока контекста для файла"""
    lp = latest_path(chat_id, filename)
    if not lp.exists():
        return ""
    lang = detect_language(filename)
    code = lp.read_text(encoding="utf-8")
    return f"<<<CONTEXT:FILE {filename}>>>\n```{lang}\n{code}\n```\n<<<END>>>"

def _call_adapter(raw_task: str, context_block: str, mode_tag: str, output_pref: str) -> dict:
    """Вызов PROMPT-ADAPTER для подготовки промпта"""
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
    
    try:
        # Вызываем OpenAI API с GPT-5
        response = _openai_create(ADAPTER_MODEL, adapter_prompt)
        
        # Извлекаем текст из ответа
        if hasattr(response, 'choices') and response.choices:
            text = response.choices[0].message.content
        else:
            raise ValueError("Invalid response format from OpenAI")
        
        # Пытаемся распарсить JSON
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            # Если JSON обернут в markdown блок кода
            code_block_match = re.search(r'```(?:json)?\n(.*?)\n```', text, re.DOTALL)
            if code_block_match:
                return json.loads(code_block_match.group(1))
            
            # Последняя попытка - извлечь через extract_code
            inner = extract_code(text)
            return json.loads(inner)
            
    except Exception as e:
        logger.error(f"Adapter call failed: {e}")
        # Возвращаем минимальный валидный объект
        return {
            "messages": [
                {"role": "system", "content": "Generate code based on user request using GPT-5"},
                {"role": "user", "content": raw_task}
            ],
            "response_contract": {"mode": output_pref}
        }

def _call_codegen_from_messages(messages: list[dict]) -> str:
    """Генерация кода на основе подготовленных сообщений"""
    try:
        response = _openai_create(CODEGEN_MODEL, messages)
        
        # Извлекаем текст из ответа
        if hasattr(response, 'choices') and response.choices:
            text = response.choices[0].message.content
            return text
        else:
            raise ValueError("Invalid response format from OpenAI")
            
    except Exception as e:
        logger.error(f"Codegen call failed: {e}")
        return "# Error generating code"

def _apply_files_json(chat_id: int, active_filename: str, files_obj: list[dict]) -> Path:
    """Применение FILES_JSON результата с безопасной обработкой путей"""
    active_written = None
    base_dir = chat_dir(chat_id)
    
    for item in files_obj:
        raw_path = item.get("path", "").strip()
        content = item.get("content", "")
        
        if not raw_path:
            continue
        
        # Безопасная обработка пути
        safe_output_path = safe_path_join(base_dir, raw_path)
        if safe_output_path is None:
            logger.warning(f"Skipping unsafe path: {raw_path}")
            continue
        
        try:
            # Создаем директории если нужно
            safe_output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Записываем содержимое
            safe_output_path.write_text(content, encoding="utf-8")
            logger.info(f"Written file: {safe_output_path}")
            
            # Проверяем, является ли это активным файлом
            if Path(raw_path).name == active_filename:
                active_written = version_current_file(chat_id, active_filename, content)
                
        except Exception as e:
            logger.error(f"Failed to write file {raw_path}: {e}")
            continue
    
    # Если активный файл не был записан, но есть файлы - используем первый
    if active_written is None and files_obj:
        first = files_obj[0]
        content = first.get("content", "")
        active_written = version_current_file(chat_id, active_filename, content)
    
    return active_written or latest_path(chat_id, active_filename)

def _infer_output_pref(raw_text: str, has_context: bool) -> str:
    """Определение предпочтительного формата вывода"""
    if has_context and (DIFF_BLOCK_RE.search(raw_text) or UNIFIED_DIFF_HINT_RE.search(raw_text) or GIT_DIFF_HINT_RE.search(raw_text)):
        return "UNIFIED_DIFF"
    return ADAPTER_OUTPUT_PREF

# ---------- AUDIT LOG ----------
AUDIT_DB = OUTPUT_DIR / "audit.db"

def _audit_connect():
    """Подключение к БД аудита"""
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
    """Обрезка строки до лимита"""
    if s is None: return None
    if len(s) <= limit: return s
    return s[:limit]

def _file_meta(path: Optional[Path]) -> tuple[Optional[str], Optional[int]]:
    """Метаданные файла (хеш и размер)"""
    if not path or not path.exists():
        return None, None
    b = path.stat().st_size
    h = _sha256_file(path)
    return h, b

def audit_event(chat_id: int, event_type: str, active_file: Optional[str] = None,
                model: Optional[str] = None, prompt: Optional[str] = None,
                output_path: Optional[Path] = None, meta: Optional[dict] = None):
    """Запись события в аудит лог"""
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

# ---------- ДЕКОРАТОР ДЛЯ БЕЗОПАСНОСТИ УЗЛОВ ----------
def safe_node(func):
    """Декоратор для безопасной обработки ошибок в узлах графа"""
    def wrapper(state: GraphState) -> GraphState:
        try:
            return func(state)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            
            # Формируем понятное сообщение об ошибке
            error_msg = f"❌ Ошибка в {func.__name__}: "
            
            if "api_key" in str(e).lower():
                error_msg += "Проблема с API ключом OpenAI"
            elif "rate" in str(e).lower():
                error_msg += "Превышен лимит запросов к API"
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
    """Парсинг входящего сообщения для определения команды"""
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
    """Создание нового файла с безопасным именем"""
    chat_id = state["chat_id"]
    raw_filename = (state.get("arg") or "main.py").strip()
    
    # Очищаем имя файла
    filename = sanitize_filename(raw_filename)
    
    # Определяем язык
    language = detect_language(filename)
    
    # Создаем файл-заглушку
    ensure_latest_placeholder(chat_id, filename, language)
    
    state["active_file"] = filename
    state["reply_text"] = f"✅ Файл создан/активирован: `{filename}`\n🔤 Язык: {language}"
    
    # Если имя было изменено, сообщаем
    if filename != raw_filename:
        state["reply_text"] += f"\n⚠️ Имя файла было очищено от небезопасных символов"
    
    audit_event(chat_id, "CREATE", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_switch(state: GraphState) -> GraphState:
    """Переключение на существующий файл"""
    chat_id = state["chat_id"]
    filename = (state.get("arg") or "").strip()
    
    if not filename:
        state["reply_text"] = "Укажи имя: `/switch app.py`"
        return state
    
    # Очищаем имя файла
    filename = sanitize_filename(filename)
    
    if not latest_path(chat_id, filename).exists():
        state["reply_text"] = f"Файл `{filename}` ещё не создан. Используй `/create {filename}`."
        return state
    
    state["active_file"] = filename
    state["reply_text"] = f"🔀 Переключился на `{filename}`."
    audit_event(chat_id, "SWITCH", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_files(state: GraphState) -> GraphState:
    """Показать список файлов"""
    files = list_files(state["chat_id"])
    if not files:
        state["reply_text"] = "Файлов пока нет. Начни с `/create app.py`."
    else:
        state["reply_text"] = "🗂 Файлы:\n" + "\n".join(f"- {f}" for f in files)
    audit_event(state["chat_id"], "FILES", active_file=state.get("active_file"), model=state.get("model"))
    return state

@safe_node
def node_model(state: GraphState) -> GraphState:
    """Показать информацию о модели (только GPT-5)"""
    state["reply_text"] = (
        f"🧠 Используется модель: `GPT-5`\n\n"
        f"ℹ️ Это единственная поддерживаемая модель.\n"
        f"Все запросы автоматически направляются на GPT-5."
    )
    audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model="gpt-5")
    return state

@safe_node
def node_reset(state: GraphState) -> GraphState:
    """Сброс состояния чата"""
    state["active_file"] = None
    state["model"] = DEFAULT_MODEL
    state["reply_text"] = "♻️ Сбросил состояние чата. Начни с `/create <filename>`."
    audit_event(state["chat_id"], "RESET")
    return state

@safe_node
def node_generate(state: GraphState) -> GraphState:
    """Генерация кода с помощью GPT-5"""
    chat_id = state["chat_id"]
    active = state.get("active_file")
    
    # Автоматически создаем файл если его нет
    if not active:
        active = "main.py"
        ensure_latest_placeholder(chat_id, active, detect_language(active))
        state["active_file"] = active
        logger.info(f"Auto-created file: {active}")

    # Всегда используем GPT-5
    model = "gpt-5"
    state["model"] = model
    
    raw_user_text = state["input_text"]
    
    # Проверяем наличие текущего файла
    lp = latest_path(chat_id, active)
    existed_before = lp.exists()
    ensure_latest_placeholder(chat_id, active, detect_language(active))
    current_text = lp.read_text(encoding="utf-8") if lp.exists() else ""

    language = detect_language(active)
    has_context = existed_before and not _is_placeholder_or_empty(current_text)
    context_block = _build_context_block(chat_id, active) if has_context else ""
    mode_tag = "DIFF_PATCH" if has_context else "NEW_FILE"
    output_pref = _infer_output_pref(raw_user_text, has_context)

    logger.info(f"Generating for {active} with GPT-5, mode {mode_tag}")

    try:
        # Шаг 1: PROMPT-ADAPTER
        adapter_obj = _call_adapter(raw_user_text, context_block, mode_tag, output_pref)
        messages = adapter_obj.get("messages") or []
        
        if not messages:
            raise ValueError("Adapter returned empty messages")

        # Шаг 2: Codegen с GPT-5
        codegen_text = _call_codegen_from_messages(messages)
        
        if not codegen_text or codegen_text == "# Error generating code":
            raise ValueError("Failed to generate code")

        # Применение результата
        mode = (adapter_obj.get("response_contract") or {}).get("mode", output_pref)
        updated_path = None

        if (mode or "").upper() == "FILES_JSON":
            try:
                obj = json.loads(codegen_text)
            except json.JSONDecodeError:
                obj = json.loads(extract_code(codegen_text))
            
            files = obj.get("files") or []
            if not files:
                raise ValueError("No files in response")
                
            updated_path = _apply_files_json(chat_id, active, files)

        elif (mode or "").upper() == "UNIFIED_DIFF":
            # Пока используем fallback на полный файл
            logger.info("UNIFIED_DIFF mode - using full file replacement")
            code = extract_code(codegen_text)
            updated_path = version_current_file(chat_id, active, code)

        else:
            # Режим по умолчанию - простой код
            code = extract_code(codegen_text)
            updated_path = version_current_file(chat_id, active, code)

        # Формируем ответ
        rel = latest_path(chat_id, active).relative_to(OUTPUT_DIR)
        state["reply_text"] = (
            f"✅ Обновил `{active}` через PROMPT-ADAPTER v3\n"
            f"🧠 Модель: `GPT-5`\n"
            f"📁 Контракт: `{(mode or output_pref)}`\n"
            f"💾 Сохранено: `{rel}`\n\n"
            f"Отправь следующий промпт или используй команды."
        )
        
        # Аудит успешной генерации
        audit_event(
            chat_id, "GENERATE",
            active_file=active, 
            model="gpt-5",
            prompt=json.dumps(adapter_obj, ensure_ascii=False)[:4000],
            output_path=updated_path,
            meta={
                "adapter_mode": mode_tag, 
                "contract_mode": mode or output_pref,
                "success": True
            }
        )
        
    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        
        # Аудит неудачной генерации
        audit_event(
            chat_id, "GENERATE_ERROR",
            active_file=active,
            model="gpt-5",
            meta={"error": str(e)[:500]}
        )
        
        # Перебрасываем для обработки декоратором
        raise

    return state

# --------- DOWNLOAD ---------
def _iter_selected_files(base: Path, arg: Optional[str]) -> Iterable[Path]:
    """Итератор по выбранным файлам"""
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
    """Создание ZIP архива с выбранными файлами"""
    base = chat_dir(chat_id)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = base / f"export-{ts}.zip"
    to_pack = list(_iter_selected_files(base, arg))
    
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in to_pack:
            z.write(p, arcname=p.name)
    
    logger.info(f"Created ZIP archive: {out.name} with {len(to_pack)} files")
    return out

@safe_node
def node_download(state: GraphState) -> GraphState:
    """Создание архива для скачивания"""
    chat_id = state["chat_id"]
    arg = state.get("arg")
    
    try:
        z = _make_zip(chat_id, arg)
        state["file_to_send"] = str(z)
        sel = arg or "all"
        state["reply_text"] = f"📦 Подготовил архив `{z.name}` ({sel})."
        audit_event(chat_id, "DOWNLOAD", active_file=state.get("active_file"), model="gpt-5", output_path=z)
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        state["reply_text"] = f"❌ Не удалось создать архив: {str(e)[:200]}"
    
    return state

# ---------- РОУТЕР И СБОРКА ПРИЛОЖЕНИЯ ----------
def router(state: GraphState) -> str:
    """Роутер для определения следующего узла"""
    return state["command"]

def build_app():
    """Сборка LangGraph приложения с checkpointer"""
    sg = StateGraph(GraphState)
    
    # Добавляем узлы
    sg.add_node("parse", parse_message)
    sg.add_node("CREATE", node_create)
    sg.add_node("SWITCH", node_switch)
    sg.add_node("FILES", node_files)
    sg.add_node("MODEL", node_model)
    sg.add_node("RESET", node_reset)
    sg.add_node("GENERATE", node_generate)
    sg.add_node("DOWNLOAD", node_download)

    # Устанавливаем точку входа
    sg.set_entry_point("parse")
    
    # Добавляем условные переходы от парсера к узлам
    sg.add_conditional_edges("parse", router, {
        "CREATE": "CREATE",
        "SWITCH": "SWITCH",
        "FILES": "FILES",
        "MODEL": "MODEL",
        "RESET": "RESET",
        "GENERATE": "GENERATE",
        "DOWNLOAD": "DOWNLOAD",
    })
    
    # Все узлы ведут к концу
    for node in ("CREATE", "SWITCH", "FILES", "MODEL", "RESET", "GENERATE", "DOWNLOAD"):
        sg.add_edge(node, END)

    # Настройка checkpointer
    if _CHECKPOINTER_KIND == "sqlite":
        try:
            from langgraph.checkpoint.sqlite import SqliteSaver
            
            db_path = OUTPUT_DIR / "langgraph.db"
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Создаем checkpointer с SQLite
            checkpointer = SqliteSaver(str(db_path))
            
            logger.info(f"Using SQLite checkpointer at {db_path}")
            
        except Exception as e:
            logger.warning(f"Failed to create SQLite checkpointer: {e}, falling back to MemorySaver")
            from langgraph.checkpoint.memory import MemorySaver
            checkpointer = MemorySaver()
    else:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        logger.info("Using MemorySaver (non-persistent)")

    # Компилируем граф с checkpointer
    compiled_app = sg.compile(checkpointer=checkpointer)
    
    logger.info("LangGraph application compiled successfully with GPT-5 support")
    return compiled_app

# ---------- ИНИЦИАЛИЗАЦИЯ ----------
APP = build_app()

# Экспорт для bot.py
__all__ = ['APP', 'DEFAULT_MODEL', 'VALID_MODELS']

logger.info(f"Graph app initialized. Model: GPT-5 only. Output dir: {OUTPUT_DIR}")
