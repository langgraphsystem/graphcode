from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json, logging, random, threading
from pathlib import Path
from typing import TypedDict, Optional, Iterable, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI

# ---------- OPTIONAL: Anthropic ----------
try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None  # type: ignore

# ---------- LOGGING ----------
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ---------- CONSTANTS ----------
class Command(str, Enum):
    CREATE = "CREATE"
    SWITCH = "SWITCH"
    FILES = "FILES"
    MODEL = "MODEL"
    LLM = "LLM"
    RUN = "RUN"
    RESET = "RESET"
    GENERATE = "GENERATE"
    DOWNLOAD = "DOWNLOAD"

class OutputPreference(str, Enum):
    FILES_JSON = "FILES_JSON"
    UNIFIED_DIFF = "UNIFIED_DIFF"
    CODE_ONLY = "CODE_ONLY"

# ---------- CONFIG ----------
@dataclass
class Config:
    output_dir: Path
    prompt_file_path: Path
    adapter_model: str = "gpt-5"
    codegen_model_default: str = "gpt-5"
    adapter_targets: str = "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict"
    adapter_constraints: str = "No secrets; reasonable perf; minimal deps; production quality"
    adapter_test_policy: str = "COMPREHENSIVE_TESTS"
    adapter_output_lang: str = "EN"
    adapter_output_pref: OutputPreference = OutputPreference.FILES_JSON
    request_timeout: int = 300
    max_file_size: int = 10_000_000  # 10MB
    max_archive_size: int = 50_000_000  # 50MB
    def __post_init__(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if not self.prompt_file_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {self.prompt_file_path}")

config = Config(
    output_dir=Path(os.getenv("OUTPUT_DIR", "./out")).resolve(),
    prompt_file_path=Path(os.getenv("PROMPT_FILE_PATH", "./config/prompt_adapter.json")),
    adapter_model=os.getenv("ADAPTER_MODEL", "gpt-5"),
    codegen_model_default=os.getenv("CODEGEN_MODEL", "gpt-5"),
    adapter_targets=os.getenv("ADAPTER_TARGETS", "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict"),
    adapter_constraints=os.getenv("ADAPTER_CONSTRAINTS", "No secrets; reasonable perf; minimal deps; production quality"),
    adapter_test_policy=os.getenv("ADAPTER_TEST_POLICY", "COMPREHENSIVE_TESTS"),
    adapter_output_lang=os.getenv("ADAPTER_OUTPUT_LANG", "EN"),
    adapter_output_pref=OutputPreference(os.getenv("ADAPTER_OUTPUT_PREF", "FILES_JSON")),
    request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
)

# ---------- MODELS ----------
VALID_MODELS = {"gpt-5"}
VALID_CODEGEN_MODELS = {"gpt-5", "claude-opus-4-1-20250805"}

# Production defaults
FINAL_REASONING = {"effort": "high"}  # на этапе Adapter можно больше мыслить
FINAL_VERBOSITY = "low"               # GPT-5: low|medium|high (через text.verbosity)

# ---------- CLIENTS ----------
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=config.request_timeout)
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY")) if os.getenv("ANTHROPIC_API_KEY") and Anthropic else None

# ---------- FILE UTILS ----------
EXT2LANG = {
    ".py": "python", ".js": "javascript", ".ts": "typescript", ".html": "html",
    ".css": "css", ".json": "json", ".yml": "yaml", ".yaml": "yaml",
    ".sh": "bash", ".sql": "sql", ".txt": "text", ".rs": "rust",
    ".go": "go", ".java": "java", ".cpp": "cpp", ".c": "c",
    ".rb": "ruby", ".php": "php", ".swift": "swift", ".kt": "kotlin",
    ".jsx": "javascript", ".tsx": "typescript", ".md": "markdown",
}
def detect_language(filename: str) -> str:
    return EXT2LANG.get(Path(filename).suffix.lower(), "text")
def sanitize_filename(filename: str) -> str:
    unsafe = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    clean = filename
    for ch in unsafe: clean = clean.replace(ch, '_')
    clean = re.sub(r'\s+', '_', clean.strip())
    if len(clean) > 255:
        name, ext = os.path.splitext(clean)
        clean = name[:255 - len(ext)] + ext
    if not clean or clean in ['.', '..']: clean = 'unnamed_file'
    return clean
def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    try:
        clean = relative_path.strip().lstrip('/\\')
        if '..' in clean or clean.startswith('/'):
            logger.warning(f"Unsafe path: {relative_path}")
            return None
        full = base_dir / clean
        try:
            full.resolve().relative_to(base_dir.resolve())
        except ValueError:
            logger.warning(f"Path outside base dir: {relative_path}")
            return None
        return full
    except Exception as e:
        logger.error(f"Error processing path: {e}")
        return None
def chat_dir(chat_id: int) -> Path:
    p = config.output_dir / str(chat_id); p.mkdir(parents=True, exist_ok=True); return p
def latest_path(chat_id: int, filename: str) -> Path:
    return chat_dir(chat_id) / f"latest-{filename}"
def ensure_latest_placeholder(chat_id: int, filename: str, language: str) -> Path:
    lp = latest_path(chat_id, filename)
    if lp.exists(): return lp
    stubs = {'python': "# -*- coding: utf-8 -*-\n# Auto-generated file\n", 'javascript': "// Auto-generated file\n",
             'typescript': "// Auto-generated file\n", 'html': "<!DOCTYPE html>\n<html>\n<head>\n    <title>Generated</title>\n</head>\n<body>\n</body>\n</html>\n",
             'css': "/* Auto-generated file */\n", 'json': "{}\n", 'yaml': "# Auto-generated file\n", 'bash': "#!/usr/bin/env bash\n# Auto-generated file\n",
             'sql': "-- Auto-generated file\n", 'rust': "// Auto-generated file\n", 'go': "// Auto-generated file\npackage main\n",
             'java': "// Auto-generated file\npublic class Main {}\n", 'text': ""}
    try: lp.write_text(stubs.get(language, ""), encoding="utf-8")
    except Exception as e: logger.error(f"Failed to create placeholder: {e}"); lp.touch()
    return lp
def list_files(chat_id: int) -> List[str]:
    base = chat_dir(chat_id)
    try: return sorted([p.name for p in base.iterdir() if p.is_file()])
    except Exception as e: logger.error(f"Failed to list files: {e}"); return []
def version_current_file(chat_id: int, filename: str, new_content: str) -> Path:
    lp = latest_path(chat_id, filename)
    old = lp.read_text(encoding="utf-8") if lp.exists() else ""
    if hashlib.sha256(old.encode()).hexdigest() == hashlib.sha256(new_content.encode()).hexdigest(): return lp
    ts = time.strftime("%Y%m%d-%H%M%S")
    ver = chat_dir(chat_id) / f"{ts}-{filename}"
    try: ver.write_text(new_content, encoding="utf-8"); lp.write_text(new_content, encoding="utf-8"); logger.info(f"Created version: {ver.name}")
    except Exception as e: logger.error(f"Failed to version file: {e}"); raise
    return lp

# ---------- EXTRACTION ----------
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)
DIFF_BLOCK_RE = re.compile(r"```(diff|patch)\n(.*?)```", re.DOTALL | re.IGNORECASE)
UNIFIED_DIFF_HINT_RE = re.compile(r"(?m)^(--- |\+\+\+ |@@ )")
GIT_DIFF_HINT_RE = re.compile(r"(?m)^diff --git ")
def extract_code(text: str) -> str:
    m = CODE_BLOCK_RE.search(text); return m.group(2).strip() if m else text.strip()
def extract_diff_and_spec(text: str) -> Tuple[str, str]:
    diff_parts: List[str] = []
    def grab(m: re.Match) -> str: diff_parts.append(m.group(2).strip()); return ""
    text_wo = DIFF_BLOCK_RE.sub(grab, text)
    diff_text = "\n\n".join(diff_parts).strip()
    if not diff_text and (GIT_DIFF_HINT_RE.search(text_wo) or UNIFIED_DIFF_HINT_RE.search(text_wo)): return "", text_wo.strip()
    return text_wo.strip(), diff_text
def is_placeholder_or_empty(content: str) -> bool:
    if not content.strip(): return True
    if "Auto-generated" in content or "created via" in content: return True
    return len(content.strip()) < 20

# ---------- SCHEMAS ----------
ADAPTER_JSON_SCHEMA = {
    "name": "final_prompt_bundle",
    "schema": {
        "type": "object",
        "properties": {
            "system": {"type": "string"},
            "developer": {"type": "string"},
            "user": {"type": "string"},
            "constraints": {"type": "string"},
            "non_goals": {"type": "string"},
            "tests": {"type": "array", "items": {"type": "string"}},
            "output_contract": {"type": "string"}
        },
        "required": ["system", "developer", "user", "constraints", "tests", "output_contract"],
        "additionalProperties": False
    },
    "strict": True
}
FILES_JSON_SCHEMA = {
    "name": "code_files",
    "schema": {
        "type": "object",
        "properties": {
            "files": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"}
                    },
                    "required": ["path", "content"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["files"],
        "additionalProperties": False
    },
    "strict": True
}

# ---------- RATE / ERROR HELPERS ----------
# NB: не маскируем любые ошибки под "Rate limit exceeded"
def _status_code_of(exc) -> int | None:
    for attr in ("response", "http_response"):
        r = getattr(exc, attr, None)
        if r is None: continue
        try: return getattr(r, "status_code", None) or getattr(r, "status", None)
        except Exception: pass
    return None
def _log_rate_headers(exc: Exception) -> None:
    headers = None
    for attr in ("response", "http_response", "headers"):
        obj = getattr(exc, attr, None)
        if obj is None: continue
        try: headers = obj.headers if hasattr(obj, "headers") else obj
        except Exception: pass
    if not headers: return
    try:
        rid = headers.get("x-request-id")
        ra  = headers.get("retry-after")
        rem_req = headers.get("x-ratelimit-remaining-requests")
        rem_tok = headers.get("x-ratelimit-remaining-tokens")
        reset_req = headers.get("x-ratelimit-reset-requests")
        reset_tok = headers.get("x-ratelimit-reset-tokens")
        logger.warning(f"Rate headers: x-request-id={rid} retry-after={ra} "
                       f"remain(req/tok)={rem_req}/{rem_tok} reset(req/tok)={reset_req}/{reset_tok}")
    except Exception:
        pass
def _sleep_with_retry_after(e, attempt, base=1.0, cap=8.0):
    # 1) Retry-After, если доступен
    retry_after = None
    for attr in ("headers", "response", "http_response"):
        obj = getattr(e, attr, None)
        if obj is None: continue
        try:
            retry_after = (obj.get("retry-after") if hasattr(obj, "get") else obj.headers.get("retry-after"))
        except Exception:
            pass
        if retry_after: break
    if retry_after:
        try:
            delay = float(retry_after)
            time.sleep(delay); return
        except Exception:
            pass
    # 2) Иначе — экспоненциальный бэкофф + джиттер
    delay = min(cap, (2 ** attempt) * base) + random.uniform(0, 0.5)
    time.sleep(delay)
def pretty_api_error(e: Exception) -> str:
    sc = _status_code_of(e)
    msg = str(e)
    if sc == 429: return "Rate limit exceeded (429)"
    if sc == 401: return "Auth error (401)"
    if sc == 403: return "Permission denied (403)"
    if sc == 400: return "Bad request (400)"
    if sc and 500 <= sc < 600: return f"Server error ({sc})"
    return msg[:200]

# ---------- API CALL (Responses API) ----------
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def openai_responses_call(
    model: str,
    messages: List[Dict[str, str]],
    response_format: Optional[Dict] = None,
    max_output_tokens: Optional[int] = None,
    *,
    override_reasoning: Dict[str, Any] | None = None,
    override_text: Dict[str, Any] | None = None,
    temperature: float | None = 0.1,
) -> Any:
    """
    Корректный вызов Responses API:
    - messages → instructions + input (см. migrate-to-responses)
    - reasoning.effort и text.verbosity (GPT-5)
    - мягкие ретраи только для реального 429 (Retry-After + backoff)
    """
    # Разнести roles
    sys_parts: list[str] = []
    input_items: list[dict[str, str]] = []
    for m in messages:
        role = (m.get("role") or "").lower()
        content = m.get("content") or ""
        if role in ("system", "developer"):
            sys_parts.append(content)
        else:
            input_items.append({"role": role or "user", "content": content})

    instructions = "\n\n".join(filter(str.strip, sys_parts)) or None
    _input: str | list[dict[str, str]]
    if len(input_items) == 1 and "content" in input_items[0]:
        _input = input_items[0]["content"]
    else:
        _input = input_items

    kwargs: dict[str, Any] = {"model": model, "input": _input}
    if instructions: kwargs["instructions"] = instructions
    if response_format: kwargs["response_format"] = response_format
    if max_output_tokens is not None: kwargs["max_output_tokens"] = max_output_tokens
    if temperature is not None: kwargs["temperature"] = float(temperature)

    reasoning_cfg = override_reasoning or FINAL_REASONING
    if reasoning_cfg: kwargs["reasoning"] = reasoning_cfg
    text_cfg = override_text or ({"verbosity": FINAL_VERBOSITY} if FINAL_VERBOSITY else None)
    if text_cfg: kwargs["text"] = text_cfg

    try:
        return openai_client.responses.create(**kwargs)
    except Exception as e:
        # если это настоящий 429 — уважаем заголовки и делаем 1–2 повтора
        if _status_code_of(e) == 429:
            _log_rate_headers(e)
            for attempt in range(1, 3):
                _sleep_with_retry_after(e, attempt, base=1.0, cap=8.0)
                try:
                    return openai_client.responses.create(**kwargs)
                except Exception as e2:
                    if _status_code_of(e2) != 429:
                        raise
                    e = e2
        raise

# ---------- Anthropic fallback ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8), retry=retry_if_exception_type(Exception), reraise=True)
def anthropic_call(model: str, messages: List[Dict[str, str]]) -> str:
    if anthropic_client is None:
        raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY")
    system_parts: List[str], user_parts: List[str] = [], []
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system": system_parts.append(content)
        elif role == "user": user_parts.append(content)
    system_text = "\n\n".join(filter(str.strip, system_parts))
    user_text = "\n\n".join(filter(str.strip, user_parts))
    logger.info(f"Calling Anthropic API with model {model}")
    resp = anthropic_client.messages.create(
        model=model, system=system_text if system_text else None,
        temperature=0.0, max_tokens=8192, messages=[{"role": "user", "content": user_text}],
    )
    chunks: List[str] = []
    for block in getattr(resp, "content", []):
        if hasattr(block, "text"): chunks.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text": chunks.append(block.get("text", ""))
    return "".join(chunks).strip()

# ---------- PROMPT MGMT ----------
class PromptAdapterFile(BaseModel):
    template: str
    version: Optional[str] = None
    description: Optional[str] = None

_PROMPT_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "template": None}
def load_prompt_template() -> str:
    path = config.prompt_file_path; mtime = path.stat().st_mtime
    if (_PROMPT_CACHE["path"] == str(path) and _PROMPT_CACHE["mtime"] == mtime and _PROMPT_CACHE["template"]):
        return _PROMPT_CACHE["template"]
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = PromptAdapterFile(**data); tmpl = cfg.template
    for tag in ["<<<RAW_TASK>>>", "<<<MODE>>>", "<<<OUTPUT_PREF>>>", "<<<OUTPUT_LANG>>>"]:
        if tag not in tmpl: raise ValueError(f"Prompt template missing required tag: {tag}")
    _PROMPT_CACHE.update({"path": str(path), "mtime": mtime, "template": tmpl})
    return tmpl
def render_adapter_prompt(raw_task: str, context_block: str, mode_tag: str, output_pref: str) -> str:
    template = load_prompt_template()
    repl = {
        "<<<RAW_TASK>>>": raw_task, "<<<CONTEXT>>>": context_block or "(none)", "<<<MODE>>>": mode_tag,
        "<<<TARGETS>>>": config.adapter_targets, "<<<CONSTRAINTS>>>": config.adapter_constraints,
        "<<<TEST_POLICY>>>": config.adapter_test_policy, "<<<OUTPUT_PREF>>>": output_pref, "<<<OUTPUT_LANG>>>": config.adapter_output_lang,
    }
    for k, v in repl.items(): template = template.replace(k, v)
    return template
def build_context_block(chat_id: int, filename: str) -> str:
    lp = latest_path(chat_id, filename)
    if not lp.exists(): return ""
    lang = detect_language(filename); code = lp.read_text(encoding="utf-8")
    return f"""<<<CONTEXT:FILE {filename}>>>
```{lang}
{code}
```"""

# ---------- QUALITY ----------
def validate_prompt_bundle(bundle: Dict[str, Any]) -> None:
    required = ["system", "developer", "user", "constraints", "tests", "output_contract"]
    for f in required:
        if f not in bundle or not str(bundle[f]).strip():
            raise ValueError(f"Adapter JSON missing or empty: {f}")
    if not isinstance(bundle["tests"], list) or len(bundle["tests"]) < 3:
        raise ValueError("Adapter JSON must contain at least 3 tests")
    text_concat = " ".join([bundle.get("system",""), bundle.get("developer",""), bundle.get("user",""),
                            bundle.get("constraints",""), bundle.get("non_goals",""), " ".join(bundle.get("tests",[]))]).lower()
    for term in ["todo", "placeholder", "tbd", "xxx", "fixme"]:
        if term in text_concat:
            raise ValueError(f"Adapter JSON contains forbidden term '{term}' - not production ready")
# ---------- STATE ----------
class GraphState(TypedDict, total=False):
    chat_id: int
    input_text: str
    command: str
    arg: Optional[str]
    active_file: Optional[str]
    model: str
    codegen_model: str
    pending_messages: Optional[List[Dict[str, str]]]
    pending_mode: Optional[str]
    pending_prompt_sha: Optional[str]
    pending_context: Optional[str]
    reply_text: str
    file_to_send: Optional[str]
    status_msgs: List[str]
# ---------- STATUS & SAFE WRAPPER ----------
def push_status(state: GraphState, msg: str) -> None:
    try:
        if "status_msgs" not in state: state["status_msgs"] = []
        if len(msg) > 500: msg = msg[:497] + "..."
        state["status_msgs"].append(msg)
    except Exception as e:
        logger.error(f"Failed to push status: {e}")
def set_reply_from_status(state: GraphState, headline: str | None = None) -> None:
    """Собираем ответ пользователю после каждого действия."""
    lines = state.get("status_msgs", [])
    block = ""
    if lines:
        block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(lines))
    prefix = (headline + "\n\n") if headline else ""
    state["reply_text"] = f"{prefix}{block}".strip() if block else (headline or "")
def safe_node(func):
    def wrapper(state: GraphState) -> GraphState:
        try:
            out = func(state)
            # после успешного действия — всегда ответ пользователю
            if "reply_text" not in out or not out["reply_text"]:
                set_reply_from_status(out)
            return out
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            err = f"❌ Error in {func.__name__}: {pretty_api_error(e)}"
            push_status(state, err)
            state["reply_text"] = err
            return state
    wrapper.__name__ = func.__name__
    return wrapper
# ---------- AUDIT ----------
def audit_connect() -> sqlite3.Connection:
    audit_db = config.output_dir / "audit.db"
    conn = sqlite3.connect(audit_db)
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
    conn.execute("CREATE INDEX IF NOT EXISTS idx_chat_id ON events(chat_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_ts ON events(ts)")
    return conn

def audit_event(chat_id: int, event_type: str, active_file: Optional[str] = None,
                model: Optional[str] = None, prompt: Optional[str] = None,
                output_path: Optional[Path] = None, meta: Optional[Dict] = None) -> None:
    conn = audit_connect()
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        sha, size = None, None
        if output_path and output_path.exists():
            size = output_path.stat().st_size
            sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        if prompt and len(prompt) > 4000: prompt = prompt[:4000]
        conn.execute(
            """INSERT INTO events
            (ts, chat_id, event_type, active_file, model, prompt,
             output_path, output_sha256, output_bytes, meta)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, chat_id, event_type, active_file, model, prompt,
             str(output_path) if output_path else None, sha, size, json.dumps(meta or {}))
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")
    finally:
        conn.close()
# ---------- ADAPTER / CODEGEN ----------
def validate_and_build_messages(bundle: Dict[str, Any]) -> List[Dict[str, str]]:
    validate_prompt_bundle(bundle)
    dev_content = bundle["developer"].strip()
    extra = []
    if bundle.get("constraints"): extra.append(f"Constraints:\n{bundle['constraints'].strip()}")
    if bundle.get("non_goals"): extra.append(f"Non-Goals:\n{bundle['non_goals'].strip()}")
    tests = bundle.get("tests", [])
    if tests: extra.append("Acceptance Tests:\n- " + "\n- ".join(tests))
    if bundle.get("output_contract"): extra.append(f"Output Contract:\n{bundle['output_contract'].strip()}")
    if extra: dev_content += "\n\n" + "\n\n".join(extra)
    return [{"role": "system", "content": bundle["system"]},
            {"role": "system", "content": dev_content},
            {"role": "user",   "content": bundle["user"]}]
def call_adapter(prompt_text: str, state: GraphState) -> Dict[str, Any]:
    """Шаг 1: строго структурированный промпт-бандл (JSON). С ответом после запроса/получения."""
    push_status(state, "📤 Sending adapter request…")
    set_reply_from_status(state, "Working…")   # немедленный ответ пользователю
    resp = openai_responses_call(
        config.adapter_model,
        messages=[
            {"role": "system", "content": "You are a Prompt Adapter for production code generation. Return ONLY valid JSON matching the schema."},
            {"role": "user", "content": prompt_text}
        ],
        response_format={"type": "json_schema", "json_schema": ADAPTER_JSON_SCHEMA, "strict": True},
        max_output_tokens=2000,
        override_reasoning={"effort": "medium"},  # больше рассуждений для качества ТЗ
        override_text={"verbosity": "medium"},
        temperature=0.1,
    )
    push_status(state, "✅ Adapter response received")
    set_reply_from_status(state)
    if hasattr(resp, "output_parsed") and resp.output_parsed:
        bundle = resp.output_parsed
    else:
        txt = getattr(resp, "output_text", "")
        if not txt: raise ValueError("Empty adapter response")
        bundle = json.loads(txt)
    messages = validate_and_build_messages(bundle)
    return {
        "messages": messages,
        "response_contract": {"mode": bundle.get("output_contract", "FILES_JSON")},
        "constraints": bundle.get("constraints", ""),
        "non_goals": bundle.get("non_goals", ""),
        "tests": bundle.get("tests", []),
    }
_CODEGEN_GATE = threading.Semaphore(1)  # исключаем скрытую конкуренцию
def get_provider_from_model(model: str) -> str:
    return "anthropic" if model.startswith("claude") else "openai"
def call_codegen(messages: List[Dict[str, str]], mode: Optional[str], model: str, state: GraphState) -> str:
    """Шаг 2: генерируем файлы проекта. С ответами после запроса/получения."""
    provider = get_provider_from_model(model)
    with _CODEGEN_GATE:
        if provider == "openai":
            response_format = None
            if mode and mode.upper() == "FILES_JSON":
                response_format = {"type": "json_schema", "json_schema": FILES_JSON_SCHEMA, "strict": True}
            push_status(state, f"📤 Codegen request → {model}")
            set_reply_from_status(state, "Running codegen…")
            resp = openai_responses_call(
                model, messages=messages, response_format=response_format,
                max_output_tokens=3500, override_reasoning={"effort": "minimal"},
                override_text={"verbosity": "low"}, temperature=0.1,
            )
            push_status(state, "✅ Codegen response received")
            set_reply_from_status(state)
            txt = getattr(resp, "output_text", None)
            if not txt: raise ValueError("Empty codegen output")
            return txt
        else:
            push_status(state, f"📤 Codegen request → {model} (Anthropic)")
            set_reply_from_status(state, "Running codegen…")
            txt = anthropic_call(model, messages)
            push_status(state, "✅ Codegen response received")
            set_reply_from_status(state)
            if not txt: raise ValueError("Empty codegen output from Anthropic")
            return txt
# ---------- FILE APPLICATION ----------
def apply_files_json(chat_id: int, active_filename: str, files_obj: List[Dict[str, str]], state: GraphState) -> Path:
    active_written = None
    base_dir = chat_dir(chat_id)
    for item in files_obj:
        raw_path = item.get("path", "").strip()
        content = item.get("content", "")
        if not raw_path: continue
        safe_output_path = safe_path_join(base_dir, raw_path)
        if safe_output_path is None:
            logger.warning(f"Skipping unsafe path: {raw_path}")
            push_status(state, f"⚠️ Skipped unsafe path: {raw_path}")
            continue
        try:
            if len(content.encode('utf-8')) > config.max_file_size:
                logger.warning(f"File too large, skipping: {raw_path}")
                push_status(state, f"⚠️ Skipped large file: {raw_path}")
                continue
            safe_output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_output_path.write_text(content, encoding="utf-8")
            push_status(state, f"💾 Written file: {raw_path}")
            set_reply_from_status(state)
            if Path(raw_path).name == active_filename:
                active_written = version_current_file(chat_id, active_filename, content)
        except Exception as e:
            logger.error(f"Failed to write file {raw_path}: {e}")
            push_status(state, f"❌ Failed to write: {raw_path}: {e}")
            set_reply_from_status(state)
            continue
    if active_written is None and files_obj:
        first = files_obj[0]; content = first.get("content", "")
        active_written = version_current_file(chat_id, active_filename, content)
        push_status(state, f"💾 Updated active file from first item: {active_filename}")
        set_reply_from_status(state)
    return active_written or latest_path(chat_id, active_filename)
def infer_output_preference(raw_text: str, has_context: bool) -> str:
    if has_context and (DIFF_BLOCK_RE.search(raw_text) or UNIFIED_DIFF_HINT_RE.search(raw_text) or GIT_DIFF_HINT_RE.search(raw_text)):
        return OutputPreference.UNIFIED_DIFF.value
    return config.adapter_output_pref.value
# ---------- GRAPH NODES ----------
def parse_message(state: GraphState) -> GraphState:
    text = state["input_text"].strip()
    state["command"] = Command.GENERATE.value
    state["arg"] = None
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None
        mapping = {"/create": Command.CREATE, "/switch": Command.SWITCH, "/files": Command.FILES,
                   "/model": Command.MODEL, "/llm": Command.LLM, "/run": Command.RUN,
                   "/reset": Command.RESET, "/download": Command.DOWNLOAD}
        state["command"] = mapping.get(cmd, Command.GENERATE).value
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
    state.setdefault("codegen_model", config.codegen_model_default)
    push_status(state, f"✅ Created/activated file {filename} (language: {language})")
    if filename != raw_filename: push_status(state, "⚠️ Filename was sanitized for safety")
    set_reply_from_status(state)
    audit_event(chat_id, "CREATE", active_file=filename, model=state.get("model"))
    return state
@safe_node
def node_switch(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]; filename = (state.get("arg") or "").strip()
    if not filename:
        push_status(state, "❗ Please specify filename: /switch app.py")
        set_reply_from_status(state); return state
    filename = sanitize_filename(filename)
    if not latest_path(chat_id, filename).exists():
        push_status(state, f"❗ File {filename} doesn't exist. Use /create {filename} first.")
        set_reply_from_status(state); return state
    state["active_file"] = filename
    push_status(state, f"🔀 Switched to {filename}")
    set_reply_from_status(state)
    audit_event(chat_id, "SWITCH", active_file=filename, model=state.get("model"))
    return state
@safe_node
def node_files(state: GraphState) -> GraphState:
    files = list_files(state["chat_id"])
    if not files: push_status(state, "No files yet. Start with /create app.py")
    else: push_status(state, "🗂 Files:\n" + "\n".join(f"- {f}" for f in files))
    set_reply_from_status(state)
    audit_event(state["chat_id"], "FILES", active_file=state.get("active_file"))
    return state
@safe_node
def node_model(state: GraphState) -> GraphState:
    cg_model = state.get("codegen_model") or config.codegen_model_default
    push_status(state, f"🧠 Adapter: GPT-5 (reasoning.effort={FINAL_REASONING.get('effort','minimal')}, text.verbosity={FINAL_VERBOSITY})")
    push_status(state, f"🧩 Codegen default: {cg_model}")
    push_status(state, f"🔧 Select codegen: /llm <{'|'.join(sorted(VALID_CODEGEN_MODELS))}> or /run")
    set_reply_from_status(state)
    audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model="gpt-5")
    return state
@safe_node
def node_llm(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]; arg = (state.get("arg") or "").strip(); pending = state.get("pending_messages")
    if not arg:
        current = state.get("codegen_model") or config.codegen_model_default
        push_status(state, "Select a model for CODE GENERATION:")
        push_status(state, "Available:\n- " + "\n- ".join(sorted(VALID_CODEGEN_MODELS)))
        push_status(state, f"Current default: {current}")
        if pending: push_status(state, "💡 Prepared prompt present. Generation will start after selection.")
        set_reply_from_status(state); return state
    model = arg
    if model not in VALID_CODEGEN_MODELS:
        push_status(state, f"Model not supported for codegen: {model}")
        push_status(state, f"Available: {', '.join(sorted(VALID_CODEGEN_MODELS))}")
        set_reply_from_status(state); return state
    if model.startswith("claude") and anthropic_client is None:
        push_status(state, "Claude selected but Anthropic not configured. Install 'anthropic' and set ANTHROPIC_API_KEY.")
        set_reply_from_status(state); return state
    state["codegen_model"] = model
    if pending:
        try:
            mode = state.get("pending_mode") or config.adapter_output_pref.value
            messages = pending
            push_status(state, f"▶️ Running codegen with selected model: {model}")
            set_reply_from_status(state, "Starting codegen…")
            codegen_text = call_codegen(messages, mode=mode, model=model, state=state)
            if not codegen_text or codegen_text == "# Error generating code": raise ValueError("Failed to generate code")
            active = state.get("active_file") or "main.py"
            updated_path = None
            if mode.upper() == "FILES_JSON":
                try: obj = json.loads(codegen_text)
                except json.JSONDecodeError: obj = json.loads(extract_code(codegen_text))
                files = obj.get("files", [])
                if not files: raise ValueError("No files in response")
                push_status(state, f"🧩 Applying FILES_JSON: {len(files)} file(s)")
                set_reply_from_status(state)
                updated_path = apply_files_json(chat_id, active, files, state)
            elif mode.upper() == "UNIFIED_DIFF":
                push_status(state, "🧩 Applying UNIFIED_DIFF (fallback: full replacement)")
                set_reply_from_status(state)
                code = extract_code(codegen_text)
                updated_path = version_current_file(chat_id, active, code)
                push_status(state, f"💾 Updated {active} (UNIFIED_DIFF fallback)")
                set_reply_from_status(state)
            else:
                push_status(state, "🧩 Applying direct code output")
                set_reply_from_status(state)
                code = extract_code(codegen_text)
                updated_path = version_current_file(chat_id, active, code)
                push_status(state, f"💾 Updated {active}")
                set_reply_from_status(state)
            rel = latest_path(chat_id, active).relative_to(config.output_dir)
            status_lines = state.get("status_msgs", [])
            status_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(status_lines))
            state["reply_text"] = (
                f"{status_block}\n\n"
                f"✅ Updated {active} via PROMPT-ADAPTER v3\n"
                f"🧠 Adapter: GPT-5 Pro/Thinking Pro\n"
                f"🧩 Codegen LLM: {model}\n"
                f"📄 Contract: {mode}\n"
                f"💾 Saved: {rel}\n\n"
                f"Commands: /files, /switch , /download"
            )
            audit_event(chat_id, "GENERATE", active_file=active, model=model,
                        prompt=json.dumps({"pending_mode": mode, "pending_sha": state.get("pending_prompt_sha")}, ensure_ascii=False)[:4000],
                        output_path=updated_path, meta={"adapter_ready": True, "success": True})
        finally:
            state.pop("pending_messages", None)
            state.pop("pending_mode", None)
            state.pop("pending_prompt_sha", None)
            state.pop("pending_context", None)
    else:
        push_status(state, f"🔧 Codegen model set to: {model} (will be used for next generation)")
        set_reply_from_status(state)
        audit_event(chat_id, "LLM_SET", active_file=state.get("active_file"), model=model)
    return state
@safe_node
def node_run(state: GraphState) -> GraphState:
    if not state.get("pending_messages"):
        push_status(state, "No prepared prompt. Send a task first for the adapter.")
        set_reply_from_status(state); return state
    model = state.get("codegen_model") or config.codegen_model_default
    state["arg"] = model
    return node_llm(state)
@safe_node
def node_reset(state: GraphState) -> GraphState:
    state["active_file"] = None
    state["model"] = config.adapter_model
    state["codegen_model"] = config.codegen_model_default
    for k in ["pending_messages","pending_mode","pending_prompt_sha","pending_context"]:
        state.pop(k, None)
    push_status(state, "♻️ State reset. Start with /create ")
    set_reply_from_status(state)
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
    state["model"] = config.adapter_model
    raw_user_text = state["input_text"]
    lp = latest_path(chat_id, active)
    existed_before = lp.exists()
    ensure_latest_placeholder(chat_id, active, detect_language(active))
    current_text = lp.read_text(encoding="utf-8") if lp.exists() else ""
    has_context = existed_before and not is_placeholder_or_empty(current_text)
    context_block = build_context_block(chat_id, active) if has_context else ""
    mode_tag = "DIFF_PATCH" if has_context else "NEW_FILE"
    output_pref = infer_output_preference(raw_user_text, has_context)
    push_status(state, f"📩 User request ({len(raw_user_text)} chars)")
    push_status(state, f"🧠 Adapter: GPT-5 (mode={mode_tag})")
    try:
        adapter_prompt = render_adapter_prompt(raw_user_text, context_block, mode_tag, output_pref)
        push_status(state, f"✅ Loaded external prompt: {config.prompt_file_path.resolve()}")
        sha = hashlib.sha256(adapter_prompt.encode('utf-8')).hexdigest()
        push_status(state, f"📤 Sending adapter prompt (hash: {sha[:10]}...)")
        set_reply_from_status(state, "Preparing structured prompt…")
        adapter_result = call_adapter(adapter_prompt, state)
        messages = adapter_result.get("messages", [])
        if not messages: raise ValueError("Adapter returned empty messages")
        mode = adapter_result.get("response_contract", {}).get("mode", output_pref)
        state["pending_messages"] = messages
        state["pending_mode"] = mode
        state["pending_prompt_sha"] = sha
        state["pending_context"] = "present" if has_context else "absent"
        audit_event(chat_id, "ADAPTER_READY", active_file=active, model=config.adapter_model,
                    prompt=json.dumps({"mode": mode, "sha": sha[:16]}, ensure_ascii=False)[:4000],
                    meta={"has_context": has_context})
        options = " | ".join(sorted(VALID_CODEGEN_MODELS))
        push_status(state, "✅ Structured prompt ready.")
        push_status(state, "Choose LLM for code generation (user decision - no auto-run):")
        push_status(state, f"→ /llm <{options}>  (recommended)")
        default_model = state.get("codegen_model") or config.codegen_model_default
        push_status(state, f"→ /run  (use current: {default_model})")
        push_status(state, "After selection, generation and file updates will begin.")
        set_reply_from_status(state)
    except Exception as e:
        audit_event(chat_id, "ADAPTER_ERROR", active_file=active, model=config.adapter_model, meta={"error": str(e)[:500]})
        raise
    return state
@safe_node
def node_download(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]; arg = state.get("arg")
    try:
        archive_path = make_archive(chat_id, arg)
        state["file_to_send"] = str(archive_path)
        selection = arg or "all"
        push_status(state, f"📦 Prepared archive {archive_path.name} ({selection})")
        set_reply_from_status(state)
        audit_event(chat_id, "DOWNLOAD", active_file=state.get("active_file"),
                    model=state.get("codegen_model") or config.codegen_model_default, output_path=archive_path)
    except Exception as e:
        push_status(state, f"❌ Failed to create archive: {str(e)[:200]}")
        set_reply_from_status(state)
    return state
# ---------- ARCHIVE ----------
def iter_selected_files(base: Path, arg: Optional[str]) -> Iterable[Path]:
    try:
        files = [p for p in base.iterdir() if p.is_file()]
        if not arg: return sorted(files)
        arg = arg.strip().lower()
        if arg == "latest": return sorted([p for p in files if p.name.startswith("latest-")])
        elif arg == "versions": return sorted([p for p in files if not p.name.startswith("latest-")])
        else:
            return sorted([p for p in files if p.name == f"latest-{arg}" or p.name.endswith(f"-{arg}")])
    except Exception as e:
        logger.error(f"Failed to select files: {e}")
        return []

def make_archive(chat_id: int, arg: Optional[str]) -> Path:
    base = chat_dir(chat_id); ts = time.strftime("%Y%m%d-%H%M%S"); out = base / f"export-{ts}.zip"
    to_pack = list(iter_selected_files(base, arg))
    if not to_pack: raise ValueError("No files to archive")
    total_size = sum(p.stat().st_size for p in to_pack)
    if total_size > config.max_archive_size: raise ValueError(f"Archive too large: {total_size} bytes")
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in to_pack: z.write(p, arcname=p.name)
    logger.info(f"Created ZIP archive: {out.name} with {len(to_pack)} files")
    return out
# ---------- GRAPH ----------
def router(state: GraphState) -> str:
    return state["command"]

def build_app() -> Any:
    sg = StateGraph(GraphState)
    sg.add_node("parse", parse_message)
    sg.add_node("CREATE", node_create)
    sg.add_node("SWITCH", node_switch)
    sg.add_node("FILES", node_files)
    sg.add_node("MODEL", node_model)
    sg.add_node("LLM", node_llm)
    sg.add_node("RUN", node_run)
    sg.add_node("RESET", node_reset)
    sg.add_node("GENERATE", node_generate)
    sg.add_node("DOWNLOAD", node_download)
    sg.set_entry_point("parse")
    sg.add_conditional_edges("parse", router, {
        Command.CREATE.value: "CREATE",
        Command.SWITCH.value: "SWITCH",
        Command.FILES.value: "FILES",
        Command.MODEL.value: "MODEL",
        Command.LLM.value: "LLM",
        Command.RUN.value: "RUN",
        Command.RESET.value: "RESET",
        Command.GENERATE.value: "GENERATE",
        Command.DOWNLOAD.value: "DOWNLOAD",
    })
    for node in ["CREATE", "SWITCH", "FILES", "MODEL", "LLM", "RUN", "RESET", "GENERATE", "DOWNLOAD"]:
        sg.add_edge(node, END)
    checkpointer = MemorySaver()
    logger.info("Using MemorySaver for state management")
    compiled_app = sg.compile(checkpointer=checkpointer)
    logger.info("LangGraph application compiled successfully")
    return compiled_app
# ---------- INIT ----------
APP = build_app()
__all__ = ['APP', 'VALID_MODELS', 'VALID_CODEGEN_MODELS', 'config']

logger.info("Graph app initialized. Adapter: %s. Codegen models: %s. Output dir: %s",
            config.adapter_model, sorted(VALID_CODEGEN_MODELS), config.output_dir)
