from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json, logging
from pathlib import Path
from typing import TypedDict, Optional, Iterable, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel
from openai import OpenAI

# Optional Anthropic client (for Claude codegen)
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

# ---------- CONFIGURATION ----------
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
            # Create a dummy prompt file if it doesn't exist to make the script runnable
            logger.warning(f"Prompt file not found: {self.prompt_file_path}. Creating a dummy file.")
            dummy_prompt = {
                "template": "RAW_TASK: <<<RAW_TASK>>>\nMODE: <<<MODE>>>\nOUTPUT_PREF: <<<OUTPUT_PREF>>>\nOUTPUT_LANG: <<<OUTPUT_LANG>>>\nCONTEXT: <<<CONTEXT>>>\nTARGETS: <<<TARGETS>>>\nCONSTRAINTS: <<<CONSTRAINTS>>>\nTEST_POLICY: <<<TEST_POLICY>>>",
                "version": "1.0-dummy"
            }
            self.prompt_file_path.parent.mkdir(parents=True, exist_ok=True)
            self.prompt_file_path.write_text(json.dumps(dummy_prompt, indent=2))


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

# ---------- MODELS CONFIG ----------
VALID_MODELS = {"gpt-5"}
VALID_CODEGEN_MODELS = {
    "gpt-5",
    "claude-opus-4-1-20250805",
}

FINAL_REASONING = {"effort": "high"}
FINAL_VERBOSITY = "low"

# ---------- CLIENTS ----------
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=config.request_timeout
)

anthropic_client = None
if os.getenv("ANTHROPIC_API_KEY") and Anthropic is not None:
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

gateway_client = OpenAI(
    api_key=os.getenv("AI_GATEWAY_API_KEY") or os.getenv("OPENAI_API_KEY"),
    base_url=os.getenv("AI_GATEWAY_BASE_URL") or None,
    timeout=config.request_timeout
)

# ---------- FILE UTILITIES ----------
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
    unsafe_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    clean_name = filename
    for ch in unsafe_chars:
        clean_name = clean_name.replace(ch, '_')
    clean_name = re.sub(r'\s+', '_', clean_name.strip())
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length - len(ext)] + ext
    if not clean_name or clean_name in ['.', '..']:
        clean_name = 'unnamed_file'
    return clean_name

def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    try:
        clean_path = relative_path.strip().lstrip('/\\')
        if '..' in clean_path or clean_path.startswith('/'):
            logger.warning(f"Potentially unsafe path rejected: {relative_path}")
            return None
        full_path = base_dir / clean_path
        full_path.resolve().relative_to(base_dir.resolve())
        return full_path
    except ValueError:
        logger.warning(f"Path outside base directory rejected: {relative_path}")
        return None
    except Exception as e:
        logger.error(f"Error processing path: {e}")
        return None

def chat_dir(chat_id: int) -> Path:
    p = config.output_dir / str(chat_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def latest_path(chat_id: int, filename: str) -> Path:
    return chat_dir(chat_id) / f"latest-{filename}"

def ensure_latest_placeholder(chat_id: int, filename: str, language: str) -> Path:
    lp = latest_path(chat_id, filename)
    if lp.exists():
        return lp
    stubs = {
        'python': "# -*- coding: utf-8 -*-\n# Auto-generated file\n",
        'javascript': "// Auto-generated file\n",
        'html': "<!DOCTYPE html><html><head><title>Generated</title></head><body></body></html>",
        'css': "/* Auto-generated file */",
        'json': "{}",
        'bash': "#!/usr/bin/env bash\n",
    }
    lp.write_text(stubs.get(language, "# Auto-generated file\n"), encoding="utf-8")
    return lp

def list_files(chat_id: int) -> List[str]:
    base = chat_dir(chat_id)
    return sorted([p.name for p in base.iterdir() if p.is_file()])

def version_current_file(chat_id: int, filename: str, new_content: str) -> Path:
    lp = latest_path(chat_id, filename)
    if lp.exists() and hashlib.sha256(lp.read_bytes()).hexdigest() == hashlib.sha256(new_content.encode()).hexdigest():
        return lp
    ts = time.strftime("%Y%m%d-%H%M%S")
    ver = chat_dir(chat_id) / f"{ts}-{filename}"
    ver.write_text(new_content, encoding="utf-8")
    lp.write_text(new_content, encoding="utf-8")
    logger.info(f"Created version: {ver.name}")
    return lp

# ---------- CODE EXTRACTION ----------
CODE_BLOCK_RE = re.compile(r"```(?:[a-zA-Z0-9_+-]+)?\n(.*?)\n?```", re.DOTALL)

def extract_code(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    return match.group(1).strip() if match else text.strip()

def is_placeholder_or_empty(content: str) -> bool:
    return not content.strip() or "Auto-generated" in content

# ---------- SCHEMAS ----------
ADAPTER_JSON_SCHEMA = {
    "name": "final_prompt_bundle", "schema": {
        "type": "object", "properties": {
            "system": {"type": "string"}, "developer": {"type": "string"},
            "user": {"type": "string"}, "constraints": {"type": "string"},
            "non_goals": {"type": "string"}, "tests": {"type": "array", "items": {"type": "string"}},
            "output_contract": {"type": "string"}
        }, "required": ["system", "developer", "user", "tests", "output_contract"],
    }, "strict": True
}
FILES_JSON_SCHEMA = {
    "name": "code_files", "schema": {
        "type": "object", "properties": {
            "files": {"type": "array", "items": {
                "type": "object", "properties": {
                    "path": {"type": "string"}, "content": {"type": "string"}
                }, "required": ["path", "content"],
            }}
        }, "required": ["files"],
    }, "strict": True
}

# ---------- API CALLS ----------
@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def openai_responses_call(model: str, messages: List[Dict[str, str]], **kwargs) -> Any:
    logger.info(f"Calling OpenAI Responses API with model {model}")
    sys_parts = [m["content"] for m in messages if m["role"] in ("system", "developer")]
    input_items = [m for m in messages if m["role"] not in ("system", "developer")]
    _input = input_items[0]["content"] if len(input_items) == 1 else input_items
    
    client = gateway_client if kwargs.pop("use_gateway", False) else openai_client
    return client.responses.create(
        model=model, input=_input,
        instructions="\n\n".join(sys_parts) or None,
        reasoning=kwargs.pop("override_reasoning", FINAL_REASONING),
        text=kwargs.pop("override_text", {"verbosity": FINAL_VERBOSITY}),
        temperature=0.1, **kwargs
    )

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=8), reraise=True)
def anthropic_call(model: str, messages: List[Dict[str, str]]) -> str:
    if not anthropic_client:
        raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY")
    logger.info(f"Calling Anthropic API with model {model}")
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    user_msgs = [m for m in messages if m["role"] != "system"]
    
    resp = anthropic_client.messages.create(
        model=model, system=system or None, temperature=0.0, max_tokens=8192,
        messages=user_msgs
    )
    return "".join(block.text for block in resp.content if hasattr(block, "text")).strip()

# ---------- PROMPT MANAGEMENT ----------
_PROMPT_CACHE: Dict[str, Any] = {}
def load_prompt_template() -> str:
    path = config.prompt_file_path
    mtime = path.stat().st_mtime
    if _PROMPT_CACHE.get("mtime") == mtime:
        return _PROMPT_CACHE["template"]
    
    data = json.loads(path.read_text(encoding="utf-8"))
    template = PromptAdapterFile(**data).template
    if not all(tag in template for tag in ["<<<RAW_TASK>>>", "<<<MODE>>>"]):
        raise ValueError("Prompt template missing required tags")
    
    _PROMPT_CACHE["mtime"] = mtime
    _PROMPT_CACHE["template"] = template
    return template

def render_adapter_prompt(raw_task: str, context_block: str, mode_tag: str) -> str:
    template = load_prompt_template()
    replacements = {
        "<<<RAW_TASK>>>": raw_task,
        "<<<CONTEXT>>>": context_block or "(none)",
        "<<<MODE>>>": mode_tag,
        "<<<TARGETS>>>": config.adapter_targets,
        "<<<CONSTRAINTS>>>": config.adapter_constraints,
        "<<<TEST_POLICY>>>": config.adapter_test_policy,
        "<<<OUTPUT_PREF>>>": config.adapter_output_pref.value,
        "<<<OUTPUT_LANG>>>": config.adapter_output_lang,
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template

def build_context_block(chat_id: int, filename: str) -> str:
    lp = latest_path(chat_id, filename)
    if not lp.exists(): return ""
    code = lp.read_text(encoding="utf-8")
    return f"<<<CONTEXT:FILE {filename}>>>\n```{detect_language(filename)}\n{code}\n```\n<<<END>>>"

# ---------- QUALITY VALIDATION ----------
def validate_prompt_bundle(bundle: Dict[str, Any]) -> None:
    if not all(bundle.get(f) for f in ["system", "developer", "user", "tests", "output_contract"]):
        raise ValueError("Adapter JSON missing or empty required fields")
    if not isinstance(bundle["tests"], list) or len(bundle["tests"]) < 3:
        raise ValueError("Adapter JSON must contain at least 3 tests")
    if any(term in json.dumps(bundle).lower() for term in ["todo", "tbd", "xxx", "fixme"]):
        raise ValueError("Adapter JSON contains forbidden placeholder terms")

# ---------- ADAPTER & CODEGEN LOGIC ----------
def call_adapter(prompt_text: str) -> Dict[str, Any]:
    try:
        resp = openai_responses_call(
            config.adapter_model,
            messages=[
                {"role": "system", "content": "You are a Prompt Adapter for production code generation. Return ONLY valid JSON."},
                {"role": "user", "content": prompt_text}
            ],
            response_format={"type": "json_schema", "json_schema": ADAPTER_JSON_SCHEMA},
            max_output_tokens=2000,
            override_reasoning={"effort": "medium"},
        )
        bundle = resp.output_parsed if hasattr(resp, "output_parsed") else json.loads(getattr(resp, "output_text", "{}"))
        validate_prompt_bundle(bundle)
        
        dev_parts = [
            bundle["developer"], f"Constraints:\n{bundle.get('constraints', 'N/A')}",
            f"Non-Goals:\n{bundle.get('non_goals', 'N/A')}",
            "Acceptance Tests:\n- " + "\n- ".join(bundle["tests"]),
            f"Output Contract:\n{bundle['output_contract']}"
        ]
        messages = [
            {"role": "system", "content": bundle["system"]},
            {"role": "developer", "content": "\n\n".join(dev_parts)},
            {"role": "user", "content": bundle["user"]},
        ]
        return {"messages": messages, "response_contract": {"mode": bundle["output_contract"]}}
    except Exception as e:
        logger.error(f"Adapter call failed, falling back to basic prompt: {e}", exc_info=True)
        return {
            "messages": [
                {"role": "system", "content": "Generate production-quality code based on user request."},
                {"role": "user", "content": prompt_text}
            ],
            "response_contract": {"mode": config.adapter_output_pref.value}
        }

def call_codegen(messages: List[Dict[str, str]], mode: str, model: str) -> str:
    provider = "anthropic" if model.startswith("claude") and not os.getenv("AI_GATEWAY_BASE_URL") else "openai"
    if provider == "openai":
        kwargs = { "use_gateway": model.startswith("claude") }
        if mode == "FILES_JSON":
            kwargs["response_format"] = {"type": "json_schema", "json_schema": FILES_JSON_SCHEMA}
        
        api_model = f"anthropic/{model}" if kwargs.get("use_gateway") else model
        resp = openai_responses_call(api_model, messages, max_output_tokens=3500, **kwargs)
        return getattr(resp, "output_text", "# Error: Empty codegen output")
    else:
        return anthropic_call(model, messages)

# ---------- FILE APPLICATION ----------
def apply_files_json(chat_id: int, active_filename: str, files_obj: List[Dict[str, str]]) -> Path:
    base_dir = chat_dir(chat_id)
    active_written = None
    for item in files_obj:
        path, content = item.get("path", ""), item.get("content", "")
        if not path: continue
        safe_output_path = safe_path_join(base_dir, path)
        if not safe_output_path:
            logger.warning(f"Skipping unsafe path: {path}")
            continue
        safe_output_path.parent.mkdir(parents=True, exist_ok=True)
        safe_output_path.write_text(content, encoding="utf-8")
        logger.info(f"Written file: {safe_output_path}")
        if Path(path).name == active_filename:
            active_written = version_current_file(chat_id, active_filename, content)
    
    if not active_written and files_obj: # Fallback to first file if active wasn't specified
        first_file = files_obj[0]
        return version_current_file(chat_id, active_filename, first_file.get("content", ""))
        
    return active_written or latest_path(chat_id, active_filename)

# ---------- AUDIT LOGGING ----------
# ИЗМЕНЕНИЕ: Функция теперь сама управляет соединением с БД.
def audit_event(chat_id: int, event_type: str, **kwargs) -> None:
    """Manages its own DB connection to log an audit event."""
    db_path = config.output_dir / "audit.db"
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, ts TEXT, chat_id INTEGER, event_type TEXT,
                    active_file TEXT, model TEXT, prompt TEXT, output_path TEXT,
                    output_sha256 TEXT, output_bytes INTEGER, meta TEXT
                )
            """)
            
            data = {
                "ts": time.strftime("%Y-%m-%d %H:%M:%S"), "chat_id": chat_id,
                "event_type": event_type, "meta": json.dumps(kwargs.pop("meta", {}))
            }
            data.update(kwargs)
            
            if "output_path" in data and data["output_path"]:
                p = Path(data["output_path"])
                if p.exists():
                    data["output_bytes"] = p.stat().st_size
                    data["output_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
            
            cols = ", ".join(data.keys())
            placeholders = ", ".join("?" for _ in data)
            conn.execute(f"INSERT INTO events ({cols}) VALUES ({placeholders})", list(data.values()))
            conn.commit()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")


# ---------- GRAPH STATE & NODES ----------
class GraphState(TypedDict, total=False):
    # ИЗМЕНЕНИЕ: Удаляем 'db' из состояния, чтобы оно было сериализуемым.
    chat_id: int
    input_text: str
    command: Command
    arg: Optional[str]
    active_file: Optional[str]
    codegen_model: str
    pending_messages: Optional[List[Dict[str, str]]]
    pending_mode: Optional[str]
    reply_text: str
    file_to_send: Optional[str]
    status_msgs: List[str]

def safe_node(func):
    def wrapper(state: GraphState) -> GraphState:
        try:
            return func(state)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            state["reply_text"] = f"❌ Error in {func.__name__}: {str(e)[:200]}"
            return state
    return wrapper

def push_status(state: GraphState, msg: str) -> None:
    state.setdefault("status_msgs", []).append(msg)

@safe_node
def entry_point(state: GraphState) -> GraphState:
    text = state["input_text"].strip()
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd_str = parts[0].upper().replace("/", "")
        state["command"] = Command[cmd_str] if cmd_str in Command.__members__ else Command.GENERATE
        state["arg"] = parts[1].strip() if len(parts) > 1 else None
    else:
        state["command"] = Command.GENERATE
    
    state["status_msgs"] = []
    # ИЗМЕНЕНИЕ: Удаляем управление БД из состояния.
    return state

@safe_node
def node_create_switch(state: GraphState) -> GraphState:
    filename = sanitize_filename((state.get("arg") or "main.py").strip())
    is_create = state["command"] == Command.CREATE
    
    if is_create or not latest_path(state["chat_id"], filename).exists():
        ensure_latest_placeholder(state["chat_id"], filename, detect_language(filename))
        state["reply_text"] = f"✅ File created/activated: {filename}"
    else:
        state["reply_text"] = f"🔀 Switched to {filename}"
        
    state["active_file"] = filename
    state.setdefault("codegen_model", config.codegen_model_default)
    audit_event(state["chat_id"], state["command"].value, active_file=filename)
    return state

@safe_node
def node_files(state: GraphState) -> GraphState:
    files = list_files(state["chat_id"])
    state["reply_text"] = "🗂 Files:\n" + "\n".join(f"- {f}" for f in files) if files else "No files yet."
    return state

@safe_node
def node_model_info(state: GraphState) -> GraphState:
    cg_model = state.get("codegen_model") or config.codegen_model_default
    state["reply_text"] = (
        f"🧠 Adapter: {config.adapter_model}\n"
        f"🧩 Codegen: {cg_model}\n"
        f"🔧 To change: /llm <{'|'.join(sorted(VALID_CODEGEN_MODELS))}>"
    )
    return state

@safe_node
def node_llm(state: GraphState) -> GraphState:
    arg = (state.get("arg") or "").strip()
    if not arg or arg not in VALID_CODEGEN_MODELS:
        state["reply_text"] = f"Invalid model. Available: {', '.join(sorted(VALID_CODEGEN_MODELS))}"
        return state
    
    if arg.startswith("claude") and not (anthropic_client or os.getenv("AI_GATEWAY_BASE_URL")):
        state["reply_text"] = "Claude is not configured. Set ANTHROPIC_API_KEY or AI_GATEWAY_BASE_URL."
        return state
        
    state["codegen_model"] = arg
    audit_event(state["chat_id"], "LLM_SET", model=arg)
    
    if state.get("pending_messages"):
        state["command"] = Command.RUN
        return node_run(state)
    else:
        state["reply_text"] = f"🔧 Codegen model set to: {arg}"
        return state

@safe_node
def node_run(state: GraphState) -> GraphState:
    messages = state.get("pending_messages")
    if not messages:
        state["reply_text"] = "No prepared prompt. Send a task first."
        return state

    chat_id = state["chat_id"]
    active_file = state.get("active_file", "main.py")
    model = state.get("codegen_model", config.codegen_model_default)
    mode = state.get("pending_mode", config.adapter_output_pref.value)
    push_status(state, f"▶️ Running codegen with model: {model}")

    codegen_text = call_codegen(messages, mode=mode, model=model)
    if codegen_text.startswith("# Error"):
        raise ValueError(codegen_text)
    
    push_status(state, "✔️ Codegen model responded successfully.")

    if mode == "FILES_JSON":
        files_obj = json.loads(extract_code(codegen_text)).get("files", [])
        updated_path = apply_files_json(chat_id, active_file, files_obj)
    else:
        updated_path = version_current_file(chat_id, active_file, extract_code(codegen_text))
    
    push_status(state, f"✔️ Code applied to file: {updated_path.name}")
    
    rel_path = updated_path.relative_to(config.output_dir)
    status_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(state.get("status_msgs", [])))
    state["reply_text"] = (
        f"🧭 **Execution Status:**\n{status_block}\n\n"
        f"✅ **Done!** File `{active_file}` was updated.\n"
        f"🧩 **Codegen:** `{model}`\n"
        f"💾 **Saved:** `{rel_path}`"
    )
    audit_event(chat_id, "GENERATE_SUCCESS", model=model, output_path=str(updated_path))
    
    state.pop("pending_messages", None)
    state.pop("pending_mode", None)
    return state

@safe_node
def node_reset(state: GraphState) -> GraphState:
    keys_to_clear = ["active_file", "pending_messages", "pending_mode"]
    for key in keys_to_clear:
        state.pop(key, None)
    state["codegen_model"] = config.codegen_model_default
    state["reply_text"] = "♻️ State reset."
    audit_event(state["chat_id"], "RESET")
    return state

@safe_node
def node_generate(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    active_file = state.get("active_file")
    if not active_file:
        state = node_create_switch(state)
        active_file = state["active_file"]
    
    push_status(state, f"📩 User request received ({len(state['input_text'])} chars)")
    
    context_block = build_context_block(chat_id, active_file)
    mode_tag = "DIFF_PATCH" if context_block else "NEW_FILE"
    push_status(state, f"🧠 Calling adapter model (mode: {mode_tag})")
    
    adapter_prompt = render_adapter_prompt(state["input_text"], context_block, mode_tag)
    adapter_result = call_adapter(adapter_prompt)
    
    if not adapter_result.get("messages"):
        state["reply_text"] = "❌ Error: Adapter model failed to process the request. Please try rephrasing."
        return state

    push_status(state, "✔️ Adapter model responded successfully.")
    
    state["pending_messages"] = adapter_result["messages"]
    state["pending_mode"] = adapter_result["response_contract"]["mode"]
    
    audit_event(chat_id, "ADAPTER_READY", model=config.adapter_model)
    push_status(state, "✅ Structured prompt is ready.")
    
    status_block = "\n".join(f"{i+1}. {line}" for i, line in enumerate(state.get("status_msgs", [])))
    state["reply_text"] = (
        f"🧭 **Preparation Status:**\n{status_block}\n\n"
        "**Choose a model for code generation:**\n"
        f"→ `/llm <{'|'.join(sorted(VALID_CODEGEN_MODELS))}>`\n"
        f"→ `/run` (use current: `{state.get('codegen_model', config.codegen_model_default)}`)"
    )
    return state

@safe_node
def node_download(state: GraphState) -> GraphState:
    chat_id = state["chat_id"]
    base = chat_dir(chat_id)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out_path = base / f"export-{ts}.zip"

    files_to_pack = [p for p in base.iterdir() if p.is_file()]
    if not files_to_pack: raise ValueError("No files to archive")
    
    with zipfile.ZipFile(out_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in files_to_pack: zf.write(p, arcname=p.name)

    state["file_to_send"] = str(out_path)
    state["reply_text"] = f"📦 Prepared archive: {out_path.name}"
    audit_event(chat_id, "DOWNLOAD", output_path=str(out_path))
    return state

@safe_node
def exit_point(state: GraphState) -> GraphState:
    # ИЗМЕНЕНИЕ: Больше не нужно закрывать соединение, так как оно не хранится в состоянии.
    return state

# ---------- GRAPH BUILDER ----------
def build_app() -> Any:
    sg = StateGraph(GraphState)
    sg.add_node("entry", entry_point)
    sg.add_node(Command.CREATE.name, node_create_switch)
    sg.add_node(Command.SWITCH.name, node_create_switch)
    sg.add_node(Command.FILES.name, node_files)
    sg.add_node(Command.MODEL.name, node_model_info)
    sg.add_node(Command.LLM.name, node_llm)
    sg.add_node(Command.RUN.name, node_run)
    sg.add_node(Command.RESET.name, node_reset)
    sg.add_node(Command.GENERATE.name, node_generate)
    sg.add_node(Command.DOWNLOAD.name, node_download)
    sg.add_node("exit", exit_point)

    sg.set_entry_point("entry")
    sg.add_conditional_edges("entry", lambda state: state["command"].name)
    
    for cmd_name in [c.name for c in Command if c not in [Command.LLM]]:
        sg.add_edge(cmd_name, "exit")
    
    sg.add_conditional_edges(
        Command.LLM.name,
        lambda state: state.get("command", Command.LLM).name,
        {Command.RUN.name: Command.RUN.name, Command.LLM.name: "exit"}
    )
    
    sg.add_edge("exit", END)

    db_path = config.output_dir / "checkpoints.sqlite"
    checkpointer = SqliteSaver.from_conn_string(str(db_path))
    logger.info(f"Using SqliteSaver for persistent state management (DB: {db_path})")
    
    return sg.compile(checkpointer=checkpointer)

# ---------- INITIALIZATION ----------
APP = build_app()
__all__ = ['APP', 'config']
logger.info(
    "Graph app initialized. Adapter: %s. Codegen models: %s. Output dir: %s",
    config.adapter_model, sorted(VALID_CODEGEN_MODELS), config.output_dir
)
