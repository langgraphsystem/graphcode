from __future__ import annotations
import os, re, time, hashlib, sqlite3, zipfile, json, logging
from pathlib import Path
from typing import TypedDict, Optional, Iterable, List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from pydantic import BaseModel, Field, validator
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

# ---------- MODELS CONFIG ----------
VALID_MODELS = {"gpt-5"}
VALID_CODEGEN_MODELS = {
    "gpt-5",
    "claude-opus-4-1-20250805",
}

# Production reasoning/verbosity settings
FINAL_REASONING = {"effort": "high"}
FINAL_VERBOSITY = "compact"

# ---------- CLIENTS ----------
openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=config.request_timeout
)

anthropic_client = None
if os.getenv("ANTHROPIC_API_KEY") and Anthropic is not None:
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

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
    """Detect programming language from file extension."""
    return EXT2LANG.get(Path(filename).suffix.lower(), "text")

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe filesystem operations."""
    unsafe_chars = ['..', '/', '\\', ':', '*', '?', '"', '<', '>', '|']
    clean_name = filename
    for ch in unsafe_chars:
        clean_name = clean_name.replace(ch, '_')
    
    # Handle whitespace
    clean_name = re.sub(r'\s+', '_', clean_name.strip())
    
    # Limit length
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length - len(ext)] + ext
    
    # Ensure non-empty
    if not clean_name or clean_name in ['.', '..']:
        clean_name = 'unnamed_file'
    
    return clean_name

def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    """Safely join paths preventing directory traversal."""
    try:
        clean_path = relative_path.strip().lstrip('/\\')
        
        # Check for dangerous patterns
        if '..' in clean_path or clean_path.startswith('/'):
            logger.warning(f"Potentially unsafe path rejected: {relative_path}")
            return None
        
        full_path = base_dir / clean_path
        
        # Ensure path stays within base directory
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
    """Get or create chat-specific directory."""
    p = config.output_dir / str(chat_id)
    p.mkdir(parents=True, exist_ok=True)
    return p

def latest_path(chat_id: int, filename: str) -> Path:
    """Get path for latest version of a file."""
    return chat_dir(chat_id) / f"latest-{filename}"

def ensure_latest_placeholder(chat_id: int, filename: str, language: str) -> Path:
    """Create placeholder file if it doesn't exist."""
    lp = latest_path(chat_id, filename)
    if lp.exists():
        return lp
    
    stubs = {
        'python': "# -*- coding: utf-8 -*-\n# Auto-generated file\n",
        'javascript': "// Auto-generated file\n",
        'typescript': "// Auto-generated file\n",
        'html': "<!DOCTYPE html>\n<html>\n<head>\n    <title>Generated</title>\n</head>\n<body>\n</body>\n</html>\n",
        'css': "/* Auto-generated file */\n",
        'json': "{}\n",
        'yaml': "# Auto-generated file\n",
        'bash': "#!/usr/bin/env bash\n# Auto-generated file\n",
        'sql': "-- Auto-generated file\n",
        'rust': "// Auto-generated file\n",
        'go': "// Auto-generated file\npackage main\n",
        'java': "// Auto-generated file\npublic class Main {}\n",
        'text': "",
    }
    
    try:
        lp.write_text(stubs.get(language, ""), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to create placeholder: {e}")
        lp.touch()
    
    return lp

def list_files(chat_id: int) -> List[str]:
    """List all files in chat directory."""
    base = chat_dir(chat_id)
    try:
        return sorted([p.name for p in base.iterdir() if p.is_file()])
    except Exception as e:
        logger.error(f"Failed to list files: {e}")
        return []

def version_current_file(chat_id: int, filename: str, new_content: str) -> Path:
    """Version a file with timestamp and update latest."""
    lp = latest_path(chat_id, filename)
    
    # Check if content changed
    old = lp.read_text(encoding="utf-8") if lp.exists() else ""
    if hashlib.sha256(old.encode()).hexdigest() == hashlib.sha256(new_content.encode()).hexdigest():
        return lp
    
    # Create versioned copy
    ts = time.strftime("%Y%m%d-%H%M%S")
    ver = chat_dir(chat_id) / f"{ts}-{filename}"
    
    try:
        ver.write_text(new_content, encoding="utf-8")
        lp.write_text(new_content, encoding="utf-8")
        logger.info(f"Created version: {ver.name}")
    except Exception as e:
        logger.error(f"Failed to version file: {e}")
        raise
    
    return lp

# ---------- CODE EXTRACTION ----------
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)
DIFF_BLOCK_RE = re.compile(r"```(diff|patch)\n(.*?)```", re.DOTALL | re.IGNORECASE)
UNIFIED_DIFF_HINT_RE = re.compile(r"(?m)^(--- |\+\+\+ |@@ )")
GIT_DIFF_HINT_RE = re.compile(r"(?m)^diff --git ")

def extract_code(text: str) -> str:
    """Extract code from markdown code block or return as-is."""
    m = CODE_BLOCK_RE.search(text)
    return m.group(2).strip() if m else text.strip()

def extract_diff_and_spec(text: str) -> Tuple[str, str]:
    """Extract diff blocks and remaining specification."""
    diff_parts: List[str] = []
    
    def grab_diff(m: re.Match) -> str:
        diff_parts.append(m.group(2).strip())
        return ""
    
    text_without_diff = DIFF_BLOCK_RE.sub(grab_diff, text)
    diff_text = "\n\n".join(diff_parts).strip()
    
    # Check for diff patterns outside code blocks
    if not diff_text and (GIT_DIFF_HINT_RE.search(text_without_diff) or 
                          UNIFIED_DIFF_HINT_RE.search(text_without_diff)):
        return "", text_without_diff.strip()
    
    return text_without_diff.strip(), diff_text

def is_placeholder_or_empty(content: str) -> bool:
    """Check if content is empty or placeholder."""
    if not content.strip():
        return True
    if "Auto-generated" in content or "created via" in content:
        return True
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

# ---------- API CALLS ----------
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
    max_output_tokens: Optional[int] = None
) -> Any:
    """Call OpenAI Responses API with retry logic."""
    try:
        logger.info(f"Calling OpenAI Responses API with model {model}")
        kwargs = {
            "model": model,
            "reasoning": FINAL_REASONING,
            "verbosity": FINAL_VERBOSITY,
            "input": messages,
        }
        if response_format:
            kwargs["response_format"] = response_format
        if max_output_tokens is not None:
            kwargs["max_output_tokens"] = max_output_tokens
        
        return openai_client.responses.create(**kwargs)
    except Exception as e:
        logger.error(f"OpenAI Responses API error: {e}")
        raise

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    retry=retry_if_exception_type(Exception),
    reraise=True
)
def anthropic_call(model: str, messages: List[Dict[str, str]]) -> str:
    """Call Anthropic API with retry logic."""
    if anthropic_client is None:
        raise RuntimeError("Anthropic client not initialized. Set ANTHROPIC_API_KEY")
    
    # Separate system and user messages
    system_parts: List[str] = []
    user_parts: List[str] = []
    
    for m in messages:
        role, content = m.get("role", "user"), m.get("content", "")
        if role == "system":
            system_parts.append(content)
        elif role == "user":
            user_parts.append(content)
    
    system_text = "\n\n".join(filter(str.strip, system_parts))
    user_text = "\n\n".join(filter(str.strip, user_parts))
    
    logger.info(f"Calling Anthropic API with model {model}")
    
    resp = anthropic_client.messages.create(
        model=model,
        system=system_text if system_text else None,
        temperature=0.0,
        max_tokens=8192,
        messages=[{"role": "user", "content": user_text}],
    )
    
    # Extract text from response
    chunks: List[str] = []
    for block in getattr(resp, "content", []):
        if hasattr(block, "text"):
            chunks.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            chunks.append(block.get("text", ""))
    
    return "".join(chunks).strip()

# ---------- PROMPT MANAGEMENT ----------
class PromptAdapterFile(BaseModel):
    template: str
    version: Optional[str] = None
    description: Optional[str] = None

_PROMPT_CACHE: Dict[str, Any] = {"path": None, "mtime": None, "template": None}

def load_prompt_template() -> str:
    """Load and cache prompt template from file."""
    path = config.prompt_file_path
    mtime = path.stat().st_mtime
    
    # Use cache if valid
    if (_PROMPT_CACHE["path"] == str(path) and 
        _PROMPT_CACHE["mtime"] == mtime and 
        _PROMPT_CACHE["template"]):
        return _PROMPT_CACHE["template"]
    
    # Load and validate
    data = json.loads(path.read_text(encoding="utf-8"))
    cfg = PromptAdapterFile(**data)
    tmpl = cfg.template
    
    # Validate required tags
    required_tags = ["<<<RAW_TASK>>>", "<<<MODE>>>", "<<<OUTPUT_PREF>>>", "<<<OUTPUT_LANG>>>"]
    for tag in required_tags:
        if tag not in tmpl:
            raise ValueError(f"Prompt template missing required tag: {tag}")
    
    # Update cache
    _PROMPT_CACHE.update({"path": str(path), "mtime": mtime, "template": tmpl})
    return tmpl

def render_adapter_prompt(
    raw_task: str,
    context_block: str,
    mode_tag: str,
    output_pref: str
) -> str:
    """Render prompt template with substitutions."""
    template = load_prompt_template()
    
    replacements = {
        "<<<RAW_TASK>>>": raw_task,
        "<<<CONTEXT>>>": context_block or "(none)",
        "<<<MODE>>>": mode_tag,
        "<<<TARGETS>>>": config.adapter_targets,
        "<<<CONSTRAINTS>>>": config.adapter_constraints,
        "<<<TEST_POLICY>>>": config.adapter_test_policy,
        "<<<OUTPUT_PREF>>>": output_pref,
        "<<<OUTPUT_LANG>>>": config.adapter_output_lang,
    }
    
    for key, value in replacements.items():
        template = template.replace(key, value)
    
    return template

def build_context_block(chat_id: int, filename: str) -> str:
    """Build context block from existing file."""
    lp = latest_path(chat_id, filename)
    if not lp.exists():
        return ""
    
    lang = detect_language(filename)
    code = lp.read_text(encoding="utf-8")
    
    return f"""<<<CONTEXT:FILE {filename}>>>
```{lang}
{code}
```
<<<END>>>"""

# ---------- QUALITY VALIDATION ----------
def validate_prompt_bundle(bundle: Dict[str, Any]) -> None:
    """Validate adapter response quality."""
    required_fields = ["system", "developer", "user", "constraints", "tests", "output_contract"]
    
    for field in required_fields:
        if field not in bundle or not str(bundle[field]).strip():
            raise ValueError(f"Adapter JSON missing or empty: {field}")
    
    # Validate tests
    if not isinstance(bundle["tests"], list) or len(bundle["tests"]) < 3:
        raise ValueError("Adapter JSON must contain at least 3 tests")
    
    # Check for placeholders
    text_concat = " ".join([
        bundle.get("system", ""),
        bundle.get("developer", ""),
        bundle.get("user", ""),
        bundle.get("constraints", ""),
        bundle.get("non_goals", ""),
        " ".join(bundle.get("tests", []))
    ]).lower()
    
    forbidden_terms = ["todo", "placeholder", "tbd", "xxx", "fixme"]
    for term in forbidden_terms:
        if term in text_concat:
            raise ValueError(f"Adapter JSON contains forbidden term '{term}' - not production ready")

# ---------- ADAPTER LOGIC ----------
def call_adapter(prompt_text: str) -> Dict[str, Any]:
    """Call adapter and return structured prompt."""
    try:
        resp = openai_responses_call(
            config.adapter_model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a Prompt Adapter for production code generation. Return ONLY valid JSON matching the schema."
                },
                {
                    "role": "user",
                    "content": prompt_text
                }
            ],
            response_format={"type": "json_schema", "json_schema": ADAPTER_JSON_SCHEMA}
        )
        
        # Extract parsed response
        if hasattr(resp, "output_parsed") and resp.output_parsed:
            bundle = resp.output_parsed
        else:
            txt = getattr(resp, "output_text", "")
            if not txt:
                raise ValueError("Empty adapter response")
            bundle = json.loads(txt)
        
        # Validate quality
        validate_prompt_bundle(bundle)
        
        # Build final messages
        dev_content = bundle["developer"].strip()
        extra = []
        
        if bundle.get("constraints"):
            extra.append(f"Constraints:\n{bundle['constraints'].strip()}")
        if bundle.get("non_goals"):
            extra.append(f"Non-Goals:\n{bundle['non_goals'].strip()}")
        
        tests = bundle.get("tests", [])
        if tests:
            extra.append("Acceptance Tests:\n- " + "\n- ".join(tests))
        
        if bundle.get("output_contract"):
            extra.append(f"Output Contract:\n{bundle['output_contract'].strip()}")
        
        if extra:
            dev_content += "\n\n" + "\n\n".join(extra)
        
        messages = [
            {"role": "system", "content": bundle["system"]},
            {"role": "system", "content": dev_content},
            {"role": "user", "content": bundle["user"]},
        ]
        
        return {
            "messages": messages,
            "response_contract": {"mode": bundle.get("output_contract", "FILES_JSON")},
            "constraints": bundle.get("constraints", ""),
            "non_goals": bundle.get("non_goals", ""),
            "tests": tests
        }
        
    except Exception as e:
        logger.error(f"Adapter call failed: {e}", exc_info=True)
        # Fallback to simple prompt
        return {
            "messages": [
                {"role": "system", "content": "Generate production-quality code based on user request"},
                {"role": "user", "content": prompt_text}
            ],
            "response_contract": {"mode": config.adapter_output_pref.value}
        }

# ---------- CODEGEN LOGIC ----------
def get_provider_from_model(model: str) -> str:
    """Determine provider from model name."""
    return "anthropic" if model.startswith("claude") else "openai"

def call_codegen(
    messages: List[Dict[str, str]],
    mode: Optional[str] = None,
    model: str = "gpt-5"
) -> str:
    """Call code generation with specified model."""
    try:
        provider = get_provider_from_model(model)
        
        if provider == "openai":
            response_format = None
            if mode and mode.upper() == "FILES_JSON":
                response_format = {"type": "json_schema", "json_schema": FILES_JSON_SCHEMA}
            
            resp = openai_responses_call(
                model,
                messages=messages,
                response_format=response_format
            )
            
            txt = getattr(resp, "output_text", None)
            if not txt:
                raise ValueError("Empty codegen output")
            return txt
        
        else:  # Anthropic
            txt = anthropic_call(model, messages)
            if not txt:
                raise ValueError("Empty codegen output from Anthropic")
            return txt
            
    except Exception as e:
        logger.error(f"Codegen call failed: {e}", exc_info=True)
        return "# Error generating code"

# ---------- FILE APPLICATION ----------
def apply_files_json(
    chat_id: int,
    active_filename: str,
    files_obj: List[Dict[str, str]]
) -> Path:
    """Apply FILES_JSON response to filesystem."""
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
            # Check file size limit
            if len(content.encode('utf-8')) > config.max_file_size:
                logger.warning(f"File too large, skipping: {raw_path}")
                continue
            
            safe_output_path.parent.mkdir(parents=True, exist_ok=True)
            safe_output_path.write_text(content, encoding="utf-8")
            logger.info(f"Written file: {safe_output_path}")
            
            if Path(raw_path).name == active_filename:
                active_written = version_current_file(chat_id, active_filename, content)
                
        except Exception as e:
            logger.error(f"Failed to write file {raw_path}: {e}")
            continue
    
    # Ensure active file is written
    if active_written is None and files_obj:
        first = files_obj[0]
        content = first.get("content", "")
        active_written = version_current_file(chat_id, active_filename, content)
    
    return active_written or latest_path(chat_id, active_filename)

def infer_output_preference(raw_text: str, has_context: bool) -> str:
    """Infer output preference from text patterns."""
    if has_context and (DIFF_BLOCK_RE.search(raw_text) or 
                       UNIFIED_DIFF_HINT_RE.search(raw_text) or 
                       GIT_DIFF_HINT_RE.search(raw_text)):
        return OutputPreference.UNIFIED_DIFF.value
    return config.adapter_output_pref.value

# ---------- AUDIT LOGGING ----------
def audit_connect() -> sqlite3.Connection:
    """Connect to audit database."""
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

def audit_event(
    chat_id: int,
    event_type: str,
    active_file: Optional[str] = None,
    model: Optional[str] = None,
    prompt: Optional[str] = None,
    output_path: Optional[Path] = None,
    meta: Optional[Dict] = None
) -> None:
    """Log audit event to database."""
    conn = audit_connect()
    try:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Calculate file hash if exists
        sha, size = None, None
        if output_path and output_path.exists():
            size = output_path.stat().st_size
            sha = hashlib.sha256(output_path.read_bytes()).hexdigest()
        
        # Truncate prompt if too long
        if prompt and len(prompt) > 4000:
            prompt = prompt[:4000]
        
        conn.execute(
            """INSERT INTO events 
               (ts, chat_id, event_type, active_file, model, prompt, 
                output_path, output_sha256, output_bytes, meta)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (ts, chat_id, event_type, active_file, model, prompt,
             str(output_path) if output_path else None, sha, size,
             json.dumps(meta or {}))
        )
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to log audit event: {e}")
    finally:
        conn.close()

# ---------- GRAPH STATE ----------
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

# ---------- NODE DECORATOR ----------
def safe_node(func):
    """Decorator for safe node execution with error handling."""
    def wrapper(state: GraphState) -> GraphState:
        try:
            return func(state)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {e}", exc_info=True)
            
            error_msg = f"❌ Error in {func.__name__}: "
            if "api key" in str(e).lower():
                error_msg += "API key issue"
            elif "rate" in str(e).lower():
                error_msg += "Rate limit exceeded"
            elif "timeout" in str(e).lower():
                error_msg += "Request timeout"
            elif "json" in str(e).lower():
                error_msg += "JSON parsing error"
            else:
                error_msg += str(e)[:200]
            
            state["reply_text"] = error_msg
            return state
    
    wrapper.__name__ = func.__name__
    return wrapper

def push_status(state: GraphState, msg: str) -> None:
    """Add status message to state."""
    try:
        if "status_msgs" not in state:
            state["status_msgs"] = []
        
        if len(msg) > 500:
            msg = msg[:497] + "..."
        
        state["status_msgs"].append(msg)
    except Exception as e:
        logger.error(f"Failed to push status: {e}")

# ---------- GRAPH NODES ----------
def parse_message(state: GraphState) -> GraphState:
    """Parse incoming message to determine command."""
    text = state["input_text"].strip()
    state["command"] = Command.GENERATE.value
    state["arg"] = None
    
    if text.startswith("/"):
        parts = text.split(maxsplit=1)
        cmd = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else None
        
        command_mapping = {
            "/create": Command.CREATE,
            "/switch": Command.SWITCH,
            "/files": Command.FILES,
            "/model": Command.MODEL,
            "/llm": Command.LLM,
            "/run": Command.RUN,
            "/reset": Command.RESET,
            "/download": Command.DOWNLOAD,
        }
        
        state["command"] = command_mapping.get(cmd, Command.GENERATE).value
        state["arg"] = arg
    
    return state

@safe_node
def node_create(state: GraphState) -> GraphState:
    """Create or activate a file."""
    chat_id = state["chat_id"]
    raw_filename = (state.get("arg") or "main.py").strip()
    filename = sanitize_filename(raw_filename)
    language = detect_language(filename)
    
    ensure_latest_placeholder(chat_id, filename, language)
    state["active_file"] = filename
    state.setdefault("codegen_model", config.codegen_model_default)
    
    state["reply_text"] = f"✅ File created/activated: {filename}\n📤 Language: {language}"
    push_status(state, f"✅ Created/activated file {filename} (language: {language})")
    
    if filename != raw_filename:
        state["reply_text"] += f"\n⚠️ Filename was sanitized for safety"
    
    audit_event(chat_id, "CREATE", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_switch(state: GraphState) -> GraphState:
    """Switch to an existing file."""
    chat_id = state["chat_id"]
    filename = (state.get("arg") or "").strip()
    
    if not filename:
        state["reply_text"] = "Please specify filename: /switch app.py"
        return state
    
    filename = sanitize_filename(filename)
    if not latest_path(chat_id, filename).exists():
        state["reply_text"] = f"File {filename} doesn't exist. Use /create {filename} first."
        return state
    
    state["active_file"] = filename
    state["reply_text"] = f"🔀 Switched to {filename}"
    push_status(state, f"🔀 Switched to file {filename}")
    
    audit_event(chat_id, "SWITCH", active_file=filename, model=state.get("model"))
    return state

@safe_node
def node_files(state: GraphState) -> GraphState:
    """List all files in chat directory."""
    files = list_files(state["chat_id"])
    
    if not files:
        state["reply_text"] = "No files yet. Start with /create app.py"
    else:
        state["reply_text"] = "🗂 Files:\n" + "\n".join(f"- {f}" for f in files)
    
    audit_event(state["chat_id"], "FILES", active_file=state.get("active_file"))
    return state

@safe_node
def node_model(state: GraphState) -> GraphState:
    """Show current model configuration."""
    cg_model = state.get("codegen_model") or config.codegen_model_default
    
    state["reply_text"] = (
        f"🧠 Adapter: GPT-5 (Pro/Thinking Pro)\n"
        f"   reasoning.effort=high, verbosity=compact\n"
        f"🧩 Codegen (default): {cg_model}\n"
        f"🔧 To select codegen model: /llm <{'|'.join(sorted(VALID_CODEGEN_MODELS))}> or /run"
    )
    
    audit_event(state["chat_id"], "MODEL", active_file=state.get("active_file"), model="gpt-5")
    return state

@safe_node
def node_llm(state: GraphState) -> GraphState:
    """Select LLM for code generation and optionally run pending prompt."""
    chat_id = state["chat_id"]
    arg = (state.get("arg") or "").strip()
    pending = state.get("pending_messages")
    
    if not arg:
        current = state.get("codegen_model") or config.codegen_model_default
        msg = (
            "Select a model for CODE GENERATION:\n"
            "Available models:\n- " + "\n- ".join(sorted(VALID_CODEGEN_MODELS)) +
            f"\n\nCurrent default: {current}"
        )
        if pending:
            msg += "\n\n💡 You have a prepared prompt. After selecting, generation will start immediately."
        state["reply_text"] = msg
        return state
    
    model = arg
    if model not in VALID_CODEGEN_MODELS:
        state["reply_text"] = (
            f"Model not supported for codegen: {model}\n"
            f"Available: {', '.join(sorted(VALID_CODEGEN_MODELS))}"
        )
        return state
    
    if model.startswith("claude") and anthropic_client is None:
        state["reply_text"] = (
            "Claude selected but Anthropic not configured. "
            "Please install 'anthropic' package and set ANTHROPIC_API_KEY."
        )
        return state
    
    # Set as default
    state["codegen_model"] = model
    
    # If we have pending adapter, run immediately
    if pending:
        try:
            mode = state.get("pending_mode") or config.adapter_output_pref.value
            messages = pending
            
            push_status(state, f"▶️ Running codegen with selected model: {model}")
            
            codegen_text = call_codegen(messages, mode=mode, model=model)
            if not codegen_text or codegen_text == "# Error generating code":
                raise ValueError("Failed to generate code")
            
            active = state.get("active_file") or "main.py"
            updated_path = None
            
            # Apply based on mode
            if mode.upper() == "FILES_JSON":
                try:
                    obj = json.loads(codegen_text)
                except json.JSONDecodeError:
                    obj = json.loads(extract_code(codegen_text))
                
                files = obj.get("files", [])
                if not files:
                    raise ValueError("No files in response")
                
                push_status(state, f"🧩 Applying FILES_JSON: {len(files)} file(s)")
                updated_path = apply_files_json(chat_id, active, files)
                
            elif mode.upper() == "UNIFIED_DIFF":
                push_status(state, "🧩 Applying UNIFIED_DIFF (fallback: full replacement)")
                code = extract_code(codegen_text)
                updated_path = version_current_file(chat_id, active, code)
                
            else:
                push_status(state, "🧩 Applying direct code output")
                code = extract_code(codegen_text)
                updated_path = version_current_file(chat_id, active, code)
            
            # Build response
            rel = latest_path(chat_id, active).relative_to(config.output_dir)
            status_lines = state.get("status_msgs", [])
            status_block = ""
            if status_lines:
                status_block = "🧭 Execution status:\n"
                status_block += "\n".join(f"{i+1}. {line}" for i, line in enumerate(status_lines))
                status_block += "\n\n"
            
            state["reply_text"] = (
                f"{status_block}"
                f"✅ Updated {active} via PROMPT-ADAPTER v3\n"
                f"🧠 Adapter: GPT-5 Pro/Thinking Pro\n"
                f"🧩 Codegen LLM: {model}\n"
                f"📄 Contract: {mode}\n"
                f"💾 Saved: {rel}\n\n"
                f"Commands: /files, /switch <file>, /download"
            )
            
            audit_event(
                chat_id, "GENERATE",
                active_file=active,
                model=model,
                prompt=json.dumps({
                    "pending_mode": mode,
                    "pending_sha": state.get("pending_prompt_sha"),
                }, ensure_ascii=False)[:4000],
                output_path=updated_path,
                meta={"adapter_ready": True, "success": True}
            )
            
        finally:
            # Clear pending state
            state.pop("pending_messages", None)
            state.pop("pending_mode", None)
            state.pop("pending_prompt_sha", None)
            state.pop("pending_context", None)
    else:
        # No pending - just set default
        state["reply_text"] = f"🔧 Codegen model set to: {model} (will be used for next generation)"
        audit_event(chat_id, "LLM_SET", active_file=state.get("active_file"), model=model)
    
    return state

@safe_node
def node_run(state: GraphState) -> GraphState:
    """Run pending prompt with current model."""
    if not state.get("pending_messages"):
        state["reply_text"] = "No prepared prompt. Send a task first for the adapter."
        return state
    
    model = state.get("codegen_model") or config.codegen_model_default
    state["arg"] = model
    return node_llm(state)

@safe_node
def node_reset(state: GraphState) -> GraphState:
    """Reset chat state."""
    state["active_file"] = None
    state["model"] = config.adapter_model
    state["codegen_model"] = config.codegen_model_default
    
    # Clear pending
    state.pop("pending_messages", None)
    state.pop("pending_mode", None)
    state.pop("pending_prompt_sha", None)
    state.pop("pending_context", None)
    
    state["reply_text"] = "♻️ State reset. Start with /create <filename>"
    audit_event(state["chat_id"], "RESET")
    return state

@safe_node
def node_generate(state: GraphState) -> GraphState:
    """Generate code via adapter + codegen pipeline."""
    chat_id = state["chat_id"]
    active = state.get("active_file")
    
    # Auto-create main.py if needed
    if not active:
        active = "main.py"
        ensure_latest_placeholder(chat_id, active, detect_language(active))
        state["active_file"] = active
        logger.info(f"Auto-created file: {active}")
    
    # Force adapter model
    state["model"] = config.adapter_model
    raw_user_text = state["input_text"]
    
    # Check context
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
        # Call adapter
        adapter_prompt = render_adapter_prompt(raw_user_text, context_block, mode_tag, output_pref)
        push_status(state, f"✅ Loaded external prompt: {config.prompt_file_path.resolve()}")
        
        sha = hashlib.sha256(adapter_prompt.encode('utf-8')).hexdigest()
        push_status(state, f"📤 Sending adapter prompt (hash: {sha[:10]}...)")
        
        adapter_result = call_adapter(adapter_prompt)
        messages = adapter_result.get("messages", [])
        if not messages:
            raise ValueError("Adapter returned empty messages")
        
        mode = adapter_result.get("response_contract", {}).get("mode", output_pref)
        
        # Store pending and ask user to choose model
        state["pending_messages"] = messages
        state["pending_mode"] = mode
        state["pending_prompt_sha"] = sha
        state["pending_context"] = "present" if has_context else "absent"
        
        audit_event(
            chat_id, "ADAPTER_READY",
            active_file=active,
            model=config.adapter_model,
            prompt=json.dumps({"mode": mode, "sha": sha[:16]}, ensure_ascii=False)[:4000],
            meta={"has_context": has_context}
        )
        
        # Build response
        options = " | ".join(sorted(VALID_CODEGEN_MODELS))
        status_lines = state.get("status_msgs", [])
        status_block = ""
        if status_lines:
            status_block = "🧭 Status:\n"
            status_block += "\n".join(f"{i+1}. {line}" for i, line in enumerate(status_lines))
            status_block += "\n\n"
        
        default_model = state.get("codegen_model") or config.codegen_model_default
        state["reply_text"] = (
            f"{status_block}"
            "✅ Structured prompt ready.\n"
            "Choose LLM for code generation (user decision - no auto-run):\n"
            f"→ /llm <{options}>  (recommended)\n"
            f"→ /run  (use current: {default_model})\n\n"
            "After selection, generation and file updates will begin."
        )
        
    except Exception as e:
        logger.error(f"Adapter stage failed: {e}", exc_info=True)
        audit_event(
            chat_id, "ADAPTER_ERROR",
            active_file=active,
            model=config.adapter_model,
            meta={"error": str(e)[:500]}
        )
        raise
    
    return state

@safe_node
def node_download(state: GraphState) -> GraphState:
    """Create downloadable archive."""
    chat_id = state["chat_id"]
    arg = state.get("arg")
    
    try:
        archive_path = make_archive(chat_id, arg)
        state["file_to_send"] = str(archive_path)
        selection = arg or "all"
        state["reply_text"] = f"📦 Prepared archive {archive_path.name} ({selection})"
        push_status(state, f"📦 Created archive {archive_path.name} (filter: {selection})")
        
        audit_event(
            chat_id, "DOWNLOAD",
            active_file=state.get("active_file"),
            model=state.get("codegen_model") or config.codegen_model_default,
            output_path=archive_path
        )
    except Exception as e:
        logger.error(f"Failed to create archive: {e}")
        state["reply_text"] = f"❌ Failed to create archive: {str(e)[:200]}"
    
    return state

# ---------- ARCHIVE CREATION ----------
def iter_selected_files(base: Path, arg: Optional[str]) -> Iterable[Path]:
    """Iterate files based on selection criteria."""
    try:
        files = [p for p in base.iterdir() if p.is_file()]
        if not arg:
            return sorted(files)
        
        arg = arg.strip().lower()
        if arg == "latest":
            return sorted([p for p in files if p.name.startswith("latest-")])
        elif arg == "versions":
            return sorted([p for p in files if not p.name.startswith("latest-")])
        else:
            # Specific file pattern
            return sorted([
                p for p in files 
                if p.name == f"latest-{arg}" or p.name.endswith(f"-{arg}")
            ])
    except Exception as e:
        logger.error(f"Failed to select files: {e}")
        return []

def make_archive(chat_id: int, arg: Optional[str]) -> Path:
    """Create ZIP archive of selected files."""
    base = chat_dir(chat_id)
    ts = time.strftime("%Y%m%d-%H%M%S")
    out = base / f"export-{ts}.zip"
    
    to_pack = list(iter_selected_files(base, arg))
    if not to_pack:
        raise ValueError("No files to archive")
    
    # Check total size
    total_size = sum(p.stat().st_size for p in to_pack)
    if total_size > config.max_archive_size:
        raise ValueError(f"Archive too large: {total_size} bytes")
    
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in to_pack:
            z.write(p, arcname=p.name)
    
    logger.info(f"Created ZIP archive: {out.name} with {len(to_pack)} files")
    return out

# ---------- GRAPH BUILDER ----------
def router(state: GraphState) -> str:
    """Route to appropriate node based on command."""
    return state["command"]

def build_app() -> Any:
    """Build the LangGraph application."""
    sg = StateGraph(GraphState)
    
    # Add nodes
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
    
    # Set entry point
    sg.set_entry_point("parse")
    
    # Add conditional routing
    sg.add_conditional_edges(
        "parse",
        router,
        {
            Command.CREATE.value: "CREATE",
            Command.SWITCH.value: "SWITCH",
            Command.FILES.value: "FILES",
            Command.MODEL.value: "MODEL",
            Command.LLM.value: "LLM",
            Command.RUN.value: "RUN",
            Command.RESET.value: "RESET",
            Command.GENERATE.value: "GENERATE",
            Command.DOWNLOAD.value: "DOWNLOAD",
        }
    )
    
    # Add edges to END
    for node in ["CREATE", "SWITCH", "FILES", "MODEL", "LLM", "RUN", "RESET", "GENERATE", "DOWNLOAD"]:
        sg.add_edge(node, END)
    
    # Initialize checkpointer
    checkpointer = MemorySaver()
    logger.info("Using MemorySaver for state management")
    
    # Compile graph
    compiled_app = sg.compile(checkpointer=checkpointer)
    logger.info("LangGraph application compiled successfully")
    
    return compiled_app

# ---------- INITIALIZATION ----------
APP = build_app()

__all__ = ['APP', 'VALID_MODELS', 'VALID_CODEGEN_MODELS', 'config']

logger.info(
    "Graph app initialized. Adapter: %s. Codegen models: %s. Output dir: %s",
    config.adapter_model,
    sorted(VALID_CODEGEN_MODELS),
    config.output_dir
)
