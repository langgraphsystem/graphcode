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
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
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
    unsafe_chars = ["..", "/", "\\", ":", "*", "?", "\"", "<", ">", "|"]
    clean_name = filename
    for ch in unsafe_chars:
        clean_name = clean_name.replace(ch, "_")
    
    # Handle whitespace
    clean_name = re.sub(r"\s+", "_", clean_name.strip())
    
    # Limit length
    max_length = 255
    if len(clean_name) > max_length:
        name, ext = os.path.splitext(clean_name)
        clean_name = name[:max_length - len(ext)] + ext
    
    # Ensure non-empty
    if not clean_name or clean_name in [".", ".."]:
        clean_name = "unnamed_file"
    
    return clean_name

def safe_path_join(base_dir: Path, relative_path: str) -> Optional[Path]:
    """Safely join paths preventing directory traversal."""
    try:
        clean_path = relative_path.strip().lstrip("/\\")
        
        # Check for dangerous patterns
        if ".." in clean_path or clean_path.startswith("/"):
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
        "python": "# -*- coding: utf-8 -*-\n# Auto-generated file\n",
        "javascript": "// Auto-generated file\n",
        "typescript": "// Auto-generated file\n",
        "html": "<!DOCTYPE html>\n<html>\n<head>\n    <title>Generated</title>\n</head>\n<body>\n</body>\n</html>\n",
        "css": "/* Auto-generated file */\n",
        "json": "{}\n",
        "yaml": "# Auto-generated file\n",
        "bash": "#!/usr/bin/env bash\n# Auto-generated file\n",
        "sql": "-- Auto-generated file\n",
        "rust": "// Auto-generated file\n",
        "go": "// Auto-generated file\npackage main\n",
        "java": "// Auto-generated file\npublic class Main {}\n",
        "text": "",
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

# ---------- ANTHROPIC API CALLS ----------
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
    """Call adapter and return structured prompt - FIXED VERSION."""
    try:
        logger.info("Calling adapter with CORRECT OpenAI Responses API format")
        
        # ИСПРАВЛЕННЫЙ вызов OpenAI Responses API
        resp = openai_client.responses.create(
            model=config.adapter_model,
            instructions="Return ONLY JSON per the schema. No prose outside JSON.",
            input=prompt_text,
            response_format={
                "type": "json_schema",
                "json_schema": ADAPTER_JSON_SCHEMA,
                "strict": True
            },
            reasoning={"effort": "medium"},
            text={"verbosity": "medium"},
            temperature=0.1,
            max_output_tokens=2000
        )
        
        # Extract parsed response
        bundle = None
        if hasattr(resp, "output_parsed") and resp.output_parsed:
            bundle = resp.output_parsed
            logger.info("✅ Got structured response from output_parsed")
        else:
            # Fallback to text parsing
            output_text = getattr(resp, "output_text", "")
            if not output_text:
                raise ValueError("Empty adapter response")
            
            logger.info("⚠️ Using fallback text parsing")
            try:
                bundle = json.loads(output_text)
            except json.JSONDecodeError:
                # Try to extract JSON from markdown block
                import re
                match = re.search(r"```(?:json)?\n?(.*?)```", output_text, re.DOTALL)
                if match:
                    bundle = json.loads(match.group(1).strip())
                else:
                    raise ValueError("Cannot parse adapter response as JSON")
        
        if not bundle:
            raise ValueError("Failed to get structured response from adapter")
        
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
    """Call code generation with specified model - FIXED VERSION."""
    try:
        provider = get_provider_from_model(model)
        
        if provider == "openai":
            # Convert messages to correct format for OpenAI Responses API
            system_parts = []
            user_parts = []
            
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                
                if role == "system":
                    system_parts.append(content)
                else:
                    user_parts.append(content)
            
            instructions = "\n\n".join(system_parts) if system_parts else (
                "Generate clean, production-ready code based on user requirements."
            )
            input_text = "\n\n".join(user_parts)
            
            # ИСПРАВЛЕННЫЙ вызов для кодогенерации
            if mode and mode.upper() == "FILES_JSON":
                logger.info("Calling codegen with FILES_JSON mode")
                resp = openai_client.responses.create(
                    model=model,
                    instructions="Return ONLY JSON per the schema. Production-grade code; include tests.",
                    input=input_text,
                    response_format={
                        "type": "json_schema",
                        "json_schema": FILES_JSON_SCHEMA,
                        "strict": True
                    },
                    reasoning={"effort": "minimal"},
                    text={"verbosity": "low"},
                    temperature=0.1,
                    max_output_tokens=3500
                )
                
                # Return structured JSON response
                if hasattr(resp, "output_parsed") and resp.output_parsed:
                    return json.dumps(resp.output_parsed, ensure_ascii=False, indent=2)
                else:
                    output_text = getattr(resp, "output_text", "")
                    if not output_text:
                        raise ValueError("Empty codegen output")
                    return output_text
            else:
                # For non-JSON modes
                logger.info(f"Calling codegen with {mode} mode")
                resp = openai_client.responses.create(
                    model=model,
                    instructions=instructions,
                    input=input_text,
                    reasoning={"effort": "minimal"},
                    text={"verbosity": "low"},
                    temperature=0.1,
                    max_output_tokens=3500
                )
                
                output_text = getattr(resp, "output_text", "")
                if not output_text:
                    raise ValueError("Empty codegen output")
                return output_text
        
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
            if len(content.encode("utf-8")) > config.max_file_size:
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
    conn = sqlite3.connect(audit_db)  # ИСПРАВЛЕНО: добавлено создание соединения
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

# ========== SIMPLE GRAPH APP IMPLEMENTATION (ADD TO END OF FILE) ==========

SESSION_FILE = "session.json"

@dataclass
class Session:
    active_file: Optional[str] = None
    model: str = "gpt-5"
    last_prompt: Optional[str] = None

def _session_path(chat_id: int) -> Path:
    return chat_dir(chat_id) / SESSION_FILE

def load_session(chat_id: int) -> Session:
    path = _session_path(chat_id)
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Session(
                active_file=data.get("active_file"),
                model=data.get("model", "gpt-5"),
                last_prompt=data.get("last_prompt"),
            )
        except Exception as e:
            logger.warning(f"Failed to load session: {e}")
    return Session()

def save_session(chat_id: int, s: Session) -> None:
    path = _session_path(chat_id)
    try:
        path.write_text(json.dumps({
            "active_file": s.active_file,
            "model": s.model,
            "last_prompt": s.last_prompt,
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.error(f"Failed to save session: {e}")

def _zip_chat_dir(chat_id: int) -> Optional[Path]:
    base = chat_dir(chat_id)
    zip_path = base / f"chat_{chat_id}.zip"
    try:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            total_size = 0
            for p in base.rglob("*"):
                if p.is_file():
                    bs = p.stat().st_size
                    total_size += bs
                    if total_size > config.max_archive_size:
                        logger.warning("Archive too big, stopping add.")
                        break
                    zf.write(p, arcname=p.relative_to(base))
        return zip_path
    except Exception as e:
        logger.error(f"Zip error: {e}")
        return None

class SimpleGraphApp:
    """
    Минимальная синхронная реализация граф-приложения.
    Ожидает state={"chat_id": int, "input_text": str}
    """
    def invoke(self, state: Dict[str, Any], config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        chat_id = int(state.get("chat_id"))
        text: str = (state.get("input_text") or "").strip()
        if not text:
            return {"reply_text": "Пустой запрос."}

        session = load_session(chat_id)

        # --- Команды ---
        if text.startswith("/create"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                return {"reply_text": "Укажи имя файла: /create <filename>"}
            filename = sanitize_filename(parts[1])
            lang = detect_language(filename)
            ensure_latest_placeholder(chat_id, filename, lang)
            session.active_file = filename
            save_session(chat_id, session)
            reply = (
                f"Структурированный промпт готов для файла `{filename}`.\n"
                f"Выберите LLM для генерации кода."
            )
            return {"reply_text": reply}

        if text.startswith("/switch"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                return {"reply_text": "Укажи имя файла: /switch <filename>"}
            filename = sanitize_filename(parts[1])
            path = latest_path(chat_id, filename)
            if not path.exists():
                return {"reply_text": f"Файл `{filename}` не найден. Сначала /create {filename}"}
            session.active_file = filename
            save_session(chat_id, session)
            return {"reply_text": f"Текущий файл: `{filename}`"}

        if text.startswith("/files"):
            names = list_files(chat_id)
            if not names:
                return {"reply_text": "Файлов пока нет. Используйте /create <filename>."}
            lst = "\n".join(f"- {n}" for n in names)
            return {"reply_text": f"Файлы:\n{lst}"}

        if text.startswith("/model"):
            return {"reply_text": f"Текущая модель: {session.model}\nДоступные: {', '.join(sorted(VALID_CODEGEN_MODELS))}"}

        if text.startswith("/llm"):
            parts = text.split(maxsplit=1)
            if len(parts) < 2:
                return {"reply_text": f"Укажи модель: /llm <model>\nДоступные: {', '.join(sorted(VALID_CODEGEN_MODELS))}"}
            model = parts[1].strip()
            if model not in VALID_CODEGEN_MODELS:
                return {"reply_text": f"Неизвестная модель: {model}\nДоступные: {', '.join(sorted(VALID_CODEGEN_MODELS))}"}
            session.model = model
            save_session(chat_id, session)
            return {"reply_text": f"Модель установлена: {model}"}

        if text.startswith("/reset"):
            session = Session()
            save_session(chat_id, session)
            return {"reply_text": "Сессия сброшена."}

        if text.startswith("/download"):
            zp = _zip_chat_dir(chat_id)
            if not zp or not zp.exists():
                return {"reply_text": "Не удалось собрать архив."}
            # bot.py умеет отправлять файл, если сюда передать путь
            return {"reply_text": f"Готов архив: {zp.name}", "file_to_send": str(zp)}

        if text.startswith("/run"):
            # запускаем генерацию по последнему запросу пользователя (если он был),
            # либо сообщаем, что нужно прислать задачу.
            if not session.active_file:
                return {"reply_text": "Сначала выбери файл: /create <filename>"}
            if not session.last_prompt:
                return {"reply_text": "Пришли текст задачи (без команды), затем /run."}
            return self._generate(chat_id, session, session.last_prompt)

        # --- Обычный текст: считаем это задачей для генерации ---
        if not session.active_file:
            return {"reply_text": "Сначала создай/выбери файл: /create <filename> или /switch <filename>."}

        # Сохраняем последний промпт пользователя
        session.last_prompt = text
        save_session(chat_id, session)

        # Подсказка интерфейсу: можно предложить выбрать LLM, если ещё не выбрана
        if session.model not in VALID_CODEGEN_MODELS:
            hint = "Выберите LLM для генерации кода."
        else:
            hint = "Готово. Используй /run для генерации кода."
        return {
            "reply_text": (
                "Структурированный промпт готов.\n"
                f"{hint}\n"
                "Подсказка: /llm gpt-5 или /llm claude-opus-4-1-20250805"
            )
        }

    # --- Генерация кода ---
    def _generate(self, chat_id: int, session: Session, user_task: str) -> Dict[str, Any]:
        try:
            filename = session.active_file or "main.py"
            ctx = build_context_block(chat_id, filename)
            mode = config.adapter_output_pref.value  # по умолчанию FILES_JSON
            mode = infer_output_preference(user_task, has_context=bool(ctx))

            # 1) Сформировать адаптированный промпт
            adapter_prompt = render_adapter_prompt(
                raw_task=user_task,
                context_block=ctx,
                mode_tag=mode,
                output_pref=mode
            )
            adapter_out = call_adapter(adapter_prompt)
            messages = adapter_out.get("messages", [])
            response_contract = adapter_out.get("response_contract", {}) or {}
            mode = (response_contract.get("mode") or mode or "FILES_JSON").upper()

            # 2) Кодогенерация
            code_output = call_codegen(messages, mode=mode, model=session.model)

            # 3) Применение результата
            file_to_send: Optional[Path] = None
            reply_lines: List[str] = []

            if mode == "FILES_JSON":
                try:
                    parsed = json.loads(code_output)
                    files = parsed.get("files", [])
                    out_path = apply_files_json(chat_id, filename, files)
                    file_to_send = out_path
                    reply_lines.append("Код сгенерирован и сохранён (FILES_JSON).")
                except Exception as e:
                    logger.error(f"FILES_JSON parse/apply error: {e}")
                    # как fallback — просто записать в активный файл
                    out_path = version_current_file(chat_id, filename, code_output)
                    file_to_send = out_path
                    reply_lines.append("Не удалось распарсить FILES_JSON, записан raw вывод в активный файл.")
            else:
                # Режимы diff/code-only — сейчас записываем как контент в активный файл
                out_path = version_current_file(chat_id, filename, code_output)
                file_to_send = out_path
                reply_lines.append(f"Код сгенерирован в режиме {mode} и записан в `{filename}`.")

            # 4) Аудит
            try:
                audit_event(
                    chat_id=chat_id,
                    event_type="GENERATE",
                    active_file=filename,
                    model=session.model,
                    prompt=user_task,
                    output_path=file_to_send,
                    meta={"mode": mode}
                )
            except Exception as e:
                logger.warning(f"Audit event failed: {e}")

            reply = "\n".join(reply_lines)
            return {"reply_text": reply, "file_to_send": str(file_to_send) if file_to_send else None}

        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return {"reply_text": f"Ошибка генерации: {e}"}


# Экспортируем билдер и объект APP — чтобы bot.py смог их импортировать
def build_app() -> SimpleGraphApp:
    return SimpleGraphApp()

APP = build_app()
