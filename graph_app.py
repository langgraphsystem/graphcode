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

# ----- LOGGING -----

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ----- CONSTANTS -----

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

# ----- CONFIGURATION -----

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
    adapter_targets=os.getenv(
        "ADAPTER_TARGETS",
        "Python 3.11; Ruff+Black; Pydantic v2; asyncio; type hints strict"
    ),
    adapter_constraints=os.getenv(
        "ADAPTER_CONSTRAINTS",
        "No secrets; reasonable perf; minimal deps; production quality"
    ),
    adapter_test_policy=os.getenv("ADAPTER_TEST_POLICY", "COMPREHENSIVE_TESTS"),
    adapter_output_lang=os.getenv("ADAPTER_OUTPUT_LANG", "EN"),
    adapter_output_pref=OutputPreference(os.getenv("ADAPTER_OUTPUT_PREF", "FILES_JSON")),
    request_timeout=int(os.getenv("REQUEST_TIMEOUT", "300")),
)

# ----- MODELS CONFIG -----

VALID_MODELS = {"gpt-5"}
VALID_CODEGEN_MODELS = {
    "gpt-5",
    "claude-opus-4-1-20250805",
}

# ----- CLIENTS -----

openai_client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    timeout=config.request_timeout
)

anthropic_client = None
if os.getenv("ANTHROPIC_API_KEY") and Anthropic is not None:
    anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# ----- FILE UTILITIES -----

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

# ----- CODE EXTRACTION -----

CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]+)?\n(.*?)```", re.DOTALL)
DIFF_BLOCK_RE = re.compile(r"```(diff|patch)\n(.*?)```", re.DOTALL | re.IGNORECASE)
UNIFIED_DIFF_HINT_RE = re.compile(r"(?m)^(\-\-\- |\+\+\+ |@@ )")
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
        return True    # ... (код продолжаетcя без типографских кавычек)