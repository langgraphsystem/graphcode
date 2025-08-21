from **future** import annotations
import asyncio
import importlib
import logging
import os
import sys
import signal
from pathlib import Path
from typing import Any, Dict, Optional, List
from dataclasses import dataclass
from enum import Enum
import time
import contextlib
import traceback

from telegram import (
Update,
InlineKeyboardMarkup,
InlineKeyboardButton,
BotCommand,
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
Application,
ApplicationBuilder,
CallbackQueryHandler,
CommandHandler,
ContextTypes,
MessageHandler,
filters,
)
from telegram.error import TelegramError, NetworkError, TimedOut

# =========================

# Configuration

# =========================

@dataclass
class BotConfig:
token: str
app_module: str = "graph_app"
app_name: str = "APP"
log_level: str = "INFO"
max_message_length: int = 4096
typing_delay: float = 0.5
error_retry_count: int = 3
error_retry_delay: float = 1.0
graph_timeout: float = 60.0
allowed_users: Optional[List[int]] = None
checkpoint_ns: str = “prod-bot”
safe_mode: bool = False

```
def __post_init__(self):
    if not self.token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN is required")
```

# Load configuration from environment

config = BotConfig(
token=os.environ.get(“TELEGRAM_BOT_TOKEN”, “”),
app_module=os.environ.get(“APP_MODULE”, “graph_app”),
app_name=os.environ.get(“APP_NAME”, “APP”),
log_level=os.environ.get(“LOG_LEVEL”, “INFO”).upper(),
graph_timeout=float(os.environ.get(“GRAPH_TIMEOUT”, “60.0”)),
checkpoint_ns=os.environ.get(“CHECKPOINT_NS”, “prod-bot”),
safe_mode=os.environ.get(“BOT_SAFE_MODE”, “false”).lower() == “true”,
allowed_users=None
)

# =========================

# Logging Setup

# =========================

logging.basicConfig(
level=config.log_level,
format=”%(asctime)s - %(name)s - %(levelname)s - %(message)s”,
handlers=[
logging.StreamHandler(),
logging.FileHandler(“bot.log”, encoding=“utf-8”)
]
)
logger = logging.getLogger(“bot”)

# Reduce noise from telegram library

logging.getLogger(“httpx”).setLevel(logging.WARNING)
logging.getLogger(“telegram”).setLevel(logging.INFO)

# =========================

# Safe Graph Import

# =========================

APP = None
GRAPH_ERROR = None

def safe_import_graph():
“”“Безопасный импорт graph приложения.”””
global APP, GRAPH_ERROR

```
try:
    logger.info(f"Попытка импорта {config.app_module}.{config.app_name}...")
    _mod = importlib.import_module(config.app_module)
    APP = getattr(_mod, config.app_name)
    logger.info(f"✅ Успешно импортирован {config.app_name} из {config.app_module}")
    return True
except ImportError as e:
    GRAPH_ERROR = f"Модуль не найден: {e}"
    logger.error(f"❌ Ошибка импорта модуля: {e}")
    return False
except AttributeError as e:
    GRAPH_ERROR = f"Объект {config.app_name} не найден в {config.app_module}: {e}"
    logger.error(f"❌ Ошибка доступа к объекту: {e}")
    return False
except Exception as e:
    GRAPH_ERROR = f"Неожиданная ошибка при импорте: {e}"
    logger.error(f"❌ Критическая ошибка импорта: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    return False
```

# Попытка импорта при запуске

GRAPH_AVAILABLE = safe_import_graph()

if not GRAPH_AVAILABLE:
logger.warning(f”⚠️ Graph недоступен: {GRAPH_ERROR}”)
logger.warning(“🔄 Бот будет работать в ограниченном режиме”)

# =========================

# UI Constants

# =========================

class Emoji:
LOADING = “⏳”
OK = “✅”
ERROR = “❌”
WARNING = “⚠️”
INFO = “ℹ️”
ROBOT = “🤖”
FILES = “🗂”
CODE = “💻”
SETTINGS = “⚙️”

HELP_TEXT = “””
🤖 **AI Code Generator on LangGraph**

**Basic Commands:**
• `/start` — Initialize bot
• `/help` — Show this help
• `/status` — Show system status
• `/diagnostics` — Run diagnostics

**Code Generation Commands:**
• `/create <file>` — Create/activate file
• `/switch <file>` — Switch to existing file  
• `/files` — List all files
• `/model` — View current models
• `/llm <model>` — Select code generation model
• `/run` — Execute prepared prompt
• `/reset` — Reset state
• `/download [filter]` — Download files as archive

**Available Models:**
• GPT-5 (default)
• Claude Opus 4.1 (`claude-opus-4-1-20250805`)

**How to Use:**

1. Start with `/create calculator.py`
1. Send your task description
1. Bot prepares structured prompt via adapter
1. Choose LLM: GPT-5 or Claude Opus 4.1
1. Code is generated and saved

**Download Options:**
• `/download` — all files
• `/download latest` — only latest versions
• `/download versions` — only versioned files
• `/download <filename>` — specific file
“””

def get_status_keyboard() -> InlineKeyboardMarkup:
“”“Клавиатура