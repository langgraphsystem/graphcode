from __future__ import annotations
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
    checkpoint_ns: str = "prod-bot"
    safe_mode: bool = False

    def __post_init__(self):
        if not self.token:
            raise RuntimeError("TELEGRAM_BOT_TOKEN is required")


# Load configuration from environment
config = BotConfig(
    token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
    app_module=os.environ.get("APP_MODULE", "graph_app"),
    app_name=os.environ.get("APP_NAME", "APP"),
    log_level=os.environ.get("LOG_LEVEL", "INFO").upper(),
    graph_timeout=float(os.environ.get("GRAPH_TIMEOUT", "60.0")),
    checkpoint_ns=os.environ.get("CHECKPOINT_NS", "prod-bot"),
    safe_mode=os.environ.get("BOT_SAFE_MODE", "false").lower() == "true",
    allowed_users=None,
)

# =========================
# Logging Setup
# =========================

logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("bot")

# Reduce noise from telegram library
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

# =========================
# Safe Graph Import
# =========================

APP = None
GRAPH_ERROR = None

def safe_import_graph():
    """Безопасный импорт graph приложения."""
    global APP, GRAPH_ERROR
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


# Попытка импорта при запуске
GRAPH_AVAILABLE = safe_import_graph()

if not GRAPH_AVAILABLE:
    logger.warning(f"⚠️ Graph недоступен: {GRAPH_ERROR}")
    logger.warning("🔄 Бот будет работать в ограниченном режиме")

# =========================
# UI Constants
# =========================

class Emoji:
    LOADING = "⏳"
    OK = "✅"
    ERROR = "❌"
    WARNING = "⚠️"
    INFO = "ℹ️"
    ROBOT = "🤖"
    FILES = "🗂"
    CODE = "💻"
    SETTINGS = "⚙️"

HELP_TEXT = """
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
2. Send your task description
3. Bot prepares structured prompt via adapter
4. Choose LLM: GPT-5 or Claude Opus 4.1
5. Code is generated and saved

**Download Options:**
• `/download` — all files
• `/download latest` — only latest versions
• `/download versions` — only versioned files
• `/download <filename>` — specific file
"""

def get_status_keyboard() -> InlineKeyboardMarkup:
    """Клавиатура для статуса."""
    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(text="🔄 Retry Graph", callback_data="retry_graph"),
            InlineKeyboardButton(text="📊 Diagnostics", callback_data="run_diagnostics"),
        ],
        [
            InlineKeyboardButton(text=f"{Emoji.INFO} Help", callback_data="cmd_help"),
        ],
    ])

def get_llm_keyboard() -> InlineKeyboardMarkup:
    """Generate LLM selection keyboard."""
    if not GRAPH_AVAILABLE:
        return InlineKeyboardMarkup([
            [InlineKeyboardButton(text="❌ Graph not available", callback_data="graph_error")]
        ])

    return InlineKeyboardMarkup([
        [
            InlineKeyboardButton(text="🧠 GPT-5", callback_data="choose_gpt5"),
            InlineKeyboardButton(text="🎭 Claude Opus 4.1", callback_data="choose_claude"),
        ],
        [
            InlineKeyboardButton(text="▶️ Run with current", callback_data="run_current"),
        ],
        [
            InlineKeyboardButton(text=f"{Emoji.INFO} Help", callback_data="cmd_help"),
            InlineKeyboardButton(text=f"{Emoji.FILES} Files", callback_data="cmd_files"),
            InlineKeyboardButton(text=f"{Emoji.SETTINGS} Model Info", callback_data="cmd_model"),
        ],
    ])

# =========================
# User Authorization
# =========================

def is_authorized(user_id: int) -> bool:
    """Check if user is authorized to use the bot."""
    if config.allowed_users is None:
        return True
    return user_id in config.allowed_users

async def check_authorization(update: Update) -> bool:
    """Check authorization and send message if not authorized."""
    if not update.effective_user:
        return False

    if not is_authorized(update.effective_user.id):
        if update.effective_message:
            await update.effective_message.reply_text(
                f"{Emoji.ERROR} You are not authorized to use this bot.\n"
                "Contact the administrator for access."
            )
        return False
    return True

# =========================
# Graph Integration
# =========================

def get_config_for_chat(chat_id: int, additional_config: Optional[Dict] = None) -> Dict[str, Any]:
    """Generate configuration for LangGraph with optional extensions."""
    base_config = {
        "configurable": {
            "thread_id": f"tg-{chat_id}",
            "checkpoint_ns": config.checkpoint_ns,
            "user_id": str(chat_id),
            "session_type": "telegram_bot",
            "bot_version": "3.1",
        }
    }
    if additional_config:
        base_config["configurable"].update(additional_config)
    return base_config

async def invoke_graph_with_retry(
    chat_id: int,
    text: str,
    max_retries: int = 3,
    timeout: float = None
) -> Dict[str, Any]:
    """Invoke graph with retry logic and proper error handling."""
    if not GRAPH_AVAILABLE:
        raise RuntimeError(f"Graph недоступен: {GRAPH_ERROR}")

    if timeout is None:
        timeout = config.graph_timeout

    state = {"chat_id": chat_id, "input_text": text}
    cfg = get_config_for_chat(chat_id)
    last_error = None

    for attempt in range(max_retries):
        try:
            logger.info(f"Graph invoke attempt {attempt + 1}/{max_retries} for chat_id: {chat_id}")
            loop = asyncio.get_running_loop()
            result: Dict[str, Any] = await asyncio.wait_for(
                loop.run_in_executor(None, lambda: APP.invoke(state, config=cfg)),
                timeout=timeout,
            )
            if result:
                logger.info(f"Successfully invoked graph for chat_id: {chat_id}")
                return result
            else:
                logger.warning(f"Empty result from graph for chat_id: {chat_id}")
                return {"reply_text": "Получен пустой результат от Graph."}
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt + 1} for chat_id: {chat_id}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)
        except Exception as e:
            last_error = e
            logger.error(f"Error on attempt {attempt + 1} for chat_id: {chat_id}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)

    logger.error(f"All {max_retries} attempts failed for chat_id: {chat_id}. Last error: {last_error}")
    raise last_error if last_error else Exception("Unknown error occurred")

# =========================
# Diagnostics
# =========================

async def run_diagnostics() -> str:
    """Запуск диагностики системы."""
    results = []

    results.append("🔍 **ДИАГНОСТИКА СИСТЕМЫ**\n")
    env_checks = {
        "TELEGRAM_BOT_TOKEN": bool(os.getenv("TELEGRAM_BOT_TOKEN")),
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "ANTHROPIC_API_KEY": bool(os.getenv("ANTHROPIC_API_KEY")),
    }

    results.append("🔑 **Переменные окружения:**")
    for var, status in env_checks.items():
        emoji = "✅" if status else "❌"
        results.append(f"{emoji} {var}")

    file_checks = {
        "graph_app.py": Path("graph_app.py").exists(),
        "config/prompt_adapter.json": Path("config/prompt_adapter.json").exists(),
        "prompt_adapter.json": Path("prompt_adapter.json").exists(),
    }

    results.append("\n📁 **Файлы:**")
    for file, exists in file_checks.items():
        emoji = "✅" if exists else "❌"
        results.append(f"{emoji} {file}")

    results.append(f"\n🧠 **Graph Engine:**")
    if GRAPH_AVAILABLE:
        results.append("✅ Доступен и импортирован")
    else:
        results.append(f"❌ Недоступен: {GRAPH_ERROR}")

    results.append(f"\n🔌 **API тесты:**")

    try:
        if os.getenv("OPENAI_API_KEY"):
            from openai import OpenAI
            client = OpenAI()
            results.append("✅ OpenAI клиент инициализирован")
        else:
            results.append("⚠️ OpenAI ключ не установлен")
    except Exception as e:
        results.append(f"❌ OpenAI ошибка: {str(e)[:50]}")

    try:
        if os.getenv("ANTHROPIC_API_KEY"):
            from anthropic import Anthropic
            client = Anthropic()
            results.append("✅ Anthropic клиент инициализирован")
        else:
            results.append("⚠️ Anthropic ключ не установлен")
    except Exception as e:
        results.append(f"❌ Anthropic ошибка: {str(e)[:50]}")

    return "\n".join(results)

# =========================
# Message Handling
# =========================

async def send_long_message(
    update: Update,
    text: str,
    parse_mode: Optional[str] = None,
    reply_markup: Optional[InlineKeyboardMarkup] = None
) -> None:
    """Send long message splitting into chunks if needed."""
    if not update.effective_message:
        return

    msg = update.effective_message
    limit = config.max_message_length

    if len(text) <= limit:
        try:
            await msg.reply_text(
                text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        except TelegramError as e:
            logger.error(f"Failed to send message: {e}")
            await msg.reply_text(
                text,
                disable_web_page_preview=True,
                reply_markup=reply_markup,
            )
        return

    chunks = [text[i:i + limit] for i in range(0, len(text), limit)]
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        try:
            await msg.reply_text(
                chunk,
                parse_mode=parse_mode if is_last else None,
                disable_web_page_preview=True,
                reply_markup=reply_markup if is_last else None,
            )
        except TelegramError:
            await msg.reply_text(
                chunk,
                disable_web_page_preview=True,
                reply_markup=reply_markup if is_last else None,
            )
        if not is_last:
            await asyncio.sleep(0.1)

def detect_llm_choice_needed(reply: str) -> bool:
    """Check if response indicates LLM choice is needed."""
    if not reply:
        return False

    markers = [
        "Structured prompt ready",
        "Choose LLM for code generation",
        "→ /llm",
        "→ /run",
        "user decision",
        "select llm",
        "choose model",
        "Структурированный промпт готов",
        "Выберите LLM для генерации кода",
        "Использовать текущую",
        "выбрать модель",
        "выберите llm",
        "model selection",
        "выбор модели",
    ]
    reply_lower = reply.lower()
    return any(marker.lower() in reply_lower for marker in markers)

async def send_file_if_exists(
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    file_path: str
) -> bool:
    """Send file if it exists."""
    try:
        path = Path(file_path)
        if not path.exists():
            return False

        if path.stat().st_size > 50 * 1024 * 1024:
            await context.bot.send_message(
                chat_id,
                f"{Emoji.WARNING} File too large to send: {path.name}",
            )
            return False

        with open(path, "rb") as f:
            await context.bot.send_document(
                chat_id,
                document=f,
                filename=path.name,
                caption=f"📦 Archive: {path.name}",
            )
        return True
    except Exception as e:
        logger.error(f"Failed to send file: {e}")
        return False

# =========================
# Command Handlers
# =========================

async def on_start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /start command."""
    if not await check_authorization(update):
        return

    user = update.effective_user
    status_emoji = "✅" if GRAPH_AVAILABLE else "⚠️"
    status_text = "Полная функциональность" if GRAPH_AVAILABLE else "Ограниченный режим"
    welcome_text = (
        f"{Emoji.OK} Welcome{f', {user.first_name}' if user else ''}!\n\n"
        f"{Emoji.ROBOT} AI Code Generator powered by LangGraph\n"
        f"Статус: {status_emoji} {status_text}\n\n"
        "Доступные команды:\n"
        "• `/help` - полная справка\n"
        "• `/status` - статус системы\n"
        "• `/diagnostics` - диагностика проблем\n"
    )

    if GRAPH_AVAILABLE:
        welcome_text += "\n• `/create calculator.py` - начать кодирование"
    else:
        welcome_text += f"\n⚠️ Graph недоступен: {GRAPH_ERROR[:100]}"

    await update.message.reply_text(welcome_text)

async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if not await check_authorization(update):
        return

    help_text = HELP_TEXT
    if not GRAPH_AVAILABLE:
        help_text += f"\n\n⚠️ **Внимание:** Graph Engine недоступен.\n"
        help_text += f"Причина: {GRAPH_ERROR}\n"
        help_text += "Команды генерации кода не работают.\n"
        help_text += "Используйте `/diagnostics` для проверки."

    await send_long_message(
        update,
        help_text,
        parse_mode=ParseMode.MARKDOWN,
    )

async def on_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command."""
    if not await check_authorization(update):
        return

    status_text = (
        f"**📊 BOT STATUS**\n\n"
        f"🤖 Bot: ✅ Running\n"
        f"🧠 Graph Engine: {('✅ Available' if GRAPH_AVAILABLE else '❌ Unavailable')}\n"
        f"📊 Log Level: {config.log_level}\n"
        f"🔌 Module: {config.app_module}.{config.app_name}\n"
        f"⏱️ Timeout: {config.graph_timeout}s\n"
        f"🏷️ Namespace: {config.checkpoint_ns}\n"
    )
    if not GRAPH_AVAILABLE:
        status_text += f"\n❌ **Graph Error:**\n{GRAPH_ERROR}"

    await update.message.reply_text(
        status_text,
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=get_status_keyboard(),
    )

async def on_diagnostics(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /diagnostics command."""
    if not await check_authorization(update):
        return

    loading = await update.message.reply_text(f"{Emoji.LOADING} Запуск диагностики...")
    try:
        diagnostics_result = await run_diagnostics()
        await loading.delete()
        await send_long_message(
            update,
            diagnostics_result,
            parse_mode=ParseMode.MARKDOWN,
        )
    except Exception as e:
        await loading.delete()
        await update.message.reply_text(
            f"{Emoji.ERROR} Ошибка диагностики: {str(e)[:200]}"
        )

async def on_graph_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle graph-dependent commands."""
    if not await check_authorization(update):
        return

    if not GRAPH_AVAILABLE:
        await update.message.reply_text(
            f"{Emoji.ERROR} **Graph Engine недоступен**\n\n"
            f"Причина: {GRAPH_ERROR}\n\n"
            "Что делать:\n"
            "1. Проверьте `/diagnostics`\n"
            "2. Убедитесь что graph_app.py существует\n"
            "3. Проверьте API ключи в .env\n"
            "4. Перезапустите бота",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=get_status_keyboard(),
        )
        return

    await process_graph_command(update, context)

async def process_graph_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process command through graph engine."""
    chat_id = update.effective_chat.id
    payload = (update.message.text or "").strip()
    if not payload:
        await update.message.reply_text(
            f"{Emoji.WARNING} Empty message. Send a command or task description."
        )
        return

    typing_task = asyncio.create_task(
        context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
    )
    loading = await update.message.reply_text(f"{Emoji.LOADING} Processing request...")

    try:
        start_time = time.time()
        result = await invoke_graph_with_retry(chat_id, payload)
        elapsed = time.time() - start_time
        logger.info(f"Graph processed in {elapsed:.2f}s for chat {chat_id}")
        reply = result.get("reply_text", "Completed.")

        if detect_llm_choice_needed(reply):
            await send_long_message(update, reply)
            await update.message.reply_text(
                "Select model for code generation or run with current:",
                reply_markup=get_llm_keyboard(),
            )
        else:
            await send_long_message(update, reply)

        file_to_send = result.get("file_to_send")
        if file_to_send:
            await send_file_if_exists(context, chat_id, file_to_send)
    except Exception as e:
        logger.exception(f"Graph command error for chat {chat_id}")
        error_msg = str(e)
        if "graph недоступен" in error_msg.lower():
            error_detail = "Graph engine недоступен"
        elif "timeout" in error_msg.lower():
            error_detail = "Request timeout"
        elif "api" in error_msg.lower():
            error_detail = "API connection issue"
        else:
            error_detail = "Processing failed"

        await update.message.reply_text(
            f"{Emoji.ERROR} {error_detail}\n"
            f"Details: {error_msg[:200]}"
        )
    finally:
        typing_task.cancel()
        with contextlib.suppress(Exception):
            await loading.delete()

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    query = update.callback_query
    if not query or not await check_authorization(update):
        if query:
            await query.answer("Unauthorized", show_alert=True)
        return

    await query.answer()
    data = query.data or ""
    if data == "retry_graph":
        global GRAPH_AVAILABLE, GRAPH_ERROR
        GRAPH_AVAILABLE = safe_import_graph()
        status = "✅ Успешно!" if GRAPH_AVAILABLE else f"❌ {GRAPH_ERROR}"
        await query.message.reply_text(f"🔄 Повторная попытка: {status}")
    elif data == "run_diagnostics":
        await on_diagnostics(update, context)
    elif data == "cmd_help":
        await on_help(update, context)
    elif data.startswith("choose_") or data == "run_current":
        if not GRAPH_AVAILABLE:
            await query.message.reply_text(
                f"{Emoji.ERROR} Graph недоступен для выбора модели"
            )
        else:
            command_map = {
                "choose_gpt5": "/llm gpt-5",
                "choose_claude": "/llm claude-opus-4-1-20250805",
                "run_current": "/run",
            }
            mapped_command = command_map.get(data)
            if mapped_command:
                update.message.text = mapped_command
                await process_graph_command(update, context)

async def on_error(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle unhandled errors."""
    logger.exception(f"Unhandled error: {context.error}")
    try:
        if isinstance(update, Update) and update.effective_chat:
            error_text = f"{Emoji.ERROR} Internal error occurred.\n"
            if isinstance(context.error, NetworkError):
                error_text += "Network issue detected. Please try again."
            elif isinstance(context.error, TimedOut):
                error_text += "Request timed out. Please try again."
            else:
                error_text += "Please try again or send /help"
            await context.bot.send_message(
                chat_id=update.effective_chat.id,
                text=error_text,
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# =========================
# Bot Setup
# =========================

async def post_init(application: Application) -> None:
    """Post-initialization setup."""
    commands = [
        BotCommand("start", "Initialize bot"),
        BotCommand("help", "Show help"),
        BotCommand("status", "Show system status"),
        BotCommand("diagnostics", "Run diagnostics"),
    ]
    if GRAPH_AVAILABLE:
        commands.extend([
            BotCommand("create", "Create/activate file"),
            BotCommand("switch", "Switch to file"),
            BotCommand("files", "List files"),
            BotCommand("model", "View models"),
            BotCommand("llm", "Select LLM"),
            BotCommand("run", "Run generation"),
            BotCommand("reset", "Reset state"),
            BotCommand("download", "Download files"),
        ])
    await application.bot.set_my_commands(commands)
    logger.info(f"Bot commands registered ({len(commands)} commands)")

def build_application() -> Application:
    """Build the Telegram bot application."""
    builder = ApplicationBuilder().token(config.token)
    builder.connection_pool_size(8)
    builder.connect_timeout(30.0)
    builder.read_timeout(30.0)
    builder.write_timeout(30.0)
    builder.pool_timeout(10.0)

    app = builder.build()

    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("status", on_status))
    app.add_handler(CommandHandler("diagnostics", on_diagnostics))

    if GRAPH_AVAILABLE:
        for cmd in ["create", "switch", "files", "model", "llm", "run", "reset", "download"]:
            app.add_handler(CommandHandler(cmd, on_graph_command))
        app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, process_graph_command))
    else:
        for cmd in ["create", "switch", "files", "model", "llm", "run", "reset", "download"]:
            app.add_handler(CommandHandler(cmd, on_graph_command))

    app.add_handler(CallbackQueryHandler(on_callback))
    app.add_error_handler(on_error)
    app.post_init = post_init
    return app

async def main() -> None:
    """Main entry point."""
    logger.info(f"Starting bot (Graph: {'✅' if GRAPH_AVAILABLE else '❌'})")
    app = build_application()
    await app.initialize()
    await app.start()
    try:
        logger.info("Bot polling started...")
        await app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True,
        )
        stop_event = asyncio.Event()

        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}")
            stop_event.set()

        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, signal_handler)

        await stop_event.wait()
    except Exception as e:
        logger.critical(f"Critical error: {e}")
    finally:
        logger.info("Stopping bot...")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        logger.info("Bot stopped gracefully")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot terminated by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)