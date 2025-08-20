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
    graph_timeout: float = 60.0  # Timeout for graph operations
    allowed_users: Optional[List[int]] = None  # None = allow all
    checkpoint_ns: str = "prod-bot"  # Checkpoint namespace
    
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
    allowed_users=None  # Could parse from env: os.environ.get("ALLOWED_USERS", "").split(",")
)

# =========================
# Logging Setup
# =========================
logging.basicConfig(
    level=config.log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("bot.log", encoding="utf-8")
    ]
)
logger = logging.getLogger("bot")

# Reduce noise from telegram library
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("telegram").setLevel(logging.INFO)

# =========================
# Dynamic Import of Graph APP
# =========================
try:
    _mod = importlib.import_module(config.app_module)
    APP = getattr(_mod, config.app_name)
    logger.info(f"Successfully imported {config.app_name} from {config.app_module}")
except Exception as e:
    logger.critical(f"Failed to import graph APP from {config.app_module}.{config.app_name}: {e}")
    raise RuntimeError(f"Failed to import graph APP: {e}")

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
• `/create <file>` — Create/activate file
• `/switch <file>` — Switch to existing file  
• `/files` — List all files
• `/model` — View current models
• `/llm <model>` — Select code generation model
• `/run` — Execute prepared prompt
• `/reset` — Reset state
• `/download [filter]` — Download files as archive

**How to Use:**
1. Start with `/create app.py`
2. Send your task description
3. Bot prepares structured prompt via adapter
4. Choose LLM: GPT-5 or Claude Opus 4.1
5. Code is generated and saved

**Available Models:**
• GPT-5 (default)
• Claude Opus 4.1 (`claude-opus-4-1-20250805`)

**Download Options:**
• `/download` — all files
• `/download latest` — only latest versions
• `/download versions` — only versioned files
• `/download <filename>` — specific file
"""

# Dynamic keyboard generation
def get_llm_keyboard() -> InlineKeyboardMarkup:
    """Generate LLM selection keyboard."""
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
# Enhanced Graph Helpers
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
    if timeout is None:
        timeout = config.graph_timeout
    
    state = {"chat_id": chat_id, "input_text": text}
    cfg = get_config_for_chat(chat_id)
    
    last_error = None
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Graph invoke attempt {attempt + 1}/{max_retries} for chat_id: {chat_id}")
            
            # Run in executor to avoid blocking with timeout
            loop = asyncio.get_running_loop()
            result: Dict[str, Any] = await asyncio.wait_for(
                loop.run_in_executor(
                    None, 
                    lambda: APP.invoke(state, config=cfg)
                ),
                timeout=timeout
            )
            
            if result:
                logger.info(f"Successfully invoked graph for chat_id: {chat_id}")
                return result
            else:
                logger.warning(f"Empty result from graph for chat_id: {chat_id}")
                return {}
                
        except asyncio.TimeoutError as e:
            last_error = e
            logger.warning(f"Timeout on attempt {attempt + 1} for chat_id: {chat_id}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # exponential backoff
                
        except Exception as e:
            last_error = e
            logger.error(f"Error on attempt {attempt + 1} for chat_id: {chat_id}: {str(e)}")
            if attempt < max_retries - 1:
                await asyncio.sleep(2 ** attempt)  # exponential backoff
    
    # Если все попытки неудачны
    logger.error(f"All {max_retries} attempts failed for chat_id: {chat_id}. Last error: {last_error}")
    raise last_error if last_error else Exception("Unknown error occurred")

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
    
    # Clean up text for Telegram
    text = text.replace("```", "```\n") if "```" in text else text
    
    if len(text) <= limit:
        try:
            await msg.reply_text(
                text,
                parse_mode=parse_mode,
                disable_web_page_preview=True,
                reply_markup=reply_markup
            )
        except TelegramError as e:
            logger.error(f"Failed to send message: {e}")
            # Retry without parse mode
            await msg.reply_text(
                text,
                disable_web_page_preview=True,
                reply_markup=reply_markup
            )
        return
    
    # Split into chunks
    chunks = []
    current_chunk = ""
    
    for line in text.split('\n'):
        if len(current_chunk) + len(line) + 1 > limit:
            chunks.append(current_chunk)
            current_chunk = line
        else:
            current_chunk += '\n' + line if current_chunk else line
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Send chunks
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        try:
            await msg.reply_text(
                chunk,
                parse_mode=parse_mode if is_last else None,
                disable_web_page_preview=True,
                reply_markup=reply_markup if is_last else None
            )
        except TelegramError:
            await msg.reply_text(
                chunk,
                disable_web_page_preview=True,
                reply_markup=reply_markup if is_last else None
            )
        
        if not is_last:
            await asyncio.sleep(0.1)  # Small delay between chunks

def detect_llm_choice_needed(reply: str) -> bool:
    """Check if response indicates LLM choice is needed."""
    if not reply:
        return False
        
    markers = [
        # EN
        "Structured prompt ready",
        "Choose LLM for code generation",
        "→ /llm",
        "→ /run",
        "user decision",
        "select llm",
        "choose model",
        
        # RU (совпадает с сообщениями из графа)
        "Структурированный промпт готов",
        "Выберите LLM для генерации кода",
        "Использовать текущую",
        "выбрать модель",
        "выберите llm",
        
        # Дополнительные паттерны
        "model selection",
        "выбор модели"
    ]
    
    # Приводим к нижнему регистру для более надёжного поиска
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
        
        # Check file size
        if path.stat().st_size > 50 * 1024 * 1024:  # 50MB limit
            await context.bot.send_message(
                chat_id,
                f"{Emoji.WARNING} File too large to send: {path.name}"
            )
            return False
        
        with open(path, 'rb') as f:
            await context.bot.send_document(
                chat_id,
                document=f,
                filename=path.name,
                caption=f"📦 Archive: {path.name}"
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
    welcome_text = (
        f"{Emoji.OK} Welcome{f', {user.first_name}' if user else ''}!\n\n"
        f"{Emoji.ROBOT} I'm an AI code generator powered by LangGraph.\n"
        "I use GPT-5 adapter with your choice of GPT-5 or Claude for generation.\n\n"
        "Send /help for commands or start with:\n"
        "→ `/create app.py` to begin coding"
    )
    
    await update.message.reply_text(welcome_text, parse_mode=ParseMode.MARKDOWN)

async def on_help(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /help command."""
    if not await check_authorization(update):
        return
    
    await update.message.reply_text(
        HELP_TEXT,
        parse_mode=ParseMode.MARKDOWN,
        disable_web_page_preview=True
    )

async def on_status(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle /status command - show bot status."""
    if not await check_authorization(update):
        return
    
    try:
        # Test graph availability with short timeout
        test_result = await invoke_graph_with_retry(
            update.effective_chat.id,
            "/model",
            max_retries=1,
            timeout=10.0
        )
        graph_status = "✅ Operational" if test_result else "⚠️ Limited"
    except asyncio.TimeoutError:
        graph_status = "⏱️ Slow response"
    except Exception:
        graph_status = "❌ Unavailable"
    
    status_text = (
        f"**Bot Status**\n\n"
        f"🤖 Bot: ✅ Running\n"
        f"🧠 Graph Engine: {graph_status}\n"
        f"📊 Log Level: {config.log_level}\n"
        f"🔌 Module: {config.app_module}.{config.app_name}\n"
        f"⏱️ Timeout: {config.graph_timeout}s\n"
        f"🏷️ Namespace: {config.checkpoint_ns}"
    )
    
    await update.message.reply_text(status_text, parse_mode=ParseMode.MARKDOWN)

async def on_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle inline button callbacks."""
    query = update.callback_query
    if not query:
        return
    
    # Check authorization
    if not await check_authorization(update):
        await query.answer("Unauthorized", show_alert=True)
        return
    
    await query.answer()
    
    chat_id = update.effective_chat.id
    data = query.data or ""
    
    # Map callback data to commands
    command_map = {
        "cmd_help": "/help",
        "cmd_files": "/files",
        "cmd_model": "/model",
        "choose_gpt5": "/llm gpt-5",
        "choose_claude": "/llm claude-opus-4-1-20250805",
        "run_current": "/run",
    }
    
    mapped_command = command_map.get(data)
    if not mapped_command:
        await query.message.reply_text(f"{Emoji.ERROR} Unknown command")
        return
    
    # Process as text command
    await process_text(
        query.message,
        context,
        chat_id,
        mapped_command
    )

async def on_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Handle all graph commands."""
    if not await check_authorization(update):
        return
    
    await on_text(update, context)

async def on_text(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    override_text: Optional[str] = None
) -> None:
    """Handle text messages and commands."""
    if not await check_authorization(update):
        return
    
    if not update.message:
        return
    
    chat_id = update.effective_chat.id
    payload = override_text or (update.message.text or "").strip()
    
    await process_text(update.message, context, chat_id, payload)

async def process_text(
    message: Any,
    context: ContextTypes.DEFAULT_TYPE,
    chat_id: int,
    payload: str
) -> None:
    """Process text through graph engine with enhanced error handling."""
    if not payload:
        await message.reply_text(
            f"{Emoji.WARNING} Empty message. Send a command or task description."
        )
        return
    
    # Show typing indicator
    typing_task = None
    try:
        typing_task = asyncio.create_task(
            context.bot.send_chat_action(chat_id=chat_id, action=ChatAction.TYPING)
        )
    except Exception as e:
        logger.warning(f"Failed to send typing indicator: {e}")
    
    # Send loading message
    loading = None
    try:
        loading = await message.reply_text(f"{Emoji.LOADING} Processing request...")
    except Exception as e:
        logger.error(f"Failed to send loading message: {e}")
    
    try:
        # Invoke graph
        start_time = time.time()
        result = await invoke_graph_with_retry(chat_id, payload)
        elapsed = time.time() - start_time
        
        logger.info(f"Graph processed in {elapsed:.2f}s for chat {chat_id}")
        
    except asyncio.TimeoutError:
        logger.warning(f"Graph timeout for chat {chat_id}")
        if loading:
            with contextlib.suppress(Exception):
                await loading.delete()
        
        await message.reply_text(
            f"{Emoji.WARNING} Request timed out after {config.graph_timeout}s\n"
            "The operation may be too complex. Try simplifying your request."
        )
        return
        
    except Exception as e:
        logger.exception(f"Graph invoke error for chat {chat_id}")
        if loading:
            with contextlib.suppress(Exception):
                await loading.delete()
        
        error_msg = str(e)
        if "api" in error_msg.lower():
            error_detail = "API connection issue"
        elif "timeout" in error_msg.lower():
            error_detail = "Request timeout"
        elif "rate" in error_msg.lower():
            error_detail = "Rate limit exceeded"
        elif "key" in error_msg.lower():
            error_detail = "API key issue"
        else:
            error_detail = "Processing failed"
        
        await message.reply_text(
            f"{Emoji.ERROR} {error_detail}\n"
            f"Details: {error_msg[:200]}"
        )
        return
    
    finally:
        # Cancel typing task
        if typing_task:
            typing_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await typing_task
    
    # Delete loading message
    if loading:
        with contextlib.suppress(Exception):
            await loading.delete()
    
    # Extract response
    reply = result.get("reply_text", "Completed.")
    file_to_send = result.get("file_to_send")
    
    # Check if LLM choice is needed
    if detect_llm_choice_needed(reply):
        await send_long_message(update, reply)
        await message.reply_text(
            "Select model for code generation or run with current:",
            reply_markup=get_llm_keyboard()
        )
    else:
        await send_long_message(update, reply)
    
    # Send file if available
    if file_to_send:
        await send_file_if_exists(context, chat_id, file_to_send)

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
                text=error_text
            )
    except Exception as e:
        logger.error(f"Failed to send error message: {e}")

# =========================
# Health Check & Monitoring
# =========================
async def health_check() -> bool:
    """Perform basic health check of the bot systems."""
    try:
        # Test graph import
        if not APP:
            return False
        
        # Could add more checks here (database, APIs, etc.)
        return True
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return False

async def periodic_cleanup():
    """Periodic cleanup task."""
    while True:
        try:
            await asyncio.sleep(3600)  # Run every hour
            logger.info("Running periodic cleanup...")
            
            # Here you could add cleanup logic:
            # - Clear old checkpoints
            # - Clean temporary files
            # - Log statistics
            
        except asyncio.CancelledError:
            logger.info("Cleanup task cancelled")
            break
        except Exception as e:
            logger.error(f"Cleanup task error: {e}")

# =========================
# Bot Setup
# =========================
async def post_init(application: Application) -> None:
    """Post-initialization setup."""
    # Set bot commands
    commands = [
        BotCommand("start", "Initialize bot"),
        BotCommand("help", "Show help"),
        BotCommand("create", "Create/activate file"),
        BotCommand("switch", "Switch to file"),
        BotCommand("files", "List files"),
        BotCommand("model", "View models"),
        BotCommand("llm", "Select LLM"),
        BotCommand("run", "Run generation"),
        BotCommand("reset", "Reset state"),
        BotCommand("download", "Download files"),
        BotCommand("status", "Bot status"),
    ]
    
    await application.bot.set_my_commands(commands)
    logger.info("Bot commands registered")
    
    # Start cleanup task
    application.create_task(periodic_cleanup())
    logger.info("Periodic cleanup task started")

async def shutdown(application: Application) -> None:
    """Cleanup on shutdown."""
    logger.info("Bot shutting down...")
    
    # Cancel all tasks
    for task in asyncio.all_tasks():
        if not task.done():
            task.cancel()

# =========================
# Main Application
# =========================
def build_application() -> Application:
    """Build the Telegram bot application."""
    builder = ApplicationBuilder().token(config.token)
    
    # Configure connection pool with more robust settings
    builder.connection_pool_size(8)
    builder.connect_timeout(30.0)
    builder.read_timeout(30.0)
    builder.write_timeout(30.0)
    builder.pool_timeout(10.0)
    
    app = builder.build()
    
    # Add handlers
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(CommandHandler("help", on_help))
    app.add_handler(CommandHandler("status", on_status))
    
    # Graph commands
    for cmd in ["create", "switch", "files", "model", "llm", "run", "reset", "download"]:
        app.add_handler(CommandHandler(cmd, on_command))
    
    # Callbacks
    app.add_handler(CallbackQueryHandler(on_callback))
    
    # Plain text
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    
    # Error handler
    app.add_error_handler(on_error)
    
    # Lifecycle hooks
    app.post_init = post_init
    app.post_shutdown = shutdown
    
    return app

async def main() -> None:
    """Main entry point."""
    logger.info(f"Starting bot with module: {config.app_module}.{config.app_name}")
    logger.info(f"Checkpoint namespace: {config.checkpoint_ns}")
    logger.info(f"Graph timeout: {config.graph_timeout}s")
    
    # Health check before starting
    if not await health_check():
        logger.critical("Health check failed, aborting startup")
        sys.exit(1)
    
    # Build application
    app = build_application()
    
    # Initialize
    await app.initialize()
    await app.start()
    
    try:
        logger.info("Bot polling started...")
        await app.updater.start_polling(
            allowed_updates=Update.ALL_TYPES,
            drop_pending_updates=True
        )
        
        # Keep running
        stop_event = asyncio.Event()
        
        # Handle signals
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

# =========================
# Entry Point
# =========================
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        logger.info("Bot terminated by user")
    except Exception as e:
        logger.critical(f"Fatal error: {e}")
        sys.exit(1)
