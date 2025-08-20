# bot.py
import os
import io
import asyncio
import html
import logging
import re
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder,
    MessageHandler,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    filters
)

from graph_app import APP, DEFAULT_MODEL, VALID_MODELS

# Загрузка переменных окружения
load_dotenv()

# Настройка логирования
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("tg-bot")

# Конфигурация
TELEGRAM_TOKEN = os.environ.get("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("TELEGRAM_TOKEN not found in environment variables")

# Максимальный размер файла для обработки (в байтах)
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "10485760"))  # 10MB по умолчанию

# Эмодзи для улучшения UX
EMOJI = {
    "success": "✅",
    "error": "❌",
    "warning": "⚠️",
    "file": "📄",
    "folder": "📁",
    "code": "💻",
    "download": "📥",
    "model": "🧠",
    "reset": "♻️",
    "info": "ℹ️",
    "loading": "⏳",
}

# Тексты команд (оставляем как есть; без parse_mode будет обычный текст)
HELP_TEXT = """
🤖 **Кодогенератор на LangGraph**

**Основные команды:**
• `/start` - Начало работы
• `/help` - Эта справка
• `/create <file>` - Создать/активировать файл
• `/switch <file>` - Переключиться на файл
• `/files` - Список файлов
• `/model [name]` - Выбор/просмотр модели
• `/reset` - Сброс состояния чата
• `/download [filter]` - Скачать файлы
• `/status` - Текущий статус

**Как использовать:**
1. Создайте файл: `/create app.py`
2. Отправьте описание того, что нужно сделать
3. Бот сгенерирует код с помощью AI

**Поддерживаемые форматы:**
• Текстовые сообщения с ТЗ
• `.txt` файлы со спецификацией
• `.diff` или `.patch` файлы с изменениями

**Фильтры для /download:**
• `latest` - только актуальные версии
• `versions` - только версии с временными метками
• `<filename>` - конкретный файл
• без параметра - все файлы
"""

async def invoke_graph(chat_id: int, text: str, current_model: Optional[str] = None):
    """Вызов LangGraph с обработкой ошибок"""
    try:
        model = current_model or DEFAULT_MODEL
        # Привязываем состояние к чату через thread_id
        result = await asyncio.to_thread(
            APP.invoke,
            {
                "chat_id": chat_id,
                "input_text": text,
                "model": model
            },
            {"configurable": {"thread_id": str(chat_id)}},
        )
        return result
    except Exception as e:
        logger.error(f"Graph invocation error for chat {chat_id}: {e}", exc_info=True)
        raise

async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /start"""
    user = update.effective_user
    welcome_text = (
        f"Привет, {user.first_name}! 👋\n\n"
        f"{EMOJI['code']} Я кодогенератор на базе LangGraph.\n"
        f"Использую PROMPT-ADAPTER → Codegen для создания кода.\n\n"
        f"Начни с команды /create <filename> чтобы создать файл.\n"
        f"Или используй /help для подробной справки."
    )

    # Создаем клавиатуру с основными командами
    keyboard = [
        [
            InlineKeyboardButton("📄 Создать файл", callback_data="cmd_create"),
            InlineKeyboardButton("📁 Мои файлы", callback_data="cmd_files"),
        ],
        [
            InlineKeyboardButton("🧠 Модель", callback_data="cmd_model"),
            InlineKeyboardButton("❓ Помощь", callback_data="cmd_help"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)

    await update.message.reply_text(
        welcome_text,
        reply_markup=reply_markup
    )

async def on_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработчик команды /help"""
    help_text = """
🤖 Кодогенератор на LangGraph

Основные команды:
• /start - Начало работы
• /help - Эта справка
• /create <file> - Создать/активировать файл
• /switch <file> - Переключиться на файл
• /files - Список файлов
• /model - Информация о модели
• /reset - Сброс состояния чата
• /download [filter] - Скачать файлы
• /status - Текущий статус

Как использовать:
1. Создайте файл: /create app.py
2. Отправьте описание того, что нужно сделать
3. Бот сгенерирует код с помощью AI
"""
    await update.message.reply_text(
        help_text,
        disable_web_page_preview=True
    )

async def on_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Показать текущий статус чата"""
    chat_id = update.effective_chat.id

    try:
        # Получаем текущее состояние через пустой запрос к /files
        result = await invoke_graph(chat_id, "/files")
        files_info = result.get("reply_text", "Нет файлов")

        # Пробуем получить активный файл / модель
        result_model = await invoke_graph(chat_id, "/model")
        model_info = result_model.get("reply_text", "") or ""

        # Извлекаем информацию о модели (ищем текст в обратных кавычках)
        m = re.search(r"`([^`]+)`", model_info)
        current_model = m.group(1) if m else DEFAULT_MODEL

        status_text = (
            f"{EMOJI['info']} Статус чата\n\n"
            f"{EMOJI['model']} Модель: {current_model}\n"
            f"{EMOJI['folder']} Файлы:\n{files_info}"
        )

    except Exception as e:
        logger.error(f"Status check error: {e}", exc_info=True)
        status_text = f"{EMOJI['error']} Не удалось получить статус"

    await update.message.reply_text(status_text)

def _decode_bytes(b: bytes) -> str:
    """Декодирование байтов с автоопределением кодировки"""
    encodings = ["utf-8-sig", "utf-8", "cp1251", "cp1252", "latin-1", "iso-8859-1"]
    for enc in encodings:
        try:
            return b.decode(enc)
        except (UnicodeDecodeError, LookupError):
            continue
    return b.decode("utf-8", errors="ignore")

def escape_markdown_v2(text: str) -> str:
    """Экранирование специальных символов для MarkdownV2"""
    escape_chars = ['_', '*', '[', ']', '(', ')', '~', '`', '>', '#', '+', '-', '=', '|', '{', '}', '.', '!']
    for char in escape_chars:
        text = text.replace(char, f'\\{char}')
    return text

async def _handle_text_to_graph(chat_id: int, payload: str, update: Update):
    """Обработка текста через граф с отправкой результата"""
    try:
        # Показываем индикатор загрузки (без Markdown)
        loading_msg = await update.message.reply_text(
            f"{EMOJI['loading']} Обрабатываю запрос..."
        )

        # Вызываем граф
        result = await invoke_graph(chat_id, payload)

        # Удаляем сообщение о загрузке
        try:
            await loading_msg.delete()
        except:
            pass

        # Получаем ответ
        reply = result.get("reply_text", "Готово.")

        # Отправляем ответ БЕЗ ParseMode для избежания ошибок парсинга
        max_length = 4096
        if len(reply) > max_length:
            parts = [reply[i:i+max_length] for i in range(0, len(reply), max_length)]
            for part in parts:
                await update.message.reply_text(
                    part,
                    disable_web_page_preview=True
                )
        else:
            await update.message.reply_text(
                reply,
                disable_web_page_preview=True
            )

        # Если есть файл для отправки (например, архив)
        file_path = result.get("file_to_send")
        if file_path and os.path.exists(file_path):
            try:
                file_size = os.path.getsize(file_path)
                if file_size > 50 * 1024 * 1024:  # 50MB - лимит Telegram
                    await update.message.reply_text(
                        f"{EMOJI['warning']} Файл слишком большой ({file_size // (1024*1024)}MB). "
                        f"Максимум 50MB для Telegram."
                    )
                else:
                    with open(file_path, "rb") as f:
                        await update.message.reply_document(
                            document=f,
                            filename=os.path.basename(file_path),
                            caption=f"{EMOJI['download']} Архив готов"
                        )
            except Exception as e:
                logger.error(f"Failed to send file: {e}")
                await update.message.reply_text(
                    f"{EMOJI['error']} Не удалось отправить файл: {str(e)[:100]}"
                )

    except Exception as e:
        logger.exception("Error in text handling")
        error_msg = str(e)[:200]

        # Удаляем сообщение о загрузке если оно есть
        try:
            if 'loading_msg' in locals():
                await loading_msg.delete()
        except:
            pass

        # Формируем понятное сообщение об ошибке БЕЗ форматирования
        if "api_key" in error_msg.lower():
            error_text = f"{EMOJI['error']} Проблема с API ключом OpenAI"
        elif "rate" in error_msg.lower():
            error_text = f"{EMOJI['error']} Превышен лимит запросов к API. Попробуйте позже."
        elif "timeout" in error_msg.lower():
            error_text = f"{EMOJI['error']} Превышено время ожидания ответа от AI"
        else:
            error_text = f"{EMOJI['error']} Ошибка: {error_msg}"

        await update.message.reply_text(error_text)

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработчик текстовых сообщений"""
    chat_id = update.effective_chat.id
    text = update.message.text or ""

    # Логируем запрос
    user = update.effective_user
    logger.info(f"Text from {user.username or user.id}: {text[:100]}")

    await _handle_text_to_graph(chat_id, text, update)

async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработчик документов (.txt, .diff, .patch и др.)"""
    chat_id = update.effective_chat.id
    doc = update.message.document
    caption = update.message.caption or ""

    # Проверяем размер файла
    if doc.file_size > MAX_FILE_SIZE:
        await update.message.reply_text(
            f"{EMOJI['error']} Файл слишком большой ({doc.file_size // (1024*1024)}MB). "
            f"Максимум {MAX_FILE_SIZE // (1024*1024)}MB."
        )
        return

    # Логируем
    logger.info(f"Document received: {doc.file_name}, size: {doc.file_size}")

    try:
        # Скачиваем файл
        tgfile = await ctx.bot.get_file(doc.file_id)
        bio = io.BytesIO()
        await tgfile.download_to_memory(out=bio)
        content = _decode_bytes(bio.getvalue())

        # Определяем тип файла
        name = (doc.file_name or "file").lower()

        # Формируем payload
        payload = caption.strip()
        if payload:
            payload += "\n\n"

        # Добавляем метку о прикрепленном файле
        payload += f"[ATTACHED: {name}]\n"

        # Обрабатываем разные типы файлов
        if name.endswith((".diff", ".patch")):
            payload += f"```diff\n{content}\n```"
        elif name.endswith((".py", ".js", ".ts", ".java", ".cpp", ".c", ".go", ".rs")):
            ext = name.split(".")[-1]
            lang_map = {
                "py": "python", "js": "javascript", "ts": "typescript",
                "cpp": "cpp", "c": "c", "java": "java", "go": "go", "rs": "rust"
            }
            lang = lang_map.get(ext, ext)
            payload += f"```{lang}\n{content}\n```"
        elif name.endswith((".json", ".yaml", ".yml", ".toml", ".xml")):
            ext = name.split(".")[-1]
            payload += f"```{ext}\n{content}\n```"
        else:
            payload += content

        # Обрабатываем через граф
        await _handle_text_to_graph(chat_id, payload, update)

    except Exception as e:
        logger.exception("Document processing error")
        await update.message.reply_text(
            f"{EMOJI['error']} Не удалось обработать документ: {str(e)[:200]}"
        )

async def on_callback_query(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Обработчик inline кнопок"""
    query = update.callback_query
    await query.answer()

    chat_id = update.effective_chat.id
    data = query.data

    # Маппинг callback_data на команды
    command_map = {
        "cmd_create": "/create main.py",
        "cmd_files": "/files",
        "cmd_model": "/model",
        "cmd_help": "/help",
    }

    if data in command_map:
        if data == "cmd_help":
            await query.message.reply_text(
                HELP_TEXT,
                disable_web_page_preview=True
            )
        else:
            # Обрабатываем через граф
            command = command_map[data]

            # Создаем обёртку, чтобы _handle_text_to_graph ожидал update.message
            class FakeMessage:
                def __init__(self, msg):
                    self.message = msg  # telegram.Message

                # Эти методы не используются _handle_text_to_graph, но оставим про запас
                async def reply_text(self, text, **kwargs):
                    return await self.message.reply_text(text, **kwargs)

                async def reply_document(self, **kwargs):
                    return await self.message.reply_document(**kwargs)

            fake_update = FakeMessage(query.message)
            await _handle_text_to_graph(chat_id, command, fake_update)

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Глобальный обработчик ошибок"""
    logger.error("Exception while handling an update", exc_info=True)

    # Пытаемся отправить сообщение пользователю
    try:
        if update and update.effective_message:
            await update.effective_message.reply_text(
                f"{EMOJI['error']} Произошла внутренняя ошибка. Попробуйте позже."
            )
    except:
        pass

def main():
    """Главная функция запуска бота"""
    # Проверяем наличие необходимых переменных
    if not TELEGRAM_TOKEN:
        logger.error("TELEGRAM_TOKEN not found in environment")
        raise ValueError("TELEGRAM_TOKEN is required")

    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not found - API calls will fail")

    # Создаем приложение
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()

    # Регистрируем обработчики команд (block=True — чтобы не проваливались дальше в filters.COMMAND)
    app.add_handler(CommandHandler("start", on_start, block=True))
    app.add_handler(CommandHandler("help", on_help, block=True))
    app.add_handler(CommandHandler("status", on_status, block=True))

    # Обработчики сообщений
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))

    # Обработчик остальных команд (для /create, /switch, /files, /model, /reset, /download)
    app.add_handler(MessageHandler(filters.COMMAND, on_text))

    # Обработчик inline кнопок
    app.add_handler(CallbackQueryHandler(on_callback_query))

    # Глобальный обработчик ошибок
    app.add_error_handler(error_handler)

    # Запускаем бота
    try:
        handlers_count = len(app.handlers[0])
    except Exception:
        handlers_count = "unknown"

    logger.info(f"Starting bot with {handlers_count} handlers...")
    logger.info(f"Models available: {', '.join(sorted(VALID_MODELS))}")
    logger.info(f"Default model: {DEFAULT_MODEL}")

    # Запуск
    app.run_polling(
        allowed_updates=Update.ALL_TYPES,
        drop_pending_updates=True  # Игнорируем старые сообщения
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
