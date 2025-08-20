# bot.py
import os, io, asyncio, logging
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import ApplicationBuilder, MessageHandler, CommandHandler, ContextTypes, filters

from graph_app import APP, DEFAULT_MODEL

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tg-bot")

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]

async def invoke_graph(chat_id: int, text: str):
    # thread_id привязывает состояние к чатy в чекпойнтере LangGraph
    return await asyncio.to_thread(
        APP.invoke,
        {"chat_id": chat_id, "input_text": text, "model": DEFAULT_MODEL},
        {"configurable": {"thread_id": str(chat_id)}},
    )

async def on_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Привет! Я кодогенератор на LangGraph.\n"
        "Команды: /create <file>, /switch <file>, /files, /model [name], /reset, "
        "/download [latest|versions|<file>]\n\n"
        "Отправь текст ТЗ, *.txt с ТЗ или *.diff/*.patch. "
        "Модель увидит текущий файл и дифф и внесёт минимальные правки."
    )

def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8-sig", "utf-8", "cp1251", "latin-1"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode("utf-8", errors="ignore")

async def _handle_text_to_graph(chat_id: int, payload: str, update: Update):
    try:
        result = await invoke_graph(chat_id, payload)
        # Текст ответа
        reply = result.get("reply_text", "Готово.")
        await update.message.reply_text(reply, disable_web_page_preview=True)
        # Отправка ZIP, если есть
        file_path = result.get("file_to_send")
        if file_path and os.path.exists(file_path):
            try:
                await update.message.reply_document(document=open(file_path, "rb"), filename=os.path.basename(file_path))
            except Exception:
                logger.exception("send zip error")
    except Exception as e:
        logger.exception("invoke error")
        await update.message.reply_text(f"Ошибка: {e}")

async def on_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    text = update.message.text or ""
    await _handle_text_to_graph(chat_id, text, update)

async def on_document(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    """Принимает .txt (как SPEC) или .diff/.patch (как DIFF); caption добавляется сверху."""
    chat_id = update.effective_chat.id
    doc = update.message.document
    caption = update.message.caption or ""
    try:
        tgfile = await ctx.bot.get_file(doc.file_id)
        bio = io.BytesIO()
        await tgfile.download_to_memory(out=bio)
        content = _decode_bytes(bio.getvalue())

        name = (doc.file_name or "file").lower()
        # Граф сам распознает SPEC vs DIFF. Чуть помечаем для читаемости.
        payload = caption.strip()
        if payload:
            payload += "\n\n"
        payload += f"[ATTACHED:{name}]\n"
        if name.endswith((".diff", ".patch")):
            payload += f"```diff\n{content}\n```"
        else:
            payload += content

        await _handle_text_to_graph(chat_id, payload, update)

    except Exception as e:
        logger.exception("doc error")
        await update.message.reply_text(f"Не удалось прочитать документ: {e}")

def main():
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", on_start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    app.add_handler(MessageHandler(filters.Document.ALL, on_document))  # .txt / .diff / .patch
    app.add_handler(MessageHandler(filters.COMMAND, on_text))           # /create, /switch, /files, /model, /reset, /download
    app.run_polling()  # без allowed_updates — безопаснее для PTB 20.x

if __name__ == "__main__":
    main()
