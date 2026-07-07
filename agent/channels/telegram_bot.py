"""Telegram 渠道：把收到的文本交给 on_message 回调，回复其返回值。

渠道只负责收发，不掺业务逻辑——将来加新渠道（语音/QQ）实现同样的回调即可。
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Awaitable, Callable

from telegram import Update
from telegram.ext import Application, ContextTypes, MessageHandler, filters

# (text, username, chat_id, source_at, message_id) -> reply
OnMessage = Callable[[str, "str | None", int, "str | None", "str | None"], Awaitable[str]]
PostInit = Callable[[Application], Awaitable[None]]

log = logging.getLogger(__name__)


def build_app(
    token: str,
    allow_from: list[str],
    on_message: OnMessage,
    post_init: PostInit | None = None,
) -> Application:
    builder = (
        Application.builder()
        .token(token)
        .connect_timeout(float(os.environ.get("TELEGRAM_CONNECT_TIMEOUT", "10")))
        .read_timeout(float(os.environ.get("TELEGRAM_READ_TIMEOUT", "30")))
        .write_timeout(float(os.environ.get("TELEGRAM_WRITE_TIMEOUT", "20")))
        .pool_timeout(float(os.environ.get("TELEGRAM_POOL_TIMEOUT", "10")))
        .get_updates_connect_timeout(float(os.environ.get("TELEGRAM_GET_UPDATES_CONNECT_TIMEOUT", "10")))
        .get_updates_read_timeout(float(os.environ.get("TELEGRAM_GET_UPDATES_READ_TIMEOUT", "35")))
        .get_updates_write_timeout(float(os.environ.get("TELEGRAM_GET_UPDATES_WRITE_TIMEOUT", "20")))
        .get_updates_pool_timeout(float(os.environ.get("TELEGRAM_GET_UPDATES_POOL_TIMEOUT", "10")))
    )
    proxy_url = os.environ.get("TELEGRAM_PROXY_URL") or os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    if proxy_url:
        builder = builder.proxy(proxy_url).get_updates_proxy(proxy_url)
    if post_init is not None:
        builder = builder.post_init(post_init)
    app = builder.build()

    async def handle(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        if update.message is None or update.message.text is None:
            return
        user = update.effective_user.username if update.effective_user else None
        ctx.application.bot_data["telegram_last_update_monotonic"] = time.monotonic()
        ctx.application.bot_data["telegram_last_update_id"] = update.update_id
        # 白名单：allow_from 非空时只允许列表内用户
        if allow_from and user not in allow_from:
            print(
                f"Telegram 收到未授权消息: update_id={update.update_id} user={user!r}",
                flush=True,
            )
            await update.message.reply_text("（未授权用户）")
            return
        chat_id = update.effective_chat.id if update.effective_chat else 0
        source_at = update.message.date.isoformat() if update.message.date else None
        source_message_id = f"telegram:{chat_id}:{update.message.message_id}"
        text_len = len(update.message.text)
        print(
            f"Telegram 收到消息: update_id={update.update_id} chat_id={chat_id} "
            f"user={user!r} message_at={source_at} text_len={text_len}",
            flush=True,
        )
        reply = await on_message(update.message.text, user, chat_id, source_at, source_message_id)
        print(
            f"Telegram 业务处理完成: update_id={update.update_id} reply_len={len(reply)}",
            flush=True,
        )
        await update.message.reply_text(reply)
        print(f"Telegram 回复已发送: update_id={update.update_id}", flush=True)

    async def on_error(update: object, ctx: ContextTypes.DEFAULT_TYPE) -> None:
        exc = ctx.error
        exc_info = (type(exc), exc, exc.__traceback__) if exc else None
        log.error("Telegram update failed: update=%r", update, exc_info=exc_info)

    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle))
    app.add_error_handler(on_error)
    return app
