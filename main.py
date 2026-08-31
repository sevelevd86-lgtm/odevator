import os
import asyncio
import base64
import logging
from io import BytesIO

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import (
    Message,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
)
from aiogram.enums import ChatMemberStatus

from openai import OpenAI


# ============================================================
# НАСТРОЙКИ
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()

CHANNEL_USERNAME = "@musicalmeet"
CHANNEL_URL = "https://t.me/musicalmeet"

# Модель редактирования изображений
OPENAI_IMAGE_MODEL = "gpt-image-2"


# ============================================================
# ПРОВЕРКА SECRETS
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN не задан в Secrets"
    )

if not OPENAI_API_KEY:
    raise RuntimeError(
        "OPENAI_API_KEY не задан в Secrets"
    )


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

log = logging.getLogger(
    "whitebear-drip"
)


# ============================================================
# TELEGRAM
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)

dp = Dispatcher()


# ============================================================
# OPENAI
# ============================================================

openai_client = OpenAI(
    api_key=OPENAI_API_KEY
)


# ============================================================
# СОСТОЯНИЕ ПОЛЬЗОВАТЕЛЕЙ
# ============================================================

# Пользователи, которые уже проходили проверку
verified_users = set()

# Пользователи, которым сейчас обрабатывается фото
processing_users = set()


# ============================================================
# КЛАВИАТУРА ПОДПИСКИ
# ============================================================

def subscription_keyboard():

    return InlineKeyboardMarkup(
        inline_keyboard=[

            [
                InlineKeyboardButton(
                    text="📢 Подписаться на канал",
                    url=CHANNEL_URL
                )
            ],

            [
                InlineKeyboardButton(
                    text="✅ Проверить подписку",
                    callback_data="check_subscription"
                )
            ]

        ]
    )


# ============================================================
# ПРОВЕРКА ПОДПИСКИ
# ============================================================

async def check_subscription(
    user_id: int
) -> bool:

    try:

        member = await bot.get_chat_member(
            chat_id=CHANNEL_USERNAME,
            user_id=user_id
        )

        log.info(
            "Subscription check: user=%s status=%s",
            user_id,
            member.status
        )

        return member.status in {
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR
        }

    except Exception as e:

        log.exception(
            "Ошибка проверки подписки для user=%s: %r",
            user_id,
            e
        )

        return False


# ============================================================
# ИНСТРУКЦИЯ ПОСЛЕ ПОДПИСКИ
# ============================================================

async def send_photo_instruction(
    message: Message
):

    await message.answer(
        "Здравствуйте! 👋\n\n"
        "Теперь пришлите ваше фото в чат.\n\n"
        "📸 Отправьте <b>одно фото</b> человека, "
        "и я верну этого же человека, "
        "но уже в стильном <b>ДРИПЧИКЕ</b> 😎🔥\n\n"
        "Лучше всего подойдёт фото, где человека "
        "хорошо видно целиком или по пояс.",
        parse_mode="HTML"
    )


# ============================================================
# START
# ============================================================

@dp.message(
    CommandStart()
)
async def start(
    message: Message
):

    if not message.from_user:
        return

    user_id = message.from_user.id

    log.info(
        "/start от пользователя %s",
        user_id
    )

    # Даже если пользователь проверялся раньше,
    # повторно проверяем актуальную подписку.
    subscribed = await check_subscription(
        user_id
    )

    if subscribed:

        verified_users.add(
            user_id
        )

        await send_photo_instruction(
            message
        )

        return

    verified_users.discard(
        user_id
    )

    await message.answer(
        "Здравствуйте! 👋\n\n"
        "Вначале вам надо подписаться на канал.",
        reply_markup=subscription_keyboard()
    )


# ============================================================
# ПРОВЕРКА КНОПКИ
# ============================================================

@dp.callback_query(
    F.data == "check_subscription"
)
async def check_subscription_callback(
    callback: types.CallbackQuery
):

    if not callback.from_user:
        await callback.answer()
        return

    user_id = callback.from_user.id

    log.info(
        "Проверка подписки по кнопке: user=%s",
        user_id
    )

    subscribed = await check_subscription(
        user_id
    )

    if not subscribed:

        await callback.answer(
            "❌ Вы ещё не подписались на канал",
            show_alert=True
        )

        return

    verified_users.add(
        user_id
    )

    await callback.answer(
        "✅ Подписка подтверждена!"
    )

    try:

        if callback.message:

            await callback.message.edit_text(
                "Здравствуйте! 👋\n\n"
                "Теперь пришлите ваше фото в чат.\n\n"
                "📸 Отправьте <b>одно фото</b> человека, "
                "и я верну этого же человека, "
                "но уже в стильном <b>ДРИПЧИКЕ</b> 😎🔥\n\n"
                "Лучше всего подойдёт фото, где человека "
                "хорошо видно целиком или по пояс.",
                parse_mode="HTML"
            )

    except Exception as e:

        log.exception(
            "Не удалось изменить сообщение после проверки подписки: %r",
            e
        )

        if callback.message:

            await callback.message.answer(
                "Здравствуйте! 👋\n\n"
                "Теперь пришлите ваше фото в чат.\n\n"
                "📸 Отправьте <b>одно фото</b> человека, "
                "и я верну этого же человека, "
                "но уже в стильном <b>ДРИПЧИКЕ</b> 😎🔥",
                parse_mode="HTML"
            )


# ============================================================
# ПОВТОРНАЯ ПРОВЕРКА ПОДПИСКИ
# ============================================================

async def require_subscription(
    message: Message
) -> bool:

    if not message.from_user:
        return False

    user_id = message.from_user.id

    subscribed = await check_subscription(
        user_id
    )

    if not subscribed:

        verified_users.discard(
            user_id
        )

        await message.answer(
            "❌ Чтобы пользоваться ботом, "
            "сначала подпишитесь на канал.",
            reply_markup=subscription_keyboard()
        )

        return False

    verified_users.add(
        user_id
    )

    return True


# ============================================================
# OPENAI IMAGE EDIT
# ============================================================

async def edit_photo_with_openai(
    image_bytes: bytes
) -> bytes:

    prompt = """
Edit the provided photograph.

MAIN OBJECTIVE:
Keep the exact same person from the input photograph
and change ONLY their clothing into a fashionable,
modern streetwear drip outfit.

IDENTITY PRESERVATION IS EXTREMELY IMPORTANT.

Keep the same:
- person
- face
- facial features
- eyes
- nose
- mouth
- hairstyle
- skin tone
- age
- body proportions
- body shape
- pose
- hands
- fingers
- expression
- camera angle
- perspective

Do NOT replace the person with another person.

Do NOT generate a new face.

Do NOT alter the person's identity.

CLOTHING:
Replace only the original clothing with a premium,
modern streetwear outfit.

The outfit should look like realistic contemporary
"drip" fashion.

Use combinations such as:
- premium oversized hoodie
- stylish jacket
- fashionable pants
- clean modern sneakers
- tasteful chains or accessories when appropriate

The clothing must naturally fit the person's body,
pose and perspective.

REALISM:
The result must look like a real photograph.

Do NOT make it:
- cartoon
- anime
- illustration
- painting
- 3D render
- artificial-looking character

Keep the original background as close as possible.

Keep the original lighting as close as possible.

Keep the original composition and camera perspective.

Do not unnecessarily modify anything except the clothing.

The final result should look like the same photograph
of the same person, but wearing a stylish premium
streetwear drip outfit.
"""

    def make_edit():

        log.info(
            "Sending image to OpenAI. Bytes=%s",
            len(image_bytes)
        )

        image_file = BytesIO(
            image_bytes
        )

        # Важно: имя файла помогает SDK определить формат.
        image_file.name = "person.jpg"

        try:

            result = openai_client.images.edit(
                model=OPENAI_IMAGE_MODEL,
                image=image_file,
                prompt=prompt,
                size="1024x1024",
                quality="medium"
            )

        except Exception as e:

            log.exception(
                "OpenAI images.edit ERROR: %r",
                e
            )

            raise RuntimeError(
                f"Ошибка OpenAI: {e}"
            ) from e

        # ====================================================
        # ПРОВЕРКА ОТВЕТА OPENAI
        # ====================================================

        if not result:

            raise RuntimeError(
                "OpenAI вернул пустой response"
            )

        if not result.data:

            raise RuntimeError(
                "OpenAI не вернул data"
            )

        first_result = result.data[0]

        # GPT Image обычно возвращает base64.
        encoded = getattr(
            first_result,
            "b64_json",
            None
        )

        if not encoded:

            raise RuntimeError(
                "OpenAI не вернул b64_json изображения"
            )

        try:

            decoded = base64.b64decode(
                encoded
            )

        except Exception as e:

            log.exception(
                "Ошибка декодирования base64: %r",
                e
            )

            raise RuntimeError(
                "Не удалось декодировать изображение OpenAI"
            ) from e

        if not decoded:

            raise RuntimeError(
                "После декодирования изображение пустое"
            )

        log.info(
            "OpenAI image received successfully. Bytes=%s",
            len(decoded)
        )

        return decoded

    # Синхронный OpenAI SDK запускаем
    # в отдельном потоке, чтобы не блокировать Telegram.
    return await asyncio.to_thread(
        make_edit
    )


# ============================================================
# СКАЧИВАНИЕ ФОТО TELEGRAM
# ============================================================

async def download_telegram_photo(
    file_id: str
) -> bytes:

    log.info(
        "Получение Telegram file: %s",
        file_id
    )

    telegram_file = await bot.get_file(
        file_id
    )

    if not telegram_file.file_path:

        raise RuntimeError(
            "Telegram не вернул file_path"
        )

    photo_buffer = BytesIO()

    await bot.download_file(
        telegram_file.file_path,
        destination=photo_buffer
    )

    source_bytes = (
        photo_buffer.getvalue()
    )

    if not source_bytes:

        raise RuntimeError(
            "Telegram скачал пустой файл"
        )

    log.info(
        "Telegram photo downloaded. Bytes=%s",
        len(source_bytes)
    )

    return source_bytes


# ============================================================
# ФОТО
# ============================================================

@dp.message(
    F.photo
)
async def photo_handler(
    message: Message
):

    if not message.from_user:
        return

    user_id = message.from_user.id

    log.info(
        "Получено фото от user=%s",
        user_id
    )

    # ========================================================
    # ПРОВЕРКА ПОДПИСКИ
    # ========================================================

    if not await require_subscription(
        message
    ):
        return

    # ========================================================
    # ЗАЩИТА ОТ ДВОЙНОЙ ОБРАБОТКИ
    # ========================================================

    if user_id in processing_users:

        await message.answer(
            "⏳ Я ещё обрабатываю ваше предыдущее фото.\n\n"
            "Пожалуйста, дождитесь результата."
        )

        return

    processing_users.add(
        user_id
    )

    try:

        # ====================================================
        # СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЮ
        # ====================================================

        await message.answer(
            "⏳ <b>Получил фото!</b>\n\n"
            "👕 Подбираю тебе <b>ДРИПЧИК</b>...\n"
            "🔥 Наношу новый образ.\n\n"
            "Это может занять некоторое время.",
            parse_mode="HTML"
        )

        # ====================================================
        # БЕРЁМ САМОЕ КАЧЕСТВЕННОЕ ФОТО
        # ====================================================

        photo = message.photo[-1]

        log.info(
            "Telegram photo selected: "
            "file_id=%s width=%s height=%s size=%s",
            photo.file_id,
            photo.width,
            photo.height,
            photo.file_size
        )

        # ====================================================
        # СКАЧИВАЕМ ФОТО
        # ====================================================

        source_bytes = await download_telegram_photo(
            photo.file_id
        )

        # ====================================================
        # OPENAI
        # ====================================================

        log.info(
            "Начинаю обработку OpenAI для user=%s",
            user_id
        )

        edited_bytes = await edit_photo_with_openai(
            source_bytes
        )

        # ====================================================
        # ПРОВЕРКА РЕЗУЛЬТАТА
        # ====================================================

        if not edited_bytes:

            raise RuntimeError(
                "OpenAI вернул пустой результат"
            )

        log.info(
            "Обработка завершена для user=%s. "
            "Result bytes=%s",
            user_id,
            len(edited_bytes)
        )

        # ====================================================
        # ОТПРАВЛЯЕМ ФОТО
        # ====================================================

        await message.answer_photo(
            photo=types.BufferedInputFile(
                edited_bytes,
                filename="drip.png"
            ),
            caption=(
                "🔥 <b>Вот твой ДРИПЧИК!</b>\n\n"
                "😎 Образ обновлён."
            ),
            parse_mode="HTML"
        )

        log.info(
            "Результат успешно отправлен user=%s",
            user_id
        )

    except Exception as e:

        # ====================================================
        # ПОЛНЫЙ ЛОГ ОШИБКИ
        # ====================================================

        log.exception(
            "Image processing error for user=%s: %r",
            user_id,
            e
        )

        # ====================================================
        # СООБЩЕНИЕ ПОЛЬЗОВАТЕЛЮ
        # ====================================================

        await message.answer(
            "❌ <b>Не получилось обработать фото.</b>\n\n"
            "Попробуйте отправить другое фото, "
            "где человека хорошо видно.\n\n"
            "Если ошибка повторяется, попробуйте "
            "ещё раз через некоторое время.",
            parse_mode="HTML"
        )

    finally:

        processing_users.discard(
            user_id
        )

        log.info(
            "Processing lock released for user=%s",
            user_id
        )


# ============================================================
# НЕ ФОТО
# ============================================================

@dp.message()
async def other_message(
    message: Message
):

    if not message.from_user:
        return

    if not await require_subscription(
        message
    ):
        return

    await message.answer(
        "📸 Пришлите именно <b>одно фото</b> человека.\n\n"
        "После этого я сделаю ему стильный "
        "<b>ДРИПЧИК</b> 😎🔥",
        parse_mode="HTML"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    log.info(
        "========================================"
    )

    log.info(
        "🐻‍❄️ WHITE BEAR DRIP BOT"
    )

    log.info(
        "========================================"
    )

    log.info(
        "Channel: %s",
        CHANNEL_USERNAME
    )

    log.info(
        "OpenAI image model: %s",
        OPENAI_IMAGE_MODEL
    )

    log.info(
        "Bot starting..."
    )

    # Удаляем старый webhook перед polling.
    # Это предотвращает конфликт webhook/polling.
    try:

        await bot.delete_webhook(
            drop_pending_updates=False
        )

        log.info(
            "Webhook removed successfully"
        )

    except Exception as e:

        log.exception(
            "Не удалось удалить webhook: %r",
            e
        )

    # ========================================================
    # START POLLING
    # ========================================================

    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types()
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        log.info(
            "Bot stopped"
        )

    except Exception as e:

        log.exception(
            "Fatal bot error: %r",
            e
        )