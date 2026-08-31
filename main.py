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
    level=logging.INFO
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

# Пользователи, которые прошли проверку подписки
verified_users = set()

# Пользователи, которым бот сейчас обрабатывает фото
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
                    text="✅ Проверить",
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

        return member.status in {
            ChatMemberStatus.MEMBER,
            ChatMemberStatus.ADMINISTRATOR,
            ChatMemberStatus.CREATOR
        }

    except Exception as e:

        log.exception(
            "Ошибка проверки подписки: %s",
            e
        )

        return False


# ============================================================
# СООБЩЕНИЕ ПОСЛЕ ПОДПИСКИ
# ============================================================

async def send_photo_instruction(
    message: Message
):

    await message.answer(
        "Здравствуйте! 👋\n\n"
        "Теперь пришлите ваше фото в чат.\n\n"
        "📸 Отправьте **только одно фото** "
        "человека, и я верну этого же человека, "
        "но уже в стильном **ДРИПЧИКЕ** 😎🔥\n\n"
        "Лучше всего подойдёт фото, где человека "
        "хорошо видно целиком или по пояс.",
        parse_mode="Markdown"
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

    # Если пользователь уже проходил проверку
    if user_id in verified_users:

        await send_photo_instruction(
            message
        )

        return

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

        await callback.message.edit_text(
            "Здравствуйте! 👋\n\n"
            "Теперь пришлите ваше фото в чат.\n\n"
            "📸 Отправьте **только одно фото** "
            "человека, и я верну этого же человека, "
            "но уже в стильном **ДРИПЧИКЕ** 😎🔥\n\n"
            "Лучше всего подойдёт фото, где человека "
            "хорошо видно целиком или по пояс.",
            parse_mode="Markdown"
        )

    except Exception:

        await callback.message.answer(
            "Здравствуйте! 👋\n\n"
            "Теперь пришлите ваше фото в чат.\n\n"
            "📸 Отправьте **только одно фото** "
            "человека, и я верну этого же человека, "
            "но уже в стильном **ДРИПЧИКЕ** 😎🔥",
            parse_mode="Markdown"
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

    # Не полагаемся только на память процесса:
    # снова проверяем Telegram.
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

IMPORTANT:
Keep the exact same person.
Preserve their face, identity, facial features,
skin tone, hair, body proportions, pose,
hands, expression, camera angle and overall
photographic realism.

ONLY change the person's clothing.

Dress the person in a modern, stylish,
high-quality streetwear "drip" outfit.

The outfit should look natural on this exact person
and fit their body and pose correctly.

Use fashionable contemporary streetwear:
premium hoodie or jacket, stylish pants,
clean sneakers and tasteful accessories when
appropriate.

Do NOT change the person's face.
Do NOT change the person's age.
Do NOT change the person's body.
Do NOT replace the person.
Do NOT create a different person.

Keep the original background and lighting
as close to the original as possible.

The final image must look like a real photograph,
not an illustration or cartoon.

Keep everything except the clothing as close
to the original photograph as possible.
"""

    def make_edit():

        image_file = BytesIO(
            image_bytes
        )

        image_file.name = "person.jpg"

        result = openai_client.images.edit(
            model=OPENAI_IMAGE_MODEL,
            image=image_file,
            prompt=prompt,
            size="1024x1024",
            quality="medium"
        )

        if not result.data:
            raise RuntimeError(
                "OpenAI не вернул изображение"
            )

        encoded = result.data[0].b64_json

        if not encoded:
            raise RuntimeError(
                "OpenAI не вернул base64 изображения"
            )

        return base64.b64decode(
            encoded
        )

    # OpenAI SDK синхронный, поэтому
    # не блокируем Telegram event loop.
    return await asyncio.to_thread(
        make_edit
    )


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

    # Проверяем подписку
    if not await require_subscription(
        message
    ):
        return

    user_id = message.from_user.id

    # Защита от одновременной обработки
    if user_id in processing_users:

        await message.answer(
            "⏳ Я ещё обрабатываю ваше предыдущее фото."
        )

        return

    processing_users.add(
        user_id
    )

    try:

        # Telegram берёт самое большое доступное
        # разрешение фотографии.
        photo = message.photo[-1]

        await message.answer(
            "⏳ Получил фото!\n\n"
            "👕 Подбираю тебе ДРИПЧИК...\n"
            "🔥 Это может занять некоторое время."
        )

        # Получаем Telegram-файл
        telegram_file = await bot.get_file(
            photo.file_id
        )

        # Скачиваем фото в память
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
                "Не удалось скачать фотографию"
            )

        log.info(
            "Processing photo for user %s",
            user_id
        )

        # Отправляем изображение в OpenAI
        edited_bytes = await edit_photo_with_openai(
            source_bytes
        )

        # Отправляем результат
        result_photo = BytesIO(
            edited_bytes
        )

        result_photo.name = (
            "drip.jpg"
        )

        await message.answer_photo(
            photo=types.BufferedInputFile(
                edited_bytes,
                filename="drip.jpg"
            ),
            caption=(
                "🔥 Вот твой ДРИПЧИК!\n\n"
                "😎 Образ обновлён."
            )
        )

    except Exception as e:

        log.exception(
            "Image processing error: %s",
            e
        )

        await message.answer(
            "❌ Не получилось обработать фото.\n\n"
            "Попробуйте отправить другое фото "
            "и убедитесь, что на нём хорошо видно человека."
        )

    finally:

        processing_users.discard(
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
        "📸 Пришлите именно **одно фото** человека.\n\n"
        "После этого я сделаю ему стильный ДРИПЧИК 😎🔥",
        parse_mode="Markdown"
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    log.info(
        "🐻‍❄️ White Bear Drip Bot starting..."
    )

    log.info(
        "Channel: %s",
        CHANNEL_USERNAME
    )

    log.info(
        "OpenAI image model: %s",
        OPENAI_IMAGE_MODEL
    )

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