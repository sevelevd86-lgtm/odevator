import os
import asyncio
import logging
from io import BytesIO

import cv2
import numpy as np

from aiogram import Bot, Dispatcher, F, types
from aiogram.filters import CommandStart
from aiogram.types import Message


# ============================================================
# НАСТРОЙКИ
# ============================================================

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

if not BOT_TOKEN:
    raise RuntimeError("BOT_TOKEN не задан в Secrets")


CANVAS_SIZE = 1200
MAX_IMAGE_SIZE = 1600


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clothing-extractor")


# ============================================================
# TELEGRAM
# ============================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# ============================================================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# ============================================================

def decode_image(data: bytes):
    array = np.frombuffer(data, dtype=np.uint8)

    image = cv2.imdecode(
        array,
        cv2.IMREAD_COLOR
    )

    if image is None:
        raise ValueError("Не удалось открыть изображение")

    h, w = image.shape[:2]

    largest = max(h, w)

    if largest > MAX_IMAGE_SIZE:

        scale = MAX_IMAGE_SIZE / largest

        image = cv2.resize(
            image,
            (
                int(w * scale),
                int(h * scale)
            ),
            interpolation=cv2.INTER_AREA
        )

    return image


# ============================================================
# ОПРЕДЕЛЕНИЕ СВЕТЛОГО ФОНА
# ============================================================

def create_background_mask(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    lower = np.array(
        [0, 0, 200],
        dtype=np.uint8
    )

    upper = np.array(
        [180, 75, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    return mask


# ============================================================
# КОЖА
# ============================================================

def create_skin_mask(image):

    hsv = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2HSV
    )

    # Приближённое определение оттенков кожи.
    # Это обычная обработка OpenCV, не AI.
    lower = np.array(
        [0, 20, 45],
        dtype=np.uint8
    )

    upper = np.array(
        [35, 255, 255],
        dtype=np.uint8
    )

    mask = cv2.inRange(
        hsv,
        lower,
        upper
    )

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


# ============================================================
# СИЛУЭТ ЧЕЛОВЕКА
# ============================================================

def create_person_mask(image):

    h, w = image.shape[:2]

    background = create_background_mask(
        image
    )

    foreground = 255 - background

    # Убираем мелкий шум
    kernel = np.ones(
        (9, 9),
        np.uint8
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel
    )

    # ========================================================
    # GRABCUT
    # ========================================================

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    border = max(
        5,
        min(h, w) // 30
    )

    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Центральная область предполагается человеком
    x1 = int(w * 0.08)
    y1 = int(h * 0.03)
    x2 = int(w * 0.92)
    y2 = int(h * 0.98)

    mask[
        y1:y2,
        x1:x2
    ] = cv2.GC_PR_FGD

    # Пиксели, явно отличающиеся от белого,
    # считаем вероятным объектом
    mask[
        foreground > 0
    ] = cv2.GC_PR_FGD

    bg_model = np.zeros(
        (1, 65),
        np.float64
    )

    fg_model = np.zeros(
        (1, 65),
        np.float64
    )

    try:

        cv2.grabCut(
            image,
            mask,
            None,
            bg_model,
            fg_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )

    except cv2.error:

        return foreground

    result = np.where(
        (
            (mask == cv2.GC_FGD) |
            (mask == cv2.GC_PR_FGD)
        ),
        255,
        0
    ).astype(np.uint8)

    return result


# ============================================================
# УДАЛЕНИЕ ГОЛОВЫ И ОТКРЫТЫХ УЧАСТКОВ ТЕЛА
# ============================================================

def remove_body_parts(
    image,
    mask
):

    h, w = image.shape[:2]

    result = mask.copy()

    # --------------------------------------------------------
    # Верхняя часть головы
    # --------------------------------------------------------

    head_end = int(h * 0.18)

    result[
        0:head_end,
        :
    ] = 0

    # --------------------------------------------------------
    # Края — часто руки/фон
    # --------------------------------------------------------

    side = int(w * 0.05)

    result[
        :,
        0:side
    ] = 0

    result[
        :,
        w-side:w
    ] = 0

    # --------------------------------------------------------
    # Кожа
    # --------------------------------------------------------

    skin = create_skin_mask(
        image
    )

    # Удаляем кожу только там,
    # где она занимает заметную область.
    result[
        skin > 0
    ] = 0

    # --------------------------------------------------------
    # Сглаживаем маску
    # --------------------------------------------------------

    kernel = np.ones(
        (7, 7),
        np.uint8
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel
    )

    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return result


# ============================================================
# ПОИСК ОБЛАСТЕЙ ОДЕЖДЫ
# ============================================================

def find_components(mask):

    num_labels, labels, stats, centers = (
        cv2.connectedComponentsWithStats(
            mask,
            connectivity=8
        )
    )

    components = []

    image_area = (
        mask.shape[0] *
        mask.shape[1]
    )

    min_area = max(
        500,
        int(image_area * 0.003)
    )

    for i in range(1, num_labels):

        x = stats[
            i,
            cv2.CC_STAT_LEFT
        ]

        y = stats[
            i,
            cv2.CC_STAT_TOP
        ]

        w = stats[
            i,
            cv2.CC_STAT_WIDTH
        ]

        h = stats[
            i,
            cv2.CC_STAT_HEIGHT
        ]

        area = stats[
            i,
            cv2.CC_STAT_AREA
        ]

        if area < min_area:
            continue

        component_mask = np.where(
            labels == i,
            255,
            0
        ).astype(np.uint8)

        components.append({
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area,
            "mask": component_mask
        })

    components.sort(
        key=lambda x: x["area"],
        reverse=True
    )

    return components


# ============================================================
# ПОЛУЧЕНИЕ ОБЪЕКТА
# ============================================================

def extract_component(
    image,
    component
):

    x = component["x"]
    y = component["y"]
    w = component["w"]
    h = component["h"]

    padding = max(
        5,
        int(min(w, h) * 0.03)
    )

    x1 = max(
        0,
        x - padding
    )

    y1 = max(
        0,
        y - padding
    )

    x2 = min(
        image.shape[1],
        x + w + padding
    )

    y2 = min(
        image.shape[0],
        y + h + padding
    )

    crop = image[
        y1:y2,
        x1:x2
    ]

    crop_mask = component["mask"][
        y1:y2,
        x1:x2
    ]

    # Сглаживание края
    crop_mask = cv2.GaussianBlur(
        crop_mask,
        (5, 5),
        0
    )

    rgba = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2BGRA
    )

    rgba[:, :, 3] = crop_mask

    return rgba


# ============================================================
# ОПРЕДЕЛЕНИЕ ПОЛОЖЕНИЯ ПРЕДМЕТА
# ============================================================

def classify_component(component):

    w = component["w"]
    h = component["h"]

    ratio = w / max(
        1,
        h
    )

    # Длинный вертикальный объект
    # скорее всего штаны
    if (
        h > w * 1.35
        and ratio < 0.75
    ):
        return "pants"

    # Широкий невысокий объект
    # скорее всего обувь
    if (
        ratio > 1.25
        and h < w * 0.75
    ):
        return "shoes"

    return "top"


# ============================================================
# ПОДГОНКА ПОД РАЗМЕР
# ============================================================

def fit_object(
    rgba,
    max_width,
    max_height
):

    h, w = rgba.shape[:2]

    scale = min(
        max_width / max(1, w),
        max_height / max(1, h)
    )

    new_w = max(
        1,
        int(w * scale)
    )

    new_h = max(
        1,
        int(h * scale)
    )

    return cv2.resize(
        rgba,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )


# ============================================================
# ВСТАВКА RGBA
# ============================================================

def paste_rgba(
    canvas,
    rgba,
    x,
    y
):

    h, w = rgba.shape[:2]

    x1 = max(
        0,
        x
    )

    y1 = max(
        0,
        y
    )

    x2 = min(
        canvas.shape[1],
        x + w
    )

    y2 = min(
        canvas.shape[0],
        y + h
    )

    if x1 >= x2 or y1 >= y2:
        return

    source = rgba[
        y1-y:y2-y,
        x1-x:x2-x
    ]

    alpha = (
        source[:, :, 3:4]
        .astype(np.float32)
        / 255.0
    )

    foreground = (
        source[:, :, :3]
        .astype(np.float32)
    )

    background = (
        canvas[y1:y2, x1:x2]
        .astype(np.float32)
    )

    result = (
        foreground * alpha
        +
        background * (1-alpha)
    )

    canvas[
        y1:y2,
        x1:x2
    ] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


# ============================================================
# СОЗДАНИЕ ФИНАЛЬНОЙ КАРТИНКИ
# ============================================================

def create_catalog(
    image
):

    # Получаем силуэт
    person_mask = create_person_mask(
        image
    )

    # Убираем голову/кожу
    clothing_mask = remove_body_parts(
        image,
        person_mask
    )

    # Ищем крупные отдельные области
    components = find_components(
        clothing_mask
    )

    if not components:

        raise RuntimeError(
            "Не удалось найти одежду"
        )

    # Берём максимум 6 крупных элементов
    components = components[:6]

    items = []

    for component in components:

        item_type = classify_component(
            component
        )

        rgba = extract_component(
            image,
            component
        )

        items.append({
            "type": item_type,
            "rgba": rgba,
            "component": component
        })

    # ========================================================
    # ИЩЕМ:
    # TOP
    # PANTS
    # SHOES
    # ========================================================

    tops = [
        x for x in items
        if x["type"] == "top"
    ]

    pants = [
        x for x in items
        if x["type"] == "pants"
    ]

    shoes = [
        x for x in items
        if x["type"] == "shoes"
    ]

    # Если классификатор не смог найти штаны,
    # используем самый высокий вертикальный объект.
    if not pants:

        vertical = [
            x for x in items
            if x["component"]["h"]
            >
            x["component"]["w"]
        ]

        if vertical:

            vertical.sort(
                key=lambda x: x["component"]["area"],
                reverse=True
            )

            pants = [
                vertical[0]
            ]

            if vertical[0] in tops:
                tops.remove(
                    vertical[0]
                )

    # ========================================================
    # БЕЛЫЙ ФОН
    # ========================================================

    canvas = np.full(
        (
            CANVAS_SIZE,
            CANVAS_SIZE,
            3
        ),
        255,
        dtype=np.uint8
    )

    # ========================================================
    # ВЕРХ
    # ========================================================

    if tops:

        top = tops[0]

        rgba = fit_object(
            top["rgba"],
            600,
            480
        )

        x = (
            40
            +
            (540 - rgba.shape[1]) // 2
        )

        y = (
            40
            +
            (480 - rgba.shape[0]) // 2
        )

        paste_rgba(
            canvas,
            rgba,
            x,
            y
        )

    # ========================================================
    # ШТАНЫ
    #
    # ВАЖНО:
    # штаны находятся ПОД верхом.
    # ========================================================

    if pants:

        pant = pants[0]

        rgba = fit_object(
            pant["rgba"],
            500,
            570
        )

        x = (
            60
            +
            (500 - rgba.shape[1]) // 2
        )

        y = (
            525
            +
            (570 - rgba.shape[0]) // 2
        )

        paste_rgba(
            canvas,
            rgba,
            x,
            y
        )

    # ========================================================
    # КРОССОВКИ
    #
    # Возле штанов справа.
    # ========================================================

    if shoes:

        shoe = shoes[0]

        rgba = fit_object(
            shoe["rgba"],
            500,
            360
        )

        x = (
            650
            +
            (500 - rgba.shape[1]) // 2
        )

        y = (
            620
            +
            (360 - rgba.shape[0]) // 2
        )

        paste_rgba(
            canvas,
            rgba,
            x,
            y
        )

    # Если кроссовки не распознаны,
    # дополнительные предметы можно разместить справа.
    if not shoes:

        remaining = [
            x for x in items
            if x not in tops[:1]
            and x not in pants[:1]
        ]

        if remaining:

            item = remaining[0]

            rgba = fit_object(
                item["rgba"],
                500,
                360
            )

            x = (
                650
                +
                (500 - rgba.shape[1]) // 2
            )

            y = (
                620
                +
                (360 - rgba.shape[0]) // 2
            )

            paste_rgba(
                canvas,
                rgba,
                x,
                y
            )

    # ========================================================
    # СОХРАНЕНИЕ
    # ========================================================

    success, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            95
        ]
    )

    if not success:

        raise RuntimeError(
            "Не удалось сохранить изображение"
        )

    return encoded.tobytes()


# ============================================================
# START
# ============================================================

@dp.message(
    CommandStart()
)
async def start(
    message: Message
):

    await message.answer(
        "👕 <b>Clothing Extractor</b>\n\n"
        "Пришли фотографию человека в одежде.\n\n"
        "Я попробую выделить одежду и сделать "
        "каталожную раскладку:\n\n"
        "👕 верх\n"
        "👖 штаны\n"
        "👟 обувь\n\n"
        "Финальное изображение будет на белом фоне."
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

    status = await message.answer(
        "⏳ Получил фотографию...\n\n"
        "👕 Выделяю одежду\n"
        "👖 Отделяю штаны\n"
        "👟 Ищу обувь\n"
        "🧼 Делаю белый фон..."
    )

    try:

        photo = message.photo[-1]

        telegram_file = await bot.get_file(
            photo.file_id
        )

        buffer = BytesIO()

        await bot.download_file(
            telegram_file.file_path,
            destination=buffer
        )

        data = buffer.getvalue()

        if not data:

            raise RuntimeError(
                "Фотография пустая"
            )

        image = decode_image(
            data
        )

        # Обработка OpenCV выполняется
        # в отдельном потоке.
        result = await asyncio.to_thread(
            create_catalog,
            image
        )

        await message.answer_photo(
            photo=types.BufferedInputFile(
                result,
                filename="clothing.jpg"
            ),
            caption=(
                "✅ Готово!\n\n"
                "👕 Одежда выделена и размещена "
                "на белом фоне."
            )
        )

        try:
            await status.delete()
        except Exception:
            pass

    except Exception as e:

        log.exception(
            "Ошибка: %s",
            e
        )

        await message.answer(
            "❌ Не получилось выделить одежду.\n\n"
            "Попробуй фотографию, где человека "
            "хорошо видно целиком, а фон не слишком "
            "сложный."
        )


# ============================================================
# ДРУГИЕ СООБЩЕНИЯ
# ============================================================

@dp.message()
async def other_message(
    message: Message
):

    await message.answer(
        "📸 Пришли фотографию человека в одежде."
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    log.info(
        "👕 Clothing Extractor starting..."
    )

    log.info(
        "AI generation: OFF"
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