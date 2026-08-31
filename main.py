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


# Размер готовой картинки
CANVAS_SIZE = 1200

# Максимальный размер входной картинки для обработки
MAX_INPUT_SIZE = 1600


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("clothing-layout-bot")


# ============================================================
# TELEGRAM
# ============================================================

bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()


# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ============================================================

def resize_for_processing(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    longest = max(h, w)

    if longest <= MAX_INPUT_SIZE:
        return img

    scale = MAX_INPUT_SIZE / longest
    return cv2.resize(
        img,
        (int(w * scale), int(h * scale)),
        interpolation=cv2.INTER_AREA
    )


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError("Не удалось открыть изображение")

    return resize_for_processing(image)


def white_background_mask(image: np.ndarray) -> np.ndarray:
    """
    Полностью обычная компьютерная обработка:
    ищем пиксели, близкие к белому, и считаем их фоном.
    Никаких AI/нейросетей здесь нет.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Белый/очень светлый фон
    white = cv2.inRange(
        hsv,
        np.array([0, 0, 205], dtype=np.uint8),
        np.array([180, 70, 255], dtype=np.uint8)
    )

    foreground = 255 - white

    kernel = np.ones((5, 5), np.uint8)
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )
    foreground = cv2.morphologyEx(
        foreground,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    # Убираем мелкий мусор
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        foreground,
        connectivity=8
    )

    clean = np.zeros_like(foreground)
    image_area = image.shape[0] * image.shape[1]
    min_area = max(150, int(image_area * 0.0015))

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            clean[labels == i] = 255

    return clean


def grabcut_mask(image: np.ndarray) -> np.ndarray:
    """
    Запасной вариант без AI.
    OpenCV GrabCut использует классический алгоритм сегментации.
    """
    h, w = image.shape[:2]

    mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    # Уверенный фон по краям
    border = max(5, min(h, w) // 35)
    mask[:border, :] = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border] = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD

    # Центральная область — вероятный объект
    x1 = int(w * 0.08)
    y1 = int(h * 0.06)
    x2 = int(w * 0.92)
    y2 = int(h * 0.94)

    mask[y1:y2, x1:x2] = cv2.GC_PR_FGD

    # Если фон почти белый, сразу считаем непохожие на белый
    # пиксели вероятным передним планом.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    not_white = cv2.inRange(
        hsv,
        np.array([0, 35, 20], dtype=np.uint8),
        np.array([180, 255, 240], dtype=np.uint8)
    )

    mask[not_white > 0] = cv2.GC_PR_FGD

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            None,
            bgd_model,
            fgd_model,
            5,
            cv2.GC_INIT_WITH_MASK
        )
    except cv2.error:
        # Если GrabCut не смог обработать фото,
        # возвращаем простую маску.
        return not_white

    result = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0
    ).astype(np.uint8)

    kernel = np.ones((5, 5), np.uint8)
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )
    result = cv2.morphologyEx(
        result,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )

    return result


def build_foreground(image: np.ndarray):
    """
    Получаем одежду/предмет без фона.
    """
    mask = white_background_mask(image)

    image_area = image.shape[0] * image.shape[1]
    mask_area = cv2.countNonZero(mask)

    # Если простая обработка нашла слишком мало/много,
    # используем классический GrabCut.
    ratio = mask_area / max(1, image_area)

    if ratio < 0.01 or ratio > 0.92:
        mask = grabcut_mask(image)

    # Ещё раз чистим маску
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=2
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    # Оставляем достаточно крупные области.
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask,
        connectivity=8
    )

    components = []
    min_area = max(300, int(image_area * 0.002))

    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        if area < min_area:
            continue

        # Не берём огромную область, если это почти весь кадр:
        # GrabCut/фон мог слиться с предметом.
        component_mask = np.where(labels == i, 255, 0).astype(np.uint8)

        pad = max(4, int(min(w, h) * 0.025))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(image.shape[1], x + w + pad)
        y2 = min(image.shape[0], y + h + pad)

        crop = image[y1:y2, x1:x2]
        crop_mask = component_mask[y1:y2, x1:x2]

        components.append({
            "image": crop,
            "mask": crop_mask,
            "x": x,
            "y": y,
            "w": w,
            "h": h,
            "area": area
        })

    # Если отдельных объектов много, оставляем самые крупные.
    components.sort(key=lambda c: c["area"], reverse=True)
    components = components[:8]

    if not components:
        # Последний вариант: считаем весь предмет центральным.
        return [{
            "image": image,
            "mask": np.ones(
                image.shape[:2],
                dtype=np.uint8
            ) * 255,
            "x": 0,
            "y": 0,
            "w": image.shape[1],
            "h": image.shape[0],
            "area": image.shape[0] * image.shape[1]
        }]

    return components


def object_type(component) -> str:
    """
    Простая геометрическая классификация.
    Это НЕ AI.
    Нужна только для аккуратной раскладки вещей.
    """
    w = component["w"]
    h = component["h"]
    area = component["area"]

    ratio = w / max(1, h)
    canvas_area = component["image"].shape[0] * component["image"].shape[1]
    fill = area / max(1, canvas_area)

    # Длинный вертикальный объект — брюки/штаны.
    if ratio < 0.62 and h > w * 1.35:
        return "pants"

    # Маленький вытянутый объект — чаще обувь/аксессуар.
    if ratio > 1.35 and h < w * 0.75:
        return "shoes"

    # Небольшой компактный объект — аксессуар.
    if fill < 0.18 and max(w, h) < 0.45 * max(
        component["image"].shape[0],
        component["image"].shape[1]
    ):
        return "accessory"

    return "top"


def add_shadow(canvas: np.ndarray, x: int, y: int, w: int, h: int):
    """
    Лёгкая тень под предметом для аккуратного каталожного вида.
    """
    overlay = canvas.copy()

    cx = x + w // 2
    cy = y + h - max(8, h // 35)

    axes = (
        max(10, w // 3),
        max(5, h // 18)
    )

    cv2.ellipse(
        overlay,
        (cx, cy),
        axes,
        0,
        0,
        360,
        (225, 225, 225),
        -1
    )

    overlay = cv2.GaussianBlur(
        overlay,
        (0, 0),
        12
    )

    # Тень очень слабая
    canvas[:] = cv2.addWeighted(
        canvas,
        0.88,
        overlay,
        0.12,
        0
    )


def fit_rgba_to_box(
    rgba: np.ndarray,
    box_w: int,
    box_h: int
) -> np.ndarray:
    h, w = rgba.shape[:2]

    scale = min(
        box_w / max(1, w),
        box_h / max(1, h)
    )

    nw = max(1, int(w * scale))
    nh = max(1, int(h * scale))

    return cv2.resize(
        rgba,
        (nw, nh),
        interpolation=cv2.INTER_AREA
    )


def paste_rgba(
    canvas: np.ndarray,
    rgba: np.ndarray,
    x: int,
    y: int
):
    """
    Накладывает PNG-подобный RGBA объект на белый фон.
    """
    h, w = rgba.shape[:2]

    if x >= canvas.shape[1] or y >= canvas.shape[0]:
        return

    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(canvas.shape[1], x + w)
    y2 = min(canvas.shape[0], y + h)

    if x2 <= x1 or y2 <= y1:
        return

    src = rgba[
        y1 - y:y2 - y,
        x1 - x:x2 - x
    ]

    alpha = (
        src[:, :, 3:4].astype(np.float32) / 255.0
    )

    rgb = src[:, :, :3].astype(np.float32)
    dst = canvas[y1:y2, x1:x2].astype(np.float32)

    result = (
        rgb * alpha +
        dst * (1.0 - alpha)
    )

    canvas[y1:y2, x1:x2] = np.clip(
        result,
        0,
        255
    ).astype(np.uint8)


def component_to_rgba(component):
    image = component["image"]
    mask = component["mask"]

    # Небольшое сглаживание краёв.
    alpha = cv2.GaussianBlur(
        mask,
        (5, 5),
        0
    )

    bgr = image
    rgba = cv2.cvtColor(
        bgr,
        cv2.COLOR_BGR2BGRA
    )
    rgba[:, :, 3] = alpha

    return rgba


def make_catalog_image(image: np.ndarray) -> bytes:
    """
    Основная функция.

    Если на фото одна вещь:
        она аккуратно помещается на белый фон.

    Если на фото несколько раздельных вещей:
        они раскладываются по категориям:
        верх / низ / обувь / аксессуары.

    Никакой генерации изображения и никакой AI-модели.
    """
    components = build_foreground(image)

    typed = []
    for component in components:
        component["type"] = object_type(component)
        typed.append(component)

    # Сортируем так, чтобы крупные предметы были основными.
    typed.sort(
        key=lambda c: c["area"],
        reverse=True
    )

    # Если найден только один предмет — не меняем его
    # положение/ориентацию, просто делаем красивый белый фон.
    if len(typed) == 1:
        comp = typed[0]
        rgba = component_to_rgba(comp)

        # Максимальная зона для одной вещи.
        rgba = fit_rgba_to_box(
            rgba,
            820,
            980
        )

        canvas = np.full(
            (CANVAS_SIZE, CANVAS_SIZE, 3),
            255,
            dtype=np.uint8
        )

        x = (CANVAS_SIZE - rgba.shape[1]) // 2
        y = (CANVAS_SIZE - rgba.shape[0]) // 2

        paste_rgba(
            canvas,
            rgba,
            x,
            y
        )

    else:
        # Каталожная раскладка.
        # Верх: одежда/аксессуары.
        # Низ: штаны/обувь.
        canvas = np.full(
            (CANVAS_SIZE, CANVAS_SIZE, 3),
            255,
            dtype=np.uint8
        )

        tops = [
            c for c in typed
            if c["type"] == "top"
        ]
        pants = [
            c for c in typed
            if c["type"] == "pants"
        ]
        shoes = [
            c for c in typed
            if c["type"] == "shoes"
        ]
        accessories = [
            c for c in typed
            if c["type"] == "accessory"
        ]

        # Если классификация неидеальна, крупные предметы
        # всё равно будут показаны.
        if not tops:
            tops = typed[:1]

        # ---- ВЕРХ ----
        top_positions = [
            (55, 55, 520, 450),
            (625, 55, 520, 450),
        ]

        top_items = tops[:2] + accessories[:1]

        # Если аксессуар есть, стараемся поставить его справа сверху.
        for index, comp in enumerate(top_items[:2]):
            box = top_positions[index]

            rgba = component_to_rgba(comp)
            rgba = fit_rgba_to_box(
                rgba,
                box[2],
                box[3]
            )

            x = box[0] + (box[2] - rgba.shape[1]) // 2
            y = box[1] + (box[3] - rgba.shape[0]) // 2

            paste_rgba(canvas, rgba, x, y)

        # ---- НИЗ / ШТАНЫ ----
        if pants:
            comp = pants[0]
            rgba = component_to_rgba(comp)
            rgba = fit_rgba_to_box(
                rgba,
                500,
                570
            )

            x = 70 + (500 - rgba.shape[1]) // 2
            y = 545 + (570 - rgba.shape[0]) // 2

            paste_rgba(canvas, rgba, x, y)

        # ---- ОБУВЬ ----
        shoe_items = shoes[:2]

        if shoe_items:
            if len(shoe_items) == 1:
                boxes = [(650, 600, 470, 360)]
            else:
                boxes = [
                    (620, 610, 250, 330),
                    (850, 610, 250, 330)
                ]

            for index, comp in enumerate(shoe_items):
                box = boxes[index]

                rgba = component_to_rgba(comp)
                rgba = fit_rgba_to_box(
                    rgba,
                    box[2],
                    box[3]
                )

                x = box[0] + (box[2] - rgba.shape[1]) // 2
                y = box[1] + (box[3] - rgba.shape[0]) // 2

                paste_rgba(canvas, rgba, x, y)

        # ---- ДОПОЛНИТЕЛЬНЫЕ ПРЕДМЕТЫ ----
        # Если осталось несколько крупных предметов,
        # размещаем их в свободной зоне без изменения содержимого.
        used_ids = set(
            id(x) for x in top_items[:2] + pants[:1] + shoe_items
        )

        leftovers = [
            c for c in typed
            if id(c) not in used_ids
        ][:3]

        free_boxes = [
            (40, 940, 340, 210),
            (430, 940, 340, 210),
            (820, 940, 340, 210),
        ]

        for index, comp in enumerate(leftovers):
            if index >= len(free_boxes):
                break

            box = free_boxes[index]

            rgba = component_to_rgba(comp)
            rgba = fit_rgba_to_box(
                rgba,
                box[2],
                box[3]
            )

            x = box[0] + (box[2] - rgba.shape[1]) // 2
            y = box[1] + (box[3] - rgba.shape[0]) // 2

            paste_rgba(canvas, rgba, x, y)

    # Сохраняем JPEG.
    output = BytesIO()
    success, encoded = cv2.imencode(
        ".jpg",
        canvas,
        [
            int(cv2.IMWRITE_JPEG_QUALITY),
            95
        ]
    )

    if not success:
        raise RuntimeError("Не удалось создать итоговое изображение")

    output.write(encoded.tobytes())
    return output.getvalue()


# ============================================================
# START
# ============================================================

@dp.message(CommandStart())
async def start(message: Message):
    await message.answer(
        "👕 <b>Clothing Layout Bot</b>\n\n"
        "Отправь мне фото одежды или вещи.\n\n"
        "Я без ИИ-моделей обработаю изображение, "
        "уберу обычный фон и размещу вещи на чистом "
        "белом фоне в каталожном виде.\n\n"
        "📸 Можно отправить одно фото с одной или "
        "несколькими вещами."
    )


# ============================================================
# PHOTO
# ============================================================

@dp.message(F.photo)
async def photo_handler(message: Message):
    if not message.photo:
        return

    status = await message.answer(
        "⏳ Обрабатываю фото...\n"
        "Убираю фон и раскладываю вещи."
    )

    try:
        photo = message.photo[-1]

        telegram_file = await bot.get_file(
            photo.file_id
        )

        source = BytesIO()

        await bot.download_file(
            telegram_file.file_path,
            destination=source
        )

        image_bytes = source.getvalue()

        if not image_bytes:
            raise RuntimeError("Пустой файл изображения")

        image = decode_image(image_bytes)

        result = await asyncio.to_thread(
            make_catalog_image,
            image
        )

        await message.answer_photo(
            photo=types.BufferedInputFile(
                result,
                filename="clothing_catalog.jpg"
            ),
            caption=(
                "✅ Готово!\n\n"
                "👕 Вещи размещены на белом фоне.\n"
                "Без генерации изображения и без AI-модели."
            )
        )

        try:
            await status.delete()
        except Exception:
            pass

    except Exception as e:
        log.exception(
            "Ошибка обработки фото: %s",
            e
        )

        await message.answer(
            "❌ Не удалось обработать фото.\n\n"
            "Попробуй отправить более чёткое фото, "
            "желательно с хорошо отделённой от фона одеждой."
        )


# ============================================================
# OTHER
# ============================================================

@dp.message()
async def other_message(message: Message):
    await message.answer(
        "📸 Отправь мне фото одежды.\n\n"
        "Я размещу вещи на чистом белом фоне."
    )


# ============================================================
# MAIN
# ============================================================

async def main():
    log.info("👕 Clothing Layout Bot starting...")
    log.info("AI image generation: OFF")
    await dp.start_polling(
        bot,
        allowed_updates=dp.resolve_used_update_types()
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bot stopped")