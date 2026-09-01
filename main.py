import asyncio
import os
from datetime import datetime

import ccxt.async_support as ccxt
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

SECOND_EXCHANGE_NAME = os.getenv(
    "SECOND_EXCHANGE",
    "mexc"
).lower()

SYMBOL = os.getenv(
    "SYMBOL",
    "BTC/USDT"
)

MIN_NET_PROFIT = float(
    os.getenv(
        "MIN_NET_PROFIT_PERCENT",
        "0.30"
    )
)

TRADE_SIZE = float(
    os.getenv(
        "TRADE_SIZE_USDT",
        "50"
    )
)

SCAN_INTERVAL = int(
    os.getenv(
        "SCAN_INTERVAL_SECONDS",
        "5"
    )
)

PAPER_MODE = os.getenv(
    "PAPER_MODE",
    "true"
).lower() == "true"


# ============================================================
# BOT
# ============================================================

if not BOT_TOKEN:
    raise RuntimeError(
        "BOT_TOKEN не найден. Заполни его в .env"
    )

bot = Bot(
    token=BOT_TOKEN
)

dp = Dispatcher()


# ============================================================
# EXCHANGES
# ============================================================

exchange_classes = {
    "mexc": ccxt.mexc,
    "okx": ccxt.okx,
    "binance": ccxt.binance,
    "gate": ccxt.gate,
    "kucoin": ccxt.kucoin,
}

if SECOND_EXCHANGE_NAME not in exchange_classes:
    raise RuntimeError(
        f"Неизвестная биржа: {SECOND_EXCHANGE_NAME}"
    )


bybit = ccxt.bybit({
    "enableRateLimit": True,
})

second_exchange = exchange_classes[
    SECOND_EXCHANGE_NAME
]({
    "enableRateLimit": True,
})


# ============================================================
# STATE
# ============================================================

auto_scan_running = False
auto_scan_task = None

paper_balance = 1000.0
paper_profit = 0.0
paper_trades = 0

last_scan = None
last_opportunity = None


# ============================================================
# KEYBOARD
# ============================================================

def main_keyboard():

    keyboard = InlineKeyboardBuilder()

    keyboard.button(
        text="🔎 Сканировать",
        callback_data="scan"
    )

    keyboard.button(
        text="📊 Статус",
        callback_data="status"
    )

    keyboard.button(
        text="▶️ Автоскан",
        callback_data="autostart"
    )

    keyboard.button(
        text="⏹ Остановить",
        callback_data="autostop"
    )

    keyboard.button(
        text="💰 PAPER баланс",
        callback_data="balance"
    )

    keyboard.button(
        text="📈 Статистика",
        callback_data="stats"
    )

    keyboard.button(
        text="⚙️ Настройки",
        callback_data="settings"
    )

    keyboard.adjust(
        2,
        2,
        2,
        1
    )

    return keyboard.as_markup()


# ============================================================
# PRICE
# ============================================================

async def get_ticker(exchange, symbol):

    ticker = await exchange.fetch_ticker(
        symbol
    )

    bid = ticker.get("bid")
    ask = ticker.get("ask")
    last = ticker.get("last")

    return {
        "bid": float(bid or 0),
        "ask": float(ask or 0),
        "last": float(last or 0)
    }


# ============================================================
# ARBITRAGE CALCULATION
# ============================================================

def calculate(
    buy_price,
    sell_price,
    buy_fee=0.001,
    sell_fee=0.001
):

    if buy_price <= 0:
        return None

    if sell_price <= 0:
        return None

    gross = (
        (sell_price - buy_price)
        / buy_price
    ) * 100

    fee_percent = (
        buy_fee + sell_fee
    ) * 100

    net = (
        gross
        - fee_percent
    )

    return {
        "buy_price": buy_price,
        "sell_price": sell_price,
        "gross": gross,
        "fees": fee_percent,
        "net": net
    }


# ============================================================
# SCANNER
# ============================================================

async def scan_market():

    global last_scan
    global last_opportunity

    bybit_data, second_data = await asyncio.gather(

        get_ticker(
            bybit,
            SYMBOL
        ),

        get_ticker(
            second_exchange,
            SYMBOL
        )
    )

    opportunities = []

    # --------------------------------------------------------
    # BYBIT -> SECOND
    # --------------------------------------------------------

    result = calculate(
        buy_price=bybit_data["ask"],
        sell_price=second_data["bid"]
    )

    if result:

        result["buy_exchange"] = "Bybit"

        result["sell_exchange"] = (
            SECOND_EXCHANGE_NAME.upper()
        )

        opportunities.append(
            result
        )

    # --------------------------------------------------------
    # SECOND -> BYBIT
    # --------------------------------------------------------

    result = calculate(
        buy_price=second_data["ask"],
        sell_price=bybit_data["bid"]
    )

    if result:

        result["buy_exchange"] = (
            SECOND_EXCHANGE_NAME.upper()
        )

        result["sell_exchange"] = "Bybit"

        opportunities.append(
            result
        )

    opportunities.sort(
        key=lambda x: x["net"],
        reverse=True
    )

    last_scan = datetime.now()

    if opportunities:

        last_opportunity = opportunities[0]

    else:

        last_opportunity = None

    return opportunities


# ============================================================
# FORMAT OPPORTUNITY
# ============================================================

def format_opportunity(op):

    buy_exchange = op[
        "buy_exchange"
    ]

    sell_exchange = op[
        "sell_exchange"
    ]

    buy_price = op[
        "buy_price"
    ]

    sell_price = op[
        "sell_price"
    ]

    gross = op[
        "gross"
    ]

    fees = op[
        "fees"
    ]

    net = op[
        "net"
    ]

    estimated_profit = (
        TRADE_SIZE
        * net
        / 100
    )

    return (
        "🔥 <b>АРБИТРАЖ НАЙДЕН</b>\n\n"

        f"💱 Пара: <b>{SYMBOL}</b>\n\n"

        f"🟢 Купить:\n"
        f"<b>{buy_exchange}</b>\n"
        f"${buy_price:,.2f}\n\n"

        f"🔴 Продать:\n"
        f"<b>{sell_exchange}</b>\n"
        f"${sell_price:,.2f}\n\n"

        "━━━━━━━━━━━━━━\n\n"

        f"📈 Gross: <b>{gross:.3f}%</b>\n"
        f"💸 Комиссии: <b>-{fees:.3f}%</b>\n"
        f"💰 NET: <b>{net:.3f}%</b>\n\n"

        f"💵 Размер сделки: ${TRADE_SIZE:.2f}\n"
        f"📊 Расчётная прибыль: "
        f"<b>${estimated_profit:.4f}</b>\n\n"

        f"🧪 Режим: "
        f"<b>{'PAPER' if PAPER_MODE else 'LIVE'}</b>"
    )


# ============================================================
# START COMMAND
# ============================================================

@dp.message(
    Command("start")
)
async def start_command(
    message: Message
):

    await message.answer(

        "🐻‍❄️ <b>CRYPTO ARBITRAGE</b>\n\n"

        "Добро пожаловать!\n\n"

        "Я сравниваю цены между "
        "Bybit и второй биржей и "
        "ищу потенциальный арбитраж.\n\n"

        f"💱 Пара: <b>{SYMBOL}</b>\n"
        f"🧪 Режим: "
        f"<b>{'PAPER' if PAPER_MODE else 'LIVE'}</b>\n\n"

        "Выбери действие ниже 👇",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# SCAN BUTTON
# ============================================================

@dp.callback_query(
    F.data == "scan"
)
async def scan_button(
    callback: CallbackQuery
):

    await callback.answer(
        "Сканирую..."
    )

    try:

        opportunities = await scan_market()

        if not opportunities:

            text = (
                "❌ <b>Нет данных</b>\n\n"
                "Не удалось получить котировки."
            )

        else:

            best = opportunities[0]

            if best["net"] < MIN_NET_PROFIT:

                text = (

                    "🔎 <b>СКАНИРОВАНИЕ</b>\n\n"

                    f"Лучший найденный вариант:\n\n"

                    f"{best['buy_exchange']} → "
                    f"{best['sell_exchange']}\n\n"

                    f"NET: "
                    f"<b>{best['net']:.3f}%</b>\n\n"

                    f"Минимальный порог: "
                    f"{MIN_NET_PROFIT:.2f}%\n\n"

                    "❌ Сделка ниже установленного "
                    "порога."
                )

            else:

                text = format_opportunity(
                    best
                )

        await callback.message.edit_text(

            text,

            reply_markup=main_keyboard(),

            parse_mode="HTML"
        )

    except Exception as error:

        await callback.message.edit_text(

            "⚠️ <b>Ошибка сканирования</b>\n\n"

            f"<code>{error}</code>",

            reply_markup=main_keyboard(),

            parse_mode="HTML"
        )


# ============================================================
# STATUS
# ============================================================

@dp.callback_query(
    F.data == "status"
)
async def status_button(
    callback: CallbackQuery
):

    await callback.answer()

    status = (
        "🟢 Работает"
        if auto_scan_running
        else
        "🔴 Остановлен"
    )

    last_scan_text = (

        last_scan.strftime(
            "%H:%M:%S"
        )

        if last_scan

        else

        "ещё не запускался"
    )

    await callback.message.edit_text(

        "📊 <b>СТАТУС БОТА</b>\n\n"

        f"🤖 Bot: <b>ONLINE</b>\n"

        f"🔎 Автоскан: <b>{status}</b>\n"

        f"💱 Пара: <b>{SYMBOL}</b>\n"

        f"🏦 Bybit\n"

        f"🏦 {SECOND_EXCHANGE_NAME.upper()}\n\n"

        f"🧪 PAPER MODE: "
        f"<b>{PAPER_MODE}</b>\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"🎯 Минимальный NET: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"

        f"🕐 Последний скан: "
        f"<b>{last_scan_text}</b>",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# PAPER BALANCE
# ============================================================

@dp.callback_query(
    F.data == "balance"
)
async def balance_button(
    callback: CallbackQuery
):

    await callback.answer()

    await callback.message.edit_text(

        "💰 <b>PAPER БАЛАНС</b>\n\n"

        f"Начальный баланс:\n"
        f"$1,000.00\n\n"

        f"Текущий баланс:\n"
        f"<b>${paper_balance:.2f}</b>\n\n"

        f"Прибыль:\n"
        f"<b>${paper_profit:.2f}</b>\n\n"

        f"Сделок:\n"
        f"<b>{paper_trades}</b>\n\n"

        "🧪 Это виртуальный баланс.\n"
        "Реальные средства не используются.",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# STATISTICS
# ============================================================

@dp.callback_query(
    F.data == "stats"
)
async def stats_button(
    callback: CallbackQuery
):

    await callback.answer()

    roi = (
        paper_profit
        / 1000
        * 100
    )

    await callback.message.edit_text(

        "📈 <b>СТАТИСТИКА</b>\n\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.2f}</b>\n\n"

        f"💵 Прибыль: "
        f"<b>${paper_profit:.2f}</b>\n\n"

        f"📊 ROI: "
        f"<b>{roi:.3f}%</b>\n\n"

        f"🔄 Сделок: "
        f"<b>{paper_trades}</b>\n\n"

        "⚠️ Пока реальные ордера "
        "не отправляются.",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# SETTINGS
# ============================================================

@dp.callback_query(
    F.data == "settings"
)
async def settings_button(
    callback: CallbackQuery
):

    await callback.answer()

    await callback.message.edit_text(

        "⚙️ <b>НАСТРОЙКИ</b>\n\n"

        f"💱 Пара:\n"
        f"<b>{SYMBOL}</b>\n\n"

        f"🎯 Минимальный NET:\n"
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"

        f"💵 Размер сделки:\n"
        f"<b>${TRADE_SIZE:.2f}</b>\n\n"

        f"⏱ Интервал:\n"
        f"<b>{SCAN_INTERVAL} сек.</b>\n\n"

        f"🏦 Вторая биржа:\n"
        f"<b>{SECOND_EXCHANGE_NAME.upper()}</b>\n\n"

        "Изменение этих параметров "
        "делается в файле <code>.env</code>.",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# AUTO SCAN
# ============================================================

async def automatic_scanner(
    chat_id
):

    global auto_scan_running

    while auto_scan_running:

        try:

            opportunities = await scan_market()

            if opportunities:

                best = opportunities[0]

                if best["net"] >= MIN_NET_PROFIT:

                    await bot.send_message(

                        chat_id,

                        format_opportunity(
                            best
                        ),

                        parse_mode="HTML"
                    )

        except Exception as error:

            print(
                "Scanner error:",
                error
            )

        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# START AUTO SCAN
# ============================================================

@dp.callback_query(
    F.data == "autostart"
)
async def autostart_button(
    callback: CallbackQuery
):

    global auto_scan_running
    global auto_scan_task

    await callback.answer()

    if auto_scan_running:

        await callback.message.answer(
            "ℹ️ Автоскан уже работает."
        )

        return

    auto_scan_running = True

    auto_scan_task = asyncio.create_task(

        automatic_scanner(
            callback.message.chat.id
        )
    )

    await callback.message.edit_text(

        "▶️ <b>АВТОСКАН ЗАПУЩЕН</b>\n\n"

        f"Пара: {SYMBOL}\n"

        f"Интервал: "
        f"{SCAN_INTERVAL} сек.\n\n"

        f"Бот будет присылать "
        f"возможности от "
        f"<b>{MIN_NET_PROFIT:.2f}% NET</b>.",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# STOP AUTO SCAN
# ============================================================

@dp.callback_query(
    F.data == "autostop"
)
async def autostop_button(
    callback: CallbackQuery
):

    global auto_scan_running
    global auto_scan_task

    await callback.answer()

    auto_scan_running = False

    if auto_scan_task:

        auto_scan_task.cancel()

        auto_scan_task = None

    await callback.message.edit_text(

        "⏹ <b>АВТОСКАН ОСТАНОВЛЕН</b>\n\n"

        "Бот больше не сканирует рынок.",

        reply_markup=main_keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# ERROR HANDLING / UNKNOWN COMMANDS
# ============================================================

@dp.message()
async def any_message(
    message: Message
):

    await message.answer(

        "🐻‍❄️ Используй меню ниже:",

        reply_markup=main_keyboard()
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    print(
        "================================="
    )

    print(
        "🐻‍❄️ CRYPTO ARBITRAGE BOT"
    )

    print(
        "================================="
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Second exchange: "
        f"{SECOND_EXCHANGE_NAME.upper()}"
    )

    print(
        f"Paper mode: {PAPER_MODE}"
    )

    print(
        f"Min NET: "
        f"{MIN_NET_PROFIT}%"
    )

    print(
        "Bot started..."
    )

    try:

        await dp.start_polling(
            bot
        )

    finally:

        await bybit.close()

        await second_exchange.close()

        await bot.session.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )