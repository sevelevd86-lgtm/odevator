import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
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


BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
).strip()


SYMBOL = os.getenv(
    "SYMBOL",
    "BTC/USDT"
).strip()


TRADE_SIZE_USDT = float(
    os.getenv(
        "TRADE_SIZE_USDT",
        "50"
    )
)


MIN_NET_PROFIT = float(
    os.getenv(
        "MIN_NET_PROFIT_PERCENT",
        "0.30"
    )
)


SCAN_INTERVAL = float(
    os.getenv(
        "SCAN_INTERVAL_SECONDS",
        "5"
    )
)


ORDERBOOK_LEVELS = int(
    os.getenv(
        "ORDERBOOK_LEVELS",
        "20"
    )
)


TRANSFER_COST_USDT = float(
    os.getenv(
        "TRANSFER_COST_USDT",
        "1.00"
    )
)


# Приблизительные комиссии для предварительного расчёта.
# Перед реальной торговлей обязательно заменить
# на фактические комиссии аккаунтов.
FEES = {

    "coinex": float(
        os.getenv(
            "FEE_COINEX_PERCENT",
            "0.10"
        )
    ),

    "toobit": float(
        os.getenv(
            "FEE_TOOBIT_PERCENT",
            "0.10"
        )
    ),

    "weex": float(
        os.getenv(
            "FEE_WEEX_PERCENT",
            "0.10"
        )
    ),

    "1bit": float(
        os.getenv(
            "FEE_1BIT_PERCENT",
            "0.10"
        )
    ),

    "hyperliquid": float(
        os.getenv(
            "FEE_HYPERLIQUID_PERCENT",
            "0.10"
        )
    ),
}


# ============================================================
# SAFETY
# ============================================================

# Реальные ордера этой версией НЕ отправляются.
LIVE_TRADING = False


if not BOT_TOKEN:

    raise RuntimeError(
        "BOT_TOKEN не найден.\n"
        "Добавь токен в файл .env"
    )


# ============================================================
# TELEGRAM
# ============================================================

bot = Bot(
    token=BOT_TOKEN
)


dp = Dispatcher()


# ============================================================
# GLOBAL STATE
# ============================================================

auto_tasks = {}


last_scan = None


last_opportunities = []


paper_balance = 1000.0


paper_profit = 0.0


paper_trades = 0


session: Optional[
    aiohttp.ClientSession
] = None


# Hyperliquid используется через CCXT
ccxt_venues = {}


# ============================================================
# ORDER BOOK CLASS
# ============================================================

@dataclass
class Book:

    venue: str

    symbol: str

    bids: list

    asks: list

    timestamp: float


    @property
    def best_bid(self):

        if not self.bids:

            return 0.0

        return self.bids[0][0]


    @property
    def best_ask(self):

        if not self.asks:

            return 0.0

        return self.asks[0][0]


# ============================================================
# HTTP
# ============================================================

async def http_json(
    url,
    params=None
):

    async with session.get(

        url,

        params=params,

        timeout=aiohttp.ClientTimeout(
            total=8
        )

    ) as response:

        response.raise_for_status()

        return await response.json()


# ============================================================
# NORMALIZE ORDERBOOK
# ============================================================

def normalize_levels(
    rows
):

    result = []


    for row in rows or []:

        try:

            price = float(
                row[0]
            )


            amount = float(
                row[1]
            )


            if price > 0 and amount > 0:

                result.append(
                    [
                        price,
                        amount
                    ]
                )

        except Exception:

            continue


    return result


# ============================================================
# COINEX
# ============================================================

async def coinex_book():

    data = await http_json(

        "https://api.coinex.com/v2/spot/depth",

        {
            "market": "BTCUSDT",

            "limit": ORDERBOOK_LEVELS,

            "interval": "0"
        }
    )


    depth = data["data"]["depth"]


    return Book(

        venue="coinex",

        symbol=SYMBOL,

        bids=normalize_levels(
            depth.get(
                "bids"
            )
        ),

        asks=normalize_levels(
            depth.get(
                "asks"
            )
        ),

        timestamp=time.time()
    )


# ============================================================
# TOOBIT
# ============================================================

async def toobit_book():

    data = await http_json(

        "https://api.toobit.com/api/v1/depth",

        {
            "symbol": "BTCUSDT",

            "limit": ORDERBOOK_LEVELS
        }
    )


    return Book(

        venue="toobit",

        symbol=SYMBOL,

        bids=normalize_levels(
            data.get(
                "bids"
            )
        ),

        asks=normalize_levels(
            data.get(
                "asks"
            )
        ),

        timestamp=time.time()
    )


# ============================================================
# WEEX
# ============================================================

async def weex_book():

    data = await http_json(

        "https://api-spot.weex.com/api/v3/market/depth",

        {
            "symbol": "BTCUSDT",

            "limit": ORDERBOOK_LEVELS
        }
    )


    return Book(

        venue="weex",

        symbol=SYMBOL,

        bids=normalize_levels(
            data.get(
                "bids"
            )
        ),

        asks=normalize_levels(
            data.get(
                "asks"
            )
        ),

        timestamp=time.time()
    )


# ============================================================
# 1BIT
# ============================================================

async def onebit_book():

    data = await http_json(

        "https://1bit.trade/api/v2/public/markets/btcusdt/depth",

        {
            "limit": ORDERBOOK_LEVELS
        }
    )


    depth = data.get(
        "data",
        data
    )


    return Book(

        venue="1bit",

        symbol=SYMBOL,

        bids=normalize_levels(
            depth.get(
                "bids"
            )
        ),

        asks=normalize_levels(
            depth.get(
                "asks"
            )
        ),

        timestamp=time.time()
    )


# ============================================================
# HYPERLIQUID
# ============================================================

async def init_hyperliquid():

    exchange = ccxt.hyperliquid({

        "enableRateLimit": True,

        "timeout": 10000
    })


    await exchange.load_markets()


    ccxt_venues[
        "hyperliquid"
    ] = exchange


async def hyperliquid_book():

    exchange = ccxt_venues[
        "hyperliquid"
    ]


    orderbook = await exchange.fetch_order_book(

        SYMBOL,

        ORDERBOOK_LEVELS
    )


    return Book(

        venue="hyperliquid",

        symbol=SYMBOL,

        bids=normalize_levels(
            orderbook.get(
                "bids"
            )
        ),

        asks=normalize_levels(
            orderbook.get(
                "asks"
            )
        ),

        timestamp=time.time()
    )


# ============================================================
# VENUE FUNCTIONS
# ============================================================

VENUE_FUNCTIONS = {

    "coinex":
        coinex_book,

    "toobit":
        toobit_book,

    "weex":
        weex_book,

    "1bit":
        onebit_book,

    "hyperliquid":
        hyperliquid_book,
}


# ============================================================
# GET ALL ORDERBOOKS
# ============================================================

async def fetch_all_books():

    results = await asyncio.gather(

        *(
            function()

            for function
            in VENUE_FUNCTIONS.values()
        ),

        return_exceptions=True
    )


    books = {}


    for venue, result in zip(

        VENUE_FUNCTIONS,

        results
    ):

        if isinstance(
            result,
            Book
        ):

            if (

                result.best_bid > 0

                and

                result.best_ask > 0

            ):

                books[
                    venue
                ] = result

        else:

            print(
                f"[BOOK ERROR] "
                f"{venue}: "
                f"{result}"
            )


    return books


# ============================================================
# SIMULATE BUY FROM ASK SIDE
# ============================================================

def simulate_buy(

    asks,

    usdt_amount

):

    remaining_usdt = (
        usdt_amount
    )


    btc_amount = 0.0


    spent = 0.0


    for price, amount in asks:

        if remaining_usdt <= 0:

            break


        level_cost = (
            price * amount
        )


        take_cost = min(

            remaining_usdt,

            level_cost
        )


        take_amount = (
            take_cost / price
        )


        btc_amount += (
            take_amount
        )


        spent += (
            take_cost
        )


        remaining_usdt -= (
            take_cost
        )


    if (

        btc_amount <= 0

        or

        spent <= 0

    ):

        return None


    average_price = (
        spent / btc_amount
    )


    slippage = (

        (
            average_price
            /
            asks[0][0]
        )

        - 1

    ) * 100


    return {

        "base":
            btc_amount,

        "quote":
            spent,

        "avg":
            average_price,

        "slippage":
            slippage,

        "filled":
            remaining_usdt <= 0.00000001
    }


# ============================================================
# SIMULATE SELL INTO BID SIDE
# ============================================================

def simulate_sell(

    bids,

    btc_amount

):

    remaining_btc = (
        btc_amount
    )


    received_usdt = 0.0


    sold_btc = 0.0


    for price, amount in bids:

        if remaining_btc <= 0:

            break


        take_amount = min(

            remaining_btc,

            amount
        )


        sold_btc += (
            take_amount
        )


        received_usdt += (

            take_amount
            *
            price
        )


        remaining_btc -= (
            take_amount
        )


    if (

        sold_btc <= 0

        or

        received_usdt <= 0

    ):

        return None


    average_price = (

        received_usdt
        /
        sold_btc
    )


    slippage = (

        (
            bids[0][0]
            -
            average_price
        )
        /
        bids[0][0]

    ) * 100


    return {

        "base":
            sold_btc,

        "quote":
            received_usdt,

        "avg":
            average_price,

        "slippage":
            slippage,

        "filled":
            remaining_btc <= 0.00000001
    }


# ============================================================
# EVALUATE ARBITRAGE
# ============================================================

def evaluate_pair(

    buy_book,

    sell_book

):

    buy = simulate_buy(

        buy_book.asks,

        TRADE_SIZE_USDT
    )


    if not buy:

        return None


    if not buy["filled"]:

        return None


    sell = simulate_sell(

        sell_book.bids,

        buy["base"]
    )


    if not sell:

        return None


    if not sell["filled"]:

        return None


    buy_fee = (

        FEES[
            buy_book.venue
        ]
        /
        100
    )


    sell_fee = (

        FEES[
            sell_book.venue
        ]
        /
        100
    )


    gross_profit = (

        sell["quote"]
        -
        buy["quote"]
    )


    trading_fees = (

        buy["quote"]
        *
        buy_fee

        +

        sell["quote"]
        *
        sell_fee
    )


    net_profit = (

        gross_profit

        -

        trading_fees

        -

        TRANSFER_COST_USDT
    )


    net_percent = (

        net_profit
        /
        buy["quote"]

    ) * 100


    return {

        "buy_venue":
            buy_book.venue,

        "sell_venue":
            sell_book.venue,

        "buy_avg":
            buy["avg"],

        "sell_avg":
            sell["avg"],

        "buy_best":
            buy_book.best_ask,

        "sell_best":
            sell_book.best_bid,

        "gross":
            gross_profit,

        "fees":
            trading_fees,

        "transfer":
            TRANSFER_COST_USDT,

        "net_profit":
            net_profit,

        "net_percent":
            net_percent,

        "buy_slippage":
            buy["slippage"],

        "sell_slippage":
            sell["slippage"],

        "total_slippage":
            (
                buy["slippage"]
                +
                sell["slippage"]
            ),

        "base_amount":
            buy["base"]
    }


# ============================================================
# FIND ALL OPPORTUNITIES
# ============================================================

def find_opportunities(
    books
):

    opportunities = []


    venues = list(
        books.values()
    )


    for buy_book in venues:

        for sell_book in venues:

            if (

                buy_book.venue
                ==
                sell_book.venue

            ):

                continue


            result = evaluate_pair(

                buy_book,

                sell_book
            )


            if result:

                opportunities.append(
                    result
                )


    return sorted(

        opportunities,

        key=lambda x:
            x["net_percent"],

        reverse=True
    )


# ============================================================
# TELEGRAM KEYBOARD
# ============================================================

def keyboard():

    kb = InlineKeyboardBuilder()


    kb.button(

        text="🔎 Сканировать",

        callback_data="scan"
    )


    kb.button(

        text="📚 Стаканы",

        callback_data="books"
    )


    kb.button(

        text="🔥 Возможности",

        callback_data="opps"
    )


    kb.button(

        text="▶️ Автоскан",

        callback_data="auto_start"
    )


    kb.button(

        text="⏹ Стоп",

        callback_data="auto_stop"
    )


    kb.button(

        text="💰 PAPER",

        callback_data="paper"
    )


    kb.button(

        text="📈 Статистика",

        callback_data="stats"
    )


    kb.button(

        text="⚙️ Настройки",

        callback_data="settings"
    )


    kb.adjust(
        2,
        2,
        2,
        2
    )


    return kb.as_markup()


# ============================================================
# FORMAT OPPORTUNITY
# ============================================================

def format_opportunity(
    op
):

    return (

        "🔥 <b>АРБИТРАЖ НАЙДЕН</b>\n\n"

        f"💱 <b>{SYMBOL}</b>\n\n"

        f"🟢 BUY → "
        f"<b>{op['buy_venue'].upper()}</b>\n"

        f"Средняя цена: "
        f"${op['buy_avg']:,.2f}\n"

        f"Best Ask: "
        f"${op['buy_best']:,.2f}\n\n"

        f"🔴 SELL → "
        f"<b>{op['sell_venue'].upper()}</b>\n"

        f"Средняя цена: "
        f"${op['sell_avg']:,.2f}\n"

        f"Best Bid: "
        f"${op['sell_best']:,.2f}\n\n"

        "━━━━━━━━━━━━━━━━\n\n"

        f"📈 Gross: "
        f"${op['gross']:.4f}\n"

        f"💸 Trading fees: "
        f"-${op['fees']:.4f}\n"

        f"🌐 Transfer reserve: "
        f"-${op['transfer']:.2f}\n"

        f"📉 Slippage: "
        f"{op['total_slippage']:.4f}%\n\n"

        f"💰 NET: "
        f"<b>${op['net_profit']:.4f}</b>\n"

        f"📊 NET %: "
        f"<b>{op['net_percent']:.3f}%</b>\n\n"

        f"📦 BTC: "
        f"{op['base_amount']:.8f}\n\n"

        "🧪 <b>PAPER MODE</b>\n"

        "Реальные ордера не отправляются."
    )


# ============================================================
# /START
# ============================================================

@dp.message(
    Command("start")
)
async def start_command(
    message: Message
):

    await message.answer(

        "🐻‍❄️ <b>CRYPTO ARBITRAGE ENGINE</b>\n\n"

        "Я сравниваю 5 площадок:\n\n"

        "• CoinEx\n"
        "• Toobit\n"
        "• WEEX\n"
        "• 1bit\n"
        "• Hyperliquid\n\n"

        f"💱 Пара: <b>{SYMBOL}</b>\n"

        f"📦 Размер: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n\n"

        "Бот получает стаканы и считает "
        "среднюю цену исполнения, "
        "комиссии, проскальзывание и "
        "резерв на transfer/network costs.\n\n"

        "🧪 Сейчас включён PAPER MODE.",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# SCAN
# ============================================================

@dp.callback_query(
    F.data == "scan"
)
async def scan_button(
    call: CallbackQuery
):

    global last_scan
    global last_opportunities


    await call.answer(
        "Сканирую..."
    )


    try:

        books = await fetch_all_books()


        opportunities = (
            find_opportunities(
                books
            )
        )


        last_opportunities = (
            opportunities
        )


        last_scan = (
            datetime.now()
        )


        if not opportunities:

            text = (

                "❌ <b>НЕТ ДАННЫХ</b>\n\n"

                "Не удалось получить "
                "достаточно стаканов."
            )

        else:

            best = opportunities[0]


            if (

                best["net_percent"]
                <
                MIN_NET_PROFIT

            ):

                text = (

                    "🔎 <b>СКАН ЗАВЕРШЁН</b>\n\n"

                    f"Стаканов: "
                    f"<b>{len(books)}/5</b>\n\n"

                    f"Лучший NET: "
                    f"<b>{best['net_percent']:.3f}%</b>\n\n"

                    f"Порог: "
                    f"<b>{MIN_NET_PROFIT:.3f}%</b>\n\n"

                    "❌ Возможность ниже "
                    "установленного порога."
                )

            else:

                text = (
                    format_opportunity(
                        best
                    )
                )


        await call.message.edit_text(

            text,

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


    except Exception as e:

        await call.message.edit_text(

            "⚠️ <b>ОШИБКА</b>\n\n"

            f"<code>{e}</code>",

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


# ============================================================
# BOOKS
# ============================================================

@dp.callback_query(
    F.data == "books"
)
async def books_button(
    call: CallbackQuery
):

    await call.answer(
        "Получаю стаканы..."
    )


    try:

        books = (
            await fetch_all_books()
        )


        lines = [

            "📚 <b>СТАКАНЫ</b>\n",

            f"💱 {SYMBOL}\n"
        ]


        for venue in VENUE_FUNCTIONS:

            book = books.get(
                venue
            )


            if not book:

                lines.append(

                    f"🔴 <b>{venue.upper()}</b>"
                    " — нет данных\n"
                )

                continue


            spread = (

                (
                    book.best_ask
                    /
                    book.best_bid
                )

                - 1

            ) * 100


            lines.append(

                f"🟢 <b>{venue.upper()}</b>\n"

                f"ASK: "
                f"${book.best_ask:,.2f}\n"

                f"BID: "
                f"${book.best_bid:,.2f}\n"

                f"Spread: "
                f"{spread:.4f}%\n"
            )


        await call.message.edit_text(

            "\n".join(lines),

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


    except Exception as e:

        await call.message.edit_text(

            f"⚠️ Ошибка:\n"
            f"<code>{e}</code>",

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


# ============================================================
# OPPORTUNITIES
# ============================================================

@dp.callback_query(
    F.data == "opps"
)
async def opportunities_button(
    call: CallbackQuery
):

    await call.answer(
        "Считаю..."
    )


    try:

        books = (
            await fetch_all_books()
        )


        opportunities = (
            find_opportunities(
                books
            )
        )


        good = [

            x

            for x
            in opportunities

            if (

                x["net_percent"]
                >=
                MIN_NET_PROFIT

            )

        ][:10]


        if not good:

            text = (

                "❌ <b>ВОЗМОЖНОСТЕЙ НЕТ</b>\n\n"

                f"Порог: "
                f"{MIN_NET_PROFIT:.3f}%"
            )


        else:

            lines = [

                "🔥 <b>ТОП АРБИТРАЖА</b>\n"
            ]


            for index, op in enumerate(

                good,

                1
            ):

                lines.append(

                    f"{index}. "

                    f"{op['buy_venue'].upper()}"

                    " → "

                    f"{op['sell_venue'].upper()}\n"

                    f"NET: "
                    f"<b>{op['net_percent']:.3f}%</b>\n"

                    f"Profit: "
                    f"<b>${op['net_profit']:.4f}</b>\n"
                )


            text = "\n".join(
                lines
            )


        await call.message.edit_text(

            text,

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


    except Exception as e:

        await call.message.edit_text(

            f"⚠️ Ошибка:\n"
            f"<code>{e}</code>",

            reply_markup=keyboard(),

            parse_mode="HTML"
        )


# ============================================================
# PAPER
# ============================================================

@dp.callback_query(
    F.data == "paper"
)
async def paper_button(
    call: CallbackQuery
):

    await call.answer()


    await call.message.edit_text(

        "💰 <b>PAPER ACCOUNT</b>\n\n"

        f"Начальный баланс: "
        f"<b>$1,000.00</b>\n\n"

        f"Текущий баланс: "
        f"<b>${paper_balance:.2f}</b>\n\n"

        f"Прибыль: "
        f"<b>${paper_profit:.2f}</b>\n\n"

        f"Сделок: "
        f"<b>{paper_trades}</b>\n\n"

        "🧪 Виртуальный счёт.\n"
        "Реальные деньги не используются.",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# STATS
# ============================================================

@dp.callback_query(
    F.data == "stats"
)
async def stats_button(
    call: CallbackQuery
):

    await call.answer()


    roi = (

        paper_profit
        /
        1000

    ) * 100


    await call.message.edit_text(

        "📈 <b>СТАТИСТИКА</b>\n\n"

        f"💰 Баланс: "
        f"${paper_balance:.2f}\n"

        f"📈 Прибыль: "
        f"${paper_profit:.2f}\n"

        f"📊 ROI: "
        f"{roi:.3f}%\n"

        f"🔄 Сделок: "
        f"{paper_trades}\n",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# SETTINGS
# ============================================================

@dp.callback_query(
    F.data == "settings"
)
async def settings_button(
    call: CallbackQuery
):

    await call.answer()


    await call.message.edit_text(

        "⚙️ <b>НАСТРОЙКИ</b>\n\n"

        f"💱 Пара: "
        f"<b>{SYMBOL}</b>\n\n"

        f"💵 Размер: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n\n"

        f"🎯 Минимальный NET: "
        f"<b>{MIN_NET_PROFIT:.3f}%</b>\n\n"

        f"📚 Уровней стакана: "
        f"<b>{ORDERBOOK_LEVELS}</b>\n\n"

        f"🌐 Transfer reserve: "
        f"<b>${TRANSFER_COST_USDT:.2f}</b>\n\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n\n"

        "🧪 LIVE TRADING: <b>OFF</b>\n\n"

        "Настройки изменяются через .env.",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# AUTO SCANNER
# ============================================================

async def auto_loop(
    chat_id
):

    while True:

        try:

            books = (
                await fetch_all_books()
            )


            opportunities = (
                find_opportunities(
                    books
                )
            )


            if opportunities:

                best = (
                    opportunities[0]
                )


                if (

                    best["net_percent"]
                    >=
                    MIN_NET_PROFIT

                ):

                    await bot.send_message(

                        chat_id,

                        format_opportunity(
                            best
                        ),

                        parse_mode="HTML"
                    )


        except asyncio.CancelledError:

            raise


        except Exception as e:

            print(
                "[AUTO ERROR]",
                e
            )


        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# AUTO START
# ============================================================

@dp.callback_query(
    F.data == "auto_start"
)
async def auto_start(
    call: CallbackQuery
):

    chat_id = (
        call.message.chat.id
    )


    if chat_id in auto_tasks:

        await call.answer(
            "Уже запущено"
        )

        return


    auto_tasks[
        chat_id
    ] = asyncio.create_task(

        auto_loop(
            chat_id
        )
    )


    await call.answer(
        "Запущено"
    )


    await call.message.edit_text(

        "▶️ <b>АВТОСКАН ЗАПУЩЕН</b>\n\n"

        "Площадки:\n"

        "CoinEx\n"
        "Toobit\n"
        "WEEX\n"
        "1bit\n"
        "Hyperliquid\n\n"

        f"💱 {SYMBOL}\n"

        f"💵 ${TRADE_SIZE_USDT:.2f}\n"

        f"🎯 {MIN_NET_PROFIT:.3f}%\n"

        f"⏱ {SCAN_INTERVAL} сек.",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# AUTO STOP
# ============================================================

@dp.callback_query(
    F.data == "auto_stop"
)
async def auto_stop(
    call: CallbackQuery
):

    chat_id = (
        call.message.chat.id
    )


    task = auto_tasks.pop(
        chat_id,
        None
    )


    if task:

        task.cancel()


    await call.answer(
        "Остановлено"
    )


    await call.message.edit_text(

        "⏹ <b>АВТОСКАН ОСТАНОВЛЕН</b>",

        reply_markup=keyboard(),

        parse_mode="HTML"
    )


# ============================================================
# UNKNOWN MESSAGE
# ============================================================

@dp.message()
async def fallback(
    message: Message
):

    await message.answer(

        "🐻‍❄️ Используй меню:",

        reply_markup=keyboard()
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    global session


    print(
        "========================================"
    )

    print(
        "🐻‍❄️ CRYPTO ARBITRAGE ENGINE"
    )

    print(
        "========================================"
    )

    print(
        "Mode: PAPER"
    )

    print(
        "Live trading: OFF"
    )

    print(
        "Starting..."
    )


    session = (
        aiohttp.ClientSession()
    )


    try:

        await init_hyperliquid()


        print(
            "Hyperliquid loaded."
        )


        print(
            "Starting Telegram..."
        )


        await dp.start_polling(
            bot
        )


    finally:

        for task in auto_tasks.values():

            task.cancel()


        for exchange in (
            ccxt_venues.values()
        ):

            try:

                await exchange.close()

            except Exception:

                pass


        if session:

            await session.close()


        await bot.session.close()


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )