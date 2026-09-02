import asyncio
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import aiohttp
from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command
from aiogram.types import Message, CallbackQuery
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from dotenv import load_dotenv

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

SYMBOLS = [
    x.strip().upper()
    for x in os.getenv(
        "SYMBOLS",
        "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,TON/USDT,"
        "ADA/USDT,AVAX/USDT,LINK/USDT,LTC/USDT,DOT/USDT"
    ).split(",")
    if x.strip()
]

TRADE_SIZE_USDT = float(os.getenv("TRADE_SIZE_USDT", "50"))
PAPER_START_BALANCE = float(os.getenv("PAPER_START_BALANCE", "1000"))
MIN_NET_PROFIT = float(os.getenv("MIN_NET_PROFIT_PERCENT", "0.20"))
SCAN_INTERVAL = float(os.getenv("SCAN_INTERVAL_SECONDS", "5"))
ORDERBOOK_LEVELS = int(os.getenv("ORDERBOOK_LEVELS", "20"))
TRANSFER_COST_USDT = float(os.getenv("TRANSFER_COST_USDT", "1.00"))
PAPER_TRADE_COOLDOWN = int(
    os.getenv("PAPER_TRADE_COOLDOWN_SECONDS", "30")
)

FEES = {
    "coinex": float(os.getenv("FEE_COINEX_PERCENT", "0.10")),
    "toobit": float(os.getenv("FEE_TOOBIT_PERCENT", "0.10")),
    "weex": float(os.getenv("FEE_WEEX_PERCENT", "0.10")),
    "1bit": float(os.getenv("FEE_1BIT_PERCENT", "0.10")),
    "hyperliquid": float(os.getenv("FEE_HYPERLIQUID_PERCENT", "0.10")),
    "bybit": float(os.getenv("FEE_BYBIT_PERCENT", "0.10")),
    "binance": float(os.getenv("FEE_BINANCE_PERCENT", "0.10")),
    "okx": float(os.getenv("FEE_OKX_PERCENT", "0.10")),
    "gate": float(os.getenv("FEE_GATE_PERCENT", "0.10")),
    "kucoin": float(os.getenv("FEE_KUCOIN_PERCENT", "0.10")),
}

# Реальные сделки полностью выключены.
LIVE_TRADING = False

VENUE_NAMES = {
    "coinex": "COINEX",
    "toobit": "TOOBIT",
    "weex": "WEEX",
    "1bit": "1BIT",
    "hyperliquid": "HYPERLIQUID",
    "bybit": "BYBIT",
    "binance": "BINANCE",
    "okx": "OKX",
    "gate": "GATE.IO",
    "kucoin": "KUCOIN",
}


@dataclass
class Book:
    venue: str
    requested_symbol: str
    actual_symbol: str
    base: str
    quote: str
    bids: list
    asks: list
    timestamp: float

    @property
    def best_bid(self):
        return self.bids[0] if self.bids else None

    @property
    def best_ask(self):
        return self.asks[0] if self.asks else None

    @property
    def spread_percent(self):
        if not self.best_bid or not self.best_ask:
            return 0.0

        bid = self.best_bid[0]
        ask = self.best_ask[0]

        return (ask - bid) / ask * 100 if ask > 0 else 0.0


bot: Optional[Bot] = None
dp = Dispatcher()

http_session: Optional[aiohttp.ClientSession] = None

auto_scan_task: Optional[asyncio.Task] = None
auto_scan_running = False

# chat_id -> message_id
dashboard_messages = {}

last_books = {}
last_errors = {}
last_opportunities = []

market_updates = 0

paper_balance = PAPER_START_BALANCE
paper_profit = 0.0
paper_trades = 0
paper_wins = 0
paper_losses = 0
paper_volume = 0.0

paper_history = []
last_paper_trades = {}

hyperliquid_meta = None
hyperliquid_meta_time = 0


# ============================================================
# SYMBOL HELPERS
# ============================================================

def split_symbol(symbol):
    if "/" in symbol:
        return tuple(symbol.upper().split("/", 1))

    symbol = symbol.upper()

    for quote in ("USDT", "USDC"):
        if symbol.endswith(quote):
            return symbol[:-len(quote)], quote

    return symbol, "USDT"


def clean_symbol(symbol):
    return (
        symbol
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .upper()
    )


# ============================================================
# ORDERBOOK NORMALIZATION
# ============================================================

def normalize_levels(levels, reverse=False):
    result = []

    if not isinstance(levels, list):
        return result

    for item in levels:
        try:
            if isinstance(item, dict):
                p = (
                    item.get("price")
                    or item.get("px")
                    or item.get("p")
                )

                q = (
                    item.get("quantity")
                    or item.get("qty")
                    or item.get("size")
                    or item.get("sz")
                    or item.get("q")
                )

            else:
                p = item[0]
                q = item[1]

            p = float(p)
            q = float(q)

            if p > 0 and q > 0:
                result.append((p, q))

        except Exception:
            pass

    result.sort(
        key=lambda x: x[0],
        reverse=reverse
    )

    return result[:ORDERBOOK_LEVELS]


# ============================================================
# HTTP
# ============================================================

async def get_json(
    url,
    method="GET",
    params=None,
    json_data=None,
    timeout=10
):
    if http_session is None:
        raise RuntimeError(
            "HTTP session is not initialized"
        )

    try:
        timeout_obj = aiohttp.ClientTimeout(
            total=timeout
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "ArbitrageBot/3.0"
        }

        if method == "POST":
            async with http_session.post(
                url,
                json=json_data,
                params=params,
                timeout=timeout_obj,
                headers=headers
            ) as r:

                text = await r.text()

                if r.status != 200:
                    raise RuntimeError(
                        f"HTTP {r.status}: {text[:200]}"
                    )

                return await r.json(
                    content_type=None
                )

        async with http_session.get(
            url,
            params=params,
            timeout=timeout_obj,
            headers=headers
        ) as r:

            text = await r.text()

            if r.status != 200:
                raise RuntimeError(
                    f"HTTP {r.status}: {text[:200]}"
                )

            return await r.json(
                content_type=None
            )

    except asyncio.TimeoutError:
        raise RuntimeError("timeout")

    except aiohttp.ClientError as e:
        raise RuntimeError(
            f"connection error: {e}"
        )


def make_book(
    venue,
    symbol,
    actual_symbol,
    bids,
    asks,
    base,
    quote
):
    bids = normalize_levels(
        bids,
        True
    )

    asks = normalize_levels(
        asks,
        False
    )

    if not bids or not asks:
        raise RuntimeError(
            "empty order book"
        )

    return Book(
        venue,
        symbol,
        actual_symbol,
        base,
        quote,
        bids,
        asks,
        time.time()
    )


# ============================================================
# COINEX
# ============================================================

async def fetch_coinex(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    limit = min(
        [5, 10, 20, 50],
        key=lambda x: abs(
            x - ORDERBOOK_LEVELS
        )
    )

    data = await get_json(
        "https://api.coinex.com/v2/spot/depth",
        params={
            "market": clean_symbol(symbol),
            "limit": limit,
            "interval": "0"
        }
    )

    if data.get("code") not in (
        0,
        "0",
        None
    ):
        raise RuntimeError(
            str(
                data.get(
                    "message",
                    "API error"
                )
            )
        )

    d = data.get(
        "data",
        data
    ).get(
        "depth",
        data.get(
            "data",
            data
        )
    )

    return make_book(
        "coinex",
        symbol,
        symbol,
        d.get("bids", []),
        d.get("asks", []),
        base,
        quote
    )


# ============================================================
# TOOBIT
# ============================================================

async def fetch_toobit(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    limit = min(
        [5, 10, 20, 50, 100, 500, 1000],
        key=lambda x: abs(
            x - ORDERBOOK_LEVELS
        )
    )

    data = await get_json(
        "https://api.toobit.com/quote/v1/depth/merged",
        params={
            "symbol": clean_symbol(symbol),
            "scale": 0,
            "limit": limit
        }
    )

    return make_book(
        "toobit",
        symbol,
        symbol,
        data.get("b", []),
        data.get("a", []),
        base,
        quote
    )


# ============================================================
# WEEX
# ============================================================

async def fetch_weex(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    limit = (
        200
        if ORDERBOOK_LEVELS > 15
        else 15
    )

    data = await get_json(
        "https://api-spot.weex.com/api/v3/market/depth",
        params={
            "symbol": clean_symbol(symbol),
            "limit": limit
        }
    )

    if data.get("code") not in (
        None,
        0,
        "0",
        "00000"
    ):
        raise RuntimeError(
            str(
                data.get(
                    "msg",
                    "API error"
                )
            )
        )

    d = data.get(
        "data",
        data
    )

    return make_book(
        "weex",
        symbol,
        symbol,
        d.get("bids", []),
        d.get("asks", []),
        base,
        quote
    )


# ============================================================
# 1BIT
# ============================================================

async def fetch_1bit(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    market = clean_symbol(
        symbol
    ).lower()

    data = await get_json(
        f"https://1bit.trade/api/v2/public/markets/{market}/depth",
        params={
            "limit": ORDERBOOK_LEVELS
        }
    )

    d = (
        data.get("data", data)
        if isinstance(data, dict)
        else {}
    )

    d = (
        d.get("result", d)
        if isinstance(d, dict)
        else d
    )

    return make_book(
        "1bit",
        symbol,
        symbol,
        d.get("bids")
        or d.get("buy")
        or d.get("b")
        or [],
        d.get("asks")
        or d.get("sell")
        or d.get("a")
        or [],
        base,
        quote
    )


# ============================================================
# HYPERLIQUID
# ============================================================

async def hyperliquid_request(payload):
    return await get_json(
        "https://api.hyperliquid.xyz/info",
        method="POST",
        json_data=payload
    )


def normalize_hl(
    levels,
    reverse=False
):
    return normalize_levels(
        [
            {
                "px": x.get("px"),
                "sz": x.get("sz")
            }
            for x in levels
        ],
        reverse
    )


async def load_hl_meta():
    global hyperliquid_meta
    global hyperliquid_meta_time

    if (
        hyperliquid_meta
        and time.time() - hyperliquid_meta_time < 600
    ):
        return hyperliquid_meta

    data = await hyperliquid_request(
        {
            "type": "spotMeta"
        }
    )

    tokens = {
        int(t["index"]): str(
            t.get("name", "")
        ).upper()
        for t in data.get(
            "tokens",
            []
        )
        if "index" in t
    }

    result = {}

    for pair in data.get(
        "universe",
        []
    ):
        ts = pair.get(
            "tokens",
            []
        )

        if len(ts) == 2:
            b = tokens.get(
                int(ts[0]),
                ""
            )

            q = tokens.get(
                int(ts[1]),
                ""
            )

            if b and q:
                result[
                    (b, q)
                ] = str(
                    pair.get("name")
                )

    hyperliquid_meta = result
    hyperliquid_meta_time = time.time()

    return result


async def fetch_hyperliquid(symbol):
    base, quote = split_symbol(symbol)

    meta = await load_hl_meta()

    actual = (
        meta.get(
            (base, quote)
        )
        or meta.get(
            (base, "USDC")
        )
    )

    if not actual:
        raise RuntimeError(
            "pair not available"
        )

    actual_base, actual_quote = split_symbol(
        actual
    )

    data = await hyperliquid_request(
        {
            "type": "l2Book",
            "coin": actual
        }
    )

    levels = data.get(
        "levels",
        []
    )

    if len(levels) < 2:
        raise RuntimeError(
            "empty order book"
        )

    return make_book(
        "hyperliquid",
        symbol,
        actual,
        normalize_hl(
            levels[0],
            True
        ),
        normalize_hl(
            levels[1],
            False
        ),
        actual_base,
        actual_quote
    )


# ============================================================
# BYBIT
# ============================================================

async def fetch_bybit(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    data = await get_json(
        "https://api.bybit.com/v5/market/orderbook",
        params={
            "category": "spot",
            "symbol": clean_symbol(symbol),
            "limit": min(
                max(
                    ORDERBOOK_LEVELS,
                    1
                ),
                1000
            )
        }
    )

    if data.get("retCode") != 0:
        raise RuntimeError(
            str(
                data.get(
                    "retMsg",
                    "API error"
                )
            )
        )

    d = data["result"]

    return make_book(
        "bybit",
        symbol,
        symbol,
        d.get("b", []),
        d.get("a", []),
        base,
        quote
    )


# ============================================================
# BINANCE
# ============================================================

async def fetch_binance(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    data = await get_json(
        "https://api.binance.com/api/v3/depth",
        params={
            "symbol": clean_symbol(symbol),
            "limit": min(
                max(
                    ORDERBOOK_LEVELS,
                    5
                ),
                1000
            )
        }
    )

    return make_book(
        "binance",
        symbol,
        symbol,
        data.get("bids", []),
        data.get("asks", []),
        base,
        quote
    )


# ============================================================
# OKX
# ============================================================

async def fetch_okx(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    inst = f"{base}-{quote}"

    data = await get_json(
        "https://www.okx.com/api/v5/market/books",
        params={
            "instId": inst,
            "sz": min(
                max(
                    ORDERBOOK_LEVELS,
                    1
                ),
                400
            )
        }
    )

    if data.get("code") != "0":
        raise RuntimeError(
            str(
                data.get(
                    "msg",
                    "API error"
                )
            )
        )

    d = data.get(
        "data",
        []
    )

    if not d:
        raise RuntimeError(
            "empty order book"
        )

    return make_book(
        "okx",
        symbol,
        inst,
        d[0].get("bids", []),
        d[0].get("asks", []),
        base,
        quote
    )


# ============================================================
# GATE.IO
# ============================================================

async def fetch_gate(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    pair = f"{base}_{quote}"

    data = await get_json(
        "https://api.gateio.ws/api/v4/spot/order_book",
        params={
            "currency_pair": pair,
            "interval": "0",
            "limit": min(
                max(
                    ORDERBOOK_LEVELS,
                    1
                ),
                1000
            )
        }
    )

    return make_book(
        "gate",
        symbol,
        pair,
        data.get("bids", []),
        data.get("asks", []),
        base,
        quote
    )


# ============================================================
# KUCOIN
# ============================================================

async def fetch_kucoin(symbol):
    base, quote = split_symbol(symbol)

    if quote != "USDT":
        raise RuntimeError(
            "USDT spot only"
        )

    pair = f"{base}-{quote}"

    size = (
        20
        if ORDERBOOK_LEVELS <= 20
        else 100
    )

    data = await get_json(
        f"https://api.kucoin.com/api/v1/market/orderbook/level2_{size}",
        params={
            "symbol": pair
        }
    )

    if data.get("code") != "200000":
        raise RuntimeError(
            str(
                data.get(
                    "msg",
                    "API error"
                )
            )
        )

    d = data.get(
        "data",
        {}
    )

    return make_book(
        "kucoin",
        symbol,
        pair,
        d.get("bids", []),
        d.get("asks", []),
        base,
        quote
    )


# ============================================================
# EXCHANGE MAP
# ============================================================

VENUE_FUNCTIONS = {
    "coinex": fetch_coinex,
    "toobit": fetch_toobit,
    "weex": fetch_weex,
    "1bit": fetch_1bit,
    "hyperliquid": fetch_hyperliquid,
    "bybit": fetch_bybit,
    "binance": fetch_binance,
    "okx": fetch_okx,
    "gate": fetch_gate,
    "kucoin": fetch_kucoin,
}


# ============================================================
# FETCH ONE
# ============================================================

async def fetch_one(
    venue,
    symbol,
    func
):
    try:
        return (
            venue,
            symbol,
            await func(symbol),
            None
        )

    except Exception as e:
        return (
            venue,
            symbol,
            None,
            str(e)
        )


# ============================================================
# FETCH ALL BOOKS
# ============================================================

async def fetch_all_books():
    tasks = [
        fetch_one(
            venue,
            symbol,
            func
        )
        for venue, func
        in VENUE_FUNCTIONS.items()
        for symbol in SYMBOLS
    ]

    results = await asyncio.gather(
        *tasks
    )

    books = {}
    errors = {}

    for (
        venue,
        symbol,
        book,
        error
    ) in results:

        if book:
            books[
                (symbol, venue)
            ] = book

        else:
            errors[
                (symbol, venue)
            ] = error

    return books, errors


# ============================================================
# PAPER BUY
# ============================================================

def simulate_buy(
    asks,
    usdt
):
    if not asks:
        return (
            0,
            0,
            0,
            0,
            False
        )

    remain = usdt
    bought = 0
    spent = 0

    first = asks[0][0]

    for p, q in asks:
        if remain <= 0:
            break

        take = min(
            q,
            remain / p
        )

        bought += take
        spent += take * p
        remain -= take * p

    filled = (
        remain
        <= max(
            1e-6,
            usdt * 1e-6
        )
    )

    if bought <= 0:
        return (
            0,
            0,
            0,
            0,
            False
        )

    avg = spent / bought

    return (
        bought,
        spent,
        avg,
        (avg - first) / first * 100,
        filled
    )


# ============================================================
# PAPER SELL
# ============================================================

def simulate_sell(
    bids,
    amount
):
    if not bids or amount <= 0:
        return (
            0,
            0,
            0,
            0,
            False
        )

    remain = amount
    sold = 0
    received = 0

    first = bids[0][0]

    for p, q in bids:
        if remain <= 0:
            break

        take = min(
            q,
            remain
        )

        sold += take
        received += take * p
        remain -= take

    filled = (
        remain
        <= max(
            1e-9,
            amount * 1e-6
        )
    )

    if sold <= 0:
        return (
            0,
            0,
            0,
            0,
            False
        )

    avg = received / sold

    return (
        sold,
        received,
        avg,
        (first - avg) / first * 100,
        filled
    )


# ============================================================
# EVALUATE ARBITRAGE
# ============================================================

def evaluate_pair(
    symbol,
    buy,
    sell
):
    if (
        buy.quote != "USDT"
        or sell.quote != "USDT"
        or buy.base != sell.base
    ):
        return None

    (
        bought,
        buy_quote,
        buy_avg,
        buy_slip,
        bf
    ) = simulate_buy(
        buy.asks,
        TRADE_SIZE_USDT
    )

    if not bf:
        return None

    (
        sold,
        sell_quote,
        sell_avg,
        sell_slip,
        sf
    ) = simulate_sell(
        sell.bids,
        bought
    )

    if not sf:
        return None

    buy_fee = (
        buy_quote
        * FEES.get(
            buy.venue,
            0.1
        )
        / 100
    )

    sell_fee = (
        sell_quote
        * FEES.get(
            sell.venue,
            0.1
        )
        / 100
    )

    gross = (
        sell_quote
        - buy_quote
    )

    fees = (
        buy_fee
        + sell_fee
    )

    net = (
        gross
        - fees
        - TRANSFER_COST_USDT
    )

    pct = (
        net
        / buy_quote
        * 100
    )

    return {
        "symbol": symbol,
        "base": buy.base,
        "buy_exchange": buy.venue,
        "sell_exchange": sell.venue,
        "buy_price": buy_avg,
        "sell_price": sell_avg,
        "amount": bought,
        "buy_quote": buy_quote,
        "sell_quote": sell_quote,
        "gross_profit": gross,
        "fees": fees,
        "transfer_cost": TRANSFER_COST_USDT,
        "net_profit": net,
        "net_percent": pct,
        "buy_slippage": buy_slip,
        "sell_slippage": sell_slip,
    }


# ============================================================
# FIND OPPORTUNITIES
# ============================================================

def find_opportunities(books):
    out = []

    for symbol in SYMBOLS:

        sb = {
            venue: book
            for (
                s,
                venue
            ), book in books.items()
            if s == symbol
        }

        names = list(sb)

        for buy_exchange in names:

            for sell_exchange in names:

                if buy_exchange == sell_exchange:
                    continue

                result = evaluate_pair(
                    symbol,
                    sb[buy_exchange],
                    sb[sell_exchange]
                )

                if result:
                    out.append(result)

    return sorted(
        out,
        key=lambda x: x["net_percent"],
        reverse=True
    )


# ============================================================
# PAPER TRADE
# ============================================================

def execute_paper_trade(o):
    global paper_balance
    global paper_profit
    global paper_trades
    global paper_wins
    global paper_losses
    global paper_volume

    if (
        o["net_percent"] < MIN_NET_PROFIT
        or paper_balance < TRADE_SIZE_USDT
    ):
        return False

    key = (
        f'{o["symbol"]}|'
        f'{o["buy_exchange"]}|'
        f'{o["sell_exchange"]}'
    )

    now = time.time()

    if (
        now
        - last_paper_trades.get(key, 0)
        < PAPER_TRADE_COOLDOWN
    ):
        return False

    profit = o["net_profit"]

    paper_balance += profit
    paper_profit += profit
    paper_trades += 1
    paper_volume += TRADE_SIZE_USDT

    if profit >= 0:
        paper_wins += 1
    else:
        paper_losses += 1

    last_paper_trades[key] = now

    paper_history.insert(
        0,
        {
            **o,
            "time": datetime.now().strftime(
                "%H:%M:%S"
            )
        }
    )

    del paper_history[20:]

    return True


# ============================================================
# FORMATTING
# ============================================================

def fmt_price(v):
    if v < 1:
        return f"${v:,.6f}"

    return f"${v:,.2f}"


# ============================================================
# TELEGRAM KEYBOARD
# ============================================================

def keyboard():
    b = InlineKeyboardBuilder()

    buttons = [
        ("🔎 Сканировать", "scan"),
        ("📚 Стаканы", "books"),
        ("🔥 Возможности", "opps"),
        ("▶️ Автоскан", "auto_start"),
        ("⏹ Стоп", "auto_stop"),
        ("💰 PAPER", "paper"),
        ("📈 Статистика", "stats"),
        ("⚙️ Настройки", "settings"),
    ]

    for text, data in buttons:
        b.button(
            text=text,
            callback_data=data
        )

    b.adjust(2)

    return b.as_markup()


# ============================================================
# SINGLE DASHBOARD MESSAGE
# ============================================================

async def edit_dashboard(
    chat_id,
    text
):
    message_id = dashboard_messages.get(
        chat_id
    )

    # Только первое сообщение создаётся.
    if message_id is None:

        msg = await bot.send_message(
            chat_id,
            text,
            reply_markup=keyboard()
        )

        dashboard_messages[
            chat_id
        ] = msg.message_id

        return

    try:
        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=keyboard()
        )

    except Exception as e:
        # НОВОЕ СООБЩЕНИЕ НЕ СОЗДАЁМ.
        print(
            "[EDIT ERROR]",
            e
        )


# ============================================================
# BOOKS TEXT
# ============================================================

def books_text():
    lines = [
        "📚 <b>СТАКАНЫ</b>",
        ""
    ]

    for symbol in SYMBOLS:

        lines.append(
            f"🪙 <b>{symbol}</b>"
        )

        for venue in VENUE_FUNCTIONS:

            book = last_books.get(
                (symbol, venue)
            )

            if not book:

                lines.append(
                    f"🔴 {VENUE_NAMES[venue]} — нет данных"
                )

            else:

                lines.append(
                    f"🟢 {VENUE_NAMES[venue]}  "
                    f"ASK {fmt_price(book.best_ask[0])} / "
                    f"BID {fmt_price(book.best_bid[0])} / "
                    f"{book.spread_percent:.4f}%"
                )

        lines.append("")

    lines.append(
        f"🔄 Обновлений рынка: "
        f"<b>{market_updates}</b>"
    )

    return "\n".join(lines)


# ============================================================
# OPPORTUNITIES TEXT
# ============================================================

def opps_text():

    if not last_opportunities:

        return (
            f"🔥 <b>ВОЗМОЖНОСТИ</b>\n\n"
            f"Нет связок ≥ "
            f"{MIN_NET_PROFIT:.2f}%.\n\n"
            f"🔄 Обновлений рынка: "
            f"<b>{market_updates}</b>"
        )

    lines = [
        "🔥 <b>ВОЗМОЖНОСТИ</b>",
        ""
    ]

    for i, o in enumerate(
        last_opportunities[:10],
        1
    ):

        lines.append(
            f"#{i} 🪙 <b>{o['symbol']}</b>"
        )

        lines.append(
            f"🛒 "
            f"{VENUE_NAMES[o['buy_exchange']]} "
            f"→ {fmt_price(o['buy_price'])}"
        )

        lines.append(
            f"💰 "
            f"{VENUE_NAMES[o['sell_exchange']]} "
            f"→ {fmt_price(o['sell_price'])}"
        )

        lines.append(
            f"💵 Чистая: "
            f"<b>${o['net_profit']:.4f}</b>  "
            f"📈 <b>{o['net_percent']:.4f}%</b>"
        )

        lines.append("")

    lines.append(
        f"🔄 Обновлений рынка: "
        f"<b>{market_updates}</b>"
    )

    return "\n".join(lines)


# ============================================================
# PAPER TEXT
# ============================================================

def paper_text():

    roi = (
        paper_profit
        / PAPER_START_BALANCE
        * 100
        if PAPER_START_BALANCE
        else 0
    )

    lines = [
        "💰 <b>PAPER MODE</b>",
        "",
        f"Старт: "
        f"<b>${PAPER_START_BALANCE:.2f}</b>",

        f"Баланс: "
        f"<b>${paper_balance:.4f}</b>",

        f"Прибыль: "
        f"<b>${paper_profit:.4f}</b>",

        f"ROI: "
        f"<b>{roi:.4f}%</b>",

        "",

        f"Сделок: "
        f"<b>{paper_trades}</b>",

        f"Успешных: "
        f"<b>{paper_wins}</b>",

        f"Объём: "
        f"<b>${paper_volume:.2f}</b>",

        "",

        f"🔄 Обновлений рынка: "
        f"<b>{market_updates}</b>",

        "🟢 Реальные стаканы / "
        "🟡 виртуальные сделки / "
        "🔴 LIVE OFF"
    ]

    return "\n".join(lines)


# ============================================================
# SCAN
# ============================================================

async def run_scan():
    global last_books
    global last_errors
    global last_opportunities
    global market_updates

    (
        last_books,
        last_errors
    ) = await fetch_all_books()

    last_opportunities = find_opportunities(
        last_books
    )

    market_updates += 1


# ============================================================
# START
# ============================================================

@dp.message(Command("start"))
async def start(message: Message):

    try:
        await message.delete()
    except Exception:
        pass

    await edit_dashboard(
        message.chat.id,

        f"🤖 <b>ARBITRAGE BOT</b>\n\n"

        f"🪙 Активов: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"🏦 Бирж: "
        f"<b>{len(VENUE_FUNCTIONS)}</b>\n"

        f"📈 Порог: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n\n"

        f"🟢 Реальные стаканы\n"
        f"🟡 Виртуальные сделки\n"
        f"🔴 Реальные ордера OFF"
    )


# ============================================================
# MANUAL SCAN
# ============================================================

@dp.callback_query(F.data == "scan")
async def scan(c: CallbackQuery):

    await c.answer(
        "Обновляю рынок..."
    )

    try:

        await run_scan()

        best = (
            f"{last_opportunities[0]['symbol']} "
            f"{last_opportunities[0]['net_percent']:.4f}%"
            if last_opportunities
            else "нет"
        )

        await edit_dashboard(
            c.message.chat.id,

            f"🔎 <b>СКАНИРОВАНИЕ</b>\n\n"

            f"📡 Стаканов: "
            f"<b>{len(last_books)}/"
            f"{len(SYMBOLS) * len(VENUE_FUNCTIONS)}</b>\n"

            f"🔥 Возможностей: "
            f"<b>{len(last_opportunities)}</b>\n"

            f"🏆 Лучшая: "
            f"<b>{best}</b>\n\n"

            + opps_text()
        )

    except Exception as e:

        await edit_dashboard(
            c.message.chat.id,

            f"❌ <b>Ошибка</b>\n\n"
            f"<code>{str(e)[:500]}</code>"
        )


# ============================================================
# BOOKS
# ============================================================

@dp.callback_query(F.data == "books")
async def books(c: CallbackQuery):

    await c.answer(
        "Обновляю стаканы..."
    )

    try:

        await run_scan()

        await edit_dashboard(
            c.message.chat.id,
            books_text()
        )

    except Exception as e:

        await edit_dashboard(
            c.message.chat.id,

            f"❌ <b>Ошибка</b>\n\n"
            f"<code>{str(e)[:500]}</code>"
        )


# ============================================================
# OPPORTUNITIES
# ============================================================

@dp.callback_query(F.data == "opps")
async def opps(c: CallbackQuery):

    await c.answer()

    await edit_dashboard(
        c.message.chat.id,
        opps_text()
    )


# ============================================================
# PAPER
# ============================================================

@dp.callback_query(F.data == "paper")
async def paper(c: CallbackQuery):

    await c.answer()

    await edit_dashboard(
        c.message.chat.id,
        paper_text()
    )


# ============================================================
# STATUS
# ============================================================

@dp.callback_query(F.data == "status")
async def status(c: CallbackQuery):

    await c.answer()

    await edit_dashboard(
        c.message.chat.id,

        f"📊 <b>СТАТУС</b>\n\n"

        f"🪙 Активов: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"🏦 Бирж: "
        f"<b>{len(VENUE_FUNCTIONS)}</b>\n"

        f"📡 Стаканов: "
        f"<b>{len(last_books)}/"
        f"{len(SYMBOLS) * len(VENUE_FUNCTIONS)}</b>\n"

        f"🔄 Обновлений: "
        f"<b>{market_updates}</b>\n"

        f"▶️ Автоскан: "
        f"<b>{'🟢 ON' if auto_scan_running else '🔴 OFF'}</b>\n\n"

        f"💰 PAPER: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"📈 Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 Сделок: "
        f"<b>{paper_trades}</b>\n\n"

        f"🎯 Порог: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"🔴 LIVE: <b>OFF</b>"
    )


# ============================================================
# STATISTICS
# ============================================================

@dp.callback_query(F.data == "stats")
async def stats(c: CallbackQuery):

    await c.answer()

    roi = (
        paper_profit
        / PAPER_START_BALANCE
        * 100
        if PAPER_START_BALANCE
        else 0
    )

    await edit_dashboard(
        c.message.chat.id,

        f"📈 <b>СТАТИСТИКА</b>\n\n"

        f"🔄 Обновлений рынка: "
        f"<b>{market_updates}</b>\n"

        f"📡 Получено стаканов: "
        f"<b>{len(last_books)}</b>\n"

        f"🔥 Возможностей: "
        f"<b>{len(last_opportunities)}</b>\n\n"

        f"💰 Баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"💵 Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 ROI: "
        f"<b>{roi:.4f}%</b>\n"

        f"🔄 PAPER сделок: "
        f"<b>{paper_trades}</b>\n"

        f"🟢 Успешных: "
        f"<b>{paper_wins}</b>\n\n"

        f"🔴 Реальные ордера: "
        f"<b>OFF</b>"
    )


# ============================================================
# SETTINGS
# ============================================================

@dp.callback_query(F.data == "settings")
async def settings(c: CallbackQuery):

    await c.answer()

    await edit_dashboard(
        c.message.chat.id,

        "⚙️ <b>НАСТРОЙКИ</b>\n\n"

        "🪙 <b>Активы:</b>\n"

        + "\n".join(
            "• " + x
            for x in SYMBOLS
        )

        + f"\n\n"

        f"🏦 Бирж: "
        f"<b>{len(VENUE_FUNCTIONS)}</b>\n"

        f"💵 PAPER: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"📈 Порог: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"📚 Уровней: "
        f"<b>{ORDERBOOK_LEVELS}</b>\n"

        f"🔴 LIVE: <b>OFF</b>"
    )


# ============================================================
# AUTO SCANNER
# ============================================================

async def auto_scanner(chat_id):

    global auto_scan_running

    while auto_scan_running:

        try:

            await run_scan()

            good = [
                x
                for x in last_opportunities
                if x["net_percent"] >= MIN_NET_PROFIT
            ]

            traded = False

            if good:
                traded = execute_paper_trade(
                    good[0]
                )

            best = (
                good[0]
                if good
                else None
            )

            text = (
                "▶️ <b>АВТОСКАН</b>\n\n"

                f"🔄 Обновление рынка №"
                f"<b>{market_updates}</b>\n"

                f"📡 Стаканов: "
                f"<b>{len(last_books)}/"
                f"{len(SYMBOLS) * len(VENUE_FUNCTIONS)}</b>\n"

                f"🪙 Активов: "
                f"<b>{len(SYMBOLS)}</b>\n"

                f"🔥 Возможностей ≥ "
                f"{MIN_NET_PROFIT:.2f}%: "
                f"<b>{len(good)}</b>\n\n"

                f"💰 PAPER баланс: "
                f"<b>${paper_balance:.4f}</b>\n"

                f"📈 Прибыль: "
                f"<b>${paper_profit:.4f}</b>\n"

                f"📊 PAPER сделок: "
                f"<b>{paper_trades}</b>\n"
            )

            if traded and best:

                text += (
                    "\n🟢 <b>PAPER СДЕЛКА</b>\n"

                    f"🪙 {best['symbol']}\n"

                    f"🛒 "
                    f"{VENUE_NAMES[best['buy_exchange']]} "
                    f"→ "
                    f"{VENUE_NAMES[best['sell_exchange']]}\n"

                    f"💵 <b>"
                    f"+${best['net_profit']:.4f}"
                    f"</b> "

                    f"({best['net_percent']:.4f}%)\n"
                )

            elif best:

                text += (
                    f"\n🏆 Лучшая: "
                    f"<b>{best['symbol']}</b> "
                    f"{best['net_percent']:.4f}%\n"
                )

            else:

                text += (
                    "\n⏳ Подходящей "
                    "возможности пока нет.\n"
                )

            text += (
                f"\n⏱ Следующее обновление "
                f"через ~{SCAN_INTERVAL} сек.\n"

                "🔴 LIVE OFF"
            )

            # ВАЖНО:
            # редактируем существующее сообщение.
            # Новое сообщение НЕ создаётся.
            await edit_dashboard(
                chat_id,
                text
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

@dp.callback_query(F.data == "auto_start")
async def auto_start(c: CallbackQuery):

    global auto_scan_running
    global auto_scan_task

    await c.answer()

    if auto_scan_running:

        await edit_dashboard(
            c.message.chat.id,

            "▶️ <b>АВТОСКАН УЖЕ РАБОТАЕТ</b>\n\n"

            "Новое сообщение создаваться не будет.\n"

            "Счётчик обновлений будет "
            "меняться в этом сообщении."
        )

        return

    auto_scan_running = True

    await edit_dashboard(
        c.message.chat.id,

        "▶️ <b>АВТОСКАН ЗАПУЩЕН</b>\n\n"

        "Первое обновление рынка..."
    )

    auto_scan_task = asyncio.create_task(
        auto_scanner(
            c.message.chat.id
        )
    )


# ============================================================
# AUTO STOP
# ============================================================

@dp.callback_query(F.data == "auto_stop")
async def auto_stop(c: CallbackQuery):

    global auto_scan_running
    global auto_scan_task

    await c.answer()

    auto_scan_running = False

    if auto_scan_task:

        auto_scan_task.cancel()

        try:
            await auto_scan_task

        except asyncio.CancelledError:
            pass

        auto_scan_task = None

    await edit_dashboard(
        c.message.chat.id,

        f"⏹ <b>АВТОСКАН ОСТАНОВЛЕН</b>\n\n"

        f"🔄 Обновлений рынка: "
        f"<b>{market_updates}</b>\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"📈 Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 Сделок: "
        f"<b>{paper_trades}</b>"
    )


# ============================================================
# FALLBACK
# ============================================================

@dp.message()
async def fallback(message: Message):

    try:
        await message.delete()

    except Exception:
        pass

    await edit_dashboard(
        message.chat.id,

        "🤖 <b>ARBITRAGE BOT</b>\n\n"
        "Выбери действие."
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    global bot
    global http_session
    global auto_scan_running
    global auto_scan_task

    if not BOT_TOKEN:
        raise RuntimeError(
            "BOT_TOKEN не найден в .env"
        )

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        )
    )

    http_session = aiohttp.ClientSession()

    print("=" * 50)
    print("ARBITRAGE BOT 3.0")
    print(
        "Assets:",
        ", ".join(SYMBOLS)
    )
    print(
        "Exchanges:",
        ", ".join(VENUE_NAMES.values())
    )
    print(
        "Threshold:",
        MIN_NET_PROFIT,
        "%"
    )
    print(
        "Paper:",
        PAPER_START_BALANCE
    )
    print(
        "Live trading: OFF"
    )
    print("=" * 50)

    try:

        await dp.start_polling(
            bot
        )

    finally:

        auto_scan_running = False

        if auto_scan_task:

            auto_scan_task.cancel()

            try:
                await auto_scan_task

            except asyncio.CancelledError:
                pass

        if http_session:
            await http_session.close()

        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())