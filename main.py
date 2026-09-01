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

from dotenv import load_dotenv


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

SYMBOL = os.getenv("SYMBOL", "BTC/USDT").strip()

TRADE_SIZE_USDT = float(
    os.getenv("TRADE_SIZE_USDT", "50")
)

MIN_NET_PROFIT = float(
    os.getenv("MIN_NET_PROFIT_PERCENT", "0.30")
)

SCAN_INTERVAL = float(
    os.getenv("SCAN_INTERVAL_SECONDS", "5")
)

ORDERBOOK_LEVELS = int(
    os.getenv("ORDERBOOK_LEVELS", "20")
)

TRANSFER_COST_USDT = float(
    os.getenv("TRANSFER_COST_USDT", "1.00")
)

FEES = {
    "coinex": float(os.getenv("FEE_COINEX_PERCENT", "0.10")),
    "toobit": float(os.getenv("FEE_TOOBIT_PERCENT", "0.10")),
    "weex": float(os.getenv("FEE_WEEX_PERCENT", "0.10")),
    "1bit": float(os.getenv("FEE_1BIT_PERCENT", "0.10")),
    "hyperliquid": float(os.getenv("FEE_HYPERLIQUID_PERCENT", "0.10")),
}


# ВАЖНО:
# Реальная торговля отключена.
# Этот бот сейчас только получает стаканы и считает возможности.
LIVE_TRADING = False


# ============================================================
# GLOBALS
# ============================================================

bot: Optional[Bot] = None
dp = Dispatcher()

http_session: Optional[aiohttp.ClientSession] = None

auto_scan_task: Optional[asyncio.Task] = None
auto_scan_running = False

last_books = {}
last_opportunities = []

paper_balance = 1000.0
paper_profit = 0.0
paper_trades = 0


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class Book:
    venue: str
    symbol: str
    bids: list
    asks: list
    timestamp: float
    derivative: bool = False

    @property
    def best_bid(self):
        if not self.bids:
            return None
        return self.bids[0]

    @property
    def best_ask(self):
        if not self.asks:
            return None
        return self.asks[0]

    @property
    def spread_percent(self):
        if not self.best_bid or not self.best_ask:
            return 0.0

        bid = self.best_bid[0]
        ask = self.best_ask[0]

        if ask <= 0:
            return 0.0

        return (ask - bid) / ask * 100


# ============================================================
# HELPERS
# ============================================================

def normalize_levels(levels, reverse=False):
    """
    Преобразует уровни стакана к:
    [(price, amount), ...]
    """

    result = []

    if not isinstance(levels, list):
        return result

    for item in levels:
        try:
            if isinstance(item, dict):
                price = (
                    item.get("price")
                    or item.get("px")
                    or item.get("p")
                )

                amount = (
                    item.get("quantity")
                    or item.get("qty")
                    or item.get("size")
                    or item.get("sz")
                    or item.get("q")
                )

            elif isinstance(item, (list, tuple)) and len(item) >= 2:
                price = item[0]
                amount = item[1]

            else:
                continue

            price = float(price)
            amount = float(amount)

            if price <= 0 or amount <= 0:
                continue

            result.append((price, amount))

        except Exception:
            continue

    result.sort(
        key=lambda x: x[0],
        reverse=reverse
    )

    return result


async def get_json(
    url,
    method="GET",
    params=None,
    json_data=None,
    timeout=10
):
    """
    Универсальный HTTP запрос.
    """

    if http_session is None:
        raise RuntimeError("HTTP session is not initialized")

    try:
        request_timeout = aiohttp.ClientTimeout(
            total=timeout
        )

        if method.upper() == "POST":
            async with http_session.post(
                url,
                json=json_data,
                params=params,
                timeout=request_timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ArbitrageBot/1.0",
                },
            ) as response:

                text = await response.text()

                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: {text[:300]}"
                    )

                try:
                    return await response.json(
                        content_type=None
                    )
                except Exception:
                    raise RuntimeError(
                        f"Invalid JSON: {text[:300]}"
                    )

        else:
            async with http_session.get(
                url,
                params=params,
                timeout=request_timeout,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ArbitrageBot/1.0",
                },
            ) as response:

                text = await response.text()

                if response.status != 200:
                    raise RuntimeError(
                        f"HTTP {response.status}: {text[:300]}"
                    )

                try:
                    return await response.json(
                        content_type=None
                    )
                except Exception:
                    raise RuntimeError(
                        f"Invalid JSON: {text[:300]}"
                    )

    except asyncio.TimeoutError:
        raise RuntimeError("timeout")

    except aiohttp.ClientError as e:
        raise RuntimeError(
            f"connection error: {e}"
        )


# ============================================================
# SYMBOL HELPERS
# ============================================================

def clean_symbol(symbol):
    """
    BTC/USDT -> BTCUSDT
    """

    return (
        symbol
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .upper()
    )


def split_symbol(symbol):
    """
    BTC/USDT -> BTC, USDT
    """

    symbol = symbol.upper()

    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return base, quote

    if symbol.endswith("USDT"):
        return symbol[:-4], "USDT"

    if symbol.endswith("USDC"):
        return symbol[:-4], "USDC"

    return symbol, "USDT"


BASE_ASSET, QUOTE_ASSET = split_symbol(SYMBOL)


# ============================================================
# COINEX
# ============================================================

async def fetch_coinex():
    """
    CoinEx V2:

    GET /spot/depth

    market=BTCUSDT
    limit=20
    interval=0
    """

    market = clean_symbol(SYMBOL)

    url = "https://api.coinex.com/v2/spot/depth"

    # CoinEx поддерживает 5/10/20/50
    limit = min(
        [5, 10, 20, 50],
        key=lambda x: abs(x - ORDERBOOK_LEVELS)
    )

    data = await get_json(
        url,
        params={
            "market": market,
            "limit": limit,
            "interval": "0",
        },
    )

    if not isinstance(data, dict):
        raise RuntimeError("CoinEx invalid response")

    if data.get("code") not in (0, "0", None):
        raise RuntimeError(
            f"CoinEx: {data.get('message', 'API error')}"
        )

    payload = data.get("data", data)

    depth = payload.get("depth", payload)

    bids = normalize_levels(
        depth.get("bids", []),
        reverse=True
    )

    asks = normalize_levels(
        depth.get("asks", []),
        reverse=False
    )

    if not bids or not asks:
        raise RuntimeError("CoinEx empty order book")

    return Book(
        venue="coinex",
        symbol=SYMBOL,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# TOOBIT
# ============================================================

async def fetch_toobit():
    """
    Toobit current spot API:

    GET /quote/v1/depth

    symbol=BTCUSDT
    limit=20

    Response:
    {
        "t": ...,
        "b": [
            ["price", "quantity"]
        ],
        "a": [
            ["price", "quantity"]
        ]
    }
    """

    market = clean_symbol(SYMBOL)

    url = "https://api.toobit.com/quote/v1/depth"

    limit = min(
        [5, 10, 20, 50, 100, 500, 1000],
        key=lambda x: abs(x - ORDERBOOK_LEVELS)
    )

    data = await get_json(
        url,
        params={
            "symbol": market,
            "limit": limit,
        },
    )

    if not isinstance(data, dict):
        raise RuntimeError("Toobit invalid response")

    bids_raw = (
        data.get("b")
        or data.get("bids")
        or []
    )

    asks_raw = (
        data.get("a")
        or data.get("asks")
        or []
    )

    bids = normalize_levels(
        bids_raw,
        reverse=True
    )

    asks = normalize_levels(
        asks_raw,
        reverse=False
    )

    if not bids or not asks:
        raise RuntimeError(
            "Toobit empty order book"
        )

    return Book(
        venue="toobit",
        symbol=SYMBOL,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# WEEX
# ============================================================

async def fetch_weex():
    """
    WEEX Spot V3:

    GET /api/v3/market/depth

    symbol=BTCUSDT
    limit=15 or 200
    """

    market = clean_symbol(SYMBOL)

    url = "https://api-spot.weex.com/api/v3/market/depth"

    # V3 официально поддерживает 15 и 200
    limit = 200 if ORDERBOOK_LEVELS > 15 else 15

    data = await get_json(
        url,
        params={
            "symbol": market,
            "limit": limit,
        },
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "WEEX invalid response"
        )

    # Иногда API может вернуть code/msg
    if data.get("code") not in (
        None,
        0,
        "0",
        "00000",
    ):
        raise RuntimeError(
            f"WEEX: {data.get('msg', 'API error')}"
        )

    payload = data.get(
        "data",
        data
    )

    bids_raw = payload.get(
        "bids",
        []
    )

    asks_raw = payload.get(
        "asks",
        []
    )

    bids = normalize_levels(
        bids_raw,
        reverse=True
    )

    asks = normalize_levels(
        asks_raw,
        reverse=False
    )

    if not bids or not asks:
        raise RuntimeError(
            "WEEX empty order book"
        )

    return Book(
        venue="weex",
        symbol=SYMBOL,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# 1BIT
# ============================================================

async def fetch_1bit():
    """
    1bit API v2:

    GET /public/markets/{market}/depth

    market:
    btcusdt
    """

    market = (
        clean_symbol(SYMBOL)
        .lower()
    )

    url = (
        f"https://1bit.trade/api/v2/"
        f"public/markets/{market}/depth"
    )

    data = await get_json(
        url,
        params={
            "limit": ORDERBOOK_LEVELS,
        },
    )

    if not isinstance(data, dict):
        raise RuntimeError(
            "1bit invalid response"
        )

    # На случай разных вариантов envelope
    payload = data

    if isinstance(data.get("data"), dict):
        payload = data["data"]

    elif isinstance(data.get("result"), dict):
        payload = data["result"]

    bids_raw = (
        payload.get("bids")
        or payload.get("buy")
        or payload.get("b")
        or []
    )

    asks_raw = (
        payload.get("asks")
        or payload.get("sell")
        or payload.get("a")
        or []
    )

    bids = normalize_levels(
        bids_raw,
        reverse=True
    )

    asks = normalize_levels(
        asks_raw,
        reverse=False
    )

    if not bids or not asks:
        raise RuntimeError(
            f"1bit empty order book: "
            f"{str(data)[:300]}"
        )

    return Book(
        venue="1bit",
        symbol=SYMBOL,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# HYPERLIQUID
# ============================================================

async def hyperliquid_request(payload):
    """
    Hyperliquid:

    POST https://api.hyperliquid.xyz/info
    """

    return await get_json(
        "https://api.hyperliquid.xyz/info",
        method="POST",
        json_data=payload,
    )


async def get_hyperliquid_spot_coin():
    """
    Ищем BTC spot-пару автоматически.

    Hyperliquid использует spotMeta.
    Для spot coin используется имя пары
    или @index.
    """

    meta = await hyperliquid_request({
        "type": "spotMeta"
    })

    if not isinstance(meta, dict):
        raise RuntimeError(
            "Hyperliquid spotMeta invalid"
        )

    universe = meta.get(
        "universe",
        []
    )

    tokens = meta.get(
        "tokens",
        []
    )

    # Сначала ищем обычное имя BTC/USDC
    for pair in universe:
        name = str(
            pair.get("name", "")
        ).upper()

        if name in (
            "BTC/USDC",
            "BTC/USDT",
            "UBTC/USDC",
        ):
            return name

    # Потом ищем по token IDs
    token_names = {}

    for token in tokens:
        try:
            idx = int(
                token.get("index")
            )
            name = str(
                token.get("name", "")
            ).upper()

            token_names[idx] = name

        except Exception:
            continue

    for pair in universe:
        pair_tokens = pair.get(
            "tokens"
        )

        if not isinstance(
            pair_tokens,
            list
        ) or len(pair_tokens) != 2:
            continue

        try:
            base_index = int(
                pair_tokens[0]
            )

            quote_index = int(
                pair_tokens[1]
            )

        except Exception:
            continue

        base_name = token_names.get(
            base_index,
            ""
        )

        quote_name = token_names.get(
            quote_index,
            ""
        )

        if (
            base_name in (
                "BTC",
                "UBTC"
            )
            and quote_name in (
                "USDC",
                "USDT"
            )
        ):
            pair_name = pair.get(
                "name"
            )

            if pair_name:
                return str(pair_name)

    return None


async def fetch_hyperliquid():
    """
    Получаем настоящий L2 order book.

    Сначала пытаемся найти BTC spot.
    Если BTC spot не найден, используем BTC perpetual,
    но помечаем его как derivative.
    """

    spot_coin = None

    try:
        spot_coin = (
            await get_hyperliquid_spot_coin()
        )
    except Exception:
        spot_coin = None

    # --------------------------------------------------------
    # SPOT
    # --------------------------------------------------------

    if spot_coin:
        data = await hyperliquid_request({
            "type": "l2Book",
            "coin": spot_coin,
        })

        if not isinstance(data, dict):
            raise RuntimeError(
                "Hyperliquid invalid L2 response"
            )

        levels = data.get(
            "levels",
            []
        )

        if (
            isinstance(levels, list)
            and len(levels) >= 2
        ):
            bids_raw = levels[0]
            asks_raw = levels[1]

            bids = normalize_hyperliquid_levels(
                bids_raw,
                reverse=True
            )

            asks = normalize_hyperliquid_levels(
                asks_raw,
                reverse=False
            )

            if bids and asks:
                return Book(
                    venue="hyperliquid",
                    symbol=spot_coin,
                    bids=bids,
                    asks=asks,
                    timestamp=time.time(),
                    derivative=False,
                )

    # --------------------------------------------------------
    # FALLBACK: BTC PERPETUAL
    # --------------------------------------------------------

    data = await hyperliquid_request({
        "type": "l2Book",
        "coin": BASE_ASSET,
    })

    if not isinstance(data, dict):
        raise RuntimeError(
            "Hyperliquid invalid perp response"
        )

    levels = data.get(
        "levels",
        []
    )

    if (
        not isinstance(levels, list)
        or len(levels) < 2
    ):
        raise RuntimeError(
            "Hyperliquid empty order book"
        )

    bids = normalize_hyperliquid_levels(
        levels[0],
        reverse=True
    )

    asks = normalize_hyperliquid_levels(
        levels[1],
        reverse=False
    )

    if not bids or not asks:
        raise RuntimeError(
            "Hyperliquid empty order book"
        )

    return Book(
        venue="hyperliquid",
        symbol=f"{BASE_ASSET}-PERP",
        bids=bids,
        asks=asks,
        timestamp=time.time(),
        derivative=True,
    )


def normalize_hyperliquid_levels(
    levels,
    reverse=False
):
    """
    Hyperliquid format:

    {
        "px": "113377.0",
        "sz": "7.6699",
        "n": 17
    }
    """

    result = []

    if not isinstance(
        levels,
        list
    ):
        return result

    for item in levels:
        try:
            price = float(
                item["px"]
            )

            amount = float(
                item["sz"]
            )

            if price <= 0 or amount <= 0:
                continue

            result.append(
                (price, amount)
            )

        except Exception:
            continue

    result.sort(
        key=lambda x: x[0],
        reverse=reverse
    )

    return result[:ORDERBOOK_LEVELS]


# ============================================================
# FETCH ALL BOOKS
# ============================================================

VENUE_FUNCTIONS = {
    "coinex": fetch_coinex,
    "toobit": fetch_toobit,
    "weex": fetch_weex,
    "1bit": fetch_1bit,
    "hyperliquid": fetch_hyperliquid,
}


async def fetch_one_venue(name, func):
    """
    Получает стакан одной биржи.
    Ошибка одной биржи не ломает остальные.
    """

    try:
        book = await func()

        print(
            f"[OK] {name.upper()} "
            f"{book.symbol} "
            f"bid={book.best_bid[0]:.2f} "
            f"ask={book.best_ask[0]:.2f}"
        )

        return name, book, None

    except Exception as e:
        error = str(e)

        print(
            f"[ERROR] {name.upper()}: {error}"
        )

        return name, None, error


async def fetch_all_books():
    """
    Параллельно запрашиваем все биржи.
    """

    tasks = [
        fetch_one_venue(
            name,
            func
        )
        for name, func
        in VENUE_FUNCTIONS.items()
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=False
    )

    books = {}
    errors = {}

    for name, book, error in results:

        if book is not None:
            books[name] = book

        if error is not None:
            errors[name] = error

    return books, errors


# ============================================================
# ORDER BOOK SIMULATION
# ============================================================

def simulate_buy(
    asks,
    usdt_amount
):
    """
    Симуляция покупки на ASK.

    Возвращает:
    BTC amount
    USDT spent
    average price
    slippage
    filled
    """

    if not asks:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        )

    remaining = usdt_amount
    bought = 0.0
    spent = 0.0

    first_price = asks[0][0]

    for price, amount in asks:

        if remaining <= 0:
            break

        level_cost = (
            price * amount
        )

        if level_cost <= remaining:
            take_amount = amount

        else:
            take_amount = (
                remaining / price
            )

        bought += take_amount

        spent += (
            take_amount * price
        )

        remaining -= (
            take_amount * price
        )

    filled = (
        remaining <=
        max(
            0.000001,
            usdt_amount * 0.000001
        )
    )

    if bought <= 0:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        )

    avg_price = (
        spent / bought
    )

    slippage = (
        (avg_price - first_price)
        / first_price
        * 100
    )

    return (
        bought,
        spent,
        avg_price,
        slippage,
        filled,
    )


def simulate_sell(
    bids,
    base_amount
):
    """
    Симуляция продажи по BID.
    """

    if not bids or base_amount <= 0:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        )

    remaining = base_amount
    sold = 0.0
    received = 0.0

    first_price = bids[0][0]

    for price, amount in bids:

        if remaining <= 0:
            break

        take_amount = min(
            remaining,
            amount
        )

        sold += take_amount

        received += (
            take_amount * price
        )

        remaining -= take_amount

    filled = (
        remaining <=
        max(
            0.000000001,
            base_amount * 0.000001
        )
    )

    if sold <= 0:
        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False,
        )

    avg_price = (
        received / sold
    )

    slippage = (
        (first_price - avg_price)
        / first_price
        * 100
    )

    return (
        sold,
        received,
        avg_price,
        slippage,
        filled,
    )


# ============================================================
# ARBITRAGE
# ============================================================

def evaluate_pair(
    buy_book,
    sell_book
):
    """
    Считает арбитраж:

    BUY  -> asks
    SELL -> bids
    """

    # Нельзя сравнивать спот с perpetual.
    if (
        buy_book.derivative
        or sell_book.derivative
    ):
        return None

    (
        bought,
        buy_quote,
        buy_avg,
        buy_slippage,
        buy_filled,
    ) = simulate_buy(
        buy_book.asks,
        TRADE_SIZE_USDT,
    )

    if not buy_filled:
        return None

    (
        sold,
        sell_quote,
        sell_avg,
        sell_slippage,
        sell_filled,
    ) = simulate_sell(
        sell_book.bids,
        bought,
    )

    if not sell_filled:
        return None

    buy_fee_percent = FEES.get(
        buy_book.venue,
        0.10
    )

    sell_fee_percent = FEES.get(
        sell_book.venue,
        0.10
    )

    buy_fee = (
        buy_quote
        * buy_fee_percent
        / 100
    )

    sell_fee = (
        sell_quote
        * sell_fee_percent
        / 100
    )

    gross_profit = (
        sell_quote
        - buy_quote
    )

    trading_fees = (
        buy_fee
        + sell_fee
    )

    net_profit = (
        gross_profit
        - trading_fees
        - TRANSFER_COST_USDT
    )

    net_percent = (
        net_profit
        / buy_quote
        * 100
    )

    return {
        "buy_exchange": buy_book.venue,
        "sell_exchange": sell_book.venue,

        "buy_price": buy_avg,
        "sell_price": sell_avg,

        "first_buy_price":
            buy_book.best_ask[0],

        "first_sell_price":
            sell_book.best_bid[0],

        "amount_btc": bought,

        "buy_quote": buy_quote,
        "sell_quote": sell_quote,

        "gross_profit": gross_profit,
        "fees": trading_fees,

        "transfer_cost":
            TRANSFER_COST_USDT,

        "net_profit":
            net_profit,

        "net_percent":
            net_percent,

        "buy_slippage":
            buy_slippage,

        "sell_slippage":
            sell_slippage,
    }


def find_opportunities(
    books
):
    opportunities = []

    names = list(
        books.keys()
    )

    for buy_name in names:

        for sell_name in names:

            if buy_name == sell_name:
                continue

            result = evaluate_pair(
                books[buy_name],
                books[sell_name],
            )

            if result is None:
                continue

            opportunities.append(
                result
            )

    opportunities.sort(
        key=lambda x: x["net_percent"],
        reverse=True
    )

    return opportunities


# ============================================================
# TELEGRAM KEYBOARD
# ============================================================

def main_keyboard():
    builder = InlineKeyboardBuilder()

    builder.button(
        text="🔎 Сканировать",
        callback_data="scan",
    )

    builder.button(
        text="📊 Статус",
        callback_data="status",
    )

    builder.button(
        text="▶️ Автоскан",
        callback_data="auto_start",
    )

    builder.button(
        text="⏹ Остановить",
        callback_data="auto_stop",
    )

    builder.button(
        text="💰 PAPER баланс",
        callback_data="paper",
    )

    builder.button(
        text="📈 Статистика",
        callback_data="stats",
    )

    builder.button(
        text="⚙️ Настройки",
        callback_data="settings",
    )

    builder.adjust(2)

    return builder.as_markup()


def bottom_keyboard():
    builder = InlineKeyboardBuilder()

    builder.button(
        text="🔎 Сканировать",
        callback_data="scan",
    )

    builder.button(
        text="📚 Стаканы",
        callback_data="books",
    )

    builder.button(
        text="🔥 Возможности",
        callback_data="opps",
    )

    builder.button(
        text="▶️ Автоскан",
        callback_data="auto_start",
    )

    builder.button(
        text="⏹ Стоп",
        callback_data="auto_stop",
    )

    builder.button(
        text="💰 PAPER",
        callback_data="paper",
    )

    builder.button(
        text="📈 Статистика",
        callback_data="stats",
    )

    builder.button(
        text="⚙️ Настройки",
        callback_data="settings",
    )

    builder.adjust(2)

    return builder.as_markup()


# ============================================================
# FORMAT BOOKS
# ============================================================

VENUE_NAMES = {
    "coinex": "COINEX",
    "toobit": "TOOBIT",
    "weex": "WEEX",
    "1bit": "1BIT",
    "hyperliquid": "HYPERLIQUID",
}


def format_price(value):
    if value is None:
        return "—"

    return f"${value:,.2f}"


def format_books(
    books,
    errors=None
):
    errors = errors or {}

    text = (
        "📚 <b>СТАКАНЫ</b>\n\n"
    )

    text += (
        f"💱 <b>{SYMBOL}</b>\n\n"
    )

    for name in VENUE_FUNCTIONS.keys():

        title = VENUE_NAMES.get(
            name,
            name.upper()
        )

        if name not in books:

            error = errors.get(
                name,
                "нет данных"
            )

            # Чтобы Telegram не превращал
            # техническую ошибку в огромный текст.
            if len(error) > 100:
                error = error[:100] + "..."

            text += (
                f"🔴 <b>{title}</b>"
                f" — нет данных\n"
            )

            text += (
                f"   <i>{error}</i>\n\n"
            )

            continue

        book = books[name]

        bid = book.best_bid[0]
        ask = book.best_ask[0]

        bid_qty = book.best_bid[1]
        ask_qty = book.best_ask[1]

        derivative_text = ""

        if book.derivative:
            derivative_text = (
                "\n⚠️ <i>PERPETUAL</i>"
            )

        text += (
            f"🟢 <b>{title}</b>"
            f"{derivative_text}\n"
        )

        text += (
            f"ASK: "
            f"<b>{format_price(ask)}</b>"
            f" × {ask_qty:.6f}\n"
        )

        text += (
            f"BID: "
            f"<b>{format_price(bid)}</b>"
            f" × {bid_qty:.6f}\n"
        )

        text += (
            f"Spread: "
            f"{book.spread_percent:.4f}%\n"
        )

        # Суммарная ликвидность первых N уровней
        ask_liquidity = sum(
            price * amount
            for price, amount
            in book.asks[:ORDERBOOK_LEVELS]
        )

        bid_liquidity = sum(
            price * amount
            for price, amount
            in book.bids[:ORDERBOOK_LEVELS]
        )

        text += (
            f"📥 Продажи: "
            f"${ask_liquidity:,.2f}\n"
        )

        text += (
            f"📤 Покупки: "
            f"${bid_liquidity:,.2f}\n\n"
        )

    text += (
        f"🕐 Обновлено: "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )

    return text


# ============================================================
# FORMAT OPPORTUNITIES
# ============================================================

def format_opportunities(
    opportunities
):
    if not opportunities:
        return (
            "🔥 <b>ВОЗМОЖНОСТИ</b>\n\n"
            "Пока прибыльных связок нет."
        )

    text = (
        "🔥 <b>АРБИТРАЖНЫЕ ВОЗМОЖНОСТИ</b>\n\n"
    )

    for i, opp in enumerate(
        opportunities[:10],
        1
    ):

        buy_name = VENUE_NAMES.get(
            opp["buy_exchange"],
            opp["buy_exchange"].upper()
        )

        sell_name = VENUE_NAMES.get(
            opp["sell_exchange"],
            opp["sell_exchange"].upper()
        )

        emoji = (
            "🟢"
            if opp["net_percent"] >= MIN_NET_PROFIT
            else "🟡"
        )

        text += (
            f"{emoji} <b>#{i}</b>\n"
        )

        text += (
            f"🛒 Купить: "
            f"<b>{buy_name}</b>\n"
        )

        text += (
            f"   ${opp['buy_price']:,.2f}\n"
        )

        text += (
            f"💰 Продать: "
            f"<b>{sell_name}</b>\n"
        )

        text += (
            f"   ${opp['sell_price']:,.2f}\n"
        )

        text += (
            f"📦 Объём: "
            f"{opp['amount_btc']:.8f} BTC\n"
        )

        text += (
            f"📊 До комиссий: "
            f"${opp['gross_profit']:.4f}\n"
        )

        text += (
            f"💳 Комиссии: "
            f"-${opp['fees']:.4f}\n"
        )

        text += (
            f"⛽ Перевод: "
            f"-${opp['transfer_cost']:.2f}\n"
        )

        text += (
            f"💵 <b>Чистая прибыль: "
            f"${opp['net_profit']:.4f}</b>\n"
        )

        text += (
            f"📈 <b>{opp['net_percent']:.4f}%</b>\n"
        )

        text += (
            f"📉 Slippage: "
            f"{opp['buy_slippage']:.4f}% / "
            f"{opp['sell_slippage']:.4f}%\n\n"
        )

    return text


# ============================================================
# START
# ============================================================

@dp.message(Command("start"))
async def cmd_start(
    message: Message
):
    await message.answer(
        "🤖 <b>Arbitrage Bot</b>\n\n"
        f"💱 Пара: <b>{SYMBOL}</b>\n"
        f"💵 Размер сделки: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"
        f"📈 Минимальная прибыль: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"
        "Выбери действие:",
        reply_markup=main_keyboard(),
    )


# ============================================================
# SCAN
# ============================================================

@dp.callback_query(
    F.data == "scan"
)
async def callback_scan(
    callback: CallbackQuery
):
    await callback.answer(
        "Получаю стаканы..."
    )

    try:
        books, errors = (
            await fetch_all_books()
        )

        global last_books
        global last_opportunities

        last_books = books

        last_opportunities = (
            find_opportunities(
                books
            )
        )

        await callback.message.answer(
            format_opportunities(
                last_opportunities
            ),
            reply_markup=bottom_keyboard(),
        )

    except Exception as e:

        await callback.message.answer(
            f"❌ Ошибка сканирования:\n"
            f"<code>{str(e)}</code>",
            reply_markup=bottom_keyboard(),
        )


# ============================================================
# BOOKS
# ============================================================

@dp.callback_query(
    F.data == "books"
)
async def callback_books(
    callback: CallbackQuery
):
    await callback.answer(
        "Получаю стаканы..."
    )

    try:
        books, errors = (
            await fetch_all_books()
        )

        global last_books
        last_books = books

        await callback.message.answer(
            format_books(
                books,
                errors
            ),
            reply_markup=bottom_keyboard(),
        )

    except Exception as e:

        await callback.message.answer(
            f"❌ Ошибка:\n"
            f"<code>{str(e)}</code>",
            reply_markup=bottom_keyboard(),
        )


# ============================================================
# OPPORTUNITIES
# ============================================================

@dp.callback_query(
    F.data == "opps"
)
async def callback_opps(
    callback: CallbackQuery
):
    await callback.answer()

    if not last_opportunities:
        await callback.message.answer(
            "🔥 <b>ВОЗМОЖНОСТИ</b>\n\n"
            "Сначала нажми «Сканировать».",
            reply_markup=bottom_keyboard(),
        )
        return

    await callback.message.answer(
        format_opportunities(
            last_opportunities
        ),
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# STATUS
# ============================================================

@dp.callback_query(
    F.data == "status"
)
async def callback_status(
    callback: CallbackQuery
):
    await callback.answer()

    if not last_books:
        text = (
            "📊 <b>СТАТУС</b>\n\n"
            "Стаканы ещё не загружались."
        )

    else:

        online = len(
            last_books
        )

        total = len(
            VENUE_FUNCTIONS
        )

        text = (
            "📊 <b>СТАТУС</b>\n\n"
            f"🟢 Бирж онлайн: "
            f"<b>{online}/{total}</b>\n"
            f"💱 Пара: "
            f"<b>{SYMBOL}</b>\n"
            f"💵 Размер: "
            f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"
            f"⏱ Интервал: "
            f"<b>{SCAN_INTERVAL} сек.</b>\n"
        )

    await callback.message.answer(
        text,
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# AUTO SCAN
# ============================================================

async def auto_scanner(
    chat_id: int
):
    global auto_scan_running
    global last_books
    global last_opportunities

    while auto_scan_running:

        try:
            books, errors = (
                await fetch_all_books()
            )

            last_books = books

            opportunities = (
                find_opportunities(
                    books
                )
            )

            last_opportunities = (
                opportunities
            )

            good = [
                x
                for x in opportunities
                if x["net_percent"]
                >= MIN_NET_PROFIT
            ]

            if good:

                best = good[0]

                buy_name = VENUE_NAMES.get(
                    best["buy_exchange"],
                    best["buy_exchange"].upper()
                )

                sell_name = VENUE_NAMES.get(
                    best["sell_exchange"],
                    best["sell_exchange"].upper()
                )

                text = (
                    "🚨 <b>АРБИТРАЖ!</b>\n\n"

                    f"🛒 Купить: "
                    f"<b>{buy_name}</b>\n"
                    f"${best['buy_price']:,.2f}\n\n"

                    f"💰 Продать: "
                    f"<b>{sell_name}</b>\n"
                    f"${best['sell_price']:,.2f}\n\n"

                    f"📦 Объём: "
                    f"{best['amount_btc']:.8f} BTC\n"

                    f"💵 Чистая прибыль: "
                    f"<b>${best['net_profit']:.4f}</b>\n"

                    f"📈 Доходность: "
                    f"<b>{best['net_percent']:.4f}%</b>\n\n"

                    f"📉 Slippage: "
                    f"{best['buy_slippage']:.4f}% / "
                    f"{best['sell_slippage']:.4f}%\n\n"

                    "⚠️ PAPER MODE — реальные "
                    "ордера не отправляются."
                )

                await bot.send_message(
                    chat_id,
                    text,
                    reply_markup=bottom_keyboard(),
                )

        except Exception as e:

            print(
                f"[AUTO ERROR] {e}"
            )

        await asyncio.sleep(
            SCAN_INTERVAL
        )


@dp.callback_query(
    F.data == "auto_start"
)
async def callback_auto_start(
    callback: CallbackQuery
):
    global auto_scan_task
    global auto_scan_running

    await callback.answer()

    if auto_scan_running:

        await callback.message.answer(
            "▶️ Автоскан уже запущен.",
            reply_markup=bottom_keyboard(),
        )

        return

    auto_scan_running = True

    auto_scan_task = asyncio.create_task(
        auto_scanner(
            callback.from_user.id
        )
    )

    await callback.message.answer(
        "▶️ <b>Автоскан запущен</b>\n\n"
        f"Проверка каждые "
        f"<b>{SCAN_INTERVAL}</b> сек.\n\n"
        f"Сигнал от "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>.",
        reply_markup=bottom_keyboard(),
    )


@dp.callback_query(
    F.data == "auto_stop"
)
async def callback_auto_stop(
    callback: CallbackQuery
):
    global auto_scan_running
    global auto_scan_task

    await callback.answer()

    auto_scan_running = False

    if auto_scan_task:

        auto_scan_task.cancel()

        try:
            await auto_scan_task
        except asyncio.CancelledError:
            pass

        auto_scan_task = None

    await callback.message.answer(
        "⏹ <b>Автоскан остановлен.</b>",
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# PAPER
# ============================================================

@dp.callback_query(
    F.data == "paper"
)
async def callback_paper(
    callback: CallbackQuery
):
    await callback.answer()

    await callback.message.answer(
        "💰 <b>PAPER MODE</b>\n\n"
        f"Баланс: "
        f"<b>${paper_balance:.2f}</b>\n"
        f"Прибыль: "
        f"<b>${paper_profit:.2f}</b>\n"
        f"Сделок: "
        f"<b>{paper_trades}</b>\n\n"
        "Реальные деньги не используются.",
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# STATISTICS
# ============================================================

@dp.callback_query(
    F.data == "stats"
)
async def callback_stats(
    callback: CallbackQuery
):
    await callback.answer()

    best_text = "нет"

    if last_opportunities:

        best = (
            last_opportunities[0]
        )

        best_text = (
            f"{best['net_percent']:.4f}%"
        )

    await callback.message.answer(
        "📈 <b>СТАТИСТИКА</b>\n\n"
        f"Сканов: данные текущей сессии\n"
        f"Бирж получено: "
        f"<b>{len(last_books)}/"
        f"{len(VENUE_FUNCTIONS)}</b>\n"
        f"Возможностей: "
        f"<b>{len(last_opportunities)}</b>\n"
        f"Лучшая: "
        f"<b>{best_text}</b>\n"
        f"PAPER сделок: "
        f"<b>{paper_trades}</b>",
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# SETTINGS
# ============================================================

@dp.callback_query(
    F.data == "settings"
)
async def callback_settings(
    callback: CallbackQuery
):
    await callback.answer()

    fees_text = "\n".join(
        f"• {VENUE_NAMES[k]}: "
        f"{v:.3f}%"
        for k, v in FEES.items()
    )

    await callback.message.answer(
        "⚙️ <b>НАСТРОЙКИ</b>\n\n"

        f"💱 Пара: "
        f"<code>{SYMBOL}</code>\n"

        f"💵 Размер сделки: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"📚 Уровней стакана: "
        f"<b>{ORDERBOOK_LEVELS}</b>\n"

        f"📈 Минимальная прибыль: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"⛽ Резерв перевода: "
        f"<b>${TRANSFER_COST_USDT:.2f}</b>\n\n"

        "💳 <b>Комиссии:</b>\n"
        f"{fees_text}\n\n"

        f"🔐 LIVE TRADING: "
        f"<b>{'ON' if LIVE_TRADING else 'OFF'}</b>\n\n"

        "⚠️ Сейчас бот работает только "
        "с публичными стаканами и PAPER-режимом.",
        reply_markup=bottom_keyboard(),
    )


# ============================================================
# TEXT FALLBACK
# ============================================================

@dp.message()
async def fallback(
    message: Message
):
    await message.answer(
        "Выбери действие:",
        reply_markup=main_keyboard(),
    )


# ============================================================
# MAIN
# ============================================================

async def main():

    global bot
    global http_session

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN не найден в .env"
        )

    bot = Bot(
        token=BOT_TOKEN
    )

    http_session = aiohttp.ClientSession()

    print(
        "===================================="
    )

    print(
        "      ARBITRAGE BOT STARTED"
    )

    print(
        "===================================="
    )

    print(
        f"Symbol: {SYMBOL}"
    )

    print(
        f"Trade size: ${TRADE_SIZE_USDT}"
    )

    print(
        f"Orderbook levels: {ORDERBOOK_LEVELS}"
    )

    print(
        f"Live trading: {LIVE_TRADING}"
    )

    print(
        "===================================="
    )

    try:

        await dp.start_polling(
            bot
        )

    finally:

        global auto_scan_running
        global auto_scan_task

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


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:
        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\nBot stopped."
        )