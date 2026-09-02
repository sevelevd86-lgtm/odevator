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


# ============================================================
# CONFIG
# ============================================================

load_dotenv()

BOT_TOKEN = os.getenv(
    "BOT_TOKEN",
    ""
).strip()


# ============================================================
# КРИПТОВАЛЮТЫ
# ============================================================

SYMBOLS_RAW = os.getenv(
    "SYMBOLS",
    "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,TON/USDT"
)

SYMBOLS = [
    x.strip().upper()
    for x in SYMBOLS_RAW.split(",")
    if x.strip()
]


# ============================================================
# PAPER SETTINGS
# ============================================================

TRADE_SIZE_USDT = float(
    os.getenv(
        "TRADE_SIZE_USDT",
        "50"
    )
)


PAPER_START_BALANCE = float(
    os.getenv(
        "PAPER_START_BALANCE",
        "1000"
    )
)


# Минимальная чистая прибыль
MIN_NET_PROFIT = float(
    os.getenv(
        "MIN_NET_PROFIT_PERCENT",
        "0.20"
    )
)


# ============================================================
# SCANNER
# ============================================================

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


# ============================================================
# FEES
# ============================================================

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
# LIVE TRADING
# ============================================================

# Реальные ордера полностью отключены.
LIVE_TRADING = False


# ============================================================
# PAPER STATE
# ============================================================

paper_balance = PAPER_START_BALANCE

paper_profit = 0.0

paper_trades = 0

paper_wins = 0

paper_losses = 0

paper_volume = 0.0


# Последние PAPER-сделки
paper_history = []


# Защита от повторения одной
# и той же возможности.
PAPER_TRADE_COOLDOWN = 30


last_paper_trades = {}


# ============================================================
# GLOBALS
# ============================================================

bot: Optional[Bot] = None

dp = Dispatcher()

http_session: Optional[
    aiohttp.ClientSession
] = None


auto_scan_task: Optional[
    asyncio.Task
] = None


auto_scan_running = False


# ============================================================
# ОДНО ГЛАВНОЕ СООБЩЕНИЕ
# ============================================================

dashboard_messages = {}


# ============================================================
# LAST DATA
# ============================================================

last_books = {}

last_errors = {}

last_opportunities = []

last_scan_time = 0


# ============================================================
# HYPERLIQUID CACHE
# ============================================================

hyperliquid_spot_cache = {}

hyperliquid_cache_time = 0


# ============================================================
# BOOK
# ============================================================

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

        if (
            not self.best_bid
            or not self.best_ask
        ):
            return 0.0

        bid = self.best_bid[0]

        ask = self.best_ask[0]

        if ask <= 0:
            return 0.0

        return (
            (ask - bid)
            / ask
            * 100
        )


# ============================================================
# SYMBOL HELPERS
# ============================================================

def clean_symbol(symbol):

    return (
        symbol
        .replace("/", "")
        .replace("-", "")
        .replace("_", "")
        .upper()
    )


def split_symbol(symbol):

    symbol = symbol.upper()

    if "/" in symbol:

        return tuple(
            symbol.split(
                "/",
                1
            )
        )

    if symbol.endswith("USDT"):

        return (
            symbol[:-4],
            "USDT"
        )

    if symbol.endswith("USDC"):

        return (
            symbol[:-4],
            "USDC"
        )

    return (
        symbol,
        "USDT"
    )


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

        request_timeout = (
            aiohttp.ClientTimeout(
                total=timeout
            )
        )

        headers = {
            "Accept":
                "application/json",

            "User-Agent":
                "ArbitrageBot/2.0",
        }


        if method.upper() == "POST":

            async with http_session.post(
                url,
                json=json_data,
                params=params,
                timeout=request_timeout,
                headers=headers,
            ) as response:

                text = await response.text()

                if response.status != 200:

                    raise RuntimeError(
                        f"HTTP {response.status}: "
                        f"{text[:250]}"
                    )

                try:

                    return await response.json(
                        content_type=None
                    )

                except Exception:

                    raise RuntimeError(
                        "Invalid JSON: "
                        + text[:250]
                    )


        async with http_session.get(
            url,
            params=params,
            timeout=request_timeout,
            headers=headers,
        ) as response:

            text = await response.text()

            if response.status != 200:

                raise RuntimeError(
                    f"HTTP {response.status}: "
                    f"{text[:250]}"
                )

            try:

                return await response.json(
                    content_type=None
                )

            except Exception:

                raise RuntimeError(
                    "Invalid JSON: "
                    + text[:250]
                )

    except asyncio.TimeoutError:

        raise RuntimeError(
            "timeout"
        )

    except aiohttp.ClientError as e:

        raise RuntimeError(
            f"connection error: {e}"
        )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_levels(
    levels,
    reverse=False
):

    result = []

    if not isinstance(
        levels,
        list
    ):

        return result


    for item in levels:

        try:

            if isinstance(
                item,
                dict
            ):

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

            elif (
                isinstance(
                    item,
                    (list, tuple)
                )
                and len(item) >= 2
            ):

                price = item[0]

                amount = item[1]

            else:

                continue


            price = float(price)

            amount = float(amount)


            if (
                price <= 0
                or amount <= 0
            ):

                continue


            result.append(
                (
                    price,
                    amount
                )
            )

        except Exception:

            continue


    result.sort(
        key=lambda x: x[0],
        reverse=reverse
    )

    return result


# ============================================================
# COINEX
# ============================================================

async def fetch_coinex(
    symbol
):

    base, quote = split_symbol(
        symbol
    )

    if quote != "USDT":

        raise RuntimeError(
            "CoinEx scanner supports USDT pairs only"
        )


    market = clean_symbol(
        symbol
    )


    url = (
        "https://api.coinex.com/"
        "v2/spot/depth"
    )


    limit = min(
        [5, 10, 20, 50],
        key=lambda x:
        abs(
            x - ORDERBOOK_LEVELS
        )
    )


    data = await get_json(
        url,
        params={
            "market": market,
            "limit": limit,
            "interval": "0",
        },
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "invalid response"
        )


    if data.get("code") not in (
        0,
        "0",
        None,
    ):

        raise RuntimeError(
            str(
                data.get(
                    "message",
                    "API error"
                )
            )
        )


    payload = data.get(
        "data",
        data
    )


    depth = payload.get(
        "depth",
        payload
    )


    bids = normalize_levels(
        depth.get(
            "bids",
            []
        ),
        reverse=True
    )


    asks = normalize_levels(
        depth.get(
            "asks",
            []
        ),
        reverse=False
    )


    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "empty order book"
        )


    return Book(
        venue="coinex",
        requested_symbol=symbol,
        actual_symbol=symbol,
        base=base,
        quote=quote,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# TOOBIT
# ============================================================

async def fetch_toobit(
    symbol
):

    base, quote = split_symbol(
        symbol
    )

    if quote != "USDT":

        raise RuntimeError(
            "Toobit scanner supports USDT pairs only"
        )


    market = clean_symbol(
        symbol
    )


    # Актуальный merged depth
    url = (
        "https://api.toobit.com/"
        "quote/v1/depth/merged"
    )


    limit = min(
        [
            5,
            10,
            20,
            50,
            100,
            500,
            1000
        ],
        key=lambda x:
        abs(
            x - ORDERBOOK_LEVELS
        )
    )


    data = await get_json(
        url,
        params={
            "symbol": market,
            "scale": 0,
            "limit": limit,
        },
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "invalid response"
        )


    bids = normalize_levels(
        data.get(
            "b",
            []
        ),
        reverse=True
    )


    asks = normalize_levels(
        data.get(
            "a",
            []
        ),
        reverse=False
    )


    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "empty order book"
        )


    return Book(
        venue="toobit",
        requested_symbol=symbol,
        actual_symbol=symbol,
        base=base,
        quote=quote,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# WEEX
# ============================================================

async def fetch_weex(
    symbol
):

    base, quote = split_symbol(
        symbol
    )

    if quote != "USDT":

        raise RuntimeError(
            "WEEX scanner supports USDT pairs only"
        )


    market = clean_symbol(
        symbol
    )


    url = (
        "https://api-spot.weex.com/"
        "api/v3/market/depth"
    )


    limit = (
        200
        if ORDERBOOK_LEVELS > 15
        else 15
    )


    data = await get_json(
        url,
        params={
            "symbol": market,
            "limit": limit,
        },
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "invalid response"
        )


    if data.get("code") not in (
        None,
        0,
        "0",
        "00000",
    ):

        raise RuntimeError(
            str(
                data.get(
                    "msg",
                    "API error"
                )
            )
        )


    payload = data.get(
        "data",
        data
    )


    bids = normalize_levels(
        payload.get(
            "bids",
            []
        ),
        reverse=True
    )


    asks = normalize_levels(
        payload.get(
            "asks",
            []
        ),
        reverse=False
    )


    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "empty order book"
        )


    return Book(
        venue="weex",
        requested_symbol=symbol,
        actual_symbol=symbol,
        base=base,
        quote=quote,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# 1BIT
# ============================================================

async def fetch_1bit(
    symbol
):

    base, quote = split_symbol(
        symbol
    )

    if quote != "USDT":

        raise RuntimeError(
            "1bit scanner supports USDT pairs only"
        )


    market = (
        clean_symbol(
            symbol
        ).lower()
    )


    url = (
        "https://1bit.trade/"
        "api/v2/public/markets/"
        f"{market}/depth"
    )


    data = await get_json(
        url,
        params={
            "limit":
                ORDERBOOK_LEVELS
        },
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "invalid response"
        )


    payload = data


    if isinstance(
        data.get("data"),
        dict
    ):

        payload = data["data"]


    elif isinstance(
        data.get("result"),
        dict
    ):

        payload = data["result"]


    bids = normalize_levels(
        (
            payload.get("bids")
            or payload.get("buy")
            or payload.get("b")
            or []
        ),
        reverse=True
    )


    asks = normalize_levels(
        (
            payload.get("asks")
            or payload.get("sell")
            or payload.get("a")
            or []
        ),
        reverse=False
    )


    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "empty order book"
        )


    return Book(
        venue="1bit",
        requested_symbol=symbol,
        actual_symbol=symbol,
        base=base,
        quote=quote,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
    )


# ============================================================
# HYPERLIQUID
# ============================================================

async def hyperliquid_request(
    payload
):

    return await get_json(
        "https://api.hyperliquid.xyz/info",
        method="POST",
        json_data=payload,
    )


def normalize_hyperliquid_levels(
    levels,
    reverse=False
):

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


            if (
                price <= 0
                or amount <= 0
            ):

                continue


            result.append(
                (
                    price,
                    amount
                )
            )

        except Exception:

            continue


    result.sort(
        key=lambda x: x[0],
        reverse=reverse
    )


    return result[
        :ORDERBOOK_LEVELS
    ]


async def load_hyperliquid_spot_meta():

    global hyperliquid_spot_cache
    global hyperliquid_cache_time


    # Кэшируем metadata на 10 минут
    if (
        hyperliquid_spot_cache
        and
        time.time()
        - hyperliquid_cache_time
        < 600
    ):

        return hyperliquid_spot_cache


    data = await hyperliquid_request(
        {
            "type": "spotMeta"
        }
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "spotMeta invalid"
        )


    universe = data.get(
        "universe",
        []
    )


    tokens = data.get(
        "tokens",
        []
    )


    token_names = {}


    for token in tokens:

        try:

            token_names[
                int(
                    token.get(
                        "index"
                    )
                )
            ] = str(
                token.get(
                    "name",
                    ""
                )
            ).upper()

        except Exception:

            continue


    result = {}


    for pair in universe:

        name = str(
            pair.get(
                "name",
                ""
            )
        ).upper()


        pair_tokens = pair.get(
            "tokens"
        )


        if (
            not isinstance(
                pair_tokens,
                list
            )
            or len(pair_tokens) != 2
        ):

            continue


        try:

            base_name = token_names.get(
                int(pair_tokens[0]),
                ""
            )

            quote_name = token_names.get(
                int(pair_tokens[1]),
                ""
            )

        except Exception:

            continue


        if (
            not base_name
            or not quote_name
        ):

            continue


        result[
            (
                base_name,
                quote_name
            )
        ] = name


    hyperliquid_spot_cache = result

    hyperliquid_cache_time = time.time()


    return result


async def fetch_hyperliquid(
    symbol
):

    requested_base, requested_quote = (
        split_symbol(symbol)
    )


    meta = await (
        load_hyperliquid_spot_meta()
    )


    # Ищем точную пару
    # BTC/USDT, ETH/USDT и т.д.
    actual_symbol = meta.get(
        (
            requested_base,
            requested_quote
        )
    )


    # Hyperliquid часто использует USDC.
    # В таком случае разрешаем показать стакан,
    # но НЕ используем его в USDT-арбитраже.
    if actual_symbol is None:

        actual_symbol = meta.get(
            (
                requested_base,
                "USDC"
            )
        )


    if actual_symbol is None:

        raise RuntimeError(
            f"{symbol} not available"
        )


    if (
        "/" in actual_symbol
    ):

        actual_base, actual_quote = (
            split_symbol(
                actual_symbol
            )
        )

    else:

        actual_base = requested_base

        actual_quote = requested_quote


    data = await hyperliquid_request(
        {
            "type": "l2Book",
            "coin": actual_symbol,
        }
    )


    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "invalid l2Book response"
        )


    levels = data.get(
        "levels",
        []
    )


    if (
        not isinstance(
            levels,
            list
        )
        or len(levels) < 2
    ):

        raise RuntimeError(
            "empty order book"
        )


    bids = normalize_hyperliquid_levels(
        levels[0],
        reverse=True
    )


    asks = normalize_hyperliquid_levels(
        levels[1],
        reverse=False
    )


    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "empty order book"
        )


    return Book(
        venue="hyperliquid",
        requested_symbol=symbol,
        actual_symbol=actual_symbol,
        base=actual_base,
        quote=actual_quote,
        bids=bids,
        asks=asks,
        timestamp=time.time(),
        derivative=False,
    )


# ============================================================
# EXCHANGE FUNCTIONS
# ============================================================

VENUE_FUNCTIONS = {

    "coinex":
        fetch_coinex,

    "toobit":
        fetch_toobit,

    "weex":
        fetch_weex,

    "1bit":
        fetch_1bit,

    "hyperliquid":
        fetch_hyperliquid,
}


VENUE_NAMES = {

    "coinex":
        "COINEX",

    "toobit":
        "TOOBIT",

    "weex":
        "WEEX",

    "1bit":
        "1BIT",

    "hyperliquid":
        "HYPERLIQUID",
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

        book = await func(
            symbol
        )

        return (
            venue,
            symbol,
            book,
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
# FETCH ALL SYMBOLS
# ============================================================

async def fetch_all_books():

    tasks = []


    for venue, func in (
        VENUE_FUNCTIONS.items()
    ):

        for symbol in SYMBOLS:

            tasks.append(
                fetch_one(
                    venue,
                    symbol,
                    func
                )
            )


    results = await asyncio.gather(
        *tasks,
        return_exceptions=False
    )


    books = {}

    errors = {}


    for (
        venue,
        symbol,
        book,
        error
    ) in results:

        if book is not None:

            books[
                (
                    symbol,
                    venue
                )
            ] = book


        if error is not None:

            errors[
                (
                    symbol,
                    venue
                )
            ] = error


    return (
        books,
        errors
    )


# ============================================================
# SIMULATE BUY
# ============================================================

def simulate_buy(
    asks,
    usdt_amount
):

    if not asks:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False
        )


    remaining = usdt_amount

    bought = 0.0

    spent = 0.0

    first_price = asks[0][0]


    for (
        price,
        amount
    ) in asks:

        if remaining <= 0:

            break


        level_cost = (
            price
            * amount
        )


        take_amount = (
            amount
            if level_cost <= remaining
            else remaining / price
        )


        bought += take_amount

        spent += (
            take_amount
            * price
        )


        remaining -= (
            take_amount
            * price
        )


    filled = (
        remaining
        <=
        max(
            0.000001,
            usdt_amount
            * 0.000001
        )
    )


    if bought <= 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False
        )


    avg_price = (
        spent / bought
    )


    slippage = (
        (
            avg_price
            - first_price
        )
        / first_price
        * 100
    )


    return (
        bought,
        spent,
        avg_price,
        slippage,
        filled
    )


# ============================================================
# SIMULATE SELL
# ============================================================

def simulate_sell(
    bids,
    base_amount
):

    if (
        not bids
        or base_amount <= 0
    ):

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False
        )


    remaining = base_amount

    sold = 0.0

    received = 0.0

    first_price = bids[0][0]


    for (
        price,
        amount
    ) in bids:

        if remaining <= 0:

            break


        take_amount = min(
            remaining,
            amount
        )


        sold += take_amount

        received += (
            take_amount
            * price
        )

        remaining -= take_amount


    filled = (
        remaining
        <=
        max(
            0.000000001,
            base_amount
            * 0.000001
        )
    )


    if sold <= 0:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            False
        )


    avg_price = (
        received / sold
    )


    slippage = (
        (
            first_price
            - avg_price
        )
        / first_price
        * 100
    )


    return (
        sold,
        received,
        avg_price,
        slippage,
        filled
    )


# ============================================================
# EVALUATE
# ============================================================

def evaluate_pair(
    symbol,
    buy_book,
    sell_book
):

    # Нельзя сравнивать разные котировки
    if (
        buy_book.quote
        != sell_book.quote
    ):

        return None


    # Наш основной PAPER рынок USDT
    if buy_book.quote != "USDT":

        return None


    if (
        buy_book.base
        != sell_book.base
    ):

        return None


    (
        bought,
        buy_quote,
        buy_avg,
        buy_slippage,
        buy_filled
    ) = simulate_buy(
        buy_book.asks,
        TRADE_SIZE_USDT
    )


    if not buy_filled:

        return None


    (
        sold,
        sell_quote,
        sell_avg,
        sell_slippage,
        sell_filled
    ) = simulate_sell(
        sell_book.bids,
        bought
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

        "symbol":
            symbol,

        "base":
            buy_book.base,

        "quote":
            buy_book.quote,

        "buy_exchange":
            buy_book.venue,

        "sell_exchange":
            sell_book.venue,

        "buy_price":
            buy_avg,

        "sell_price":
            sell_avg,

        "amount":
            bought,

        "buy_quote":
            buy_quote,

        "sell_quote":
            sell_quote,

        "gross_profit":
            gross_profit,

        "fees":
            trading_fees,

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


# ============================================================
# FIND ALL OPPORTUNITIES
# ============================================================

def find_opportunities(
    books
):

    opportunities = []


    for symbol in SYMBOLS:

        symbol_books = {

            venue:
                book

            for (
                (book_symbol, venue),
                book
            ) in books.items()

            if book_symbol == symbol
        }


        venues = list(
            symbol_books.keys()
        )


        for buy_venue in venues:

            for sell_venue in venues:

                if (
                    buy_venue
                    == sell_venue
                ):

                    continue


                result = evaluate_pair(
                    symbol,
                    symbol_books[
                        buy_venue
                    ],
                    symbol_books[
                        sell_venue
                    ]
                )


                if result is not None:

                    opportunities.append(
                        result
                    )


    opportunities.sort(
        key=lambda x:
        x["net_percent"],
        reverse=True
    )


    return opportunities


# ============================================================
# PAPER TRADE
# ============================================================

def execute_paper_trade(
    opportunity
):

    global paper_balance
    global paper_profit
    global paper_trades
    global paper_wins
    global paper_losses
    global paper_volume


    if (
        opportunity["net_percent"]
        < MIN_NET_PROFIT
    ):

        return False


    if (
        paper_balance
        < TRADE_SIZE_USDT
    ):

        return False


    key = (
        opportunity["symbol"]
        + "|"
        + opportunity["buy_exchange"]
        + "|"
        + opportunity["sell_exchange"]
    )


    now = time.time()


    previous = last_paper_trades.get(
        key
    )


    if previous:

        if (
            now - previous
            < PAPER_TRADE_COOLDOWN
        ):

            return False


    profit = opportunity[
        "net_profit"
    ]


    paper_balance += profit

    paper_profit += profit

    paper_trades += 1

    paper_volume += (
        TRADE_SIZE_USDT
    )


    if profit >= 0:

        paper_wins += 1

    else:

        paper_losses += 1


    last_paper_trades[
        key
    ] = now


    trade = {

        "time":
            datetime.now().strftime(
                "%H:%M:%S"
            ),

        "symbol":
            opportunity["symbol"],

        "buy":
            opportunity[
                "buy_exchange"
            ],

        "sell":
            opportunity[
                "sell_exchange"
            ],

        "profit":
            profit,

        "percent":
            opportunity[
                "net_percent"
            ],
    }


    paper_history.insert(
        0,
        trade
    )


    if len(
        paper_history
    ) > 20:

        paper_history.pop()


    return True


# ============================================================
# PRICE FORMAT
# ============================================================

def price(
    value
):

    if value is None:

        return "—"

    return (
        f"${value:,.6f}"
        if value < 1
        else
        f"${value:,.2f}"
    )


# ============================================================
# MAIN KEYBOARD
# ============================================================

def main_keyboard():

    builder = (
        InlineKeyboardBuilder()
    )


    builder.button(
        text="🔎 Сканировать",
        callback_data="scan"
    )


    builder.button(
        text="📚 Стаканы",
        callback_data="books"
    )


    builder.button(
        text="🔥 Возможности",
        callback_data="opps"
    )


    builder.button(
        text="▶️ Автоскан",
        callback_data="auto_start"
    )


    builder.button(
        text="⏹ Стоп",
        callback_data="auto_stop"
    )


    builder.button(
        text="💰 PAPER",
        callback_data="paper"
    )


    builder.button(
        text="📈 Статистика",
        callback_data="stats"
    )


    builder.button(
        text="⚙️ Настройки",
        callback_data="settings"
    )


    builder.adjust(2)


    return builder.as_markup()


# ============================================================
# DASHBOARD
# ============================================================

async def show_dashboard(
    chat_id,
    text
):

    message_id = (
        dashboard_messages.get(
            chat_id
        )
    )


    # Если сообщение уже есть —
    # редактируем его.
    if message_id:

        try:

            await bot.edit_message_text(

                chat_id=chat_id,

                message_id=message_id,

                text=text,

                reply_markup=
                    main_keyboard()
            )

            return

        except Exception:

            pass


    # Если сообщения нет —
    # создаём только одно.
    message = await bot.send_message(

        chat_id,

        text,

        reply_markup=
            main_keyboard()
    )


    dashboard_messages[
        chat_id
    ] = message.message_id


# ============================================================
# START
# ============================================================

@dp.message(
    Command("start")
)
async def cmd_start(
    message: Message
):

    text = (

        "🤖 <b>ARBITRAGE BOT</b>\n\n"

        f"🪙 Монет: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"📈 Минимум: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"

        "🟢 Реальные стаканы\n"
        "🟢 Реальные цены\n"
        "🟢 Реальные объёмы\n"
        "🟡 Виртуальные сделки\n"
        "🔴 Реальные ордера OFF\n\n"

        "Выбери действие:"
    )


    # Удаляем /start, если можем
    try:

        await message.delete()

    except Exception:

        pass


    await show_dashboard(
        message.chat.id,
        text
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

    global last_books
    global last_errors
    global last_opportunities
    global last_scan_time


    await callback.answer(
        "Сканирую 6 криптовалют..."
    )


    try:

        books, errors = (
            await fetch_all_books()
        )


        last_books = books

        last_errors = errors

        last_opportunities = (
            find_opportunities(
                books
            )
        )

        last_scan_time = (
            time.time()
        )


        text = build_scan_text()


        await show_dashboard(
            callback.message.chat.id,
            text
        )


    except Exception as e:

        await show_dashboard(

            callback.message.chat.id,

            "❌ <b>ОШИБКА</b>\n\n"
            f"<code>{str(e)}</code>"
        )


# ============================================================
# BOOKS
# ============================================================

def build_books_text():

    text = (
        "📚 <b>СТАКАНЫ</b>\n\n"
    )


    for symbol in SYMBOLS:

        text += (
            f"🪙 <b>{symbol}</b>\n"
        )


        found = False


        for venue in (
            VENUE_FUNCTIONS.keys()
        ):

            book = last_books.get(
                (
                    symbol,
                    venue
                )
            )


            if book is None:

                text += (
                    f"🔴 "
                    f"{VENUE_NAMES[venue]}"
                    f" — нет данных\n"
                )

                continue


            found = True


            text += (
                f"🟢 "
                f"<b>{VENUE_NAMES[venue]}</b>"
            )


            if book.quote != "USDT":

                text += (
                    f" ({book.quote})"
                )


            text += "\n"


            text += (
                f"ASK "
                f"<b>{price(book.best_ask[0])}</b>\n"
            )


            text += (
                f"BID "
                f"<b>{price(book.best_bid[0])}</b>\n"
            )


            text += (
                f"Spread "
                f"{book.spread_percent:.4f}%\n"
            )


        if not found:

            text += (
                "⚠️ Нет доступных стаканов\n"
            )


        text += "\n"


    text += (
        f"🕐 "
        f"{datetime.now().strftime('%H:%M:%S')}"
    )


    return text


# ============================================================
# OPPORTUNITIES TEXT
# ============================================================

def build_opportunities_text():

    if not last_opportunities:

        return (

            "🔥 <b>ВОЗМОЖНОСТИ</b>\n\n"

            "Подходящих возможностей нет.\n\n"

            f"Минимум: "
            f"<b>{MIN_NET_PROFIT:.2f}%</b>"
        )


    text = (
        "🔥 <b>ВОЗМОЖНОСТИ</b>\n\n"
    )


    for i, opp in enumerate(
        last_opportunities[:10],
        1
    ):

        buy = VENUE_NAMES[
            opp["buy_exchange"]
        ]


        sell = VENUE_NAMES[
            opp["sell_exchange"]
        ]


        text += (

            f"#{i} "
            f"🪙 <b>{opp['symbol']}</b>\n"

            f"🛒 {buy} → "
            f"<b>{price(opp['buy_price'])}</b>\n"

            f"💰 {sell} → "
            f"<b>{price(opp['sell_price'])}</b>\n"

            f"📦 "
            f"{opp['amount']:.8f} "
            f"{opp['base']}\n"

            f"💵 Чистая: "
            f"<b>${opp['net_profit']:.4f}</b>\n"

            f"📈 "
            f"<b>{opp['net_percent']:.4f}%</b>\n"

            f"📉 Slippage: "
            f"{opp['buy_slippage']:.4f}% / "
            f"{opp['sell_slippage']:.4f}%\n\n"
        )


    return text


# ============================================================
# SCAN TEXT
# ============================================================

def build_scan_text():

    online = len(
        last_books
    )


    possible = len(
        last_opportunities
    )


    best = "нет"


    if last_opportunities:

        best = (
            f"{last_opportunities[0]['symbol']} "
            f"{last_opportunities[0]['net_percent']:.4f}%"
        )


    return (

        "🔎 <b>СКАНИРОВАНИЕ</b>\n\n"

        f"🪙 Монет: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"📡 Стаканов: "
        f"<b>{online}/"
        f"{len(SYMBOLS) * len(VENUE_FUNCTIONS)}</b>\n"

        f"🔥 Возможностей: "
        f"<b>{possible}</b>\n"

        f"🏆 Лучшая: "
        f"<b>{best}</b>\n\n"

        + build_opportunities_text()
    )


# ============================================================
# BOOK BUTTON
# ============================================================

@dp.callback_query(
    F.data == "books"
)
async def callback_books(
    callback: CallbackQuery
):

    await callback.answer(
        "Обновляю стаканы..."
    )


    try:

        global last_books
        global last_errors


        (
            last_books,
            last_errors
        ) = await fetch_all_books()


        await show_dashboard(

            callback.message.chat.id,

            build_books_text()
        )


    except Exception as e:

        await show_dashboard(

            callback.message.chat.id,

            "❌ <b>ОШИБКА</b>\n\n"
            f"<code>{str(e)}</code>"
        )


# ============================================================
# OPPS BUTTON
# ============================================================

@dp.callback_query(
    F.data == "opps"
)
async def callback_opps(
    callback: CallbackQuery
):

    await callback.answer()


    await show_dashboard(

        callback.message.chat.id,

        build_opportunities_text()
    )


# ============================================================
# PAPER TEXT
# ============================================================

def build_paper_text():

    roi = 0.0


    if PAPER_START_BALANCE > 0:

        roi = (
            paper_profit
            / PAPER_START_BALANCE
            * 100
        )


    text = (

        "💰 <b>PAPER MODE</b>\n\n"

        f"Старт: "
        f"<b>${PAPER_START_BALANCE:.2f}</b>\n"

        f"Баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"ROI: "
        f"<b>{roi:.4f}%</b>\n\n"

        f"Сделок: "
        f"<b>{paper_trades}</b>\n"

        f"Успешных: "
        f"<b>{paper_wins}</b>\n"

        f"Убыточных: "
        f"<b>{paper_losses}</b>\n\n"

        f"Объём: "
        f"<b>${paper_volume:.2f}</b>\n\n"

        "🟢 Рынок реальный\n"
        "🟡 Сделки виртуальные\n"
        "🔴 Деньги реальные не используются"
    )


    if paper_history:

        text += (
            "\n\n<b>Последние сделки:</b>\n"
        )


        for trade in paper_history[:5]:

            text += (

                f"• "
                f"{trade['symbol']} "
                f"{VENUE_NAMES[trade['buy']]} → "
                f"{VENUE_NAMES[trade['sell']]} "
                f"+${trade['profit']:.4f}\n"
            )


    return text


# ============================================================
# PAPER BUTTON
# ============================================================

@dp.callback_query(
    F.data == "paper"
)
async def callback_paper(
    callback: CallbackQuery
):

    await callback.answer()


    await show_dashboard(

        callback.message.chat.id,

        build_paper_text()
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


    online = len(
        last_books
    )


    total = (
        len(SYMBOLS)
        * len(VENUE_FUNCTIONS)
    )


    auto = (
        "🟢 ВКЛ"
        if auto_scan_running
        else "🔴 ВЫКЛ"
    )


    text = (

        "📊 <b>СТАТУС</b>\n\n"

        f"🪙 Криптовалют: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"📡 Стаканов: "
        f"<b>{online}/{total}</b>\n"

        f"▶️ Автоскан: "
        f"<b>{auto}</b>\n\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"📈 PAPER прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 PAPER сделок: "
        f"<b>{paper_trades}</b>\n\n"

        f"Минимум: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"Размер сделки: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n\n"

        "🔴 LIVE TRADING: "
        "<b>OFF</b>"
    )


    await show_dashboard(

        callback.message.chat.id,

        text
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


    coins = "\n".join(
        f"• {x}"
        for x in SYMBOLS
    )


    fees = "\n".join(

        f"• {VENUE_NAMES[k]}: "
        f"{v:.3f}%"

        for k, v in FEES.items()
    )


    text = (

        "⚙️ <b>НАСТРОЙКИ</b>\n\n"

        "<b>Криптовалюты:</b>\n"

        f"{coins}\n\n"

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"💰 Стартовый баланс: "
        f"<b>${PAPER_START_BALANCE:.2f}</b>\n"

        f"📈 Минимум: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"📚 Уровней: "
        f"<b>{ORDERBOOK_LEVELS}</b>\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"⛽ Резерв: "
        f"<b>${TRANSFER_COST_USDT:.2f}</b>\n\n"

        "<b>Комиссии:</b>\n"

        f"{fees}\n\n"

        "🟢 Реальные стаканы\n"
        "🟡 PAPER сделки\n"
        "🔴 LIVE OFF"
    )


    await show_dashboard(

        callback.message.chat.id,

        text
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


    roi = 0.0


    if PAPER_START_BALANCE > 0:

        roi = (
            paper_profit
            / PAPER_START_BALANCE
            * 100
        )


    best = "нет"


    if last_opportunities:

        best_opp = (
            last_opportunities[0]
        )


        best = (

            f"{best_opp['symbol']} "
            f"{best_opp['net_percent']:.4f}%"
        )


    text = (

        "📈 <b>СТАТИСТИКА</b>\n\n"

        f"🪙 Монет: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"📡 Стаканов: "
        f"<b>{len(last_books)}</b>\n"

        f"🔥 Возможностей: "
        f"<b>{len(last_opportunities)}</b>\n"

        f"🏆 Лучшая: "
        f"<b>{best}</b>\n\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"💵 Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 ROI: "
        f"<b>{roi:.4f}%</b>\n"

        f"🔄 Сделок: "
        f"<b>{paper_trades}</b>\n"

        f"🟢 Успешных: "
        f"<b>{paper_wins}</b>\n"

        f"🔴 Убыточных: "
        f"<b>{paper_losses}</b>"
    )


    await show_dashboard(

        callback.message.chat.id,

        text
    )


# ============================================================
# AUTO SCAN
# ============================================================

async def auto_scanner(
    chat_id
):

    global auto_scan_running
    global last_books
    global last_errors
    global last_opportunities
    global last_scan_time


    while auto_scan_running:

        try:

            (
                last_books,
                last_errors
            ) = await fetch_all_books()


            last_opportunities = (
                find_opportunities(
                    last_books
                )
            )


            last_scan_time = (
                time.time()
            )


            # =================================================
            # ЛУЧШАЯ ВОЗМОЖНОСТЬ
            # =================================================

            good = [

                x

                for x in last_opportunities

                if (
                    x["net_percent"]
                    >= MIN_NET_PROFIT
                )
            ]


            executed_trade = None


            # Только одна лучшая PAPER-сделка
            # за один цикл.
            if good:

                best = good[0]


                if execute_paper_trade(
                    best
                ):

                    executed_trade = best


            # =================================================
            # ОБНОВЛЯЕМ ТО ЖЕ САМОЕ СООБЩЕНИЕ
            # =================================================

            if executed_trade:

                buy_name = VENUE_NAMES[
                    executed_trade[
                        "buy_exchange"
                    ]
                ]


                sell_name = VENUE_NAMES[
                    executed_trade[
                        "sell_exchange"
                    ]
                ]


                text = (

                    "🟢 <b>PAPER СДЕЛКА</b>\n\n"

                    f"🪙 "
                    f"<b>{executed_trade['symbol']}</b>\n\n"

                    f"🛒 Купить: "
                    f"<b>{buy_name}</b>\n"

                    f"Цена: "
                    f"{price(executed_trade['buy_price'])}\n\n"

                    f"💰 Продать: "
                    f"<b>{sell_name}</b>\n"

                    f"Цена: "
                    f"{price(executed_trade['sell_price'])}\n\n"

                    f"📦 Объём: "
                    f"{executed_trade['amount']:.8f} "
                    f"{executed_trade['base']}\n"

                    f"💵 Сделка: "
                    f"${TRADE_SIZE_USDT:.2f}\n\n"

                    f"💵 Прибыль: "
                    f"<b>+${executed_trade['net_profit']:.4f}</b>\n"

                    f"📈 Доходность: "
                    f"<b>+{executed_trade['net_percent']:.4f}%</b>\n\n"

                    f"💼 PAPER баланс: "
                    f"<b>${paper_balance:.4f}</b>\n"

                    f"📊 Сделок: "
                    f"<b>{paper_trades}</b>\n\n"

                    "🔄 Автоскан продолжает работу\n"

                    "⚠️ Реальные деньги "
                    "не использовались."
                )


            else:

                online = len(
                    last_books
                )


                text = (

                    "▶️ <b>АВТОСКАН</b>\n\n"

                    f"🪙 Монет: "
                    f"<b>{len(SYMBOLS)}</b>\n"

                    f"📡 Стаканов: "
                    f"<b>{online}/"
                    f"{len(SYMBOLS) * len(VENUE_FUNCTIONS)}</b>\n"

                    f"🔥 Возможностей: "
                    f"<b>{len(last_opportunities)}</b>\n\n"

                    f"💰 PAPER баланс: "
                    f"<b>${paper_balance:.4f}</b>\n"

                    f"📈 Прибыль: "
                    f"<b>${paper_profit:.4f}</b>\n"

                    f"📊 Сделок: "
                    f"<b>{paper_trades}</b>\n\n"

                    f"🎯 Порог: "
                    f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

                    f"⏱ Следующая проверка "
                    f"примерно через "
                    f"<b>{SCAN_INTERVAL} сек.</b>\n\n"

                    "🟢 Реальные стаканы\n"
                    "🟡 Виртуальные сделки\n"
                    "🔴 LIVE OFF"
                )


            await show_dashboard(
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


            try:

                await show_dashboard(

                    chat_id,

                    "⚠️ <b>АВТОСКАН</b>\n\n"
                    f"Ошибка: "
                    f"<code>{str(e)[:300]}</code>\n\n"
                    "Повторная попытка..."
                )

            except Exception:

                pass


        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# AUTO START
# ============================================================

@dp.callback_query(
    F.data == "auto_start"
)
async def callback_auto_start(
    callback: CallbackQuery
):

    global auto_scan_running
    global auto_scan_task


    await callback.answer()


    if auto_scan_running:

        await show_dashboard(

            callback.message.chat.id,

            "▶️ <b>АВТОСКАН УЖЕ ЗАПУЩЕН</b>\n\n"

            f"🪙 Монет: "
            f"<b>{len(SYMBOLS)}</b>\n"

            f"📈 Порог: "
            f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

            f"💵 PAPER сделка: "
            f"<b>${TRADE_SIZE_USDT:.2f}</b>"
        )

        return


    auto_scan_running = True


    auto_scan_task = (
        asyncio.create_task(
            auto_scanner(
                callback.message.chat.id
            )
        )
    )


    await show_dashboard(

        callback.message.chat.id,

        "▶️ <b>АВТОСКАН ЗАПУЩЕН</b>\n\n"

        f"🪙 Монет: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"📈 Минимальная прибыль: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n\n"

        "🟢 Реальные стаканы\n"
        "🟢 Реальные цены\n"
        "🟢 Реальные объёмы\n"
        "🟡 Виртуальные сделки\n"
        "🔴 Реальные ордера OFF"
    )


# ============================================================
# AUTO STOP
# ============================================================

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


    await show_dashboard(

        callback.message.chat.id,

        "⏹ <b>АВТОСКАН ОСТАНОВЛЕН</b>\n\n"

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
async def fallback(
    message: Message
):

    try:

        await message.delete()

    except Exception:

        pass


    await show_dashboard(

        message.chat.id,

        "🤖 <b>ARBITRAGE BOT</b>\n\n"
        "Выбери действие:"
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


    # HTML включён.
    # Поэтому <b>, <i>, <code>
    # нормально отображаются Telegram.
    bot = Bot(

        token=BOT_TOKEN,

        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        )
    )


    http_session = (
        aiohttp.ClientSession()
    )


    print(
        "======================================"
    )

    print(
        "       ARBITRAGE BOT 2.0"
    )

    print(
        "======================================"
    )

    print(
        "Symbols:"
    )

    for symbol in SYMBOLS:

        print(
            " -",
            symbol
        )


    print(
        f"Trade size: "
        f"${TRADE_SIZE_USDT}"
    )

    print(
        f"Minimum profit: "
        f"{MIN_NET_PROFIT}%"
    )

    print(
        f"Interval: "
        f"{SCAN_INTERVAL}s"
    )

    print(
        f"Paper balance: "
        f"${PAPER_START_BALANCE}"
    )

    print(
        "Live trading: OFF"
    )

    print(
        "======================================"
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
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        asyncio.run(
            main()
        )

    except KeyboardInterrupt:

        print(
            "Bot stopped."
        )