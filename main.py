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

BOT_TOKEN = os.getenv("BOT_TOKEN", "").strip()

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

# ============================================================
# МИНИМАЛЬНАЯ ПРИБЫЛЬ
# ============================================================

MIN_NET_PROFIT = float(
    os.getenv(
        "MIN_NET_PROFIT_PERCENT",
        "0.20"
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


# ============================================================
# КОМИССИИ
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

# НИКОГДА не включаем реальные ордера
# в текущей версии.
LIVE_TRADING = False


# ============================================================
# PAPER TRADING
# ============================================================

PAPER_START_BALANCE = float(
    os.getenv(
        "PAPER_START_BALANCE",
        "1000"
    )
)

paper_balance = PAPER_START_BALANCE
paper_profit = 0.0
paper_trades = 0
paper_wins = 0
paper_losses = 0

paper_volume = 0.0

last_paper_trade = None

# Защита от повторного исполнения
# одной и той же возможности.
paper_trade_cooldown = 30


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

last_books = {}

last_errors = {}

last_opportunities = []

last_scan_time = 0


# ============================================================
# DATA STRUCTURE
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
# SYMBOL
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

        base, quote = symbol.split(
            "/",
            1
        )

        return base, quote

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

    return symbol, "USDT"


BASE_ASSET, QUOTE_ASSET = split_symbol(
    SYMBOL
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

        request_timeout = aiohttp.ClientTimeout(
            total=timeout
        )

        headers = {
            "Accept": "application/json",
            "User-Agent": "ArbitrageBot/1.0",
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
                        f"{text[:300]}"
                    )

                try:

                    return await response.json(
                        content_type=None
                    )

                except Exception:

                    raise RuntimeError(
                        f"Invalid JSON: "
                        f"{text[:300]}"
                    )

        else:

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
                        f"{text[:300]}"
                    )

                try:

                    return await response.json(
                        content_type=None
                    )

                except Exception:

                    raise RuntimeError(
                        f"Invalid JSON: "
                        f"{text[:300]}"
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
# NORMALIZE LEVELS
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

            elif isinstance(
                item,
                (list, tuple)
            ) and len(item) >= 2:

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

async def fetch_coinex():

    market = clean_symbol(
        SYMBOL
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
            "CoinEx invalid response"
        )

    if data.get("code") not in (
        0,
        "0",
        None,
    ):

        raise RuntimeError(
            "CoinEx: "
            + str(
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
            "CoinEx empty order book"
        )

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

    market = clean_symbol(
        SYMBOL
    )

    url = (
        "https://api.toobit.com/"
        "quote/v1/depth"
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
            "limit": limit,
        },
    )

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "Toobit invalid response"
        )

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

    if (
        not bids
        or not asks
    ):

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

    market = clean_symbol(
        SYMBOL
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
            "WEEX invalid response"
        )

    if data.get("code") not in (
        None,
        0,
        "0",
        "00000",
    ):

        raise RuntimeError(
            "WEEX: "
            + str(
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

    if (
        not bids
        or not asks
    ):

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

    market = (
        clean_symbol(
            SYMBOL
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
            "limit": ORDERBOOK_LEVELS
        },
    )

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "1bit invalid response"
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

    if (
        not bids
        or not asks
    ):

        raise RuntimeError(
            "1bit empty order book"
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


async def get_hyperliquid_spot_coin():

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
            "Hyperliquid spotMeta invalid"
        )

    universe = data.get(
        "universe",
        []
    )

    tokens = data.get(
        "tokens",
        []
    )

    # Сначала ищем BTC/USDC
    for pair in universe:

        name = str(
            pair.get(
                "name",
                ""
            )
        ).upper()

        if name in (
            "BTC/USDC",
            "BTC/USDT",
            "UBTC/USDC",
        ):

            return name

    token_names = {}

    for token in tokens:

        try:

            idx = int(
                token.get(
                    "index"
                )
            )

            name = str(
                token.get(
                    "name",
                    ""
                )
            ).upper()

            token_names[
                idx
            ] = name

        except Exception:

            continue

    for pair in universe:

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

                return str(
                    pair_name
                )

    return None


async def fetch_hyperliquid():

    spot_coin = None

    try:

        spot_coin = (
            await get_hyperliquid_spot_coin()
        )

    except Exception:

        spot_coin = None

    # ========================================================
    # SPOT
    # ========================================================

    if spot_coin:

        data = await hyperliquid_request(
            {
                "type": "l2Book",
                "coin": spot_coin,
            }
        )

        if isinstance(
            data,
            dict
        ):

            levels = data.get(
                "levels",
                []
            )

            if (
                isinstance(
                    levels,
                    list
                )
                and len(levels) >= 2
            ):

                bids = (
                    normalize_hyperliquid_levels(
                        levels[0],
                        reverse=True
                    )
                )

                asks = (
                    normalize_hyperliquid_levels(
                        levels[1],
                        reverse=False
                    )
                )

                if (
                    bids
                    and asks
                ):

                    return Book(
                        venue="hyperliquid",
                        symbol=spot_coin,
                        bids=bids,
                        asks=asks,
                        timestamp=time.time(),
                        derivative=False,
                    )

    # ========================================================
    # PERPETUAL FALLBACK
    # ========================================================

    data = await hyperliquid_request(
        {
            "type": "l2Book",
            "coin": BASE_ASSET,
        }
    )

    if not isinstance(
        data,
        dict
    ):

        raise RuntimeError(
            "Hyperliquid invalid response"
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

    if (
        not bids
        or not asks
    ):

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


# ============================================================
# EXCHANGES
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

async def fetch_one_venue(
    name,
    func
):

    try:

        book = await func()

        print(
            f"[OK] {name.upper()} "
            f"{book.symbol} "
            f"bid={book.best_bid[0]:.2f} "
            f"ask={book.best_ask[0]:.2f}"
        )

        return (
            name,
            book,
            None
        )

    except Exception as e:

        error = str(e)

        print(
            f"[ERROR] "
            f"{name.upper()}: "
            f"{error}"
        )

        return (
            name,
            None,
            error
        )


# ============================================================
# FETCH ALL
# ============================================================

async def fetch_all_books():

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

    for (
        name,
        book,
        error
    ) in results:

        if book is not None:

            books[name] = book

        if error is not None:

            errors[name] = error

    return (
        books,
        errors
    )


# ============================================================
# BUY SIMULATION
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
# SELL SIMULATION
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
# EVALUATE ARBITRAGE
# ============================================================

def evaluate_pair(
    buy_book,
    sell_book
):

    # Нельзя делать обычный spot
    # арбитраж с perpetual.
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

        "buy_exchange":
            buy_book.venue,

        "sell_exchange":
            sell_book.venue,

        "buy_price":
            buy_avg,

        "sell_price":
            sell_avg,

        "first_buy_price":
            buy_book.best_ask[0],

        "first_sell_price":
            sell_book.best_bid[0],

        "amount_btc":
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
# FIND OPPORTUNITIES
# ============================================================

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
                books[sell_name]
            )

            if result is None:

                continue

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

def paper_trade_key(
    opportunity
):

    return (
        opportunity["buy_exchange"]
        + "_"
        + opportunity["sell_exchange"]
    )


def execute_paper_trade(
    opportunity
):

    global paper_balance
    global paper_profit
    global paper_trades
    global paper_wins
    global paper_losses
    global paper_volume
    global last_paper_trade

    # ========================================================
    # Проверка минимальной прибыли
    # ========================================================

    if (
        opportunity["net_percent"]
        < MIN_NET_PROFIT
    ):

        return False, (
            "profit below threshold"
        )

    # ========================================================
    # Проверка PAPER баланса
    # ========================================================

    if (
        paper_balance
        < TRADE_SIZE_USDT
    ):

        return False, (
            "insufficient paper balance"
        )

    # ========================================================
    # Защита от повторной сделки
    # ========================================================

    current_time = time.time()

    current_key = paper_trade_key(
        opportunity
    )

    if last_paper_trade:

        last_key = (
            last_paper_trade["key"]
        )

        last_time = (
            last_paper_trade["time"]
        )

        if (
            current_key == last_key
            and
            current_time
            - last_time
            < paper_trade_cooldown
        ):

            return False, (
                "cooldown"
            )

    # ========================================================
    # ВИРТУАЛЬНАЯ СДЕЛКА
    # ========================================================

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

    last_paper_trade = {

        "key":
            current_key,

        "time":
            current_time,

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

    return True, "executed"


# ============================================================
# FORMAT PRICE
# ============================================================

def format_price(
    value
):

    if value is None:

        return "—"

    return (
        f"${value:,.2f}"
    )


# ============================================================
# FORMAT BOOKS
# ============================================================

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

            if len(error) > 100:

                error = (
                    error[:100]
                    + "..."
                )

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

        ask_liquidity = sum(
            price * amount
            for price, amount
            in book.asks[
                :ORDERBOOK_LEVELS
            ]
        )

        bid_liquidity = sum(
            price * amount
            for price, amount
            in book.bids[
                :ORDERBOOK_LEVELS
            ]
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
            "Пока подходящих связок нет.\n\n"
            f"Минимум: "
            f"<b>{MIN_NET_PROFIT:.2f}%</b>"
        )

    text = (
        "🔥 <b>АРБИТРАЖНЫЕ "
        "ВОЗМОЖНОСТИ</b>\n\n"
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
            if opp["net_percent"]
            >= MIN_NET_PROFIT
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
            f"   "
            f"${opp['buy_price']:,.2f}\n"
        )

        text += (
            f"💰 Продать: "
            f"<b>{sell_name}</b>\n"
        )

        text += (
            f"   "
            f"${opp['sell_price']:,.2f}\n"
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
            f"📈 <b>"
            f"{opp['net_percent']:.4f}%"
            f"</b>\n"
        )

        text += (
            f"📉 Slippage: "
            f"{opp['buy_slippage']:.4f}% / "
            f"{opp['sell_slippage']:.4f}%\n\n"
        )

    return text


# ============================================================
# MAIN KEYBOARD
# ============================================================

def main_keyboard():

    builder = InlineKeyboardBuilder()

    builder.button(
        text="🔎 Сканировать",
        callback_data="scan"
    )

    builder.button(
        text="📊 Статус",
        callback_data="status"
    )

    builder.button(
        text="▶️ Автоскан",
        callback_data="auto_start"
    )

    builder.button(
        text="⏹ Остановить",
        callback_data="auto_stop"
    )

    builder.button(
        text="💰 PAPER баланс",
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
# BOTTOM KEYBOARD
# ============================================================

def bottom_keyboard():

    builder = InlineKeyboardBuilder()

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
# START
# ============================================================

@dp.message(
    Command("start")
)
async def cmd_start(
    message: Message
):

    await message.answer(

        "🤖 <b>Arbitrage Bot</b>\n\n"

        f"💱 Пара: "
        f"<b>{SYMBOL}</b>\n"

        f"💵 Размер PAPER сделки: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"📈 Минимальная прибыль: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"

        "🟢 Стаканы берутся "
        "в реальном времени.\n"

        "💰 Сделки выполняются "
        "<b>виртуально</b>.\n\n"

        "Выбери действие:",

        reply_markup=main_keyboard()
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
        global last_errors
        global last_opportunities
        global last_scan_time

        last_books = books

        last_errors = errors

        last_opportunities = (
            find_opportunities(
                books
            )
        )

        last_scan_time = time.time()

        await callback.message.answer(

            format_opportunities(
                last_opportunities
            ),

            reply_markup=bottom_keyboard()
        )

    except Exception as e:

        await callback.message.answer(

            f"❌ <b>Ошибка сканирования</b>\n\n"
            f"<code>{str(e)}</code>",

            reply_markup=bottom_keyboard()
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
        global last_errors

        last_books = books

        last_errors = errors

        await callback.message.answer(

            format_books(
                books,
                errors
            ),

            reply_markup=bottom_keyboard()
        )

    except Exception as e:

        await callback.message.answer(

            f"❌ <b>Ошибка</b>\n\n"
            f"<code>{str(e)}</code>",

            reply_markup=bottom_keyboard()
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

            "Подходящих возможностей пока нет.\n\n"

            f"Минимум: "
            f"<b>{MIN_NET_PROFIT:.2f}%</b>\n\n"

            "Нажми «🔎 Сканировать».",

            reply_markup=bottom_keyboard()
        )

        return

    await callback.message.answer(

        format_opportunities(
            last_opportunities
        ),

        reply_markup=bottom_keyboard()
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

    total = len(
        VENUE_FUNCTIONS
    )

    auto_status = (
        "🟢 ВКЛЮЧЁН"
        if auto_scan_running
        else "🔴 ВЫКЛЮЧЕН"
    )

    await callback.message.answer(

        "📊 <b>СТАТУС</b>\n\n"

        f"Бирж онлайн: "
        f"<b>{online}/{total}</b>\n"

        f"Пара: "
        f"<b>{SYMBOL}</b>\n"

        f"PAPER режим: "
        f"<b>🟢 АКТИВЕН</b>\n"

        f"Реальные сделки: "
        f"<b>🔴 ОТКЛЮЧЕНЫ</b>\n\n"

        f"Автоскан: "
        f"<b>{auto_status}</b>\n"

        f"Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"Порог: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>",

        reply_markup=bottom_keyboard()
    )


# ============================================================
# AUTO SCANNER
# ============================================================

async def auto_scanner(
    chat_id: int
):

    global auto_scan_running
    global last_books
    global last_errors
    global last_opportunities
    global last_scan_time

    while auto_scan_running:

        try:

            books, errors = (
                await fetch_all_books()
            )

            last_books = books

            last_errors = errors

            opportunities = (
                find_opportunities(
                    books
                )
            )

            last_opportunities = (
                opportunities
            )

            last_scan_time = (
                time.time()
            )

            # =================================================
            # ИЩЕМ ПОДХОДЯЩИЕ ВОЗМОЖНОСТИ
            # =================================================

            good = [

                x

                for x in opportunities

                if (
                    x["net_percent"]
                    >= MIN_NET_PROFIT
                )

            ]

            # =================================================
            # PAPER TRADE
            # =================================================

            for opportunity in good:

                executed, reason = (
                    execute_paper_trade(
                        opportunity
                    )
                )

                if not executed:

                    continue

                buy_name = VENUE_NAMES.get(
                    opportunity[
                        "buy_exchange"
                    ],
                    opportunity[
                        "buy_exchange"
                    ].upper()
                )

                sell_name = VENUE_NAMES.get(
                    opportunity[
                        "sell_exchange"
                    ],
                    opportunity[
                        "sell_exchange"
                    ].upper()
                )

                profit = opportunity[
                    "net_profit"
                ]

                percent = opportunity[
                    "net_percent"
                ]

                text = (

                    "🟢 <b>PAPER СДЕЛКА</b>\n\n"

                    "📥 <b>ПОКУПКА</b>\n"

                    f"Биржа: "
                    f"<b>{buy_name}</b>\n"

                    f"Цена: "
                    f"${opportunity['buy_price']:,.2f}\n\n"

                    "📤 <b>ПРОДАЖА</b>\n"

                    f"Биржа: "
                    f"<b>{sell_name}</b>\n"

                    f"Цена: "
                    f"${opportunity['sell_price']:,.2f}\n\n"

                    f"📦 Объём: "
                    f"{opportunity['amount_btc']:.8f} BTC\n"

                    f"💵 Объём сделки: "
                    f"${TRADE_SIZE_USDT:.2f}\n\n"

                    f"💰 Чистая прибыль: "
                    f"<b>+${profit:.4f}</b>\n"

                    f"📈 Доходность: "
                    f"<b>+{percent:.4f}%</b>\n\n"

                    f"💼 PAPER баланс: "
                    f"<b>${paper_balance:.4f}</b>\n"

                    f"📊 Всего сделок: "
                    f"<b>{paper_trades}</b>\n\n"

                    "⚠️ Реальные деньги "
                    "не использовались."
                )

                await bot.send_message(
                    chat_id,
                    text,
                    reply_markup=bottom_keyboard()
                )

        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                f"[AUTO ERROR] {e}"
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
async def callback_auto_start(
    callback: CallbackQuery
):

    global auto_scan_task
    global auto_scan_running

    await callback.answer()

    if auto_scan_running:

        await callback.message.answer(

            "▶️ <b>Автоскан уже запущен.</b>\n\n"

            "Бот продолжает получать "
            "реальные стаканы и "
            "виртуально совершать "
            "PAPER-сделки.",

            reply_markup=bottom_keyboard()
        )

        return

    auto_scan_running = True

    auto_scan_task = (
        asyncio.create_task(
            auto_scanner(
                callback.from_user.id
            )
        )
    )

    await callback.message.answer(

        "▶️ <b>АВТОСКАН ЗАПУЩЕН</b>\n\n"

        f"🔄 Проверка каждые "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"📈 Минимум: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n\n"

        "🟢 Цены — реальные\n"
        "🟢 Стаканы — реальные\n"
        "🟢 Объёмы — реальные\n"
        "🟡 Сделки — виртуальные\n"
        "🔴 Реальные ордера — отключены",

        reply_markup=bottom_keyboard()
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

    await callback.message.answer(

        "⏹ <b>АВТОСКАН ОСТАНОВЛЕН</b>\n\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"📈 Прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 Сделок: "
        f"<b>{paper_trades}</b>",

        reply_markup=bottom_keyboard()
    )


# ============================================================
# PAPER BALANCE
# ============================================================

@dp.callback_query(
    F.data == "paper"
)
async def callback_paper(
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

    await callback.message.answer(

        "💰 <b>PAPER MODE</b>\n\n"

        f"Стартовый баланс: "
        f"<b>${PAPER_START_BALANCE:.2f}</b>\n\n"

        f"Текущий баланс: "
        f"<b>${paper_balance:.4f}</b>\n\n"

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

        "🟢 Рынок — реальный\n"
        "🟡 Сделки — виртуальные\n"
        "🔴 Реальные деньги — не используются",

        reply_markup=bottom_keyboard()
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

    roi = 0.0

    if PAPER_START_BALANCE > 0:

        roi = (
            paper_profit
            / PAPER_START_BALANCE
            * 100
        )

    last_trade_text = "нет"

    if last_paper_trade:

        buy_name = VENUE_NAMES.get(
            last_paper_trade["buy"],
            last_paper_trade["buy"].upper()
        )

        sell_name = VENUE_NAMES.get(
            last_paper_trade["sell"],
            last_paper_trade["sell"].upper()
        )

        last_trade_text = (
            f"{buy_name} → "
            f"{sell_name}: "
            f"+${last_paper_trade['profit']:.4f}"
        )

    await callback.message.answer(

        "📈 <b>СТАТИСТИКА</b>\n\n"

        f"Бирж получено: "
        f"<b>{len(last_books)}/"
        f"{len(VENUE_FUNCTIONS)}</b>\n"

        f"Возможностей: "
        f"<b>{len(last_opportunities)}</b>\n"

        f"Лучшая возможность: "
        f"<b>{best_text}</b>\n\n"

        f"💰 PAPER баланс: "
        f"<b>${paper_balance:.4f}</b>\n"

        f"💵 PAPER прибыль: "
        f"<b>${paper_profit:.4f}</b>\n"

        f"📊 ROI: "
        f"<b>{roi:.4f}%</b>\n"

        f"🔄 Сделок: "
        f"<b>{paper_trades}</b>\n"

        f"🟢 Успешных: "
        f"<b>{paper_wins}</b>\n"

        f"🔴 Убыточных: "
        f"<b>{paper_losses}</b>\n\n"

        f"Последняя сделка:\n"
        f"<b>{last_trade_text}</b>",

        reply_markup=bottom_keyboard()
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

        f"💵 PAPER сделка: "
        f"<b>${TRADE_SIZE_USDT:.2f}</b>\n"

        f"💰 PAPER старт: "
        f"<b>${PAPER_START_BALANCE:.2f}</b>\n"

        f"📚 Уровней стакана: "
        f"<b>{ORDERBOOK_LEVELS}</b>\n"

        f"📈 Минимум: "
        f"<b>{MIN_NET_PROFIT:.2f}%</b>\n"

        f"⏱ Интервал: "
        f"<b>{SCAN_INTERVAL} сек.</b>\n"

        f"⛽ Резерв перевода: "
        f"<b>${TRANSFER_COST_USDT:.2f}</b>\n\n"

        "💳 <b>КОМИССИИ</b>\n"

        f"{fees_text}\n\n"

        "🤖 <b>РЕЖИМ</b>\n"

        "🟢 Реальные стаканы\n"
        "🟢 Реальные цены\n"
        "🟢 Реальные объёмы\n"
        "🟡 PAPER сделки\n"
        "🔴 LIVE ордера отключены",

        reply_markup=bottom_keyboard()
    )


# ============================================================
# FALLBACK
# ============================================================

@dp.message()
async def fallback(
    message: Message
):

    await message.answer(

        "Выбери действие:",

        reply_markup=main_keyboard()
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

    # ========================================================
    # ВАЖНО:
    # Включаем HTML ParseMode.
    # Благодаря этому <b> и <i>
    # больше не будут отображаться
    # обычным текстом.
    # ========================================================

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
        "===================================="
    )

    print(
        "       ARBITRAGE BOT STARTED"
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
        f"Minimum profit: "
        f"{MIN_NET_PROFIT}%"
    )

    print(
        f"Orderbook levels: "
        f"{ORDERBOOK_LEVELS}"
    )

    print(
        f"Paper balance: "
        f"${PAPER_START_BALANCE}"
    )

    print(
        f"Live trading: "
        f"{LIVE_TRADING}"
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