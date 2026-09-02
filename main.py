import asyncio
import os
import time
import math
from dataclasses import dataclass, field
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

START_BALANCE = float(
    os.getenv("START_BALANCE_USDT", "1000")
)

SCAN_INTERVAL = float(
    os.getenv("SCAN_INTERVAL_SECONDS", "10")
)

SYMBOLS = [
    x.strip().upper()
    for x in os.getenv(
        "SYMBOLS",
        "BTC/USDT,ETH/USDT,SOL/USDT,XRP/USDT,DOGE/USDT,"
        "TON/USDT,ADA/USDT,AVAX/USDT,LINK/USDT,LTC/USDT,DOT/USDT"
    ).split(",")
    if x.strip()
]

MIN_TRADE_USDT = float(
    os.getenv("MIN_TRADE_USDT", "25")
)

MAX_TRADE_PERCENT = float(
    os.getenv("MAX_TRADE_PERCENT", "35")
)

TAKE_PROFIT_PERCENT = float(
    os.getenv("TAKE_PROFIT_PERCENT", "2.0")
)

STOP_LOSS_PERCENT = float(
    os.getenv("STOP_LOSS_PERCENT", "2.5")
)

MAX_HOLD_MINUTES = int(
    os.getenv("MAX_HOLD_MINUTES", "240")
)

MIN_SCORE_TO_BUY = float(
    os.getenv("MIN_SCORE_TO_BUY", "62")
)

MIN_SCORE_TO_SELL = float(
    os.getenv("MIN_SCORE_TO_SELL", "42")
)

FEE_PERCENT = float(
    os.getenv("FEE_PERCENT", "0.10")
)


# ============================================================
# PAPER ONLY
# ============================================================

LIVE_TRADING = False


# ============================================================
# TELEGRAM
# ============================================================

bot: Optional[Bot] = None

dp = Dispatcher()

http_session: Optional[aiohttp.ClientSession] = None


# ============================================================
# DATA
# ============================================================

@dataclass
class MarketData:
    symbol: str

    exchange: str

    price: float

    bid: float

    ask: float

    timestamp: float

    change_1m: float = 0.0
    change_5m: float = 0.0

    volatility: float = 0.0

    volume: float = 0.0

    score: float = 0.0


@dataclass
class Position:
    symbol: str

    exchange: str

    amount: float

    entry_price: float

    invested: float

    opened_at: float

    entry_score: float


@dataclass
class LearningStats:
    total_trades: int = 0

    winning_trades: int = 0

    losing_trades: int = 0

    total_profit: float = 0.0

    best_trade: float = 0.0

    worst_trade: float = 0.0

    score_adjustment: float = 0.0


# ============================================================
# WALLET
# ============================================================

class PaperWallet:

    def __init__(self):
        self.usdt = START_BALANCE

        self.positions: dict[str, Position] = {}

        self.realized_profit = 0.0

        self.total_fees = 0.0

        self.total_invested = 0.0

    def available_usdt(self):
        return self.usdt

    def can_buy(self, amount):
        return self.usdt >= amount

    def buy(
        self,
        symbol,
        exchange,
        amount_usdt,
        price,
        score
    ):

        if amount_usdt <= 0:
            return False

        if amount_usdt > self.usdt:
            amount_usdt = self.usdt

        fee = amount_usdt * FEE_PERCENT / 100

        total = amount_usdt + fee

        if total > self.usdt:
            return False

        crypto_amount = amount_usdt / price

        self.usdt -= total

        self.total_fees += fee

        self.total_invested += amount_usdt

        self.positions[symbol] = Position(
            symbol=symbol,
            exchange=exchange,
            amount=crypto_amount,
            entry_price=price,
            invested=amount_usdt,
            opened_at=time.time(),
            entry_score=score
        )

        return True

    def sell(
        self,
        symbol,
        price
    ):

        position = self.positions.get(symbol)

        if not position:
            return None

        gross = position.amount * price

        fee = gross * FEE_PERCENT / 100

        received = gross - fee

        profit = received - position.invested

        self.usdt += received

        self.realized_profit += profit

        self.total_fees += fee

        del self.positions[symbol]

        return {
            "profit": profit,
            "gross": gross,
            "fee": fee,
            "received": received,
            "position": position
        }

    def equity(self, prices):
        value = self.usdt

        for symbol, position in self.positions.items():

            price = prices.get(symbol)

            if price:
                value += position.amount * price

        return value

    def wallet_text(self, prices):

        equity = self.equity(prices)

        lines = [
            "👛 <b>ВИРТУАЛЬНЫЙ КОШЕЛЁК</b>",
            "",
            f"💵 USDT: <b>${self.usdt:.2f}</b>",
            f"💰 Стоимость портфеля: <b>${equity:.2f}</b>",
            "",
        ]

        if not self.positions:

            lines.append(
                "📭 Сейчас открытых позиций нет."
            )

        else:

            lines.append(
                "📦 <b>КУПЛЕННЫЕ АКТИВЫ</b>"
            )
            lines.append("")

            for symbol, position in self.positions.items():

                current = prices.get(
                    symbol,
                    position.entry_price
                )

                value = position.amount * current

                pnl = value - position.invested

                pnl_percent = (
                    pnl
                    / position.invested
                    * 100
                    if position.invested
                    else 0
                )

                emoji = "🟢" if pnl >= 0 else "🔴"

                lines.append(
                    f"{emoji} <b>{symbol}</b>\n"
                    f"   Количество: "
                    f"<code>{position.amount:.8f}</code>\n"
                    f"   Покупка: "
                    f"<b>${position.entry_price:.6f}</b>\n"
                    f"   Сейчас: "
                    f"<b>${current:.6f}</b>\n"
                    f"   Биржа: "
                    f"<b>{position.exchange}</b>\n"
                    f"   P/L: "
                    f"<b>{pnl:+.2f}$ "
                    f"({pnl_percent:+.2f}%)</b>\n"
                )

        return "\n".join(lines)


wallet = PaperWallet()


# ============================================================
# LEARNING
# ============================================================

learning = LearningStats()


# ============================================================
# HISTORY
# ============================================================

trade_history = []

MAX_HISTORY = 50


# ============================================================
# BOT STATE
# ============================================================

dashboard_messages = {}

auto_running = False

auto_task = None

market_updates = 0

latest_market = {}

price_history = {}

last_market_fetch = 0


# ============================================================
# EXCHANGES
# ============================================================

EXCHANGES = [
    "BYBIT",
    "BINANCE",
    "OKX",
    "GATE",
    "KUCOIN",
]


# ============================================================
# HELPERS
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

    if "/" in symbol:
        return tuple(
            symbol.upper().split("/", 1)
        )

    symbol = symbol.upper()

    for quote in [
        "USDT",
        "USDC"
    ]:

        if symbol.endswith(quote):

            return (
                symbol[:-len(quote)],
                quote
            )

    return (
        symbol,
        "USDT"
    )


def now_string():

    return datetime.now().strftime(
        "%H:%M:%S"
    )


def format_price(price):

    if price >= 1000:
        return f"${price:,.2f}"

    if price >= 1:
        return f"${price:,.4f}"

    return f"${price:.8f}"


# ============================================================
# HTTP
# ============================================================

async def get_json(
    url,
    params=None
):

    timeout = aiohttp.ClientTimeout(
        total=8
    )

    async with http_session.get(
        url,
        params=params,
        timeout=timeout,
        headers={
            "User-Agent":
                "PaperTraderBot/1.0",
            "Accept":
                "application/json"
        }
    ) as response:

        text = await response.text()

        if response.status != 200:

            raise RuntimeError(
                f"HTTP {response.status}: "
                f"{text[:200]}"
            )

        return await response.json(
            content_type=None
        )


# ============================================================
# BYBIT
# ============================================================

async def fetch_bybit(symbol):

    data = await get_json(
        "https://api.bybit.com/v5/market/tickers",
        {
            "category": "spot",
            "symbol": clean_symbol(symbol)
        }
    )

    if data.get("retCode") != 0:
        raise RuntimeError(
            data.get(
                "retMsg",
                "Bybit error"
            )
        )

    item = data["result"]["list"][0]

    price = float(
        item["lastPrice"]
    )

    bid = float(
        item["bid1Price"]
    )

    ask = float(
        item["ask1Price"]
    )

    return price, bid, ask


# ============================================================
# BINANCE
# ============================================================

async def fetch_binance(symbol):

    data = await get_json(
        "https://api.binance.com/api/v3/ticker/bookTicker",
        {
            "symbol": clean_symbol(symbol)
        }
    )

    bid = float(
        data["bidPrice"]
    )

    ask = float(
        data["askPrice"]
    )

    price = (
        bid + ask
    ) / 2

    return price, bid, ask


# ============================================================
# OKX
# ============================================================

async def fetch_okx(symbol):

    base, quote = split_symbol(
        symbol
    )

    data = await get_json(
        "https://www.okx.com/api/v5/market/ticker",
        {
            "instId":
                f"{base}-{quote}"
        }
    )

    if data.get("code") != "0":
        raise RuntimeError(
            data.get(
                "msg",
                "OKX error"
            )
        )

    item = data["data"][0]

    price = float(
        item["last"]
    )

    bid = float(
        item["bidPx"]
    )

    ask = float(
        item["askPx"]
    )

    return price, bid, ask


# ============================================================
# GATE
# ============================================================

async def fetch_gate(symbol):

    base, quote = split_symbol(
        symbol
    )

    pair = f"{base}_{quote}"

    data = await get_json(
        "https://api.gateio.ws/api/v4/spot/tickers",
        {
            "currency_pair":
                pair
        }
    )

    item = data[0]

    price = float(
        item["last"]
    )

    bid = float(
        item["highest_bid"]
    )

    ask = float(
        item["lowest_ask"]
    )

    return price, bid, ask


# ============================================================
# KUCOIN
# ============================================================

async def fetch_kucoin(symbol):

    base, quote = split_symbol(
        symbol
    )

    pair = f"{base}-{quote}"

    data = await get_json(
        "https://api.kucoin.com/api/v1/market/orderbook/level1",
        {
            "symbol":
                pair
        }
    )

    if data.get("code") != "200000":

        raise RuntimeError(
            data.get(
                "msg",
                "KuCoin error"
            )
        )

    item = data["data"]

    price = float(
        item["price"]
    )

    bid = float(
        item["bestBid"]
    )

    ask = float(
        item["bestAsk"]
    )

    return price, bid, ask


# ============================================================
# EXCHANGE FUNCTIONS
# ============================================================

EXCHANGE_FUNCTIONS = {
    "BYBIT": fetch_bybit,
    "BINANCE": fetch_binance,
    "OKX": fetch_okx,
    "GATE": fetch_gate,
    "KUCOIN": fetch_kucoin,
}


# ============================================================
# FETCH MARKET
# ============================================================

async def fetch_market():

    result = {}

    tasks = []

    for exchange, function in EXCHANGE_FUNCTIONS.items():

        for symbol in SYMBOLS:

            tasks.append(
                fetch_single(
                    exchange,
                    symbol,
                    function
                )
            )

    responses = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    for item in responses:

        if not item:
            continue

        exchange, symbol, data = item

        if data is None:
            continue

        price, bid, ask = data

        key = (
            symbol,
            exchange
        )

        result[key] = MarketData(
            symbol=symbol,
            exchange=exchange,
            price=price,
            bid=bid,
            ask=ask,
            timestamp=time.time()
        )

    return result


async def fetch_single(
    exchange,
    symbol,
    function
):

    try:

        data = await function(
            symbol
        )

        return (
            exchange,
            symbol,
            data
        )

    except Exception:

        return (
            exchange,
            symbol,
            None
        )


# ============================================================
# PRICE HISTORY
# ============================================================

def update_price_history(market):

    now = time.time()

    for (
        symbol,
        exchange
    ), data in market.items():

        key = (
            symbol,
            exchange
        )

        if key not in price_history:

            price_history[key] = []

        price_history[key].append(
            (
                now,
                data.price
            )
        )

        # Храним примерно 1 час
        cutoff = now - 3600

        price_history[key] = [
            x
            for x in price_history[key]
            if x[0] >= cutoff
        ]


def historical_change(
    symbol,
    exchange,
    seconds
):

    key = (
        symbol,
        exchange
    )

    history = price_history.get(
        key,
        []
    )

    if len(history) < 2:

        return 0.0

    now = time.time()

    target = now - seconds

    old = None

    for timestamp, price in history:

        if timestamp <= target:
            old = price

    if old is None:

        old = history[0][1]

    current = history[-1][1]

    if old <= 0:
        return 0.0

    return (
        current - old
    ) / old * 100


# ============================================================
# VOLATILITY
# ============================================================

def calculate_volatility(
    symbol,
    exchange
):

    key = (
        symbol,
        exchange
    )

    history = price_history.get(
        key,
        []
    )

    if len(history) < 5:
        return 0.0

    prices = [
        x[1]
        for x in history[-30:]
    ]

    avg = sum(prices) / len(prices)

    if avg <= 0:
        return 0.0

    variance = sum(
        (
            p - avg
        ) ** 2
        for p in prices
    ) / len(prices)

    return (
        math.sqrt(variance)
        / avg
        * 100
    )


# ============================================================
# AI-LIKE DECISION SCORE
# ============================================================

def calculate_score(
    data: MarketData
):

    change_1m = historical_change(
        data.symbol,
        data.exchange,
        60
    )

    change_5m = historical_change(
        data.symbol,
        data.exchange,
        300
    )

    volatility = calculate_volatility(
        data.symbol,
        data.exchange
    )

    data.change_1m = change_1m
    data.change_5m = change_5m
    data.volatility = volatility

    score = 50.0

    # --------------------------------------------------------
    # MOMENTUM
    # --------------------------------------------------------

    if change_1m > 0:
        score += min(
            change_1m * 5,
            12
        )

    else:
        score += max(
            change_1m * 3,
            -12
        )

    if change_5m > 0:
        score += min(
            change_5m * 3,
            15
        )

    else:
        score += max(
            change_5m * 2,
            -15
        )

    # --------------------------------------------------------
    # VOLATILITY
    # --------------------------------------------------------

    if 0.15 <= volatility <= 2.0:

        score += 5

    elif volatility > 5:

        score -= 8

    # --------------------------------------------------------
    # SPREAD
    # --------------------------------------------------------

    if data.ask > 0:

        spread = (
            data.ask - data.bid
        ) / data.ask * 100

        if spread < 0.15:
            score += 5

        elif spread > 1:
            score -= 5

    # --------------------------------------------------------
    # LEARNING ADJUSTMENT
    # --------------------------------------------------------

    score += learning.score_adjustment

    score = max(
        0,
        min(
            100,
            score
        )
    )

    data.score = score

    return score


# ============================================================
# CHOOSE BEST MARKET
# ============================================================

def choose_market(symbol):

    candidates = [
        data
        for (
            s,
            exchange
        ), data in latest_market.items()
        if s == symbol
    ]

    if not candidates:
        return None

    for data in candidates:

        calculate_score(
            data
        )

    candidates.sort(
        key=lambda x: x.score,
        reverse=True
    )

    return candidates[0]


# ============================================================
# CHOOSE TRADE SIZE
# ============================================================

def choose_trade_size(
    score
):

    available = wallet.available_usdt()

    if available < MIN_TRADE_USDT:

        return 0

    # Чем увереннее бот,
    # тем больше виртуальная сделка.

    normalized = (
        score - MIN_SCORE_TO_BUY
    ) / (
        100 - MIN_SCORE_TO_BUY
    )

    normalized = max(
        0,
        min(
            1,
            normalized
        )
    )

    percent = (
        10
        + normalized * (
            MAX_TRADE_PERCENT - 10
        )
    )

    amount = (
        available
        * percent
        / 100
    )

    amount = max(
        amount,
        MIN_TRADE_USDT
    )

    amount = min(
        amount,
        available * MAX_TRADE_PERCENT / 100
    )

    return amount


# ============================================================
# BUY DECISION
# ============================================================

def should_buy(data):

    if data is None:
        return False

    if data.score < MIN_SCORE_TO_BUY:
        return False

    if data.symbol in wallet.positions:
        return False

    if wallet.available_usdt() < MIN_TRADE_USDT:
        return False

    return True


# ============================================================
# SELL DECISION
# ============================================================

def should_sell(
    position,
    data
):

    if data is None:
        return False, "Нет данных"

    current = data.price

    pnl = (
        current
        - position.entry_price
    ) / position.entry_price * 100

    holding_minutes = (
        time.time()
        - position.opened_at
    ) / 60

    # TAKE PROFIT

    if pnl >= TAKE_PROFIT_PERCENT:

        return True, "TAKE PROFIT"

    # STOP LOSS

    if pnl <= -STOP_LOSS_PERCENT:

        return True, "STOP LOSS"

    # Сильное ухудшение оценки

    if data.score <= MIN_SCORE_TO_SELL:

        return True, "Сигнал ослаб"

    # Слишком долго держим

    if holding_minutes >= MAX_HOLD_MINUTES:

        return True, "Максимальное время"

    return False, ""


# ============================================================
# LEARNING
# ============================================================

def learn_from_trade(
    profit,
    entry_score
):

    global learning

    learning.total_trades += 1

    learning.total_profit += profit

    if profit > 0:

        learning.winning_trades += 1

        learning.best_trade = max(
            learning.best_trade,
            profit
        )

        # Если стратегия заработала,
        # немного повышаем уверенность.

        learning.score_adjustment = min(
            learning.score_adjustment + 0.15,
            5
        )

    else:

        learning.losing_trades += 1

        learning.worst_trade = min(
            learning.worst_trade,
            profit
        )

        # Если сделка неудачная,
        # снижаем агрессивность.

        learning.score_adjustment = max(
            learning.score_adjustment - 0.20,
            -5
        )


# ============================================================
# BUY
# ============================================================

async def execute_buy(
    data,
    amount
):

    if not wallet.can_buy(
        amount
    ):

        return False

    success = wallet.buy(
        symbol=data.symbol,
        exchange=data.exchange,
        amount_usdt=amount,
        price=data.ask,
        score=data.score
    )

    if not success:
        return False

    position = wallet.positions[
        data.symbol
    ]

    message = (
        "🟢 <b>БОТ КУПИЛ КРИПТУ</b>\n\n"

        f"🪙 Актив: "
        f"<b>{data.symbol}</b>\n"

        f"🏦 Биржа: "
        f"<b>{data.exchange}</b>\n"

        f"💵 Сумма: "
        f"<b>${amount:.2f}</b>\n"

        f"📦 Количество: "
        f"<code>{position.amount:.8f}</code>\n"

        f"💰 Цена покупки: "
        f"<b>{format_price(data.ask)}</b>\n"

        f"🧠 Оценка: "
        f"<b>{data.score:.1f}/100</b>\n"

        f"📈 1м: "
        f"<b>{data.change_1m:+.3f}%</b>\n"

        f"📈 5м: "
        f"<b>{data.change_5m:+.3f}%</b>\n\n"

        f"👛 Остаток USDT: "
        f"<b>${wallet.usdt:.2f}</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER MODE"
    )

    await send_trade_message(
        message
    )

    return True


# ============================================================
# SELL
# ============================================================

async def execute_sell(
    data,
    reason
):

    result = wallet.sell(
        data.symbol,
        data.bid
    )

    if result is None:
        return False

    profit = result["profit"]

    learn_from_trade(
        profit,
        result["position"].entry_score
    )

    trade_history.insert(
        0,
        {
            "time": now_string(),
            "symbol": data.symbol,
            "exchange": result["position"].exchange,
            "buy_price":
                result["position"].entry_price,
            "sell_price":
                data.bid,
            "invested":
                result["position"].invested,
            "profit": profit,
            "reason": reason
        }
    )

    del trade_history[MAX_HISTORY:]

    if profit >= 0:

        emoji = "🟢"

    else:

        emoji = "🔴"

    profit_percent = (
        profit
        / result["position"].invested
        * 100
    )

    message = (
        f"{emoji} <b>БОТ ПРОДАЛ КРИПТУ</b>\n\n"

        f"🪙 Актив: "
        f"<b>{data.symbol}</b>\n"

        f"🏦 Биржа покупки: "
        f"<b>{result['position'].exchange}</b>\n"

        f"💰 Цена покупки: "
        f"<b>{format_price(result['position'].entry_price)}</b>\n"

        f"💵 Цена продажи: "
        f"<b>{format_price(data.bid)}</b>\n"

        f"📦 Количество: "
        f"<code>{result['position'].amount:.8f}</code>\n"

        f"💸 Результат: "
        f"<b>{profit:+.2f}$ "
        f"({profit_percent:+.2f}%)</b>\n"

        f"📌 Причина: "
        f"<b>{reason}</b>\n\n"

        f"👛 Баланс: "
        f"<b>${wallet.usdt:.2f}</b>\n"

        f"🧠 Обучение: "
        f"<b>{'успешно' if profit >= 0 else 'корректирую стратегию'}</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER MODE"
    )

    await send_trade_message(
        message
    )

    return True


# ============================================================
# TRADE MESSAGE
# ============================================================

async def send_trade_message(
    text
):

    # Здесь сообщения о покупке/продаже
    # отправляются отдельно.

    for chat_id in dashboard_messages:

        try:

            await bot.send_message(
                chat_id,
                text
            )

        except Exception as e:

            print(
                "Telegram send error:",
                e
            )


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_text():

    prices = {}

    for (
        symbol,
        exchange
    ), data in latest_market.items():

        if symbol not in prices:

            prices[symbol] = data.price

    equity = wallet.equity(
        prices
    )

    roi = (
        equity
        - START_BALANCE
    ) / START_BALANCE * 100

    lines = [
        "🤖 <b>AUTONOMOUS PAPER TRADER</b>",
        "",
        f"💵 Свободно: "
        f"<b>${wallet.usdt:.2f}</b>",

        f"👛 Портфель: "
        f"<b>${equity:.2f}</b>",

        f"📈 P/L: "
        f"<b>${equity - START_BALANCE:+.2f}</b> "
        f"({roi:+.2f}%)",

        "",
        f"🔄 Обновлений: "
        f"<b>{market_updates}</b>",

        f"📊 Сделок: "
        f"<b>{learning.total_trades}</b>",

        f"🟢 Плюсовых: "
        f"<b>{learning.winning_trades}</b>",

        f"🔴 Минусовых: "
        f"<b>{learning.losing_trades}</b>",

        "",
        f"🧠 Коррекция стратегии: "
        f"<b>{learning.score_adjustment:+.2f}</b>",

        "",
        f"⚙️ Автотрейдинг: "
        f"<b>{'🟢 ON' if auto_running else '🔴 OFF'}</b>",

        "",
        "🟡 <b>PAPER MODE</b>",
        "Реальные деньги не используются."
    ]

    if wallet.positions:

        lines.append("")
        lines.append(
            "📦 <b>ОТКРЫТЫЕ ПОЗИЦИИ</b>"
        )

        for symbol, position in wallet.positions.items():

            current = prices.get(
                symbol,
                position.entry_price
            )

            pnl = (
                current
                - position.entry_price
            ) / position.entry_price * 100

            emoji = (
                "🟢"
                if pnl >= 0
                else "🔴"
            )

            lines.append(
                f"{emoji} {symbol}: "
                f"{pnl:+.2f}%"
            )

    else:

        lines.append("")
        lines.append(
            "📭 Открытых позиций нет."
        )

    return "\n".join(lines)


# ============================================================
# KEYBOARD
# ============================================================

def keyboard():

    builder = InlineKeyboardBuilder()

    buttons = [
        ("▶️ Запустить", "start_auto"),
        ("⏹ Остановить", "stop_auto"),

        ("👛 Кошелёк", "wallet"),
        ("📊 Рынок", "market"),

        ("📜 История", "history"),
        ("🧠 Обучение", "learning"),

        ("🔄 Обновить", "refresh"),
    ]

    for text, callback in buttons:

        builder.button(
            text=text,
            callback_data=callback
        )

    builder.adjust(2)

    return builder.as_markup()


# ============================================================
# DASHBOARD MESSAGE
# ============================================================

async def show_dashboard(
    chat_id,
    text=None
):

    if text is None:

        text = dashboard_text()

    message_id = dashboard_messages.get(
        chat_id
    )

    if message_id is None:

        message = await bot.send_message(
            chat_id,
            text,
            reply_markup=keyboard()
        )

        dashboard_messages[
            chat_id
        ] = message.message_id

        return

    try:

        await bot.edit_message_text(
            chat_id=chat_id,
            message_id=message_id,
            text=text,
            reply_markup=keyboard()
        )

    except Exception as e:

        # Не создаём новое dashboard-сообщение.
        print(
            "Dashboard edit error:",
            e
        )


# ============================================================
# /START
# ============================================================

@dp.message(Command("start"))
async def start_command(
    message: Message
):

    try:
        await message.delete()
    except Exception:
        pass

    await show_dashboard(
        message.chat.id
    )


# ============================================================
# REFRESH
# ============================================================

@dp.callback_query(
    F.data == "refresh"
)
async def refresh(
    callback: CallbackQuery
):

    await callback.answer(
        "Обновляю..."
    )

    await update_market()

    await show_dashboard(
        callback.message.chat.id
    )


# ============================================================
# MARKET
# ============================================================

def market_text():

    lines = [
        "📊 <b>РЫНОК</b>",
        ""
    ]

    for symbol in SYMBOLS:

        candidates = [
            data
            for (
                s,
                exchange
            ), data
            in latest_market.items()
            if s == symbol
        ]

        if not candidates:
            continue

        best = max(
            candidates,
            key=lambda x: x.score
        )

        lines.append(
            f"🪙 <b>{symbol}</b>\n"
            f"🏦 {best.exchange}\n"
            f"💰 {format_price(best.price)}\n"
            f"🧠 Score: "
            f"<b>{best.score:.1f}/100</b>\n"
            f"📈 1m: "
            f"{best.change_1m:+.3f}% | "
            f"5m: "
            f"{best.change_5m:+.3f}%\n"
        )

    return "\n".join(lines)


@dp.callback_query(
    F.data == "market"
)
async def market(
    callback: CallbackQuery
):

    await callback.answer()

    await show_dashboard(
        callback.message.chat.id,
        market_text()
    )


# ============================================================
# WALLET
# ============================================================

@dp.callback_query(
    F.data == "wallet"
)
async def wallet_callback(
    callback: CallbackQuery
):

    await callback.answer()

    prices = {}

    for (
        symbol,
        exchange
    ), data in latest_market.items():

        prices[symbol] = data.price

    await show_dashboard(
        callback.message.chat.id,
        wallet.wallet_text(
            prices
        )
    )


# ============================================================
# HISTORY
# ============================================================

@dp.callback_query(
    F.data == "history"
)
async def history(
    callback: CallbackQuery
):

    await callback.answer()

    if not trade_history:

        text = (
            "📜 <b>ИСТОРИЯ</b>\n\n"
            "Сделок пока не было."
        )

    else:

        lines = [
            "📜 <b>ИСТОРИЯ СДЕЛОК</b>",
            ""
        ]

        for trade in trade_history[:15]:

            emoji = (
                "🟢"
                if trade["profit"] >= 0
                else "🔴"
            )

            lines.append(
                f"{emoji} <b>{trade['symbol']}</b>\n"
                f"🏦 {trade['exchange']}\n"
                f"💵 "
                f"{trade['profit']:+.2f}$ "
                f"({trade['profit'] / trade['invested'] * 100:+.2f}%)\n"
                f"📌 {trade['reason']}\n"
                f"⏰ {trade['time']}\n"
            )

        text = "\n".join(lines)

    await show_dashboard(
        callback.message.chat.id,
        text
    )


# ============================================================
# LEARNING
# ============================================================

@dp.callback_query(
    F.data == "learning"
)
async def learning_callback(
    callback: CallbackQuery
):

    await callback.answer()

    winrate = (
        learning.winning_trades
        / learning.total_trades
        * 100
        if learning.total_trades
        else 0
    )

    text = (
        "🧠 <b>ОБУЧЕНИЕ БОТА</b>\n\n"

        f"📊 Всего сделок: "
        f"<b>{learning.total_trades}</b>\n"

        f"🟢 Успешных: "
        f"<b>{learning.winning_trades}</b>\n"

        f"🔴 Неудачных: "
        f"<b>{learning.losing_trades}</b>\n"

        f"🎯 Winrate: "
        f"<b>{winrate:.2f}%</b>\n\n"

        f"💰 Общий результат: "
        f"<b>${learning.total_profit:+.2f}</b>\n"

        f"🏆 Лучшая сделка: "
        f"<b>${learning.best_trade:+.2f}</b>\n"

        f"💀 Худшая сделка: "
        f"<b>${learning.worst_trade:+.2f}</b>\n\n"

        f"🧠 Коррекция score: "
        f"<b>{learning.score_adjustment:+.2f}</b>\n\n"

        "Бот постепенно меняет агрессивность "
        "стратегии на основании результатов "
        "закрытых сделок."
    )

    await show_dashboard(
        callback.message.chat.id,
        text
    )


# ============================================================
# AUTO START
# ============================================================

@dp.callback_query(
    F.data == "start_auto"
)
async def start_auto(
    callback: CallbackQuery
):

    global auto_running
    global auto_task

    await callback.answer()

    if auto_running:

        await show_dashboard(
            callback.message.chat.id,
            "🟢 <b>АВТОТРЕЙДИНГ УЖЕ РАБОТАЕТ</b>"
        )

        return

    auto_running = True

    await show_dashboard(
        callback.message.chat.id,
        "🟢 <b>АВТОТРЕЙДИНГ ЗАПУЩЕН</b>\n\n"
        "Бот начинает анализировать рынок."
    )

    auto_task = asyncio.create_task(
        trading_loop(
            callback.message.chat.id
        )
    )


# ============================================================
# AUTO STOP
# ============================================================

@dp.callback_query(
    F.data == "stop_auto"
)
async def stop_auto(
    callback: CallbackQuery
):

    global auto_running
    global auto_task

    await callback.answer()

    auto_running = False

    if auto_task:

        auto_task.cancel()

        try:
            await auto_task
        except asyncio.CancelledError:
            pass

        auto_task = None

    await show_dashboard(
        callback.message.chat.id,
        "⏹ <b>АВТОТРЕЙДИНГ ОСТАНОВЛЕН</b>\n\n"
        + dashboard_text()
    )


# ============================================================
# UPDATE MARKET
# ============================================================

async def update_market():

    global latest_market
    global market_updates

    market = await fetch_market()

    if not market:
        return

    latest_market = market

    update_price_history(
        market
    )

    for data in latest_market.values():

        calculate_score(
            data
        )

    market_updates += 1


# ============================================================
# TRADING LOOP
# ============================================================

async def trading_loop(
    chat_id
):

    global auto_running

    while auto_running:

        try:

            await update_market()

            # =================================================
            # FIRST: CHECK EXISTING POSITIONS
            # =================================================

            symbols_to_sell = []

            for symbol, position in list(
                wallet.positions.items()
            ):

                data = choose_market(
                    symbol
                )

                if data is None:
                    continue

                sell, reason = should_sell(
                    position,
                    data
                )

                if sell:

                    symbols_to_sell.append(
                        (
                            data,
                            reason
                        )
                    )

            for data, reason in symbols_to_sell:

                await execute_sell(
                    data,
                    reason
                )

            # =================================================
            # SECOND: SEARCH FOR NEW BUY
            # =================================================

            candidates = []

            for symbol in SYMBOLS:

                if symbol in wallet.positions:
                    continue

                data = choose_market(
                    symbol
                )

                if data is None:
                    continue

                if should_buy(data):

                    candidates.append(
                        data
                    )

            if candidates:

                candidates.sort(
                    key=lambda x: x.score,
                    reverse=True
                )

                best = candidates[0]

                amount = choose_trade_size(
                    best.score
                )

                if amount >= MIN_TRADE_USDT:

                    await execute_buy(
                        best,
                        amount
                    )

            # =================================================
            # UPDATE DASHBOARD
            # =================================================

            await show_dashboard(
                chat_id
            )

        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                "Trading loop error:",
                e
            )

        await asyncio.sleep(
            SCAN_INTERVAL
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
        message.chat.id
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
        token=BOT_TOKEN,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        )
    )

    http_session = aiohttp.ClientSession()

    print("=" * 60)
    print("AUTONOMOUS PAPER TRADER")
    print("=" * 60)

    print(
        "Starting balance:",
        START_BALANCE
    )

    print(
        "Assets:",
        ", ".join(SYMBOLS)
    )

    print(
        "Exchanges:",
        ", ".join(EXCHANGES)
    )

    print(
        "Scan interval:",
        SCAN_INTERVAL
    )

    print(
        "LIVE TRADING:",
        LIVE_TRADING
    )

    print("=" * 60)

    try:

        await dp.start_polling(
            bot
        )

    finally:

        if http_session:

            await http_session.close()

        await bot.session.close()


if __name__ == "__main__":

    asyncio.run(main())