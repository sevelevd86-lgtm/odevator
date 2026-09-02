import asyncio
import os
import sqlite3
import time
import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt

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

SCAN_INTERVAL = int(
    os.getenv("SCAN_INTERVAL_SECONDS", "15")
)

MIN_TRADE_USDT = float(
    os.getenv("MIN_TRADE_USDT", "20")
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

BUY_SCORE = float(
    os.getenv("BUY_SCORE", "64")
)

SELL_SCORE = float(
    os.getenv("SELL_SCORE", "43")
)

FEE_PERCENT = float(
    os.getenv("FEE_PERCENT", "0.10")
)

DB_FILE = os.getenv(
    "DATABASE_FILE",
    "trader_memory.db"
)


# ============================================================
# 50 ASSETS
# ============================================================

SYMBOLS = [
    "BTC/USDT",
    "ETH/USDT",
    "BNB/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
    "TON/USDT",

    "DOT/USDT",
    "TRX/USDT",
    "SHIB/USDT",
    "LTC/USDT",
    "BCH/USDT",
    "ATOM/USDT",
    "UNI/USDT",
    "ETC/USDT",
    "XLM/USDT",
    "NEAR/USDT",

    "APT/USDT",
    "FIL/USDT",
    "ICP/USDT",
    "ARB/USDT",
    "OP/USDT",
    "SUI/USDT",
    "INJ/USDT",
    "AAVE/USDT",
    "MKR/USDT",
    "ALGO/USDT",

    "VET/USDT",
    "HBAR/USDT",
    "EGLD/USDT",
    "SAND/USDT",
    "MANA/USDT",
    "AXS/USDT",
    "GRT/USDT",
    "THETA/USDT",
    "FTM/USDT",
    "EOS/USDT",

    "XTZ/USDT",
    "FLOW/USDT",
    "CRV/USDT",
    "LDO/USDT",
    "RUNE/USDT",
    "JASMY/USDT",
    "SEI/USDT",
    "PEPE/USDT",
    "WIF/USDT",
    "BONK/USDT",
]


# ============================================================
# 20 PUBLIC MARKET VENUES
# ============================================================

EXCHANGE_IDS = [
    "binance",
    "bybit",
    "okx",
    "kucoin",
    "gateio",
    "mexc",
    "coinex",
    "bitget",
    "htx",
    "bitmart",
    "bitrue",
    "lbank",
    "poloniex",
    "whitebit",
    "phemex",
    "xt",
    "toobit",
    "weex",
    "bitunix",
    "ascendex",
]


# ============================================================
# PAPER ONLY
# ============================================================

LIVE_TRADING = False


# ============================================================
# TELEGRAM
# ============================================================

bot: Optional[Bot] = None

dp = Dispatcher()

dashboard_messages = {}

auto_running = False

auto_task = None


# ============================================================
# GLOBAL MARKET STATE
# ============================================================

latest_market = {}

price_history = {}

market_updates = 0

last_scan_time = 0


# ============================================================
# EXCHANGE OBJECTS
# ============================================================

exchanges = {}


# ============================================================
# DATA CLASSES
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

    spread: float = 0.0

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

    best_price: float = 0.0


# ============================================================
# LEARNING STATE
# ============================================================

class LearningState:

    def __init__(self):

        self.total_trades = 0

        self.winning_trades = 0

        self.losing_trades = 0

        self.total_profit = 0.0

        self.best_trade = 0.0

        self.worst_trade = 0.0

        self.score_adjustment = 0.0

        self.good_patterns = 0

        self.bad_patterns = 0

        self.load()


    @property
    def winrate(self):

        if self.total_trades == 0:
            return 0

        return (
            self.winning_trades
            / self.total_trades
            * 100
        )


    def load(self):

        data = db_get_learning()

        if not data:
            return

        self.total_trades = data["total_trades"]

        self.winning_trades = data["winning_trades"]

        self.losing_trades = data["losing_trades"]

        self.total_profit = data["total_profit"]

        self.best_trade = data["best_trade"]

        self.worst_trade = data["worst_trade"]

        self.score_adjustment = data["score_adjustment"]

        self.good_patterns = data["good_patterns"]

        self.bad_patterns = data["bad_patterns"]


    def save(self):

        db_save_learning(self)


    def learn(
        self,
        profit,
        entry_score,
        change_1m,
        change_5m,
        volatility
    ):

        self.total_trades += 1

        self.total_profit += profit

        if profit >= 0:

            self.winning_trades += 1

            self.best_trade = max(
                self.best_trade,
                profit
            )

            self.good_patterns += 1

            self.score_adjustment += 0.15

            self.score_adjustment = min(
                self.score_adjustment,
                8
            )

            db_save_pattern(
                result="WIN",
                score=entry_score,
                change_1m=change_1m,
                change_5m=change_5m,
                volatility=volatility
            )

        else:

            self.losing_trades += 1

            self.worst_trade = min(
                self.worst_trade,
                profit
            )

            self.bad_patterns += 1

            self.score_adjustment -= 0.20

            self.score_adjustment = max(
                self.score_adjustment,
                -8
            )

            db_save_pattern(
                result="LOSS",
                score=entry_score,
                change_1m=change_1m,
                change_5m=change_5m,
                volatility=volatility
            )

        self.save()


learning = LearningState()


# ============================================================
# DATABASE
# ============================================================

db = sqlite3.connect(
    DB_FILE,
    check_same_thread=False
)

db.row_factory = sqlite3.Row


def db_init():

    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,

            exchange TEXT NOT NULL,

            entry_price REAL NOT NULL,

            exit_price REAL,

            amount REAL NOT NULL,

            invested REAL NOT NULL,

            result REAL,

            result_percent REAL,

            reason TEXT,

            entry_score REAL,

            exit_score REAL,

            opened_at TEXT,

            closed_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            result TEXT NOT NULL,

            score REAL,

            change_1m REAL,

            change_5m REAL,

            volatility REAL,

            created_at TEXT
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning (
            id INTEGER PRIMARY KEY CHECK (id = 1),

            total_trades INTEGER,

            winning_trades INTEGER,

            losing_trades INTEGER,

            total_profit REAL,

            best_trade REAL,

            worst_trade REAL,

            score_adjustment REAL,

            good_patterns INTEGER,

            bad_patterns INTEGER
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO learning
        VALUES (
            1,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0
        )
    """)

    db.commit()


def db_save_learning(state):

    cursor = db.cursor()

    cursor.execute("""
        UPDATE learning
        SET
            total_trades = ?,
            winning_trades = ?,
            losing_trades = ?,
            total_profit = ?,
            best_trade = ?,
            worst_trade = ?,
            score_adjustment = ?,
            good_patterns = ?,
            bad_patterns = ?
        WHERE id = 1
    """, (
        state.total_trades,
        state.winning_trades,
        state.losing_trades,
        state.total_profit,
        state.best_trade,
        state.worst_trade,
        state.score_adjustment,
        state.good_patterns,
        state.bad_patterns
    ))

    db.commit()


def db_get_learning():

    cursor = db.cursor()

    row = cursor.execute(
        "SELECT * FROM learning WHERE id = 1"
    ).fetchone()

    if not row:
        return None

    return dict(row)


def db_save_pattern(
    result,
    score,
    change_1m,
    change_5m,
    volatility
):

    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO patterns (
            result,
            score,
            change_1m,
            change_5m,
            volatility,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?)
    """, (
        result,
        score,
        change_1m,
        change_5m,
        volatility,
        datetime.now().isoformat()
    ))

    db.commit()


def db_save_trade(
    position,
    exit_price,
    profit,
    profit_percent,
    reason,
    exit_score
):

    cursor = db.cursor()

    cursor.execute("""
        INSERT INTO trades (
            symbol,
            exchange,
            entry_price,
            exit_price,
            amount,
            invested,
            result,
            result_percent,
            reason,
            entry_score,
            exit_score,
            opened_at,
            closed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        position.symbol,
        position.exchange,
        position.entry_price,
        exit_price,
        position.amount,
        position.invested,
        profit,
        profit_percent,
        reason,
        position.entry_score,
        exit_score,
        datetime.fromtimestamp(
            position.opened_at
        ).isoformat(),
        datetime.now().isoformat()
    ))

    db.commit()


def db_get_recent_trades(limit=15):

    cursor = db.cursor()

    return cursor.execute("""
        SELECT *
        FROM trades
        ORDER BY id DESC
        LIMIT ?
    """, (
        limit,
    )).fetchall()


def db_get_pattern_stats():

    cursor = db.cursor()

    row = cursor.execute("""
        SELECT
            COUNT(*) AS total,

            SUM(
                CASE
                    WHEN result = 'WIN'
                    THEN 1
                    ELSE 0
                END
            ) AS wins,

            SUM(
                CASE
                    WHEN result = 'LOSS'
                    THEN 1
                    ELSE 0
                END
            ) AS losses,

            AVG(
                CASE
                    WHEN result = 'WIN'
                    THEN score
                END
            ) AS avg_win_score,

            AVG(
                CASE
                    WHEN result = 'LOSS'
                    THEN score
                END
            ) AS avg_loss_score

        FROM patterns
    """).fetchone()

    return row


# ============================================================
# WALLET
# ============================================================

class PaperWallet:

    def __init__(self):

        self.usdt = START_BALANCE

        self.positions = {}

        self.realized_profit = 0.0

        self.total_fees = 0.0


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
            return False

        fee = (
            amount_usdt
            * FEE_PERCENT
            / 100
        )

        total = (
            amount_usdt
            + fee
        )

        if total > self.usdt:
            return False

        amount = (
            amount_usdt
            / price
        )

        self.usdt -= total

        self.total_fees += fee

        self.positions[symbol] = Position(
            symbol=symbol,
            exchange=exchange,
            amount=amount,
            entry_price=price,
            invested=amount_usdt,
            opened_at=time.time(),
            entry_score=score,
            best_price=price
        )

        return True


    def sell(
        self,
        symbol,
        price
    ):

        position = self.positions.get(
            symbol
        )

        if not position:
            return None

        gross = (
            position.amount
            * price
        )

        fee = (
            gross
            * FEE_PERCENT
            / 100
        )

        received = (
            gross
            - fee
        )

        profit = (
            received
            - position.invested
        )

        self.usdt += received

        self.realized_profit += profit

        self.total_fees += fee

        del self.positions[symbol]

        return {
            "position": position,
            "gross": gross,
            "fee": fee,
            "received": received,
            "profit": profit
        }


    def equity(
        self,
        prices
    ):

        total = self.usdt

        for symbol, position in self.positions.items():

            price = prices.get(
                symbol
            )

            if price:

                total += (
                    position.amount
                    * price
                )

        return total


wallet = PaperWallet()


# ============================================================
# EXCHANGE INITIALIZATION
# ============================================================

async def init_exchanges():

    for exchange_id in EXCHANGE_IDS:

        try:

            exchange_class = getattr(
                ccxt,
                exchange_id
            )

            exchange = exchange_class({
                "enableRateLimit": True,
                "timeout": 10000,
            })

            await exchange.load_markets()

            exchanges[
                exchange_id
            ] = exchange

            print(
                f"[OK] {exchange_id}"
            )

        except Exception as e:

            print(
                f"[SKIP] {exchange_id}: {e}"
            )


# ============================================================
# FETCH TICKER
# ============================================================

async def fetch_ticker(
    exchange_id,
    symbol
):

    exchange = exchanges.get(
        exchange_id
    )

    if not exchange:
        return None

    if symbol not in exchange.markets:
        return None

    try:

        ticker = await exchange.fetch_ticker(
            symbol
        )

        bid = ticker.get(
            "bid"
        )

        ask = ticker.get(
            "ask"
        )

        last = ticker.get(
            "last"
        )

        if not last:
            return None

        if not bid:
            bid = last

        if not ask:
            ask = last

        return MarketData(
            symbol=symbol,
            exchange=exchange_id.upper(),
            price=float(last),
            bid=float(bid),
            ask=float(ask),
            timestamp=time.time()
        )

    except Exception:

        return None


# ============================================================
# FETCH ALL MARKET DATA
# ============================================================

async def fetch_all_market():

    result = {}

    semaphore = asyncio.Semaphore(8)


    async def worker(
        exchange_id,
        symbol
    ):

        async with semaphore:

            return await fetch_ticker(
                exchange_id,
                symbol
            )


    tasks = []

    for exchange_id in exchanges:

        for symbol in SYMBOLS:

            tasks.append(
                worker(
                    exchange_id,
                    symbol
                )
            )


    responses = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )


    for data in responses:

        if not isinstance(
            data,
            MarketData
        ):
            continue

        result[
            (
                data.symbol,
                data.exchange
            )
        ] = data


    return result


# ============================================================
# PRICE HISTORY
# ============================================================

def update_price_history(
    market
):

    now = time.time()

    for key, data in market.items():

        if key not in price_history:

            price_history[key] = []

        price_history[key].append(
            (
                now,
                data.price
            )
        )

        cutoff = (
            now
            - 3600
        )

        price_history[key] = [
            item
            for item in price_history[key]
            if item[0] >= cutoff
        ]


def get_change(
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

    current = history[-1][1]

    target = (
        time.time()
        - seconds
    )

    old = history[0][1]

    for timestamp, price in history:

        if timestamp <= target:

            old = price

    if old <= 0:
        return 0.0

    return (
        current - old
    ) / old * 100


def get_volatility(
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

    avg = (
        sum(prices)
        / len(prices)
    )

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
# SCORING ENGINE
# ============================================================

def calculate_score(
    data
):

    data.change_1m = get_change(
        data.symbol,
        data.exchange,
        60
    )

    data.change_5m = get_change(
        data.symbol,
        data.exchange,
        300
    )

    data.volatility = get_volatility(
        data.symbol,
        data.exchange
    )

    if data.ask > 0:

        data.spread = (
            data.ask
            - data.bid
        ) / data.ask * 100

    score = 50.0


    # ========================================================
    # MOMENTUM
    # ========================================================

    if data.change_1m > 0:

        score += min(
            data.change_1m * 5,
            12
        )

    else:

        score += max(
            data.change_1m * 3,
            -12
        )


    if data.change_5m > 0:

        score += min(
            data.change_5m * 3,
            15
        )

    else:

        score += max(
            data.change_5m * 2,
            -15
        )


    # ========================================================
    # VOLATILITY
    # ========================================================

    if 0.10 <= data.volatility <= 2.5:

        score += 5

    elif data.volatility > 5:

        score -= 10


    # ========================================================
    # SPREAD
    # ========================================================

    if data.spread < 0.15:

        score += 5

    elif data.spread > 1:

        score -= 5


    # ========================================================
    # LEARNING
    # ========================================================

    score += learning.score_adjustment


    # ========================================================
    # GOOD/BAD PATTERN MEMORY
    # ========================================================

    pattern = db_get_pattern_stats()

    if pattern:

        avg_win_score = (
            pattern["avg_win_score"]
        )

        avg_loss_score = (
            pattern["avg_loss_score"]
        )

        if avg_win_score:

            if (
                data.score >= avg_win_score
            ):

                score += 2

        if avg_loss_score:

            if (
                data.score <= avg_loss_score
            ):

                score -= 2


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
# BEST MARKET
# ============================================================

def best_market(
    symbol
):

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
# TRADE SIZE
# ============================================================

def calculate_trade_size(
    score
):

    available = wallet.usdt

    if available < MIN_TRADE_USDT:

        return 0

    confidence = (
        score - BUY_SCORE
    ) / (
        100 - BUY_SCORE
    )

    confidence = max(
        0,
        min(
            1,
            confidence
        )
    )

    percent = (
        8
        + confidence
        * (
            MAX_TRADE_PERCENT - 8
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
        available
        * MAX_TRADE_PERCENT
        / 100
    )

    return amount


# ============================================================
# BUY DECISION
# ============================================================

def should_buy(
    data
):

    if not data:
        return False

    if data.symbol in wallet.positions:
        return False

    if wallet.usdt < MIN_TRADE_USDT:
        return False

    return data.score >= BUY_SCORE


# ============================================================
# SELL DECISION
# ============================================================

def should_sell(
    position,
    data
):

    if not data:

        return False, "Нет данных"

    pnl = (
        data.price
        - position.entry_price
    ) / position.entry_price * 100

    hold_minutes = (
        time.time()
        - position.opened_at
    ) / 60


    # TAKE PROFIT

    if pnl >= TAKE_PROFIT_PERCENT:

        return True, "TAKE PROFIT"


    # STOP LOSS

    if pnl <= -STOP_LOSS_PERCENT:

        return True, "STOP LOSS"


    # SCORE

    if data.score <= SELL_SCORE:

        return True, "Сигнал ослаб"


    # TIMEOUT

    if hold_minutes >= MAX_HOLD_MINUTES:

        return True, "Максимальное время"


    return False, ""


# ============================================================
# BUY
# ============================================================

async def buy_asset(
    data,
    amount
):

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


    text = (
        "🟢 <b>ПОКУПКА</b>\n\n"

        f"🪙 Актив: "
        f"<b>{data.symbol}</b>\n"

        f"🏦 Биржа: "
        f"<b>{data.exchange}</b>\n"

        f"💵 Сумма: "
        f"<b>${amount:.2f}</b>\n"

        f"📦 Количество: "
        f"<code>{position.amount:.10f}</code>\n"

        f"💰 Цена: "
        f"<b>{format_price(data.ask)}</b>\n"

        f"🧠 Score: "
        f"<b>{data.score:.2f}/100</b>\n"

        f"📈 1m: "
        f"<b>{data.change_1m:+.3f}%</b>\n"

        f"📈 5m: "
        f"<b>{data.change_5m:+.3f}%</b>\n"

        f"📊 Volatility: "
        f"<b>{data.volatility:.3f}%</b>\n\n"

        f"👛 USDT осталось: "
        f"<b>${wallet.usdt:.2f}</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER"
    )


    await send_event(
        text
    )

    return True


# ============================================================
# SELL
# ============================================================

async def sell_asset(
    data,
    reason
):

    result = wallet.sell(
        data.symbol,
        data.bid
    )

    if not result:

        return False


    position = result["position"]

    profit = result["profit"]

    profit_percent = (
        profit
        / position.invested
        * 100
    )


    db_save_trade(
        position=position,
        exit_price=data.bid,
        profit=profit,
        profit_percent=profit_percent,
        reason=reason,
        exit_score=data.score
    )


    learning.learn(
        profit=profit,
        entry_score=position.entry_score,
        change_1m=data.change_1m,
        change_5m=data.change_5m,
        volatility=data.volatility
    )


    emoji = (
        "🟢"
        if profit >= 0
        else "🔴"
    )


    text = (
        f"{emoji} <b>ПРОДАЖА</b>\n\n"

        f"🪙 Актив: "
        f"<b>{data.symbol}</b>\n"

        f"🏦 Биржа: "
        f"<b>{position.exchange}</b>\n"

        f"💰 Покупка: "
        f"<b>{format_price(position.entry_price)}</b>\n"

        f"💵 Продажа: "
        f"<b>{format_price(data.bid)}</b>\n"

        f"📦 Количество: "
        f"<code>{position.amount:.10f}</code>\n"

        f"💸 Результат: "
        f"<b>{profit:+.2f}$ "
        f"({profit_percent:+.2f}%)</b>\n"

        f"📌 Причина: "
        f"<b>{reason}</b>\n\n"

        f"👛 Баланс: "
        f"<b>${wallet.usdt:.2f}</b>\n\n"

        f"🧠 Память обновлена\n"

        f"🧠 Коррекция стратегии: "
        f"<b>{learning.score_adjustment:+.2f}</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER"
    )


    await send_event(
        text
    )

    return True


# ============================================================
# EVENT MESSAGES
# ============================================================

async def send_event(
    text
):

    for chat_id in list(
        dashboard_messages.keys()
    ):

        try:

            await bot.send_message(
                chat_id,
                text
            )

        except Exception as e:

            print(
                "Telegram error:",
                e
            )


# ============================================================
# FORMAT
# ============================================================

def format_price(
    value
):

    if value >= 1000:

        return f"${value:,.2f}"

    if value >= 1:

        return f"${value:,.4f}"

    return f"${value:.8f}"


def now_string():

    return datetime.now().strftime(
        "%H:%M:%S"
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

        prices[symbol] = data.price


    equity = wallet.equity(
        prices
    )


    pnl = (
        equity
        - START_BALANCE
    )


    roi = (
        pnl
        / START_BALANCE
        * 100
    )


    text = (
        "🤖 <b>AUTONOMOUS PAPER TRADER</b>\n\n"

        f"💵 USDT: "
        f"<b>${wallet.usdt:.2f}</b>\n"

        f"👛 Портфель: "
        f"<b>${equity:.2f}</b>\n"

        f"📈 P/L: "
        f"<b>{pnl:+.2f}$ "
        f"({roi:+.2f}%)</b>\n\n"

        f"🪙 Активов: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"🏦 Бирж подключено: "
        f"<b>{len(exchanges)}</b>\n"

        f"📡 Рыночных данных: "
        f"<b>{len(latest_market)}</b>\n"

        f"🔄 Обновлений: "
        f"<b>{market_updates}</b>\n\n"

        f"📊 Сделок: "
        f"<b>{learning.total_trades}</b>\n"

        f"🟢 Winrate: "
        f"<b>{learning.winrate:.1f}%</b>\n"

        f"💰 Общая прибыль: "
        f"<b>{learning.total_profit:+.2f}$</b>\n\n"

        f"🧠 Коррекция: "
        f"<b>{learning.score_adjustment:+.2f}</b>\n"

        f"🧠 Хороших паттернов: "
        f"<b>{learning.good_patterns}</b>\n"

        f"🧠 Плохих паттернов: "
        f"<b>{learning.bad_patterns}</b>\n\n"

        f"⚙️ Автотрейдинг: "
        f"<b>{'🟢 ON' if auto_running else '🔴 OFF'}</b>\n\n"

        "🟡 <b>PAPER MODE</b>\n"
        "Реальные деньги и ордера отключены."
    )


# ============================================================
# KEYBOARD
# ============================================================

def keyboard():

    builder = InlineKeyboardBuilder()

    buttons = [
        ("▶️ Запустить", "start"),
        ("⏹ Остановить", "stop"),

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
# DASHBOARD
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

        print(
            "Dashboard edit:",
            e
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
        "Обновляю рынок..."
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

        data = best_market(
            symbol
        )

        if not data:
            continue

        lines.append(
            f"🪙 <b>{symbol}</b>\n"
            f"🏦 {data.exchange}\n"
            f"💰 {format_price(data.price)}\n"
            f"🧠 Score: "
            f"<b>{data.score:.1f}</b>\n"
            f"📈 1m: "
            f"{data.change_1m:+.3f}% | "
            f"5m: "
            f"{data.change_5m:+.3f}%\n"
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


    equity = wallet.equity(
        prices
    )


    lines = [
        "👛 <b>ВИРТУАЛЬНЫЙ КОШЕЛЁК</b>",
        "",
        f"💵 USDT: "
        f"<b>${wallet.usdt:.2f}</b>",
        f"💰 Общая стоимость: "
        f"<b>${equity:.2f}</b>",
        ""
    ]


    if not wallet.positions:

        lines.append(
            "📭 Открытых позиций нет."
        )

    else:

        for symbol, position in wallet.positions.items():

            current = prices.get(
                symbol,
                position.entry_price
            )

            value = (
                position.amount
                * current
            )

            pnl = (
                value
                - position.invested
            )

            pnl_percent = (
                pnl
                / position.invested
                * 100
            )

            emoji = (
                "🟢"
                if pnl >= 0
                else "🔴"
            )

            lines.append(
                f"{emoji} <b>{symbol}</b>\n"
                f"🏦 {position.exchange}\n"
                f"📦 {position.amount:.8f}\n"
                f"💰 Entry: "
                f"{format_price(position.entry_price)}\n"
                f"📊 Сейчас: "
                f"{format_price(current)}\n"
                f"P/L: "
                f"<b>{pnl:+.2f}$ "
                f"({pnl_percent:+.2f}%)</b>\n"
            )


    await show_dashboard(
        callback.message.chat.id,
        "\n".join(lines)
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

    trades = db_get_recent_trades(
        15
    )


    if not trades:

        text = (
            "📜 <b>ИСТОРИЯ</b>\n\n"
            "Сделок пока нет."
        )

    else:

        lines = [
            "📜 <b>ИСТОРИЯ СДЕЛОК</b>",
            ""
        ]

        for trade in trades:

            emoji = (
                "🟢"
                if trade["result"] >= 0
                else "🔴"
            )

            lines.append(
                f"{emoji} "
                f"<b>{trade['symbol']}</b>\n"
                f"🏦 {trade['exchange']}\n"
                f"💰 "
                f"{trade['result']:+.2f}$ "
                f"({trade['result_percent']:+.2f}%)\n"
                f"📌 {trade['reason']}\n"
                f"⏰ {trade['closed_at'][-8:]}\n"
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

    pattern = db_get_pattern_stats()


    if pattern:

        avg_win = (
            pattern["avg_win_score"]
            or 0
        )

        avg_loss = (
            pattern["avg_loss_score"]
            or 0
        )

    else:

        avg_win = 0
        avg_loss = 0


    text = (
        "🧠 <b>ПАМЯТЬ И ОБУЧЕНИЕ</b>\n\n"

        f"📊 Сделок: "
        f"<b>{learning.total_trades}</b>\n"

        f"🟢 Успешных: "
        f"<b>{learning.winning_trades}</b>\n"

        f"🔴 Неудачных: "
        f"<b>{learning.losing_trades}</b>\n"

        f"🎯 Winrate: "
        f"<b>{learning.winrate:.2f}%</b>\n\n"

        f"💰 Результат: "
        f"<b>{learning.total_profit:+.2f}$</b>\n"

        f"🏆 Лучшая сделка: "
        f"<b>{learning.best_trade:+.2f}$</b>\n"

        f"💀 Худшая сделка: "
        f"<b>{learning.worst_trade:+.2f}$</b>\n\n"

        f"🧠 Хороших паттернов: "
        f"<b>{learning.good_patterns}</b>\n"

        f"🧠 Плохих паттернов: "
        f"<b>{learning.bad_patterns}</b>\n\n"

        f"📈 Средний score прибыльных: "
        f"<b>{avg_win:.2f}</b>\n"

        f"📉 Средний score убыточных: "
        f"<b>{avg_loss:.2f}</b>\n\n"

        f"⚙️ Коррекция стратегии: "
        f"<b>{learning.score_adjustment:+.2f}</b>\n\n"

        "💾 Память хранится в SQLite.\n"
        "После перезапуска история не исчезает."
    )


    await show_dashboard(
        callback.message.chat.id,
        text
    )


# ============================================================
# AUTO START
# ============================================================

@dp.callback_query(
    F.data == "start"
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
            "🟢 <b>БОТ УЖЕ РАБОТАЕТ</b>\n\n"
            "Автотрейдинг уже запущен."
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
    F.data == "stop"
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
    global last_scan_time

    market = await fetch_all_market()

    if not market:

        return False

    latest_market = market

    update_price_history(
        market
    )

    for data in latest_market.values():

        calculate_score(
            data
        )

    market_updates += 1

    last_scan_time = time.time()

    return True


# ============================================================
# TRADING LOOP
# ============================================================

async def trading_loop(
    chat_id
):

    global auto_running

    while auto_running:

        try:

            success = await update_market()

            if not success:

                await asyncio.sleep(
                    SCAN_INTERVAL
                )

                continue


            # =================================================
            # SELL EXISTING
            # =================================================

            positions_to_sell = []


            for symbol, position in list(
                wallet.positions.items()
            ):

                data = best_market(
                    symbol
                )

                if not data:
                    continue

                sell, reason = should_sell(
                    position,
                    data
                )

                if sell:

                    positions_to_sell.append(
                        (
                            data,
                            reason
                        )
                    )


            for data, reason in positions_to_sell:

                await sell_asset(
                    data,
                    reason
                )


            # =================================================
            # BUY NEW ASSET
            # =================================================

            candidates = []


            for symbol in SYMBOLS:

                if symbol in wallet.positions:

                    continue

                data = best_market(
                    symbol
                )

                if not data:
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

                amount = calculate_trade_size(
                    best.score
                )

                if amount >= MIN_TRADE_USDT:

                    await buy_asset(
                        best,
                        amount
                    )


            # =================================================
            # DASHBOARD
            # =================================================

            await show_dashboard(
                chat_id
            )


        except asyncio.CancelledError:

            raise

        except Exception as e:

            print(
                "TRADING LOOP ERROR:",
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

    global auto_running

    db_init()


    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN не найден."
        )


    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        )
    )


    print("=" * 70)

    print(
        "AUTONOMOUS PAPER TRADER"
    )

    print(
        f"Starting balance: ${START_BALANCE}"
    )

    print(
        f"Assets: {len(SYMBOLS)}"
    )

    print(
        f"Requested exchanges: {len(EXCHANGE_IDS)}"
    )

    print(
        f"Scan interval: {SCAN_INTERVAL}s"
    )

    print(
        "LIVE TRADING: OFF"
    )

    print(
        f"Database: {DB_FILE}"
    )

    print("=" * 70)


    await init_exchanges()


    print(
        f"Active exchanges: "
        f"{len(exchanges)}"
    )


    try:

        await dp.start_polling(
            bot
        )

    finally:

        auto_running = False


        if auto_task:

            auto_task.cancel()


        for exchange in exchanges.values():

            try:

                await exchange.close()

            except Exception:

                pass


        await bot.session.close()

        db.close()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":

    asyncio.run(
        main()
    )