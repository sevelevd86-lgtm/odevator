import asyncio
import json
import math
import os
import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import ccxt.async_support as ccxt
import joblib
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler

from aiogram import Bot, Dispatcher, F
from aiogram.client.default import DefaultBotProperties
from aiogram.enums import ParseMode
from aiogram.filters import Command
from aiogram.types import CallbackQuery, Message
from aiogram.utils.keyboard import InlineKeyboardBuilder

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
    os.getenv("SCAN_INTERVAL_SECONDS", "10")
)

MIN_TRADE_USDT = float(
    os.getenv("MIN_TRADE_USDT", "20")
)

MAX_TRADE_PERCENT = float(
    os.getenv("MAX_TRADE_PERCENT", "35")
)

MIN_TRADE_SCORE = float(
    os.getenv("MIN_TRADE_SCORE", "63")
)

MIN_ML_PROBABILITY = float(
    os.getenv("MIN_ML_PROBABILITY", "0.55")
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

FEE_PERCENT = float(
    os.getenv("FEE_PERCENT", "0.10")
)

DATABASE_FILE = os.getenv(
    "DATABASE_FILE",
    "trader_memory.db"
)

MODEL_FILE = os.getenv(
    "MODEL_FILE",
    "trader_model.pkl"
)

SCALER_FILE = os.getenv(
    "SCALER_FILE",
    "trader_scaler.pkl"
)


# ============================================================
# PAPER MODE
# ============================================================

LIVE_TRADING = False


# ============================================================
# 50 CRYPTOCURRENCIES
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
# 20 EXCHANGES
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
# GLOBALS
# ============================================================

bot: Optional[Bot] = None

dp = Dispatcher()

# ВАЖНО:
# db НЕ используется до init_database()

db = None

exchanges = {}

dashboard_messages = {}

latest_market = {}

price_history = {}

market_updates = 0

auto_running = False

auto_task = None


# ============================================================
# POSITION
# ============================================================

@dataclass
class Position:

    symbol: str

    exchange: str

    amount: float

    entry_price: float

    invested: float

    opened_at: float

    entry_score: float

    ml_probability: float

    features: dict


# ============================================================
# DATABASE
# ============================================================

def init_database():

    global db

    # Если по какой-то причине БД уже существует
    # — не создаём второе соединение.

    if db is not None:
        return

    db = sqlite3.connect(
        DATABASE_FILE,
        check_same_thread=False
    )

    db.row_factory = sqlite3.Row

    cursor = db.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS wallet (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            usdt REAL NOT NULL,
            realized_profit REAL NOT NULL DEFAULT 0,
            total_fees REAL NOT NULL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            symbol TEXT PRIMARY KEY,
            exchange TEXT NOT NULL,
            amount REAL NOT NULL,
            entry_price REAL NOT NULL,
            invested REAL NOT NULL,
            opened_at REAL NOT NULL,
            entry_score REAL NOT NULL,
            ml_probability REAL NOT NULL,
            features TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,

            symbol TEXT NOT NULL,
            exchange TEXT NOT NULL,

            entry_price REAL NOT NULL,
            exit_price REAL NOT NULL,

            amount REAL NOT NULL,
            invested REAL NOT NULL,

            profit REAL NOT NULL,
            profit_percent REAL NOT NULL,

            reason TEXT,

            entry_score REAL,
            exit_score REAL,

            ml_probability REAL,

            entry_features TEXT,

            opened_at REAL,
            closed_at REAL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS learning_state (
            id INTEGER PRIMARY KEY CHECK (id = 1),

            total_trades INTEGER NOT NULL DEFAULT 0,
            wins INTEGER NOT NULL DEFAULT 0,
            losses INTEGER NOT NULL DEFAULT 0,

            total_profit REAL NOT NULL DEFAULT 0,

            best_trade REAL NOT NULL DEFAULT 0,
            worst_trade REAL NOT NULL DEFAULT 0,

            model_updates INTEGER NOT NULL DEFAULT 0
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS bot_memory (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    cursor.execute("""
        INSERT OR IGNORE INTO wallet (
            id,
            usdt,
            realized_profit,
            total_fees
        )
        VALUES (1, ?, 0, 0)
    """, (
        START_BALANCE,
    ))

    cursor.execute("""
        INSERT OR IGNORE INTO learning_state (
            id
        )
        VALUES (1)
    """)

    db.commit()

    print(
        f"[DATABASE] initialized: {DATABASE_FILE}"
    )


# ============================================================
# WALLET DATABASE
# ============================================================

def load_wallet_state():

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    row = db.execute("""
        SELECT *
        FROM wallet
        WHERE id = 1
    """).fetchone()

    if row is None:

        db.execute("""
            INSERT INTO wallet (
                id,
                usdt,
                realized_profit,
                total_fees
            )
            VALUES (1, ?, 0, 0)
        """, (
            START_BALANCE,
        ))

        db.commit()

        return {
            "usdt": START_BALANCE,
            "realized_profit": 0.0,
            "total_fees": 0.0
        }

    return {
        "usdt": float(row["usdt"]),
        "realized_profit": float(
            row["realized_profit"]
        ),
        "total_fees": float(
            row["total_fees"]
        )
    }


def save_wallet_state(
    usdt,
    realized_profit,
    total_fees
):

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    db.execute("""
        UPDATE wallet
        SET
            usdt = ?,
            realized_profit = ?,
            total_fees = ?
        WHERE id = 1
    """, (
        usdt,
        realized_profit,
        total_fees
    ))

    db.commit()


# ============================================================
# POSITION DATABASE
# ============================================================

def save_position(position):

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    db.execute("""
        INSERT OR REPLACE INTO positions (
            symbol,
            exchange,
            amount,
            entry_price,
            invested,
            opened_at,
            entry_score,
            ml_probability,
            features
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        position.symbol,
        position.exchange,
        position.amount,
        position.entry_price,
        position.invested,
        position.opened_at,
        position.entry_score,
        position.ml_probability,
        json.dumps(
            position.features
        )
    ))

    db.commit()


def delete_position(symbol):

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    db.execute("""
        DELETE FROM positions
        WHERE symbol = ?
    """, (
        symbol,
    ))

    db.commit()


def load_positions():

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    rows = db.execute("""
        SELECT *
        FROM positions
    """).fetchall()

    loaded_positions = {}

    for row in rows:

        try:

            loaded_positions[
                row["symbol"]
            ] = Position(
                symbol=row["symbol"],
                exchange=row["exchange"],
                amount=float(
                    row["amount"]
                ),
                entry_price=float(
                    row["entry_price"]
                ),
                invested=float(
                    row["invested"]
                ),
                opened_at=float(
                    row["opened_at"]
                ),
                entry_score=float(
                    row["entry_score"]
                ),
                ml_probability=float(
                    row["ml_probability"]
                ),
                features=json.loads(
                    row["features"]
                )
            )

        except Exception as e:

            print(
                "[DATABASE] position load error:",
                e
            )

    return loaded_positions


# ============================================================
# LEARNING DATABASE
# ============================================================

def load_learning_state():

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    row = db.execute("""
        SELECT *
        FROM learning_state
        WHERE id = 1
    """).fetchone()

    if row is None:

        db.execute("""
            INSERT INTO learning_state (
                id
            )
            VALUES (1)
        """)

        db.commit()

        row = db.execute("""
            SELECT *
            FROM learning_state
            WHERE id = 1
        """).fetchone()

    return row


def save_learning_state(
    total_trades,
    wins,
    losses,
    total_profit,
    best_trade,
    worst_trade,
    model_updates
):

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    db.execute("""
        UPDATE learning_state
        SET
            total_trades = ?,
            wins = ?,
            losses = ?,
            total_profit = ?,
            best_trade = ?,
            worst_trade = ?,
            model_updates = ?
        WHERE id = 1
    """, (
        total_trades,
        wins,
        losses,
        total_profit,
        best_trade,
        worst_trade,
        model_updates
    ))

    db.commit()


# ============================================================
# TRADE DATABASE
# ============================================================

def save_trade(
    position,
    exit_price,
    profit,
    profit_percent,
    reason,
    exit_score
):

    if db is None:
        raise RuntimeError(
            "Database is not initialized"
        )

    db.execute("""
        INSERT INTO trades (
            symbol,
            exchange,
            entry_price,
            exit_price,
            amount,
            invested,
            profit,
            profit_percent,
            reason,
            entry_score,
            exit_score,
            ml_probability,
            entry_features,
            opened_at,
            closed_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        position.ml_probability,
        json.dumps(
            position.features
        ),
        position.opened_at,
        time.time()
    ))

    db.commit()


def get_trades():

    if db is None:
        return []

    return db.execute("""
        SELECT *
        FROM trades
        ORDER BY id DESC
        LIMIT 20
    """).fetchall()


# ============================================================
# LEARNING ENGINE
# ============================================================

class LearningEngine:

    def __init__(self):

        state = load_learning_state()

        self.total_trades = int(
            state["total_trades"]
        )

        self.wins = int(
            state["wins"]
        )

        self.losses = int(
            state["losses"]
        )

        self.total_profit = float(
            state["total_profit"]
        )

        self.best_trade = float(
            state["best_trade"]
        )

        self.worst_trade = float(
            state["worst_trade"]
        )

        self.model_updates = int(
            state["model_updates"]
        )

        self.scaler = None

        self.model = None

        self.ready = False

        self.load_model()


    @property
    def winrate(self):

        if self.total_trades == 0:

            return 0

        return (
            self.wins
            / self.total_trades
            * 100
        )


    def load_model(self):

        try:

            if (
                os.path.exists(
                    MODEL_FILE
                )
                and
                os.path.exists(
                    SCALER_FILE
                )
            ):

                self.model = joblib.load(
                    MODEL_FILE
                )

                self.scaler = joblib.load(
                    SCALER_FILE
                )

                self.ready = True

                print(
                    "[LEARNING] model loaded"
                )

        except Exception as e:

            print(
                "[LEARNING] model load error:",
                e
            )

            self.model = None

            self.scaler = None

            self.ready = False


    def features_to_vector(
        self,
        features
    ):

        return [
            float(
                features.get(
                    "change_1m",
                    0
                )
            ),

            float(
                features.get(
                    "change_5m",
                    0
                )
            ),

            float(
                features.get(
                    "change_15m",
                    0
                )
            ),

            float(
                features.get(
                    "volatility",
                    0
                )
            ),

            float(
                features.get(
                    "spread",
                    0
                )
            ),

            float(
                features.get(
                    "momentum",
                    0
                )
            ),

            float(
                features.get(
                    "rsi",
                    50
                )
            ),

            float(
                features.get(
                    "base_score",
                    50
                )
            ),

            float(
                features.get(
                    "hour_sin",
                    0
                )
            ),

            float(
                features.get(
                    "hour_cos",
                    0
                )
            )
        ]


    def train(self):

        if db is None:
            return False

        rows = db.execute("""
            SELECT
                entry_features,
                profit
            FROM trades
            WHERE entry_features IS NOT NULL
            ORDER BY id ASC
        """).fetchall()


        if len(rows) < 10:

            return False


        X = []

        y = []


        for row in rows:

            try:

                features = json.loads(
                    row["entry_features"]
                )

                X.append(
                    self.features_to_vector(
                        features
                    )
                )

                y.append(
                    1
                    if float(row["profit"]) > 0
                    else 0
                )

            except Exception:

                continue


        if len(X) < 10:

            return False


        if len(set(y)) < 2:

            return False


        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=int
        )


        self.scaler = StandardScaler()

        X_scaled = self.scaler.fit_transform(
            X
        )


        self.model = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=0.0005,
            max_iter=1000,
            random_state=42,
            class_weight="balanced"
        )


        self.model.fit(
            X_scaled,
            y
        )


        joblib.dump(
            self.model,
            MODEL_FILE
        )

        joblib.dump(
            self.scaler,
            SCALER_FILE
        )


        self.ready = True

        self.model_updates += 1


        save_learning_state(
            self.total_trades,
            self.wins,
            self.losses,
            self.total_profit,
            self.best_trade,
            self.worst_trade,
            self.model_updates
        )


        print(
            "[LEARNING] model updated"
        )

        return True


    def predict_probability(
        self,
        features
    ):

        if not self.ready:

            return 0.5


        try:

            vector = np.asarray(
                [
                    self.features_to_vector(
                        features
                    )
                ],
                dtype=float
            )


            scaled = self.scaler.transform(
                vector
            )


            probabilities = (
                self.model.predict_proba(
                    scaled
                )[0]
            )


            classes = list(
                self.model.classes_
            )


            if 1 in classes:

                index = classes.index(1)

                return float(
                    probabilities[index]
                )

        except Exception as e:

            print(
                "[LEARNING] prediction error:",
                e
            )


        return 0.5


    def record_result(
        self,
        profit
    ):

        self.total_trades += 1

        self.total_profit += profit


        if profit > 0:

            self.wins += 1

            self.best_trade = max(
                self.best_trade,
                profit
            )

        else:

            self.losses += 1

            self.worst_trade = min(
                self.worst_trade,
                profit
            )


        save_learning_state(
            self.total_trades,
            self.wins,
            self.losses,
            self.total_profit,
            self.best_trade,
            self.worst_trade,
            self.model_updates
        )


        # Переобучаем модель после сделки.

        try:

            self.train()

        except Exception as e:

            print(
                "[LEARNING] training error:",
                e
            )


# ============================================================
# GLOBAL WALLET STATE
# ============================================================

wallet_state = {
    "usdt": START_BALANCE,
    "realized_profit": 0.0,
    "total_fees": 0.0
}

wallet_usdt = START_BALANCE

wallet_realized_profit = 0.0

wallet_total_fees = 0.0

positions = {}

learning = None


# ============================================================
# MARKET DATA
# ============================================================

@dataclass
class MarketData:

    symbol: str

    exchange: str

    price: float

    bid: float

    ask: float

    volume: float

    timestamp: float

    change_1m: float = 0

    change_5m: float = 0

    change_15m: float = 0

    volatility: float = 0

    spread: float = 0

    momentum: float = 0

    rsi: float = 50

    base_score: float = 50

    ml_probability: float = 0

    final_score: float = 50


# ============================================================
# EXCHANGES
# ============================================================

async def initialize_exchanges():

    for exchange_id in EXCHANGE_IDS:

        try:

            cls = getattr(
                ccxt,
                exchange_id
            )

            exchange = cls({
                "enableRateLimit": True,
                "timeout": 10000,
            })


            await exchange.load_markets()


            exchanges[
                exchange_id
            ] = exchange


            print(
                f"[EXCHANGE] {exchange_id}: OK"
            )


        except Exception as e:

            print(
                f"[EXCHANGE] {exchange_id}: SKIP - {e}"
            )


# ============================================================
# MARKET SYMBOL
# ============================================================

def symbol_exists(
    exchange,
    symbol
):

    market = exchange.markets.get(
        symbol
    )

    if not market:

        return False

    if market.get(
        "spot"
    ) is False:

        return False

    if market.get(
        "active"
    ) is False:

        return False

    return True


# ============================================================
# FETCH MARKET
# ============================================================

async def fetch_exchange_tickers(
    exchange_id
):

    exchange = exchanges.get(
        exchange_id
    )

    if not exchange:

        return {}


    available = [
        symbol
        for symbol in SYMBOLS
        if symbol_exists(
            exchange,
            symbol
        )
    ]


    if not available:

        return {}


    try:

        if exchange.has.get(
            "fetchTickers"
        ):

            try:

                tickers = await exchange.fetch_tickers(
                    available
                )

            except Exception:

                tickers = await exchange.fetch_tickers()

        else:

            tickers = {}

            semaphore = asyncio.Semaphore(
                5
            )


            async def fetch_one(
                symbol
            ):

                async with semaphore:

                    try:

                        return (
                            symbol,
                            await exchange.fetch_ticker(
                                symbol
                            )
                        )

                    except Exception:

                        return (
                            symbol,
                            None
                        )


            responses = await asyncio.gather(
                *[
                    fetch_one(symbol)
                    for symbol in available
                ]
            )


            for symbol, ticker in responses:

                if ticker:

                    tickers[
                        symbol
                    ] = ticker


        result = {}


        for symbol, ticker in tickers.items():

            if symbol not in available:

                continue

            if not ticker:

                continue


            last = ticker.get(
                "last"
            )

            bid = ticker.get(
                "bid"
            )

            ask = ticker.get(
                "ask"
            )


            if not last:

                continue


            if not bid:

                bid = last


            if not ask:

                ask = last


            try:

                last = float(last)

                bid = float(bid)

                ask = float(ask)

            except Exception:

                continue


            if last <= 0:

                continue


            result[
                symbol
            ] = MarketData(
                symbol=symbol,
                exchange=exchange_id.upper(),
                price=last,
                bid=bid,
                ask=ask,
                volume=float(
                    ticker.get(
                        "quoteVolume"
                    )
                    or 0
                ),
                timestamp=time.time()
            )


        return result


    except Exception as e:

        print(
            f"[MARKET] {exchange_id}: {e}"
        )

        return {}


# ============================================================
# FETCH ALL MARKET DATA
# ============================================================

async def fetch_market():

    result = {}


    tasks = [
        fetch_exchange_tickers(
            exchange_id
        )
        for exchange_id in exchanges
    ]


    if not tasks:

        return {}


    responses = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )


    for response in responses:

        if not isinstance(
            response,
            dict
        ):

            continue


        for symbol, data in response.items():

            result[
                (
                    symbol,
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


# ============================================================
# HISTORICAL CHANGE
# ============================================================

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

    current = history[-1][1]

    old = history[0][1]


    for timestamp, price in history:

        if timestamp <= target:

            old = price


    if old <= 0:

        return 0


    return (
        current - old
    ) / old * 100


# ============================================================
# VOLATILITY
# ============================================================

def volatility(
    symbol,
    exchange
):

    history = price_history.get(
        (
            symbol,
            exchange
        ),
        []
    )


    if len(history) < 5:

        return 0


    prices = [
        x[1]
        for x in history[-30:]
    ]


    avg = (
        sum(prices)
        / len(prices)
    )


    if avg <= 0:

        return 0


    variance = sum(
        (
            p - avg
        ) ** 2
        for p in prices
    ) / len(prices)


    return (
        math.sqrt(
            variance
        )
        / avg
        * 100
    )


# ============================================================
# RSI
# ============================================================

def calculate_rsi(
    symbol,
    exchange
):

    history = price_history.get(
        (
            symbol,
            exchange
        ),
        []
    )


    if len(history) < 8:

        return 50


    prices = [
        x[1]
        for x in history[-30:]
    ]


    gains = []

    losses = []


    for i in range(
        1,
        len(prices)
    ):

        diff = (
            prices[i]
            - prices[i - 1]
        )


        if diff >= 0:

            gains.append(
                diff
            )

            losses.append(0)

        else:

            gains.append(0)

            losses.append(
                abs(diff)
            )


    if not gains:

        return 50


    avg_gain = (
        sum(gains[-14:])
        / min(
            len(gains),
            14
        )
    )


    avg_loss = (
        sum(losses[-14:])
        / min(
            len(losses),
            14
        )
    )


    if avg_loss == 0:

        return 100


    rs = (
        avg_gain
        / avg_loss
    )


    return (
        100
        - 100 / (
            1 + rs
        )
    )


# ============================================================
# FEATURES
# ============================================================

def calculate_features(
    data
):

    data.change_1m = historical_change(
        data.symbol,
        data.exchange,
        60
    )

    data.change_5m = historical_change(
        data.symbol,
        data.exchange,
        300
    )

    data.change_15m = historical_change(
        data.symbol,
        data.exchange,
        900
    )

    data.volatility = volatility(
        data.symbol,
        data.exchange
    )

    data.rsi = calculate_rsi(
        data.symbol,
        data.exchange
    )


    if data.ask > 0:

        data.spread = (
            data.ask
            - data.bid
        ) / data.ask * 100


    data.momentum = (
        data.change_1m * 0.5
        +
        data.change_5m * 0.3
        +
        data.change_15m * 0.2
    )


    # ========================================================
    # BASE SCORE
    # ========================================================

    score = 50.0


    # Momentum

    if data.momentum > 0:

        score += min(
            data.momentum * 5,
            15
        )

    else:

        score += max(
            data.momentum * 4,
            -15
        )


    # RSI

    if 45 <= data.rsi <= 65:

        score += 5

    elif 30 <= data.rsi < 45:

        score += 8

    elif data.rsi > 75:

        score -= 8

    elif data.rsi < 20:

        score -= 4


    # Volatility

    if (
        0.1
        <= data.volatility
        <= 3
    ):

        score += 5

    elif data.volatility > 6:

        score -= 10


    # Spread

    if data.spread < 0.15:

        score += 5

    elif data.spread > 1:

        score -= 5


    data.base_score = max(
        0,
        min(
            100,
            score
        )
    )


    features = {
        "change_1m":
            data.change_1m,

        "change_5m":
            data.change_5m,

        "change_15m":
            data.change_15m,

        "volatility":
            data.volatility,

        "spread":
            data.spread,

        "momentum":
            data.momentum,

        "rsi":
            data.rsi,

        "base_score":
            data.base_score,

        "hour_sin":
            math.sin(
                datetime.now().hour
                / 24
                * 2
                * math.pi
            ),

        "hour_cos":
            math.cos(
                datetime.now().hour
                / 24
                * 2
                * math.pi
            )
    }


    return features


# ============================================================
# FINAL SCORE
# ============================================================

def calculate_final_score(
    data
):

    features = calculate_features(
        data
    )


    # На старте learning может быть None.
    # В таком случае используем обычный score.

    if learning is None:

        ml_probability = 0.5

    else:

        ml_probability = (
            learning.predict_probability(
                features
            )
        )


    data.ml_probability = (
        ml_probability
    )


    if (
        learning is not None
        and
        learning.ready
    ):

        ml_bonus = (
            ml_probability
            - 0.5
        ) * 30


        data.final_score = max(
            0,
            min(
                100,
                data.base_score
                + ml_bonus
            )
        )

    else:

        data.final_score = (
            data.base_score
        )


    return features


# ============================================================
# BEST MARKET
# ============================================================

def get_best_asset(
    symbol
):

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

        return None, None


    best = None

    best_features = None


    for data in candidates:

        features = calculate_final_score(
            data
        )


        if (
            best is None
            or data.final_score
            > best.final_score
        ):

            best = data

            best_features = features


    return best, best_features


# ============================================================
# TRADE SIZE
# ============================================================

def calculate_trade_size(
    score
):

    available = wallet_usdt


    if available < MIN_TRADE_USDT:

        return 0


    confidence = (
        score
        - MIN_TRADE_SCORE
    ) / (
        100
        - MIN_TRADE_SCORE
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
        +
        confidence
        * (
            MAX_TRADE_PERCENT
            - 8
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
# BUY
# ============================================================

async def buy_asset(
    data,
    features,
    amount_usdt
):

    global wallet_usdt
    global wallet_total_fees


    if data.symbol in positions:

        return False


    if amount_usdt <= 0:

        return False


    if amount_usdt > wallet_usdt:

        return False


    if data.ask <= 0:

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


    if total > wallet_usdt:

        return False


    amount_crypto = (
        amount_usdt
        / data.ask
    )


    wallet_usdt -= total


    wallet_total_fees += fee


    save_wallet_state(
        wallet_usdt,
        wallet_realized_profit,
        wallet_total_fees
    )


    position = Position(
        symbol=data.symbol,
        exchange=data.exchange,
        amount=amount_crypto,
        entry_price=data.ask,
        invested=amount_usdt,
        opened_at=time.time(),
        entry_score=data.final_score,
        ml_probability=data.ml_probability,
        features=features
    )


    positions[
        data.symbol
    ] = position


    save_position(
        position
    )


    text = (
        "🟢 <b>БОТ КУПИЛ КРИПТОВАЛЮТУ</b>\n\n"

        f"🪙 <b>{data.symbol}</b>\n"

        f"🏦 Биржа: "
        f"<b>{data.exchange}</b>\n"

        f"💵 Сумма: "
        f"<b>${amount_usdt:.2f}</b>\n"

        f"📦 Количество: "
        f"<code>{amount_crypto:.10f}</code>\n"

        f"💰 Цена: "
        f"<b>{format_price(data.ask)}</b>\n\n"

        f"🧠 Score: "
        f"<b>{data.final_score:.2f}/100</b>\n"

        f"🤖 ML вероятность: "
        f"<b>{data.ml_probability * 100:.1f}%</b>\n"

        f"📈 1m: "
        f"<b>{data.change_1m:+.3f}%</b>\n"

        f"📈 5m: "
        f"<b>{data.change_5m:+.3f}%</b>\n"

        f"📈 15m: "
        f"<b>{data.change_15m:+.3f}%</b>\n"

        f"📊 RSI: "
        f"<b>{data.rsi:.1f}</b>\n"

        f"📊 Volatility: "
        f"<b>{data.volatility:.3f}%</b>\n\n"

        f"👛 USDT осталось: "
        f"<b>${wallet_usdt:.2f}</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER MODE"
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

    global wallet_usdt
    global wallet_realized_profit
    global wallet_total_fees


    position = positions.get(
        data.symbol
    )


    if not position:

        return False


    if data.bid <= 0:

        return False


    gross = (
        position.amount
        * data.bid
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


    profit_percent = (
        profit
        / position.invested
        * 100
    )


    wallet_usdt += received

    wallet_realized_profit += profit

    wallet_total_fees += fee


    save_wallet_state(
        wallet_usdt,
        wallet_realized_profit,
        wallet_total_fees
    )


    save_trade(
        position=position,
        exit_price=data.bid,
        profit=profit,
        profit_percent=profit_percent,
        reason=reason,
        exit_score=data.final_score
    )


    delete_position(
        data.symbol
    )


    del positions[
        data.symbol
    ]


    if learning is not None:

        learning.record_result(
            profit
        )


    emoji = (
        "🟢"
        if profit >= 0
        else "🔴"
    )


    learning_trades = (
        learning.total_trades
        if learning is not None
        else 0
    )

    learning_winrate = (
        learning.winrate
        if learning is not None
        else 0
    )


    text = (
        f"{emoji} <b>БОТ ПРОДАЛ КРИПТОВАЛЮТУ</b>\n\n"

        f"🪙 <b>{data.symbol}</b>\n"

        f"🏦 Биржа: "
        f"<b>{position.exchange}</b>\n\n"

        f"💰 Покупка: "
        f"<b>{format_price(position.entry_price)}</b>\n"

        f"💵 Продажа: "
        f"<b>{format_price(data.bid)}</b>\n\n"

        f"📦 Количество: "
        f"<code>{position.amount:.10f}</code>\n"

        f"💸 Результат: "
        f"<b>{profit:+.2f}$</b>\n"

        f"📊 ROI сделки: "
        f"<b>{profit_percent:+.2f}%</b>\n\n"

        f"📌 Причина: "
        f"<b>{reason}</b>\n\n"

        f"👛 Баланс: "
        f"<b>${wallet_usdt:.2f}</b>\n"

        f"🧠 Сделок в памяти: "
        f"<b>{learning_trades}</b>\n"

        f"🎯 Winrate: "
        f"<b>{learning_winrate:.1f}%</b>\n\n"

        f"⏰ {now_string()}\n"

        "🟡 PAPER MODE"
    )


    await send_event(
        text
    )


    return True


# ============================================================
# SELL DECISION
# ============================================================

def should_sell(
    position,
    data
):

    current_price = data.price


    if position.entry_price <= 0:

        return False, ""


    pnl = (
        current_price
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


    # ML SIGNAL

    if (
        learning is not None
        and learning.ready
        and data.ml_probability < 0.35
        and pnl > 0
    ):

        return True, "ML SIGNAL WEAKENED"


    # SCORE

    if data.final_score < 40:

        return True, "MARKET SCORE DROPPED"


    # MAX HOLD

    if holding_minutes >= MAX_HOLD_MINUTES:

        return True, "MAX HOLD TIME"


    return False, ""


# ============================================================
# AUTO TRADING
# ============================================================

async def trading_loop(
    chat_id
):

    global auto_running


    while auto_running:

        try:

            market_ok = await update_market()


            if not market_ok:

                await show_dashboard(
                    chat_id,
                    "⚠️ <b>Не удалось получить рынок</b>\n\n"
                    "Повторяю попытку..."
                )

                await asyncio.sleep(
                    SCAN_INTERVAL
                )

                continue


            # =================================================
            # SELL
            # =================================================

            for symbol in list(
                positions.keys()
            ):

                position = positions.get(
                    symbol
                )

                if not position:

                    continue


                data, features = get_best_asset(
                    symbol
                )


                if not data:

                    continue


                sell, reason = should_sell(
                    position,
                    data
                )


                if sell:

                    await sell_asset(
                        data,
                        reason
                    )


            # =================================================
            # BUY
            # =================================================

            candidates = []


            for symbol in SYMBOLS:

                if symbol in positions:

                    continue


                data, features = get_best_asset(
                    symbol
                )


                if not data:

                    continue


                if data.final_score < MIN_TRADE_SCORE:

                    continue


                if (
                    learning is not None
                    and learning.ready
                    and
                    data.ml_probability
                    < MIN_ML_PROBABILITY
                ):

                    continue


                candidates.append(
                    (
                        data,
                        features
                    )
                )


            if candidates:

                candidates.sort(
                    key=lambda x: (
                        x[0].final_score
                    ),
                    reverse=True
                )


                best, features = candidates[0]


                amount = calculate_trade_size(
                    best.final_score
                )


                if amount >= MIN_TRADE_USDT:

                    await buy_asset(
                        best,
                        features,
                        amount
                    )


            await show_dashboard(
                chat_id
            )


        except asyncio.CancelledError:

            raise


        except Exception as e:

            print(
                "[TRADING ERROR]",
                repr(e)
            )


            try:

                await show_dashboard(
                    chat_id,
                    "⚠️ <b>Ошибка торгового цикла</b>\n\n"
                    f"<code>{str(e)[:1000]}</code>\n\n"
                    "Бот продолжит работу."
                )

            except Exception:

                pass


        await asyncio.sleep(
            SCAN_INTERVAL
        )


# ============================================================
# UPDATE MARKET
# ============================================================

async def update_market():

    global latest_market
    global market_updates


    market = await fetch_market()


    if not market:

        return False


    latest_market = market


    update_price_history(
        market
    )


    for data in latest_market.values():

        try:

            calculate_final_score(
                data
            )

        except Exception as e:

            print(
                "[FEATURE ERROR]",
                e
            )


    market_updates += 1


    return True


# ============================================================
# PORTFOLIO VALUE
# ============================================================

def portfolio_value():

    value = wallet_usdt


    for symbol, position in positions.items():

        candidates = [
            data
            for (
                s,
                exchange
            ), data
            in latest_market.items()
            if s == symbol
        ]


        if candidates:

            # Используем медиану,
            # чтобы не завышать стоимость портфеля.

            prices = sorted(
                data.price
                for data in candidates
                if data.price > 0
            )


            if prices:

                middle = len(prices) // 2

                if len(prices) % 2:

                    price = prices[middle]

                else:

                    price = (
                        prices[middle - 1]
                        + prices[middle]
                    ) / 2

            else:

                price = position.entry_price

        else:

            price = position.entry_price


        value += (
            position.amount
            * price
        )


    return value


# ============================================================
# DASHBOARD TEXT
# ============================================================

def dashboard_text():

    equity = portfolio_value()


    pnl = (
        equity
        - START_BALANCE
    )


    roi = (
        pnl
        / START_BALANCE
        * 100
    )


    total_trades = (
        learning.total_trades
        if learning is not None
        else 0
    )


    winrate = (
        learning.winrate
        if learning is not None
        else 0
    )


    total_profit = (
        learning.total_profit
        if learning is not None
        else wallet_realized_profit
    )


    model_ready = (
        learning.ready
        if learning is not None
        else False
    )


    model_updates = (
        learning.model_updates
        if learning is not None
        else 0
    )


    text = (
        "🤖 <b>AUTONOMOUS CRYPTO TRADER</b>\n\n"

        f"💵 USDT: "
        f"<b>${wallet_usdt:.2f}</b>\n"

        f"👛 Портфель: "
        f"<b>${equity:.2f}</b>\n"

        f"📈 P/L: "
        f"<b>{pnl:+.2f}$ "
        f"({roi:+.2f}%)</b>\n\n"

        f"🪙 Монет: "
        f"<b>{len(SYMBOLS)}</b>\n"

        f"🏦 Бирж активно: "
        f"<b>{len(exchanges)}</b>\n"

        f"📡 Данных: "
        f"<b>{len(latest_market)}</b>\n"

        f"🔄 Обновлений: "
        f"<b>{market_updates}</b>\n\n"

        f"📊 Сделок: "
        f"<b>{total_trades}</b>\n"

        f"🎯 Winrate: "
        f"<b>{winrate:.1f}%</b>\n"

        f"💰 Реализовано: "
        f"<b>{total_profit:+.2f}$</b>\n\n"

        f"🧠 ML модель: "
        f"<b>{'🟢 ACTIVE' if model_ready else '🟡 Сбор данных'}</b>\n"

        f"🧠 Обновлений модели: "
        f"<b>{model_updates}</b>\n\n"

        f"⚙️ Автотрейдинг: "
        f"<b>{'🟢 ON' if auto_running else '🔴 OFF'}</b>\n\n"

        "🟡 <b>PAPER MODE</b>\n"
        "Реальные ордера отключены."
    )


    return text


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

        ("📜 Сделки", "history"),
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

    if bot is None:

        return


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

        error_text = str(e).lower()


        # Сообщение могло быть удалено.
        # В таком случае создаём новое.

        if (
            "message to edit not found"
            in error_text
            or
            "message can't be edited"
            in error_text
        ):

            try:

                message = await bot.send_message(
                    chat_id,
                    text,
                    reply_markup=keyboard()
                )

                dashboard_messages[
                    chat_id
                ] = message.message_id

            except Exception as send_error:

                print(
                    "[DASHBOARD SEND]",
                    send_error
                )

        else:

            print(
                "[DASHBOARD EDIT]",
                e
            )


# ============================================================
# TELEGRAM EVENTS
# ============================================================

async def send_event(
    text
):

    if bot is None:

        return


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
                "[TELEGRAM EVENT]",
                e
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

    # Не удаляем /start сразу:
    # сначала убеждаемся, что dashboard отправился.

    try:

        await show_dashboard(
            message.chat.id
        )

    except Exception as e:

        print(
            "[START ERROR]",
            repr(e)
        )

        try:

            await message.answer(
                "⚠️ Ошибка запуска панели:\n"
                f"<code>{str(e)[:1000]}</code>"
            )

        except Exception:

            pass

        return


    try:

        await message.delete()

    except Exception:

        pass


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


    try:

        await update_market()

    except Exception as e:

        print(
            "[REFRESH ERROR]",
            e
        )


    await show_dashboard(
        callback.message.chat.id
    )


# ============================================================
# START AUTO
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
            + dashboard_text()
        )

        return


    auto_running = True


    await show_dashboard(
        callback.message.chat.id,
        "🟢 <b>АВТОТРЕЙДИНГ ЗАПУЩЕН</b>\n\n"
        "Начинаю анализ 50 активов.\n"
        "Режим: PAPER."
    )


    auto_task = asyncio.create_task(
        trading_loop(
            callback.message.chat.id
        )
    )


# ============================================================
# STOP AUTO
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
        callback.message.chat.id
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


    lines = [
        "👛 <b>ВИРТУАЛЬНЫЙ КОШЕЛЁК</b>",
        "",
        f"💵 USDT: "
        f"<b>${wallet_usdt:.2f}</b>",
        ""
    ]


    if not positions:

        lines.append(
            "📭 Открытых позиций нет."
        )

    else:

        for symbol, position in positions.items():

            candidates = [
                data
                for (
                    s,
                    exchange
                ), data
                in latest_market.items()
                if s == symbol
            ]


            if candidates:

                prices = [
                    x.price
                    for x in candidates
                    if x.price > 0
                ]

                if prices:

                    current = (
                        sum(prices)
                        / len(prices)
                    )

                else:

                    current = position.entry_price

            else:

                current = (
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
                f"📊 Current: "
                f"{format_price(current)}\n"
                f"P/L: "
                f"<b>{pnl:+.2f}$ "
                f"({pnl_percent:+.2f}%)</b>\n"
                f"🧠 Entry score: "
                f"{position.entry_score:.1f}\n"
            )


    await show_dashboard(
        callback.message.chat.id,
        "\n".join(lines)
    )


# ============================================================
# MARKET
# ============================================================

@dp.callback_query(
    F.data == "market"
)
async def market(
    callback: CallbackQuery
):

    await callback.answer()


    lines = [
        "📊 <b>РЫНОК</b>",
        ""
    ]


    for symbol in SYMBOLS:

        data, _ = get_best_asset(
            symbol
        )


        if not data:

            continue


        lines.append(
            f"🪙 <b>{symbol}</b>\n"
            f"🏦 {data.exchange}\n"
            f"💰 {format_price(data.price)}\n"
            f"🧠 Score: "
            f"<b>{data.final_score:.1f}</b>\n"
            f"🤖 ML: "
            f"<b>{data.ml_probability * 100:.1f}%</b>\n"
            f"📈 1m: "
            f"{data.change_1m:+.3f}% | "
            f"5m: "
            f"{data.change_5m:+.3f}%\n"
        )


    if len(lines) == 2:

        lines.append(
            "⏳ Данные рынка ещё загружаются."
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


    trades = get_trades()


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
                if trade["profit"] >= 0
                else "🔴"
            )


            lines.append(
                f"{emoji} <b>{trade['symbol']}</b>\n"
                f"🏦 {trade['exchange']}\n"
                f"💰 "
                f"{trade['profit']:+.2f}$ "
                f"({trade['profit_percent']:+.2f}%)\n"
                f"📌 {trade['reason']}\n"
                f"🧠 Score: "
                f"{trade['entry_score']:.1f}\n"
                f"⏰ "
                f"{datetime.fromtimestamp(trade['closed_at']).strftime('%H:%M:%S')}\n"
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


    if learning is None:

        await show_dashboard(
            callback.message.chat.id,
            "🧠 <b>ПАМЯТЬ И ОБУЧЕНИЕ</b>\n\n"
            "Модуль обучения ещё запускается."
        )

        return


    text = (
        "🧠 <b>ПАМЯТЬ И ОБУЧЕНИЕ</b>\n\n"

        f"📊 Сделок: "
        f"<b>{learning.total_trades}</b>\n"

        f"🟢 Побед: "
        f"<b>{learning.wins}</b>\n"

        f"🔴 Убытков: "
        f"<b>{learning.losses}</b>\n"

        f"🎯 Winrate: "
        f"<b>{learning.winrate:.2f}%</b>\n\n"

        f"💰 Результат: "
        f"<b>{learning.total_profit:+.2f}$</b>\n"

        f"🏆 Лучшая: "
        f"<b>{learning.best_trade:+.2f}$</b>\n"

        f"💀 Худшая: "
        f"<b>{learning.worst_trade:+.2f}$</b>\n\n"

        f"🤖 ML модель: "
        f"<b>{'ACTIVE' if learning.ready else 'COLLECTING DATA'}</b>\n"

        f"🧠 Переобучений: "
        f"<b>{learning.model_updates}</b>\n\n"

        "Бот сохраняет признаки ситуации "
        "при покупке и результат после продажи.\n\n"

        "🟢 Удачная сделка → модель получает "
        "пример хорошей ситуации.\n"

        "🔴 Неудачная сделка → модель получает "
        "пример плохой ситуации."
    )


    await show_dashboard(
        callback.message.chat.id,
        text
    )


# ============================================================
# FALLBACK
# ============================================================

@dp.message()
async def fallback(
    message: Message
):

    try:

        await show_dashboard(
            message.chat.id
        )

    except Exception as e:

        print(
            "[FALLBACK ERROR]",
            e
        )


    try:

        await message.delete()

    except Exception:

        pass


# ============================================================
# FORMATTING
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
# MAIN
# ============================================================

async def main():

    global bot
    global learning

    global wallet_state
    global wallet_usdt
    global wallet_realized_profit
    global wallet_total_fees

    global positions

    global auto_running
    global auto_task


    # ========================================================
    # 1. BOT TOKEN
    # ========================================================

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN отсутствует в .env"
        )


    # ========================================================
    # 2. DATABASE FIRST
    # ========================================================

    print(
        "[STARTUP] Initializing database..."
    )

    init_database()


    # ========================================================
    # 3. LOAD WALLET
    # ========================================================

    print(
        "[STARTUP] Loading wallet..."
    )

    wallet_state = load_wallet_state()

    wallet_usdt = wallet_state[
        "usdt"
    ]

    wallet_realized_profit = (
        wallet_state[
            "realized_profit"
        ]
    )

    wallet_total_fees = (
        wallet_state[
            "total_fees"
        ]
    )


    # ========================================================
    # 4. LOAD POSITIONS
    # ========================================================

    print(
        "[STARTUP] Loading positions..."
    )

    positions = load_positions()


    # ========================================================
    # 5. CREATE LEARNING
    # ========================================================

    print(
        "[STARTUP] Loading learning..."
    )

    learning = LearningEngine()


    # ========================================================
    # 6. CREATE TELEGRAM BOT
    # ========================================================

    bot = Bot(
        token=BOT_TOKEN,
        default=DefaultBotProperties(
            parse_mode=ParseMode.HTML
        )
    )


    # ========================================================
    # STARTUP INFO
    # ========================================================

    print("=" * 70)

    print(
        "AUTONOMOUS CRYPTO PAPER TRADER"
    )

    print("=" * 70)

    print(
        "Starting balance:",
        START_BALANCE
    )

    print(
        "Current balance:",
        wallet_usdt
    )

    print(
        "Assets:",
        len(SYMBOLS)
    )

    print(
        "Requested exchanges:",
        len(EXCHANGE_IDS)
    )

    print(
        "Scan interval:",
        SCAN_INTERVAL
    )

    print(
        "ML model:",
        learning.ready
    )

    print(
        "Previous trades:",
        learning.total_trades
    )

    print(
        "Open positions:",
        len(positions)
    )

    print(
        "LIVE TRADING:",
        LIVE_TRADING
    )

    print("=" * 70)


    # ========================================================
    # 7. REMOVE OLD WEBHOOK
    # ========================================================

    try:

        await bot.delete_webhook(
            drop_pending_updates=True
        )

        print(
            "[TELEGRAM] webhook cleared"
        )

    except Exception as e:

        print(
            "[TELEGRAM] webhook error:",
            e
        )


    # ========================================================
    # 8. EXCHANGES
    # ========================================================

    print(
        "[STARTUP] Initializing exchanges..."
    )


    await initialize_exchanges()


    print(
        "[STARTUP] Active exchanges:",
        len(exchanges)
    )


    # ========================================================
    # 9. START TELEGRAM
    # ========================================================

    print(
        "[TELEGRAM] Starting polling..."
    )


    try:

        await dp.start_polling(
            bot
        )


    finally:

        # ====================================================
        # STOP TRADING
        # ====================================================

        auto_running = False


        if auto_task:

            auto_task.cancel()

            try:

                await auto_task

            except asyncio.CancelledError:

                pass

            auto_task = None


        # ====================================================
        # CLOSE EXCHANGES
        # ====================================================

        for exchange in exchanges.values():

            try:

                await exchange.close()

            except Exception as e:

                print(
                    "[EXCHANGE CLOSE]",
                    e
                )


        # ====================================================
        # CLOSE TELEGRAM
        # ====================================================

        if bot:

            try:

                await bot.session.close()

            except Exception:

                pass


        # ====================================================
        # CLOSE DATABASE
        # ====================================================

        if db:

            try:

                db.close()

            except Exception:

                pass


        print(
            "[SHUTDOWN] Bot stopped."
        )


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
            "Bot stopped manually."
        )

    except Exception as e:

        print(
            "=" * 70
        )

        print(
            "FATAL ERROR:"
        )

        print(
            repr(e)
        )

        print(
            "=" * 70
        )

        raise