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

from sklearn.linear_model import SGDClassifier, SGDRegressor
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
    os.getenv("SCAN_INTERVAL_SECONDS", "15")
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

RETURN_MODEL_FILE = os.getenv(
    "RETURN_MODEL_FILE",
    "trader_return_model.pkl"
)

RETURN_SCALER_FILE = os.getenv(
    "RETURN_SCALER_FILE",
    "trader_return_scaler.pkl"
)

MIN_MARKET_AGE_DAYS = int(
    os.getenv(
        "MIN_MARKET_AGE_DAYS",
        "365"
    )
)

MIN_24H_VOLUME_USDT = float(
    os.getenv(
        "MIN_24H_VOLUME_USDT",
        "1000000"
    )
)

MAX_DYNAMIC_SYMBOLS = int(
    os.getenv(
        "MAX_DYNAMIC_SYMBOLS",
        "1500"
    )
)

MAX_OPEN_POSITIONS = int(
    os.getenv(
        "MAX_OPEN_POSITIONS",
        "8"
    )
)

MIN_EXPECTED_MOVE_PERCENT = float(
    os.getenv(
        "MIN_EXPECTED_MOVE_PERCENT",
        "0.45"
    )
)

EARLY_EXIT_LOSS_PERCENT = float(
    os.getenv(
        "EARLY_EXIT_LOSS_PERCENT",
        "1.25"
    )
)

STRONG_ML_PROBABILITY = float(
    os.getenv(
        "STRONG_ML_PROBABILITY",
        "0.62"
    )
)

# Реальные сделки отключены.
LIVE_TRADING = False


# ============================================================
# EXCHANGES
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

db = None

exchanges = {}

dashboard_messages = {}

latest_market = {}

price_history = {}

market_registry = {}

market_age_cache = {}

positions = {}

learning = None

market_updates = 0

auto_running = False

auto_task = None

wallet_state = {
    "usdt": START_BALANCE,
    "realized_profit": 0.0,
    "total_fees": 0.0
}

wallet_usdt = START_BALANCE

wallet_realized_profit = 0.0

wallet_total_fees = 0.0


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

    expected_move: float

    features: dict


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

    change_1m: float = 0.0

    change_5m: float = 0.0

    change_15m: float = 0.0

    change_1h: float = 0.0

    volatility: float = 0.0

    spread: float = 0.0

    momentum: float = 0.0

    rsi: float = 50.0

    trend_strength: float = 0.0

    volume_score: float = 0.0

    base_score: float = 50.0

    ml_probability: float = 0.5

    expected_move: float = 0.0

    final_score: float = 50.0


# ============================================================
# DATABASE
# ============================================================

def init_database():

    global db

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

            model_updates INTEGER NOT NULL DEFAULT 0,

            last_learning_time REAL NOT NULL DEFAULT 0
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
        VALUES (
            1,
            ?,
            0,
            0
        )
    """, (
        START_BALANCE,
    ))

    cursor.execute("""
        INSERT OR IGNORE INTO learning_state (
            id
        )
        VALUES (
            1
        )
    """)

    learning_columns = {
        row["name"]
        for row in cursor.execute(
            "PRAGMA table_info(learning_state)"
        ).fetchall()
    }

    if "last_learning_time" not in learning_columns:

        cursor.execute("""
            ALTER TABLE learning_state
            ADD COLUMN last_learning_time
            REAL NOT NULL DEFAULT 0
        """)

    position_columns = {
        row["name"]
        for row in cursor.execute(
            "PRAGMA table_info(positions)"
        ).fetchall()
    }

    if "expected_move" not in position_columns:

        cursor.execute("""
            ALTER TABLE positions
            ADD COLUMN expected_move
            REAL NOT NULL DEFAULT 0
        """)

    db.commit()

    print(
        "[DATABASE] initialized:",
        DATABASE_FILE
    )


# ============================================================
# WALLET
# ============================================================

def load_wallet_state():

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
            VALUES (
                1,
                ?,
                0,
                0
            )
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
        "usdt": float(
            row["usdt"]
        ),

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
# POSITIONS
# ============================================================

def save_position(
    position
):

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
            expected_move,
            features
        )
        VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
    """, (
        position.symbol,
        position.exchange,
        position.amount,
        position.entry_price,
        position.invested,
        position.opened_at,
        position.entry_score,
        position.ml_probability,
        position.expected_move,
        json.dumps(
            position.features
        )
    ))

    db.commit()


def delete_position(
    symbol
):

    db.execute("""
        DELETE FROM positions
        WHERE symbol = ?
    """, (
        symbol,
    ))

    db.commit()


def load_positions():

    rows = db.execute("""
        SELECT *
        FROM positions
    """).fetchall()

    loaded = {}

    for row in rows:

        try:

            expected_move = 0.0

            if "expected_move" in row.keys():

                expected_move = float(
                    row["expected_move"]
                )

            loaded[
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

                expected_move=expected_move,

                features=json.loads(
                    row["features"]
                )
            )

        except Exception as e:

            print(
                "[DATABASE] position error:",
                e
            )

    return loaded


# ============================================================
# LEARNING DATABASE
# ============================================================

def load_learning_state():

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
            VALUES (
                1
            )
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
    model_updates,
    last_learning_time
):

    db.execute("""
        UPDATE learning_state
        SET
            total_trades = ?,
            wins = ?,
            losses = ?,
            total_profit = ?,
            best_trade = ?,
            worst_trade = ?,
            model_updates = ?,
            last_learning_time = ?
        WHERE id = 1
    """, (
        total_trades,
        wins,
        losses,
        total_profit,
        best_trade,
        worst_trade,
        model_updates,
        last_learning_time
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
        VALUES (
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?,
            ?
        )
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


def get_trades(
    limit=30
):

    return db.execute("""
        SELECT *
        FROM trades
        ORDER BY id DESC
        LIMIT ?
    """, (
        limit,
    )).fetchall()


# ============================================================
# LEARNING ENGINE
# ============================================================

class LearningEngine:

    FEATURE_COUNT = 15

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

        self.last_train = float(
            state["last_learning_time"]
        )

        self.scaler = None

        self.model = None

        self.return_scaler = None

        self.return_model = None

        self.ready = False

        self.return_ready = False

        self.load_models()


    @property
    def winrate(self):

        if self.total_trades <= 0:

            return 0.0

        return (
            self.wins
            /
            self.total_trades
            *
            100
        )


    # ========================================================
    # FEATURES
    # ========================================================

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
                    "change_1h",
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
                    "trend_strength",
                    0
                )
            ),

            float(
                features.get(
                    "volume_score",
                    0
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
            ),

            float(
                features.get(
                    "market_age_years",
                    1
                )
            ),

            float(
                features.get(
                    "entry_score",
                    50
                )
            )
        ]


    # ========================================================
    # LOAD MODELS
    # ========================================================

    def load_models(self):

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
                    "[LEARNING] classifier loaded"
                )

        except Exception as e:

            print(
                "[LEARNING] classifier load error:",
                repr(e)
            )

            self.model = None

            self.scaler = None

            self.ready = False


        try:

            if (
                os.path.exists(
                    RETURN_MODEL_FILE
                )
                and
                os.path.exists(
                    RETURN_SCALER_FILE
                )
            ):

                self.return_model = joblib.load(
                    RETURN_MODEL_FILE
                )

                self.return_scaler = joblib.load(
                    RETURN_SCALER_FILE
                )

                self.return_ready = True

                print(
                    "[LEARNING] return model loaded"
                )

        except Exception as e:

            print(
                "[LEARNING] return model load error:",
                repr(e)
            )

            self.return_model = None

            self.return_scaler = None

            self.return_ready = False


    # ========================================================
    # LOAD ALL TRADES
    # ========================================================

    def load_training_data(self):

        rows = db.execute("""
            SELECT
                entry_features,
                profit,
                profit_percent
            FROM trades
            WHERE
                entry_features IS NOT NULL
            ORDER BY id ASC
        """).fetchall()

        X = []

        y = []

        returns = []

        for row in rows:

            try:

                features = json.loads(
                    row["entry_features"]
                )

                vector = self.features_to_vector(
                    features
                )

                X.append(
                    vector
                )

                profit = float(
                    row["profit"]
                )

                profit_percent = float(
                    row["profit_percent"]
                )

                y.append(
                    1
                    if profit > 0
                    else 0
                )

                returns.append(
                    profit_percent
                )

            except Exception as e:

                print(
                    "[LEARNING] bad training row:",
                    e
                )

        return (
            X,
            y,
            returns
        )


    # ========================================================
    # FULL TRAIN
    # ========================================================

    def train_all(
        self
    ):

        if db is None:

            return False

        X, y, returns = (
            self.load_training_data()
        )

        if len(X) < 10:

            print(
                "[LEARNING] waiting for more trades:",
                len(X),
                "/ 10"
            )

            return False

        X = np.asarray(
            X,
            dtype=float
        )

        y = np.asarray(
            y,
            dtype=int
        )

        returns = np.asarray(
            returns,
            dtype=float
        )

        if len(
            set(
                y.tolist()
            )
        ) < 2:

            print(
                "[LEARNING] need both winning and losing trades"
            )

            return False

        try:

            # ==================================================
            # CLASSIFICATION MODEL
            # ==================================================

            self.scaler = StandardScaler()

            X_scaled = (
                self.scaler.fit_transform(
                    X
                )
            )

            self.model = SGDClassifier(

                loss="log_loss",

                penalty="l2",

                alpha=0.0002,

                max_iter=3000,

                tol=1e-5,

                random_state=42,

                class_weight="balanced",

                average=True
            )

            # Новые сделки получают больший вес,
            # но старые ошибки остаются в истории.
            weights = np.linspace(
                0.75,
                3.0,
                len(y)
            )

            self.model.fit(
                X_scaled,
                y,
                sample_weight=weights
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


            # ==================================================
            # RETURN MODEL
            # ==================================================

            self.return_scaler = (
                StandardScaler()
            )

            X_return = (
                self.return_scaler.fit_transform(
                    X
                )
            )

            self.return_model = SGDRegressor(

                loss="huber",

                penalty="l2",

                alpha=0.0002,

                max_iter=3000,

                tol=1e-5,

                random_state=42,

                average=True
            )

            self.return_model.fit(
                X_return,
                returns,
                sample_weight=weights
            )

            joblib.dump(
                self.return_model,
                RETURN_MODEL_FILE
            )

            joblib.dump(
                self.return_scaler,
                RETURN_SCALER_FILE
            )

            self.return_ready = True

            self.model_updates += 1

            self.last_train = time.time()

            save_learning_state(

                self.total_trades,

                self.wins,

                self.losses,

                self.total_profit,

                self.best_trade,

                self.worst_trade,

                self.model_updates,

                self.last_train
            )

            print(
                "[LEARNING] FULL RETRAIN:",
                self.model_updates,
                "samples:",
                len(X),
                "wins:",
                int(sum(y)),
                "losses:",
                int(len(y) - sum(y))
            )

            return True

        except Exception as e:

            print(
                "[LEARNING] full train error:",
                repr(e)
            )

            return False


    # ========================================================
    # PREDICT WIN PROBABILITY
    # ========================================================

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

            scaled = (
                self.scaler.transform(
                    vector
                )
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
                "[LEARNING] probability error:",
                repr(e)
            )

        return 0.5


    # ========================================================
    # PREDICT EXPECTED RETURN
    # ========================================================

    def predict_expected_return(
        self,
        features
    ):

        if not self.return_ready:

            return 0.0

        try:

            vector = np.asarray(
                [
                    self.features_to_vector(
                        features
                    )
                ],
                dtype=float
            )

            scaled = (
                self.return_scaler.transform(
                    vector
                )
            )

            prediction = (
                self.return_model.predict(
                    scaled
                )[0]
            )

            return float(
                np.clip(
                    prediction,
                    -20,
                    20
                )
            )

        except Exception as e:

            print(
                "[LEARNING] return prediction error:",
                repr(e)
            )

        return 0.0


    # ========================================================
    # RECORD RESULT
    # ========================================================

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

            self.model_updates,

            self.last_train
        )

        print(
            "[LEARNING] result:",
            f"{profit:+.4f}$"
        )

        # После каждой сделки модель заново
        # изучает всю историю.
        self.train_all()


# ============================================================
# MARKET SYMBOL FILTER
# ============================================================

def symbol_is_eligible(
    exchange,
    symbol,
    market
):

    if not market:

        return False

    if market.get(
        "spot"
    ) is not True:

        return False

    if market.get(
        "active"
    ) is False:

        return False

    if market.get(
        "quote"
    ) != "USDT":

        return False

    if ":" in symbol:

        return False

    upper = symbol.upper()

    blocked_fragments = (

        "UP/USDT",
        "DOWN/USDT",
        "BULL/USDT",
        "BEAR/USDT",
        "3L/USDT",
        "3S/USDT",
        "5L/USDT",
        "5S/USDT",
    )

    for fragment in blocked_fragments:

        if upper.endswith(
            fragment
        ):

            return False

    return True


# ============================================================
# MARKET AGE
# ============================================================

def get_created_timestamp(
    market
):

    info = market.get(
        "info",
        {}
    )

    candidates = [

        market.get(
            "created"
        ),

        market.get(
            "listing"
        ),

        market.get(
            "listTime"
        ),

        market.get(
            "onboardDate"
        ),

        info.get(
            "created"
        ),

        info.get(
            "createdAt"
        ),

        info.get(
            "listingTime"
        ),

        info.get(
            "onboardDate"
        ),
    ]

    for value in candidates:

        try:

            if value is None:

                continue

            value = float(
                value
            )

            if value > 10_000_000_000:

                value /= 1000

            if value > 1_000_000_000:

                return value

        except Exception:

            continue

    return None


async def check_market_age(
    exchange,
    symbol,
    market
):

    cache_key = (
        exchange.id,
        symbol
    )

    cached = market_age_cache.get(
        cache_key
    )

    if cached is not None:

        return cached

    now = time.time()

    required_age = (
        MIN_MARKET_AGE_DAYS
        * 86400
    )

    created = get_created_timestamp(
        market
    )

    if created is not None:

        result = (
            now - created
            >= required_age
        )

        market_age_cache[
            cache_key
        ] = result

        return result

    if not exchange.has.get(
        "fetchOHLCV"
    ):

        market_age_cache[
            cache_key
        ] = False

        return False

    try:

        candles = await exchange.fetch_ohlcv(

            symbol,

            timeframe="1d",

            limit=1000
        )

        if not candles:

            market_age_cache[
                cache_key
            ] = False

            return False

        first_timestamp = (
            candles[0][0]
            / 1000
        )

        result = (
            now - first_timestamp
            >= required_age
        )

        market_age_cache[
            cache_key
        ] = result

        return result

    except Exception:

        market_age_cache[
            cache_key
        ] = False

        return False


# ============================================================
# DYNAMIC MARKET DISCOVERY
# ============================================================

async def discover_dynamic_markets():

    discovered = {}

    for exchange_id, exchange in exchanges.items():

        for symbol, market in exchange.markets.items():

            if not symbol_is_eligible(
                exchange,
                symbol,
                market
            ):

                continue

            discovered.setdefault(
                symbol,
                []
            ).append(
                {
                    "exchange": exchange_id,
                    "market": market
                }
            )

    symbols = sorted(

        discovered.keys(),

        key=lambda symbol: (

            -len(
                discovered[
                    symbol
                ]
            ),

            symbol
        )
    )

    symbols = symbols[
        :MAX_DYNAMIC_SYMBOLS
    ]

    market_registry.clear()

    for symbol in symbols:

        market_registry[
            symbol
        ] = discovered[
            symbol
        ]

    print(
        "[DISCOVERY] symbols:",
        len(market_registry)
    )

    return list(
        market_registry.keys()
    )


async def filter_old_markets(
    symbols
):

    old_markets = []

    semaphore = asyncio.Semaphore(
        8
    )

    async def check_one(
        symbol
    ):

        async with semaphore:

            entries = (
                market_registry.get(
                    symbol,
                    []
                )
            )

            for entry in entries:

                exchange = exchanges.get(
                    entry["exchange"]
                )

                if exchange is None:

                    continue

                try:

                    if await check_market_age(
                        exchange,
                        symbol,
                        entry["market"]
                    ):

                        return symbol

                except Exception:

                    continue

            return None

    results = await asyncio.gather(

        *[
            check_one(symbol)
            for symbol in symbols
        ],

        return_exceptions=True
    )

    for result in results:

        if isinstance(
            result,
            str
        ):

            old_markets.append(
                result
            )

    print(
        "[AGE FILTER] old symbols:",
        len(old_markets)
    )

    return old_markets


# ============================================================
# FETCH TICKERS
# ============================================================

async def fetch_exchange_tickers(
    exchange_id,
    symbols
):

    exchange = exchanges.get(
        exchange_id
    )

    if exchange is None:

        return {}

    available = [

        symbol

        for symbol in symbols

        if symbol in exchange.markets

        and symbol_is_eligible(

            exchange,

            symbol,

            exchange.markets.get(
                symbol
            )
        )
    ]

    if not available:

        return {}

    try:

        if exchange.has.get(
            "fetchTickers"
        ):

            try:

                tickers = (
                    await exchange.fetch_tickers(
                        available
                    )
                )

            except Exception:

                tickers = (
                    await exchange.fetch_tickers()
                )

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

            results = await asyncio.gather(

                *[
                    fetch_one(symbol)
                    for symbol in available
                ]
            )

            for symbol, ticker in results:

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

            volume = ticker.get(
                "quoteVolume"
            )

            if not last:

                continue

            try:

                last = float(
                    last
                )

                bid = float(
                    bid or last
                )

                ask = float(
                    ask or last
                )

                volume = float(
                    volume or 0
                )

            except Exception:

                continue

            if last <= 0:

                continue

            if bid <= 0:

                continue

            if ask <= 0:

                continue

            if volume < MIN_24H_VOLUME_USDT:

                continue

            result[
                symbol
            ] = MarketData(

                symbol=symbol,

                exchange=exchange_id.upper(),

                price=last,

                bid=bid,

                ask=ask,

                volume=volume,

                timestamp=time.time()
            )

        return result

    except Exception as e:

        print(
            "[TICKER ERROR]",
            exchange_id,
            repr(e)
        )

        return {}


# ============================================================
# FETCH MARKET
# ============================================================

async def fetch_market():

    if not market_registry:

        return {}

    symbols = list(
        market_registry.keys()
    )

    tasks = [

        fetch_exchange_tickers(

            exchange_id,

            symbols
        )

        for exchange_id in exchanges
    ]

    responses = await asyncio.gather(

        *tasks,

        return_exceptions=True
    )

    result = {}

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

            price_history[
                key
            ] = []

        price_history[
            key
        ].append(
            (
                now,
                data.price
            )
        )

        cutoff = (
            now - 7200
        )

        price_history[
            key
        ] = [

            item

            for item
            in price_history[key]

            if item[0] >= cutoff
        ]


def historical_change(
    symbol,
    exchange,
    seconds
):

    history = price_history.get(

        (
            symbol,
            exchange
        ),

        []
    )

    if len(history) < 2:

        return 0.0

    now = time.time()

    target = (
        now - seconds
    )

    current = history[-1][1]

    old = history[0][1]

    for timestamp, price in history:

        if timestamp <= target:

            old = price

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

    history = price_history.get(

        (
            symbol,
            exchange
        ),

        []
    )

    if len(history) < 5:

        return 0.0

    prices = [

        x[1]

        for x
        in history[-60:]
    ]

    avg = (
        sum(prices)
        /
        len(prices)
    )

    if avg <= 0:

        return 0.0

    variance = (

        sum(

            (
                price - avg
            ) ** 2

            for price
            in prices

        )

        /

        len(prices)
    )

    return (

        math.sqrt(
            variance
        )

        /

        avg

        *

        100
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

        return 50.0

    prices = [

        x[1]

        for x
        in history[-60:]
    ]

    gains = []

    losses = []

    for i in range(
        1,
        len(prices)
    ):

        diff = (

            prices[i]

            -

            prices[i - 1]
        )

        if diff >= 0:

            gains.append(
                diff
            )

            losses.append(
                0
            )

        else:

            gains.append(
                0
            )

            losses.append(
                abs(diff)
            )

    if not gains:

        return 50.0

    period = min(
        14,
        len(gains)
    )

    avg_gain = (

        sum(
            gains[-period:]
        )

        /

        period
    )

    avg_loss = (

        sum(
            losses[-period:]
        )

        /

        period
    )

    if avg_loss == 0:

        return 100.0

    rs = (
        avg_gain
        /
        avg_loss
    )

    return (
        100
        -
        100
        /
        (
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

    data.change_1h = historical_change(

        data.symbol,

        data.exchange,

        3600
    )

    data.volatility = calculate_volatility(

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

            -

            data.bid
        ) / data.ask * 100

    data.momentum = (

        data.change_1m * 0.35

        +

        data.change_5m * 0.30

        +

        data.change_15m * 0.20

        +

        data.change_1h * 0.15
    )

    data.trend_strength = (

        max(
            data.change_5m,
            0
        )

        +

        max(
            data.change_15m,
            0
        )

        +

        max(
            data.change_1h,
            0
        )
    )

    volume_log = math.log10(
        max(
            data.volume,
            1
        )
    )

    data.volume_score = float(
        np.clip(
            (
                volume_log
                -
                math.log10(
                    MIN_24H_VOLUME_USDT
                )
            )
            * 8,
            0,
            15
        )
    )

    score = 50.0

    score += float(
        np.clip(
            data.momentum * 4.5,
            -18,
            18
        )
    )

    score += float(
        np.clip(
            data.change_15m * 1.5,
            -8,
            8
        )
    )

    score += float(
        np.clip(
            data.change_1h * 0.8,
            -8,
            8
        )
    )

    if 42 <= data.rsi <= 68:

        score += 5

    elif 30 <= data.rsi < 42:

        score += 7

    elif 68 < data.rsi <= 75:

        score += 2

    elif data.rsi > 78:

        score -= 9

    elif data.rsi < 20:

        score -= 3

    if (
        0.10
        <= data.volatility
        <= 3.0
    ):

        score += 5

    elif (
        3.0
        <
        data.volatility
        <= 6
    ):

        score += 1

    elif data.volatility > 6:

        score -= 10

    if data.spread < 0.10:

        score += 5

    elif data.spread < 0.25:

        score += 2

    elif data.spread > 1:

        score -= 8

    score += data.volume_score

    data.base_score = float(
        np.clip(
            score,
            0,
            100
        )
    )

    now_hour = datetime.now().hour

    market_entries = market_registry.get(
        data.symbol,
        []
    )

    market_age_years = 1.0

    for entry in market_entries:

        created = get_created_timestamp(
            entry["market"]
        )

        if created:

            age = (
                time.time()
                -
                created
            ) / (
                365.25
                *
                86400
            )

            market_age_years = max(
                market_age_years,
                age
            )

    features = {

        "change_1m":
            data.change_1m,

        "change_5m":
            data.change_5m,

        "change_15m":
            data.change_15m,

        "change_1h":
            data.change_1h,

        "volatility":
            data.volatility,

        "spread":
            data.spread,

        "momentum":
            data.momentum,

        "rsi":
            data.rsi,

        "trend_strength":
            data.trend_strength,

        "volume_score":
            data.volume_score,

        "base_score":
            data.base_score,

        "hour_sin":
            math.sin(
                now_hour
                /
                24
                *
                2
                *
                math.pi
            ),

        "hour_cos":
            math.cos(
                now_hour
                /
                24
                *
                2
                *
                math.pi
            ),

        "market_age_years":
            market_age_years,

        "entry_score":
            data.base_score
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

    if learning is None:

        probability = 0.5

        expected = 0.0

    else:

        probability = (
            learning.predict_probability(
                features
            )
        )

        expected = (
            learning.predict_expected_return(
                features
            )
        )

    data.ml_probability = probability

    data.expected_move = expected

    if (
        learning is not None
        and
        learning.ready
    ):

        ml_bonus = (

            probability
            -
            0.5
        ) * 35

        expected_bonus = float(
            np.clip(
                expected * 3,
                -12,
                12
            )
        )

        data.final_score = float(
            np.clip(
                data.base_score
                +
                ml_bonus
                +
                expected_bonus,
                0,
                100
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

        return (
            None,
            None
        )

    best = None

    best_features = None

    for data in candidates:

        features = calculate_final_score(
            data
        )

        if (
            best is None
            or
            data.final_score
            >
            best.final_score
        ):

            best = data

            best_features = features

    return (
        best,
        best_features
    )


# ============================================================
# TRADE SIZE
# ============================================================

def calculate_trade_size(
    score,
    expected_move,
    probability
):

    available = wallet_usdt

    if available < MIN_TRADE_USDT:

        return 0.0

    score_confidence = (

        score
        -
        MIN_TRADE_SCORE
    ) / max(
        1,
        100
        -
        MIN_TRADE_SCORE
    )

    score_confidence = float(
        np.clip(
            score_confidence,
            0,
            1
        )
    )

    ml_confidence = (

        probability
        -
        MIN_ML_PROBABILITY
    ) / max(
        0.01,
        1
        -
        MIN_ML_PROBABILITY
    )

    ml_confidence = float(
        np.clip(
            ml_confidence,
            0,
            1
        )
    )

    return_confidence = float(
        np.clip(
            expected_move / 3.0,
            0,
            1
        )
    )

    confidence = (

        score_confidence * 0.50

        +

        ml_confidence * 0.30

        +

        return_confidence * 0.20
    )

    percent = (

        6

        +

        confidence
        *
        (
            MAX_TRADE_PERCENT
            -
            6
        )
    )

    amount = (
        available
        *
        percent
        /
        100
    )

    amount = max(
        amount,
        MIN_TRADE_USDT
    )

    amount = min(
        amount,
        available
        *
        MAX_TRADE_PERCENT
        /
        100
    )

    return float(
        amount
    )


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

    if len(
        positions
    ) >= MAX_OPEN_POSITIONS:

        return False

    if amount_usdt <= 0:

        return False

    if amount_usdt > wallet_usdt:

        return False

    if data.ask <= 0:

        return False

    if (
        learning is not None
        and
        learning.ready
        and
        data.ml_probability
        <
        MIN_ML_PROBABILITY
    ):

        return False

    if (
        learning is not None
        and
        learning.return_ready
        and
        data.expected_move
        <
        MIN_EXPECTED_MOVE_PERCENT
    ):

        return False

    movement = max(

        abs(
            data.change_5m
        ),

        abs(
            data.change_15m
        ),

        abs(
            data.change_1h
        )
    )

    if movement < 0.15:

        return False

    fee = (

        amount_usdt
        *
        FEE_PERCENT
        /
        100
    )

    total = (
        amount_usdt
        +
        fee
    )

    if total > wallet_usdt:

        return False

    amount_crypto = (

        amount_usdt
        /
        data.ask
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

        expected_move=data.expected_move,

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

        f"🤖 Вероятность успеха: "
        f"<b>{data.ml_probability * 100:.1f}%</b>\n"

        f"📈 Ожидаемый ROI: "
        f"<b>{data.expected_move:+.2f}%</b>\n\n"

        f"📈 1m: "
        f"<b>{data.change_1m:+.3f}%</b>\n"

        f"📈 5m: "
        f"<b>{data.change_5m:+.3f}%</b>\n"

        f"📈 15m: "
        f"<b>{data.change_15m:+.3f}%</b>\n"

        f"📈 1h: "
        f"<b>{data.change_1h:+.3f}%</b>\n"

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
        *
        data.bid
    )

    fee = (

        gross
        *
        FEE_PERCENT
        /
        100
    )

    received = (
        gross
        -
        fee
    )

    profit = (

        received
        -
        position.invested
    )

    profit_percent = (

        profit
        /
        position.invested
        *
        100
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
        else
        "🔴"
    )

    trades_count = (
        learning.total_trades
        if learning
        else 0
    )

    winrate = (
        learning.winrate
        if learning
        else 0
    )

    updates = (
        learning.model_updates
        if learning
        else 0
    )

    text = (

        f"{emoji} "
        "<b>БОТ ПРОДАЛ КРИПТОВАЛЮТУ</b>\n\n"

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

        f"🤖 Вероятность при покупке: "
        f"<b>{position.ml_probability * 100:.1f}%</b>\n"

        f"📈 Ожидалось: "
        f"<b>{position.expected_move:+.2f}%</b>\n\n"

        f"👛 Баланс: "
        f"<b>${wallet_usdt:.2f}</b>\n\n"

        f"🧠 Сделок: "
        f"<b>{trades_count}</b>\n"

        f"🎯 Winrate: "
        f"<b>{winrate:.1f}%</b>\n"

        f"🧠 Переобучений: "
        f"<b>{updates}</b>\n\n"

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

    if position.entry_price <= 0:

        return False, ""

    current_price = data.price

    pnl = (

        current_price
        -
        position.entry_price

    ) / position.entry_price * 100

    holding_minutes = (

        time.time()
        -
        position.opened_at

    ) / 60

    if pnl >= TAKE_PROFIT_PERCENT:

        return True, "TAKE PROFIT"

    if pnl <= -STOP_LOSS_PERCENT:

        return True, "STOP LOSS"

    if (

        learning is not None

        and

        learning.return_ready

        and

        data.expected_move
        <=
        -EARLY_EXIT_LOSS_PERCENT

        and

        pnl < 0
    ):

        return (
            True,
            "MODEL EXPECTS DEEPER LOSS"
        )

    if (

        learning is not None

        and

        learning.ready

        and

        data.ml_probability < 0.32

        and

        pnl < 0
    ):

        return (
            True,
            "ML PROBABILITY COLLAPSED"
        )

    if (

        data.final_score < 38

        and

        pnl < 0
    ):

        return (
            True,
            "MARKET SCORE COLLAPSED"
        )

    if (

        pnl > 0.40

        and

        learning is not None

        and

        learning.ready

        and

        data.ml_probability < 0.40
    ):

        return (
            True,
            "PROFIT + ML REVERSAL"
        )

    if (

        learning is not None

        and

        learning.return_ready

        and

        data.expected_move < -0.50

        and

        pnl < -0.30
    ):

        return (
            True,
            "EXPECTED RETURN NEGATIVE"
        )

    if holding_minutes >= MAX_HOLD_MINUTES:

        return (
            True,
            "MAX HOLD TIME"
        )

    return False, ""


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
                repr(e)
            )

    market_updates += 1

    return True


# ============================================================
# PORTFOLIO
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

        prices = [

            data.price

            for data
            in candidates

            if data.price > 0
        ]

        if prices:

            current_price = float(
                np.median(
                    prices
                )
            )

        else:

            current_price = (
                position.entry_price
            )

        value += (

            position.amount
            *
            current_price
        )

    return value


# ============================================================
# DASHBOARD
# ============================================================

def dashboard_text():

    equity = portfolio_value()

    pnl = (
        equity
        -
        START_BALANCE
    )

    roi = (
        pnl
        /
        START_BALANCE
        *
        100
    )

    total_trades = (
        learning.total_trades
        if learning
        else 0
    )

    winrate = (
        learning.winrate
        if learning
        else 0
    )

    total_profit = (
        learning.total_profit
        if learning
        else wallet_realized_profit
    )

    model_ready = (
        learning.ready
        if learning
        else False
    )

    return_ready = (
        learning.return_ready
        if learning
        else False
    )

    model_updates = (
        learning.model_updates
        if learning
        else 0
    )

    text = (

        "🤖 <b>AUTONOMOUS CRYPTO TRADER v2</b>\n\n"

        f"💵 USDT: "
        f"<b>${wallet_usdt:.2f}</b>\n"

        f"👛 Портфель: "
        f"<b>${equity:.2f}</b>\n"

        f"📈 P/L: "
        f"<b>{pnl:+.2f}$ "
        f"({roi:+.2f}%)</b>\n\n"

        f"🪙 Найдено монет: "
        f"<b>{len(market_registry)}</b>\n"

        f"📡 Рынков с данными: "
        f"<b>{len(latest_market)}</b>\n"

        f"🏦 Бирж активно: "
        f"<b>{len(exchanges)}</b>\n"

        f"🔄 Обновлений: "
        f"<b>{market_updates}</b>\n\n"

        f"📌 Открытых позиций: "
        f"<b>{len(positions)}</b>/"
        f"{MAX_OPEN_POSITIONS}\n\n"

        f"📊 Сделок: "
        f"<b>{total_trades}</b>\n"

        f"🎯 Winrate: "
        f"<b>{winrate:.1f}%</b>\n"

        f"💰 Реализовано: "
        f"<b>{total_profit:+.2f}$</b>\n"

        f"💸 Комиссии: "
        f"<b>${wallet_total_fees:.2f}</b>\n\n"

        f"🧠 Classifier: "
        f"<b>{'ACTIVE' if model_ready else 'COLLECTING'}</b>\n"

        f"📈 Return model: "
        f"<b>{'ACTIVE' if return_ready else 'COLLECTING'}</b>\n"

        f"🧠 Переобучений: "
        f"<b>{model_updates}</b>\n\n"

        f"⚙️ Автотрейдинг: "
        f"<b>{'🟢 ON' if auto_running else '🔴 OFF'}</b>\n\n"

        f"🎯 Возраст монеты: "
        f"<b>≥ {MIN_MARKET_AGE_DAYS} дней</b>\n"

        f"💧 Мин. объём: "
        f"<b>${MIN_24H_VOLUME_USDT:,.0f}</b>\n\n"

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

        (
            "▶️ Запустить",
            "start"
        ),

        (
            "⏹ Остановить",
            "stop"
        ),

        (
            "👛 Кошелёк",
            "wallet"
        ),

        (
            "📊 Рынок",
            "market"
        ),

        (
            "📜 Сделки",
            "history"
        ),

        (
            "🧠 Обучение",
            "learning"
        ),

        (
            "🔄 Обновить",
            "refresh"
        )
    ]

    for text, callback in buttons:

        builder.button(
            text=text,
            callback_data=callback
        )

    builder.adjust(
        2
    )

    return builder.as_markup()


# ============================================================
# DASHBOARD MESSAGE
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

        try:

            message = await bot.send_message(

                chat_id,

                text,

                reply_markup=keyboard()
            )

            dashboard_messages[
                chat_id
            ] = message.message_id

        except Exception as e:

            print(
                "[DASHBOARD SEND]",
                repr(e)
            )

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

        if (

            "message to edit not found"
            in error_text

            or

            "message can't be edited"
            in error_text

            or

            "message is not modified"
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
                    repr(send_error)
                )

        else:

            print(
                "[DASHBOARD EDIT]",
                repr(e)
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
                repr(e)
            )


# ============================================================
# TRADING LOOP
# ============================================================

async def trading_loop(
    chat_id
):

    global auto_running

    while auto_running:

        try:

            # ==================================================
            # DISCOVERY
            # ==================================================

            await discover_dynamic_markets()

            # ==================================================
            # AGE FILTER
            # ==================================================

            symbols = await filter_old_markets(

                list(
                    market_registry.keys()
                )
            )

            allowed = set(
                symbols
            )

            for symbol in list(
                market_registry.keys()
            ):

                if symbol not in allowed:

                    del market_registry[
                        symbol
                    ]

            # ==================================================
            # MARKET
            # ==================================================

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

            # ==================================================
            # SELL
            # ==================================================

            for symbol in list(
                positions.keys()
            ):

                position = positions.get(
                    symbol
                )

                if not position:

                    continue

                data, _ = get_best_asset(
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

            # ==================================================
            # BUY
            # ==================================================

            if len(
                positions
            ) < MAX_OPEN_POSITIONS:

                candidates = []

                for symbol in list(
                    market_registry.keys()
                ):

                    if symbol in positions:

                        continue

                    data, features = (
                        get_best_asset(
                            symbol
                        )
                    )

                    if not data:

                        continue

                    if (
                        data.final_score
                        <
                        MIN_TRADE_SCORE
                    ):

                        continue

                    if (

                        learning is not None

                        and

                        learning.ready

                        and

                        data.ml_probability
                        <
                        MIN_ML_PROBABILITY
                    ):

                        continue

                    if (

                        learning is not None

                        and

                        learning.return_ready

                        and

                        data.expected_move
                        <
                        MIN_EXPECTED_MOVE_PERCENT
                    ):

                        continue

                    movement = max(

                        abs(
                            data.change_5m
                        ),

                        abs(
                            data.change_15m
                        ),

                        abs(
                            data.change_1h
                        )
                    )

                    if movement < 0.15:

                        continue

                    if (

                        data.change_15m
                        <
                        -0.20

                        and

                        data.change_1h
                        <
                        -0.50
                    ):

                        continue

                    candidates.append(

                        (
                            data,
                            features
                        )
                    )

                candidates.sort(

                    key=lambda item: (

                        item[0].final_score,

                        item[0].expected_move,

                        item[0].ml_probability,

                        item[0].volume

                    ),

                    reverse=True
                )

                slots = min(

                    2,

                    MAX_OPEN_POSITIONS
                    -
                    len(positions)
                )

                for data, features in (
                    candidates[:slots]
                ):

                    amount = (
                        calculate_trade_size(

                            data.final_score,

                            data.expected_move,

                            data.ml_probability
                        )
                    )

                    if amount >= MIN_TRADE_USDT:

                        await buy_asset(

                            data,

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
# START COMMAND
# ============================================================

@dp.message(
    Command("start")
)
async def start_command(
    message: Message
):

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

                "⚠️ Ошибка запуска:\n"

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

        await discover_dynamic_markets()

        symbols = await filter_old_markets(

            list(
                market_registry.keys()
            )
        )

        allowed = set(
            symbols
        )

        for symbol in list(
            market_registry.keys()
        ):

            if symbol not in allowed:

                del market_registry[
                    symbol
                ]

        await update_market()

    except Exception as e:

        print(
            "[REFRESH ERROR]",
            repr(e)
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
            +
            dashboard_text()
        )

        return

    auto_running = True

    await show_dashboard(

        callback.message.chat.id,

        "🟢 <b>АВТОТРЕЙДИНГ ЗАПУЩЕН</b>\n\n"

        "🔎 Ищу монеты автоматически.\n"

        "📅 Допускаются рынки от 1 года.\n"

        "💧 Фильтрую неликвид.\n"

        "🧠 Обучение идёт после каждой сделки.\n"

        "🟡 PAPER MODE."
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

        f"💰 Реализовано: "
        f"<b>{wallet_realized_profit:+.2f}$</b>",

        f"💸 Комиссии: "
        f"<b>${wallet_total_fees:.2f}</b>",

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

            prices = [

                data.price

                for data
                in candidates

                if data.price > 0
            ]

            if prices:

                current = float(
                    np.mean(
                        prices
                    )
                )

            else:

                current = (
                    position.entry_price
                )

            value = (

                position.amount
                *
                current
            )

            pnl = (

                value
                -
                position.invested
            )

            pnl_percent = (

                pnl
                /
                position.invested
                *
                100
            )

            emoji = (

                "🟢"

                if pnl >= 0

                else

                "🔴"
            )

            lines.append(

                f"{emoji} "
                f"<b>{symbol}</b>\n"

                f"🏦 {position.exchange}\n"

                f"📦 "
                f"{position.amount:.8f}\n"

                f"💰 Entry: "
                f"{format_price(position.entry_price)}\n"

                f"📊 Current: "
                f"{format_price(current)}\n"

                f"📈 P/L: "
                f"<b>{pnl:+.2f}$ "
                f"({pnl_percent:+.2f}%)</b>\n"

                f"🤖 ML: "
                f"{position.ml_probability * 100:.1f}%\n"

                f"📈 Expected: "
                f"{position.expected_move:+.2f}%\n"

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
async def market_callback(
    callback: CallbackQuery
):

    await callback.answer()

    lines = [

        "📊 <b>ЛУЧШИЕ РЫНКИ</b>",

        ""
    ]

    all_candidates = []

    for symbol in list(
        market_registry.keys()
    ):

        data, _ = get_best_asset(
            symbol
        )

        if not data:

            continue

        all_candidates.append(
            data
        )

    all_candidates.sort(

        key=lambda x: x.final_score,

        reverse=True
    )

    for data in all_candidates[:25]:

        lines.append(

            f"🪙 <b>{data.symbol}</b>\n"

            f"🏦 {data.exchange}\n"

            f"💰 {format_price(data.price)}\n"

            f"🧠 Score: "
            f"<b>{data.final_score:.1f}</b>\n"

            f"🤖 ML: "
            f"<b>{data.ml_probability * 100:.1f}%</b>\n"

            f"📈 Expected: "
            f"<b>{data.expected_move:+.2f}%</b>\n"

            f"📊 15m: "
            f"{data.change_15m:+.3f}% | "

            f"1h: "
            f"{data.change_1h:+.3f}%\n"

            f"💧 Volume: "
            f"${data.volume:,.0f}\n"
        )

    if len(lines) == 2:

        lines.append(
            "⏳ Рынок ещё загружается."
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

    trades = get_trades(
        30
    )

    if not trades:

        text = (

            "📜 <b>ИСТОРИЯ</b>\n\n"

            "Сделок пока нет."
        )

    else:

        lines = [

            "📜 <b>ПОСЛЕДНИЕ СДЕЛКИ</b>",

            ""
        ]

        for trade in trades:

            profit = float(
                trade["profit"]
            )

            emoji = (

                "🟢"

                if profit >= 0

                else

                "🔴"
            )

            closed_at = float(
                trade["closed_at"]
                or 0
            )

            # ВАЖНО:
            # вычисляем время отдельно,
            # чтобы не было SyntaxError
            # из-за многострочного f-string.
            trade_time = datetime.fromtimestamp(
                closed_at
            ).strftime(
                "%d.%m %H:%M:%S"
            )

            profit_percent = float(
                trade["profit_percent"]
                or 0
            )

            entry_score = float(
                trade["entry_score"]
                or 0
            )

            ml_probability = float(
                trade["ml_probability"]
                or 0
            )

            reason = (
                trade["reason"]
                or "UNKNOWN"
            )

            lines.append(

                f"{emoji} "
                f"<b>{trade['symbol']}</b>\n"

                f"🏦 {trade['exchange']}\n"

                f"💰 "
                f"{profit:+.2f}$ "

                f"({profit_percent:+.2f}%)\n"

                f"📌 "
                f"{reason}\n"

                f"🧠 Entry score: "
                f"{entry_score:.1f}\n"

                f"🤖 ML: "
                f"{ml_probability * 100:.1f}%\n"

                f"⏰ "
                f"{trade_time}\n"
            )

        text = "\n".join(
            lines
        )

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

            "🧠 <b>ОБУЧЕНИЕ</b>\n\n"

            "Модуль ещё запускается."
        )

        return

    text = (

        "🧠 <b>ПАМЯТЬ И ОБУЧЕНИЕ v2</b>\n\n"

        f"📊 Всего сделок: "
        f"<b>{learning.total_trades}</b>\n"

        f"🟢 Побед: "
        f"<b>{learning.wins}</b>\n"

        f"🔴 Убытков: "
        f"<b>{learning.losses}</b>\n"

        f"🎯 Winrate: "
        f"<b>{learning.winrate:.2f}%</b>\n\n"

        f"💰 Общий результат: "
        f"<b>{learning.total_profit:+.2f}$</b>\n"

        f"🏆 Лучшая сделка: "
        f"<b>{learning.best_trade:+.2f}$</b>\n"

        f"💀 Худшая сделка: "
        f"<b>{learning.worst_trade:+.2f}$</b>\n\n"

        f"🤖 Classifier: "
        f"<b>{'ACTIVE' if learning.ready else 'COLLECTING'}</b>\n"

        f"📈 Return model: "
        f"<b>{'ACTIVE' if learning.return_ready else 'COLLECTING'}</b>\n"

        f"🧠 Переобучений: "
        f"<b>{learning.model_updates}</b>\n\n"

        "Модель получает:\n"

        "• выигрышные сделки\n"

        "• убыточные сделки\n"

        "• размер прибыли/убытка\n"

        "• движение 1m/5m/15m/1h\n"

        "• RSI\n"

        "• волатильность\n"

        "• spread\n"

        "• momentum\n"

        "• тренд\n"

        "• ликвидность\n"

        "• возраст рынка\n"

        "• время сделки\n\n"

        "🔁 После каждой закрытой сделки "
        "модель заново просматривает всю "
        "историю trades."
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
            repr(e)
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
# INITIALIZE EXCHANGES
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

                "timeout": 15000
            })

            await exchange.load_markets()

            exchanges[
                exchange_id
            ] = exchange

            print(

                f"[EXCHANGE] "
                f"{exchange_id}: OK "
                f"markets={len(exchange.markets)}"
            )

        except Exception as e:

            print(

                f"[EXCHANGE] "
                f"{exchange_id}: SKIP - "
                f"{repr(e)}"
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
    # TOKEN
    # ========================================================

    if not BOT_TOKEN:

        raise RuntimeError(
            "BOT_TOKEN отсутствует в .env"
        )

    # ========================================================
    # DATABASE
    # ========================================================

    print(
        "[STARTUP] Initializing database..."
    )

    init_database()

    # ========================================================
    # WALLET
    # ========================================================

    print(
        "[STARTUP] Loading wallet..."
    )

    wallet_state = (
        load_wallet_state()
    )

    wallet_usdt = (
        wallet_state["usdt"]
    )

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
    # POSITIONS
    # ========================================================

    print(
        "[STARTUP] Loading positions..."
    )

    positions = load_positions()

    # ========================================================
    # LEARNING
    # ========================================================

    print(
        "[STARTUP] Loading learning..."
    )

    learning = LearningEngine()

    # ========================================================
    # BOT
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
        "AUTONOMOUS CRYPTO PAPER TRADER v2"
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
        "Requested exchanges:",
        len(EXCHANGE_IDS)
    )

    print(
        "Scan interval:",
        SCAN_INTERVAL
    )

    print(
        "Minimum market age:",
        MIN_MARKET_AGE_DAYS,
        "days"
    )

    print(
        "Minimum volume:",
        MIN_24H_VOLUME_USDT
    )

    print(
        "Maximum dynamic symbols:",
        MAX_DYNAMIC_SYMBOLS
    )

    print(
        "Maximum positions:",
        MAX_OPEN_POSITIONS
    )

    print(
        "ML classifier:",
        learning.ready
    )

    print(
        "ML return model:",
        learning.return_ready
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
    # WEBHOOK
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
            repr(e)
        )

    # ========================================================
    # EXCHANGES
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
    # DISCOVERY
    # ========================================================

    print(
        "[STARTUP] Discovering markets..."
    )

    try:

        await discover_dynamic_markets()

    except Exception as e:

        print(
            "[DISCOVERY ERROR]",
            repr(e)
        )

    # ========================================================
    # POLLING
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
        # STOP AUTO
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
                    repr(e)
                )

        # ====================================================
        # CLOSE BOT
        # ====================================================

        if bot:

            try:

                await bot.session.close()

            except Exception:

                pass

        # ====================================================
        # DATABASE
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

        print("=" * 70)

        print(
            "FATAL ERROR:"
        )

        print(
            repr(e)
        )

        print("=" * 70)

        raise