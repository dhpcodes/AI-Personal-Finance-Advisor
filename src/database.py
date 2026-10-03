import sqlite3
import json
import uuid
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from src.config import config
from src.utils import logger

class DatabaseManager:
    """SQLite Persistence Layer with strict user-data isolation."""

    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = config.raw_data_path.parent.parent / "advisor_app.db"
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def get_connection(self) -> sqlite3.Connection:
        """Get database connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Initialize SQLite database tables if they do not exist."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # 1. Users table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # 2. Transactions table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS transactions (
                    transaction_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    date TEXT,
                    description TEXT NOT NULL,
                    amount REAL DEFAULT 0.0,
                    currency TEXT DEFAULT 'USD',
                    country TEXT DEFAULT 'USA',
                    transaction_type TEXT DEFAULT 'expense',
                    predicted_category TEXT,
                    final_category TEXT,
                    prediction_confidence REAL DEFAULT 1.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # 3. Feedback table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    transaction_id TEXT,
                    description TEXT NOT NULL,
                    original_prediction TEXT NOT NULL,
                    corrected_category TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # 4. Behavior Profiles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS behavior_profiles (
                    user_id TEXT PRIMARY KEY,
                    calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    total_income REAL,
                    total_expense REAL,
                    net_cash_flow REAL,
                    average_transaction REAL,
                    spending_variability REAL,
                    behavior_features TEXT,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            # 5. Forecast Results table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS forecast_results (
                    forecast_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    forecast_date TEXT NOT NULL,
                    target_period TEXT NOT NULL,
                    category TEXT NOT NULL,
                    predicted_amount REAL NOT NULL,
                    lower_bound REAL,
                    upper_bound REAL,
                    method TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users(user_id)
                )
            """)
            
            conn.commit()
            logger.info(f"Initialized SQLite database schema at {self.db_path}")

    def ensure_user(self, user_id: str) -> str:
        """Create user record if not exists."""
        if not user_id:
            user_id = "default_user"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (user_id) VALUES (?)", (user_id,))
            conn.commit()
        return user_id

    def add_transaction(
        self,
        user_id: str,
        description: str,
        amount: float = 0.0,
        date: str = None,
        currency: str = "USD",
        country: str = "USA",
        predicted_category: str = "Uncategorized",
        final_category: str = None,
        confidence: float = 1.0,
        transaction_type: str = "expense"
    ) -> str:
        """Add single transaction record tied strictly to user_id."""
        self.ensure_user(user_id)
        tx_id = str(uuid.uuid4())
        date = date or datetime.now().strftime("%Y-%m-%d")
        final_cat = final_category or predicted_category
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO transactions (
                    transaction_id, user_id, date, description, amount, currency,
                    country, transaction_type, predicted_category, final_category, prediction_confidence
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (tx_id, user_id, date, description, amount, currency, country, transaction_type, predicted_category, final_cat, confidence))
            conn.commit()
        return tx_id

    def add_transactions_batch(self, user_id: str, df: pd.DataFrame) -> int:
        """Bulk insert transactions DataFrame for a user."""
        self.ensure_user(user_id)
        count = 0
        with self.get_connection() as conn:
            cursor = conn.cursor()
            for _, row in df.iterrows():
                tx_id = str(uuid.uuid4())
                desc = str(row.get("transaction_description", row.get("description", "Unknown")))
                amt = float(row.get("amount", 0.0)) if pd.notna(row.get("amount")) else 0.0
                dt = str(row.get("date", datetime.now().strftime("%Y-%m-%d")))
                curr = str(row.get("currency", "USD"))
                cntry = str(row.get("country", "USA"))
                pred_cat = str(row.get("predicted_category", row.get("category", "Uncategorized")))
                final_cat = str(row.get("final_category", pred_cat))
                conf = float(row.get("confidence", 1.0)) if pd.notna(row.get("confidence")) else 1.0
                tx_type = "income" if pred_cat.lower() == "income" or amt > 0 else "expense"
                
                cursor.execute("""
                    INSERT INTO transactions (
                        transaction_id, user_id, date, description, amount, currency,
                        country, transaction_type, predicted_category, final_category, prediction_confidence
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (tx_id, user_id, dt, desc, amt, curr, cntry, tx_type, pred_cat, final_cat, conf))
                count += 1
            conn.commit()
        return count

    def get_user_transactions(self, user_id: str) -> pd.DataFrame:
        """Strictly retrieve transactions for user_id only (User Isolation)."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM transactions WHERE user_id = ? ORDER BY date DESC, created_at DESC",
                conn,
                params=(user_id,)
            )
        return df

    def record_feedback(
        self,
        user_id: str,
        transaction_id: Optional[str],
        description: str,
        original_prediction: str,
        corrected_category: str
    ) -> str:
        """Record user classification correction feedback."""
        self.ensure_user(user_id)
        fb_id = str(uuid.uuid4())
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO feedback (
                    feedback_id, user_id, transaction_id, description, original_prediction, corrected_category
                ) VALUES (?, ?, ?, ?, ?, ?)
            """, (fb_id, user_id, transaction_id, description, original_prediction, corrected_category))
            
            # Also update final_category in transactions table if transaction_id provided
            if transaction_id:
                cursor.execute("""
                    UPDATE transactions SET final_category = ? WHERE transaction_id = ? AND user_id = ?
                """, (corrected_category, transaction_id, user_id))
                
            conn.commit()
        logger.info(f"Recorded feedback for user '{user_id}': '{description}' -> '{corrected_category}'")
        return fb_id

    def get_user_feedback(self, user_id: str) -> pd.DataFrame:
        """Get all feedback records for user_id."""
        with self.get_connection() as conn:
            df = pd.read_sql_query(
                "SELECT * FROM feedback WHERE user_id = ? ORDER BY created_at DESC",
                conn,
                params=(user_id,)
            )
        return df

    def save_behavior_profile(self, user_id: str, profile_dict: Dict[str, Any]) -> None:
        """Save or update derived behavior profile for user_id."""
        self.ensure_user(user_id)
        tot_inc = float(profile_dict.get("total_income", 0.0))
        tot_exp = float(profile_dict.get("total_expense", 0.0))
        net_flow = float(profile_dict.get("net_cash_flow", 0.0))
        avg_tx = float(profile_dict.get("average_transaction", 0.0))
        var = float(profile_dict.get("spending_variability", 0.0))
        feat_json = json.dumps(profile_dict)
        
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO behavior_profiles (
                    user_id, calculated_at, total_income, total_expense, net_cash_flow,
                    average_transaction, spending_variability, behavior_features
                ) VALUES (?, CURRENT_TIMESTAMP, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(user_id) DO UPDATE SET
                    calculated_at = CURRENT_TIMESTAMP,
                    total_income = excluded.total_income,
                    total_expense = excluded.total_expense,
                    net_cash_flow = excluded.net_cash_flow,
                    average_transaction = excluded.average_transaction,
                    spending_variability = excluded.spending_variability,
                    behavior_features = excluded.behavior_features
            """, (user_id, tot_inc, tot_exp, net_flow, avg_tx, var, feat_json))
            conn.commit()

    def get_behavior_profile(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get behavior profile dict for user_id."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT behavior_features FROM behavior_profiles WHERE user_id = ?", (user_id,))
            row = cursor.fetchone()
            if row and row["behavior_features"]:
                return json.loads(row["behavior_features"])
        return None

# Global Instance
db_manager = DatabaseManager()
