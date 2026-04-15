"""
Database Models - SQLite
Fixed: SHA-256 -> bcrypt, commit-after-read bug, SQL injection in LIKE, XSS, credential logging
"""

import sqlite3
import os
import logging
from contextlib import contextmanager
from typing import List, Dict, Optional
from datetime import datetime
import html

logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
DB_PATH = os.path.join(BASE_DIR, 'data', 'college_chatbot.db')


@contextmanager
def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        logger.error(f"Database error: {e}")
        raise
    finally:
        conn.close()


def _hash_password(password: str) -> str:
    """FIX [Critical]: Use bcrypt instead of plain SHA-256"""
    try:
        import bcrypt
        return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()
    except ImportError:
        # Fallback: use hashlib with salt (still better than plain sha256)
        import hashlib, secrets
        salt = secrets.token_hex(16)
        hashed = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return f"pbkdf2${salt}${hashed.hex()}"


def _verify_password(password: str, stored_hash: str) -> bool:
    """Verify password against stored hash"""
    try:
        import bcrypt
        if stored_hash.startswith('$2'):
            return bcrypt.checkpw(password.encode(), stored_hash.encode())
    except ImportError:
        pass

    if stored_hash.startswith('pbkdf2$'):
        import hashlib
        _, salt, hash_hex = stored_hash.split('$', 2)
        check = hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000)
        return check.hex() == hash_hex

    # Legacy: plain sha256 (for migration)
    import hashlib
    if hashlib.sha256(password.encode()).hexdigest() == stored_hash:
        return True
    return False


def _sanitize(text: str) -> str:
    """FIX [Major]: Sanitize text to prevent XSS"""
    if text is None:
        return ""
    return html.escape(str(text))


def init_database():
    import sys
    sys.path.insert(0, BASE_DIR)
    from data.schema import SCHEMA_SQL, SAMPLE_FAQS

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)

    with get_db_connection() as conn:
        conn.executescript(SCHEMA_SQL)
        logger.info("Database schema created")

        cursor = conn.execute("SELECT COUNT(*) FROM faqs")
        count = cursor.fetchone()[0]

        if count == 0:
            for faq in SAMPLE_FAQS:
                conn.execute(
                    "INSERT INTO faqs (category, question, answer, keywords) VALUES (?, ?, ?, ?)",
                    (_sanitize(faq['category']), _sanitize(faq['question']),
                     _sanitize(faq['answer']), _sanitize(faq.get('keywords', '')))
                )
            logger.info(f"Inserted {len(SAMPLE_FAQS)} sample FAQs")

        cursor = conn.execute("SELECT COUNT(*) FROM admin_users")
        if cursor.fetchone()[0] == 0:
            password_hash = _hash_password("admin123")
            conn.execute(
                "INSERT INTO admin_users (username, password_hash) VALUES (?, ?)",
                ("admin", password_hash)
            )
            # FIX [Major]: Don't log the default password
            logger.info("Default admin user created (username: admin). Change password immediately!")


# ─── FAQ Operations ─────────────────────────────────────────────────

def get_all_faqs(category: Optional[str] = None) -> List[Dict]:
    with get_db_connection() as conn:
        if category:
            cursor = conn.execute(
                "SELECT * FROM faqs WHERE category = ? AND is_active = 1 ORDER BY category, id",
                (category,))
        else:
            cursor = conn.execute(
                "SELECT * FROM faqs WHERE is_active = 1 ORDER BY category, id")
        return [dict(row) for row in cursor.fetchall()]


def get_faq_by_id(faq_id: int) -> Optional[Dict]:
    with get_db_connection() as conn:
        cursor = conn.execute("SELECT * FROM faqs WHERE id = ? AND is_active = 1", (faq_id,))
        row = cursor.fetchone()
        return dict(row) if row else None


def create_faq(category: str, question: str, answer: str, keywords: str = "") -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO faqs (category, question, answer, keywords) VALUES (?, ?, ?, ?)",
            (_sanitize(category), _sanitize(question), _sanitize(answer), _sanitize(keywords)))
        return cursor.lastrowid


def update_faq(faq_id: int, category: str, question: str, answer: str, keywords: str = "") -> bool:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """UPDATE faqs SET category=?, question=?, answer=?, keywords=?,
               updated_at=CURRENT_TIMESTAMP WHERE id=?""",
            (_sanitize(category), _sanitize(question), _sanitize(answer), _sanitize(keywords), faq_id))
        return cursor.rowcount > 0


def delete_faq(faq_id: int) -> bool:
    with get_db_connection() as conn:
        cursor = conn.execute("UPDATE faqs SET is_active=0 WHERE id=?", (faq_id,))
        return cursor.rowcount > 0


def search_faqs(query: str) -> List[Dict]:
    # FIX [Critical]: Escape LIKE metacharacters to prevent SQL injection
    escaped = query.replace('%', '\\%').replace('_', '\\_')
    search_term = f"%{escaped}%"
    with get_db_connection() as conn:
        cursor = conn.execute(
            """SELECT * FROM faqs
               WHERE is_active = 1 AND (
                   question LIKE ? ESCAPE '\\' OR answer LIKE ? ESCAPE '\\' OR keywords LIKE ? ESCAPE '\\'
               ) ORDER BY category""",
            (search_term, search_term, search_term))
        return [dict(row) for row in cursor.fetchall()]


# ─── Chat History ────────────────────────────────────────────────────

def save_chat_message(session_id, user_message, bot_response, intent=None, confidence=None, ip_address=None) -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            """INSERT INTO chat_history
               (session_id, user_message, bot_response, intent, confidence, ip_address)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (session_id, _sanitize(user_message), bot_response, intent, confidence, ip_address))
        return cursor.lastrowid


def get_chat_history(session_id=None, limit=50, offset=0) -> List[Dict]:
    with get_db_connection() as conn:
        if session_id:
            cursor = conn.execute(
                "SELECT * FROM chat_history WHERE session_id = ? ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (session_id, limit, offset))
        else:
            cursor = conn.execute(
                "SELECT * FROM chat_history ORDER BY timestamp DESC LIMIT ? OFFSET ?",
                (limit, offset))
        return [dict(row) for row in cursor.fetchall()]


def get_chat_stats() -> Dict:
    with get_db_connection() as conn:
        total = conn.execute("SELECT COUNT(*) FROM chat_history").fetchone()[0]
        today = conn.execute(
            "SELECT COUNT(*) FROM chat_history WHERE DATE(timestamp) = DATE('now')").fetchone()[0]
        top_intents = conn.execute(
            """SELECT intent, COUNT(*) as count FROM chat_history
               WHERE intent IS NOT NULL AND intent != 'unknown'
               GROUP BY intent ORDER BY count DESC LIMIT 5""").fetchall()
        avg_confidence = conn.execute(
            "SELECT AVG(confidence) FROM chat_history WHERE confidence IS NOT NULL").fetchone()[0]
        return {
            'total_messages': total,
            'messages_today': today,
            'top_intents': [dict(row) for row in top_intents],
            'avg_confidence': round(avg_confidence or 0, 3)
        }


# ─── Admin Auth ──────────────────────────────────────────────────────

def verify_admin(username: str, password: str) -> bool:
    # FIX [Major]: Separate the SELECT and UPDATE into different blocks
    # to avoid commit-after-read bug
    with get_db_connection() as conn:
        cursor = conn.execute(
            "SELECT id, password_hash FROM admin_users WHERE username = ?",
            (username,))
        row = cursor.fetchone()

    if not row:
        return False

    if not _verify_password(password, row['password_hash']):
        return False

    # Only update last_login if credentials are valid
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE admin_users SET last_login = CURRENT_TIMESTAMP WHERE username = ?",
            (username,))
    return True


def change_admin_password(username: str, new_password: str) -> bool:
    password_hash = _hash_password(new_password)
    with get_db_connection() as conn:
        cursor = conn.execute(
            "UPDATE admin_users SET password_hash = ? WHERE username = ?",
            (password_hash, username))
        return cursor.rowcount > 0


# ─── Feedback ────────────────────────────────────────────────────────

def save_feedback(chat_id: int, rating: int, comment: str = None) -> int:
    with get_db_connection() as conn:
        cursor = conn.execute(
            "INSERT INTO feedback (chat_history_id, rating, comment) VALUES (?, ?, ?)",
            (chat_id, rating, _sanitize(comment) if comment else None))
        return cursor.lastrowid


def get_feedback_stats() -> Dict:
    with get_db_connection() as conn:
        avg_rating = conn.execute("SELECT AVG(rating) FROM feedback").fetchone()[0]
        distribution = conn.execute(
            "SELECT rating, COUNT(*) as count FROM feedback GROUP BY rating ORDER BY rating").fetchall()
        return {
            'average_rating': round(avg_rating or 0, 2),
            'distribution': [dict(row) for row in distribution]
        }
