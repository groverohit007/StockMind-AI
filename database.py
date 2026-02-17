# database.py
# User Database & Subscription Management
# StockMind-AI Production

import sqlite3
import bcrypt
from datetime import datetime, timedelta
import os

DATABASE_PATH = "stockmind.db"
MASTER_USER_EMAIL = "groverohit0@gmail.com"
MASTER_USER_PASSWORD = "Rohit@93"

def init_database():
    """Initialize database with all required tables."""
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Users table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            is_admin INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            subscription_tier TEXT DEFAULT 'free',
            subscription_status TEXT DEFAULT 'active',
            stripe_customer_id TEXT,
            last_login TIMESTAMP
        )
    ''')
    
    # Usage tracking table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS usage (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            date DATE NOT NULL,
            predictions_used INTEGER DEFAULT 0,
            predictions_limit INTEGER DEFAULT 2,
            FOREIGN KEY (user_id) REFERENCES users(id),
            UNIQUE(user_id, date)
        )
    ''')
    
    # Subscriptions table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS subscriptions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            stripe_subscription_id TEXT UNIQUE,
            status TEXT DEFAULT 'active',
            current_period_end TIMESTAMP,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            cancelled_at TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    # Prediction history table (optional but useful)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            timeframe TEXT NOT NULL,
            signal TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')
    
    conn.commit()

    # Safe migrations for existing deployments
    cursor.execute("PRAGMA table_info(users)")
    user_columns = [col[1] for col in cursor.fetchall()]
    if 'is_admin' not in user_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN is_admin INTEGER DEFAULT 0")

    # App settings table for admin-managed API keys and settings
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS app_settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    conn.commit()
    ensure_master_user(conn)
    conn.close()
    print("✅ Database initialized successfully!")


def ensure_master_user(conn=None):
    """Create or update the requested master/admin user."""
    close_conn = False
    if conn is None:
        conn = sqlite3.connect(DATABASE_PATH)
        close_conn = True

    cursor = conn.cursor()
    password_hash = bcrypt.hashpw(MASTER_USER_PASSWORD.encode('utf-8'), bcrypt.gensalt())

    cursor.execute("SELECT id FROM users WHERE email = ?", (MASTER_USER_EMAIL.lower(),))
    existing = cursor.fetchone()

    if existing:
        cursor.execute(
            "UPDATE users SET password_hash = ?, is_admin = 1, subscription_tier = 'premium', subscription_status = 'active' WHERE id = ?",
            (password_hash, existing[0])
        )
    else:
        cursor.execute(
            """INSERT INTO users (email, password_hash, is_admin, subscription_tier, subscription_status)
               VALUES (?, ?, 1, 'premium', 'active')""",
            (MASTER_USER_EMAIL.lower(), password_hash)
        )

    conn.commit()
    if close_conn:
        conn.close()

# ============================================================================
# USER MANAGEMENT
# ============================================================================

def create_user(email, password):
    """Create new user with hashed password."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Hash password
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        
        is_admin = 1 if email.lower() == MASTER_USER_EMAIL.lower() else 0

        cursor.execute(
            "INSERT INTO users (email, password_hash, is_admin) VALUES (?, ?, ?)",
            (email.lower(), password_hash, is_admin)
        )
        
        user_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        return user_id
    
    except sqlite3.IntegrityError:
        return None  # Email already exists
    except Exception as e:
        print(f"Error creating user: {e}")
        return None

def verify_user(email, password):
    """Verify user credentials."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT id, password_hash FROM users WHERE email = ?",
            (email.lower(),)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            user_id, password_hash = result
            if bcrypt.checkpw(password.encode('utf-8'), password_hash):
                # Update last login
                update_last_login(user_id)
                return user_id
        
        return None
    
    except Exception as e:
        print(f"Error verifying user: {e}")
        return None

def get_user_info(user_id):
    """Get user information."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT id, email, is_admin, subscription_tier, subscription_status, 
               stripe_customer_id, created_at, last_login 
               FROM users WHERE id = ?""",
            (user_id,)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {
                'id': result[0],
                'email': result[1],
                'is_admin': bool(result[2]),
                'subscription_tier': result[3],
                'subscription_status': result[4],
                'stripe_customer_id': result[5],
                'created_at': result[6],
                'last_login': result[7]
            }
        
        return None
    
    except Exception as e:
        print(f"Error getting user info: {e}")
        return None

def update_last_login(user_id):
    """Update user's last login timestamp."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET last_login = ? WHERE id = ?",
            (datetime.now(), user_id)
        )
        
        conn.commit()
        conn.close()
    
    except Exception as e:
        print(f"Error updating last login: {e}")

# ============================================================================
# SUBSCRIPTION MANAGEMENT
# ============================================================================

def update_subscription(user_id, tier='premium', status='active', stripe_customer_id=None, stripe_subscription_id=None):
    """Update user subscription."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Update user
        cursor.execute(
            """UPDATE users 
               SET subscription_tier = ?, subscription_status = ?, stripe_customer_id = ?
               WHERE id = ?""",
            (tier, status, stripe_customer_id, user_id)
        )
        
        # Add subscription record if stripe_subscription_id provided
        if stripe_subscription_id:
            cursor.execute(
                """INSERT OR REPLACE INTO subscriptions 
                   (user_id, stripe_subscription_id, status, current_period_end)
                   VALUES (?, ?, ?, ?)""",
                (user_id, stripe_subscription_id, status, datetime.now() + timedelta(days=30))
            )
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error updating subscription: {e}")
        return False

def cancel_subscription(user_id):
    """Cancel user subscription (keep access until period end)."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            "UPDATE users SET subscription_status = 'cancelled' WHERE id = ?",
            (user_id,)
        )
        
        cursor.execute(
            "UPDATE subscriptions SET cancelled_at = ? WHERE user_id = ?",
            (datetime.now(), user_id)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error cancelling subscription: {e}")
        return False

# ============================================================================
# USAGE TRACKING
# ============================================================================

def get_usage_today(user_id):
    """Get user's usage for today."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        cursor.execute(
            """SELECT predictions_used, predictions_limit 
               FROM usage 
               WHERE user_id = ? AND date = ?""",
            (user_id, today)
        )
        
        result = cursor.fetchone()
        conn.close()
        
        if result:
            return {'used': result[0], 'limit': result[1]}
        else:
            # First prediction today
            return {'used': 0, 'limit': 2}
    
    except Exception as e:
        print(f"Error getting usage: {e}")
        return {'used': 0, 'limit': 2}

def increment_usage(user_id):
    """Increment prediction count for today."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        today = datetime.now().date()
        
        # Get user subscription tier
        user_info = get_user_info(user_id)
        
        if user_info and user_info['subscription_tier'] == 'premium':
            limit = 999999  # Unlimited
        else:
            limit = 2  # Free tier
        
        # Insert or update usage
        cursor.execute(
            """INSERT INTO usage (user_id, date, predictions_used, predictions_limit)
               VALUES (?, ?, 1, ?)
               ON CONFLICT(user_id, date) 
               DO UPDATE SET predictions_used = predictions_used + 1""",
            (user_id, today, limit)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error incrementing usage: {e}")
        return False

def can_make_prediction(user_id):
    """Check if user can make a prediction."""
    user_info = get_user_info(user_id)
    
    # Premium users always can
    if user_info and user_info['subscription_tier'] == 'premium':
        return True
    
    # Free users: check daily limit
    usage = get_usage_today(user_id)
    return usage['used'] < usage['limit']

def get_remaining_predictions(user_id):
    """Get remaining predictions for today."""
    user_info = get_user_info(user_id)
    
    if user_info and user_info['subscription_tier'] == 'premium':
        return 'Unlimited'
    
    usage = get_usage_today(user_id)
    remaining = max(0, usage['limit'] - usage['used'])
    
    return remaining

# ============================================================================
# PREDICTION HISTORY (Optional)
# ============================================================================

def save_prediction(user_id, ticker, timeframe, signal, confidence):
    """Save prediction to history."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """INSERT INTO predictions (user_id, ticker, timeframe, signal, confidence)
               VALUES (?, ?, ?, ?, ?)""",
            (user_id, ticker, timeframe, signal, confidence)
        )
        
        conn.commit()
        conn.close()
        
        return True
    
    except Exception as e:
        print(f"Error saving prediction: {e}")
        return False

def get_user_prediction_history(user_id, limit=10):
    """Get user's recent predictions."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT ticker, timeframe, signal, confidence, created_at
               FROM predictions
               WHERE user_id = ?
               ORDER BY created_at DESC
               LIMIT ?""",
            (user_id, limit)
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'ticker': r[0],
                'timeframe': r[1],
                'signal': r[2],
                'confidence': r[3],
                'created_at': r[4]
            }
            for r in results
        ]
    
    except Exception as e:
        print(f"Error getting prediction history: {e}")
        return []

# ============================================================================
# ADMIN FUNCTIONS
# ============================================================================

def get_all_users():
    """Get all users (admin only)."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        cursor.execute(
            """SELECT id, email, is_admin, subscription_tier, subscription_status, created_at
               FROM users
               ORDER BY created_at DESC"""
        )
        
        results = cursor.fetchall()
        conn.close()
        
        return [
            {
                'id': r[0],
                'email': r[1],
                'is_admin': bool(r[2]),
                'tier': r[3],
                'status': r[4],
                'created_at': r[5]
            }
            for r in results
        ]
    
    except Exception as e:
        print(f"Error getting users: {e}")
        return []

def get_stats():
    """Get platform statistics."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        
        # Total users
        cursor.execute("SELECT COUNT(*) FROM users")
        total_users = cursor.fetchone()[0]
        
        # Premium users
        cursor.execute("SELECT COUNT(*) FROM users WHERE subscription_tier = 'premium'")
        premium_users = cursor.fetchone()[0]
        
        # Today's predictions
        today = datetime.now().date()
        cursor.execute("SELECT SUM(predictions_used) FROM usage WHERE date = ?", (today,))
        today_predictions = cursor.fetchone()[0] or 0
        
        # Total predictions
        cursor.execute("SELECT COUNT(*) FROM predictions")
        total_predictions = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_users': total_users,
            'free_users': total_users - premium_users,
            'premium_users': premium_users,
            'today_predictions': today_predictions,
            'total_predictions': total_predictions,
            'revenue_monthly': premium_users * 17  # £17 per premium user
        }
    
    except Exception as e:
        print(f"Error getting stats: {e}")
        return {}


def is_admin_user(user_id):
    """Check whether user has admin privileges."""
    user = get_user_info(user_id)
    return bool(user and user.get('is_admin'))


def update_user_password(user_id, new_password):
    """Update password for an existing user."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        password_hash = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute("UPDATE users SET password_hash = ? WHERE id = ?", (password_hash, user_id))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error updating password: {e}")
        return False


def set_app_setting(key, value):
    """Create/update app setting value."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO app_settings (key, value, updated_at)
               VALUES (?, ?, CURRENT_TIMESTAMP)
               ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = CURRENT_TIMESTAMP""",
            (key, value)
        )
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error setting app setting: {e}")
        return False


def get_app_settings():
    """Return all app settings as a dictionary."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT key, value FROM app_settings ORDER BY key")
        rows = cursor.fetchall()
        conn.close()
        return {k: v for k, v in rows}
    except Exception as e:
        print(f"Error getting app settings: {e}")
        return {}


def get_all_user_emails():
    """Get all registered user emails for admin marketing view."""
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT email FROM users ORDER BY created_at DESC")
        results = [r[0] for r in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        print(f"Error getting user emails: {e}")
        return []

# ============================================================================
# INITIALIZATION
# ============================================================================

if __name__ == "__main__":
    print("🔧 Initializing StockMind-AI Database...")
    init_database()
    
    # Create test user (optional)
    test_user = create_user("test@stockmind.ai", "testpassword123")
    if test_user:
        print(f"✅ Test user created (ID: {test_user})")
        print("   Email: test@stockmind.ai")
        print("   Password: testpassword123")
    
    print("\n📊 Database ready!")
    print(f"   Location: {DATABASE_PATH}")
    print("\n💡 You can now run your Streamlit app!")
