"""
SQLite Database Manager untuk User Authentication
"""

import sqlite3
import os
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path


# Database path
DB_PATH = Path(__file__).parent / "users.db"


def init_database():
    """Initialize database dengan tables yang diperlukan"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Table: users
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            full_name TEXT NOT NULL,
            role TEXT DEFAULT 'user',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_login TIMESTAMP,
            is_active BOOLEAN DEFAULT 1,
            created_by TEXT
        )
    """)
    
    # Table: login_history
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS login_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            login_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            success BOOLEAN,
            ip_address TEXT,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    """)
    
    # Index untuk performa
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_username ON users(username)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_login_time ON login_history(login_time)
    """)
    
    conn.commit()
    conn.close()
    
    print(f"✅ Database initialized: {DB_PATH}")


def create_user(username: str, password_hash: str, full_name: str, 
                role: str = 'user', created_by: str = 'system') -> bool:
    """
    Buat user baru
    
    Args:
        username: Username (unique)
        password_hash: Hashed password
        full_name: Nama lengkap
        role: Role user (admin, doctor, staff)
        created_by: Username pembuat
        
    Returns:
        True jika berhasil, False jika gagal
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO users (username, password_hash, full_name, role, created_by)
            VALUES (?, ?, ?, ?, ?)
        """, (username, password_hash, full_name, role, created_by))
        
        conn.commit()
        conn.close()
        
        print(f"✅ User created: {username} ({role})")
        return True
        
    except sqlite3.IntegrityError:
        print(f"❌ Username already exists: {username}")
        return False
    except Exception as e:
        print(f"❌ Error creating user: {e}")
        return False


def get_user(username: str) -> Optional[Dict]:
    """
    Get user by username
    
    Args:
        username: Username
        
    Returns:
        User dict or None
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM users WHERE username = ? AND is_active = 1
        """, (username,))
        
        row = cursor.fetchone()
        conn.close()
        
        if row:
            return dict(row)
        return None
        
    except Exception as e:
        print(f"❌ Error getting user: {e}")
        return None


def update_last_login(username: str):
    """Update last login timestamp"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users 
            SET last_login = CURRENT_TIMESTAMP 
            WHERE username = ?
        """, (username,))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error updating last login: {e}")


def log_login_attempt(username: str, success: bool, user_id: Optional[int] = None,
                      ip_address: str = 'localhost'):
    """
    Log login attempt ke database
    
    Args:
        username: Username yang mencoba login
        success: True jika berhasil, False jika gagal
        user_id: User ID (jika berhasil)
        ip_address: IP address
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO login_history (user_id, username, success, ip_address)
            VALUES (?, ?, ?, ?)
        """, (user_id, username, success, ip_address))
        
        conn.commit()
        conn.close()
        
    except Exception as e:
        print(f"❌ Error logging login attempt: {e}")


def get_all_users() -> List[Dict]:
    """Get semua users (untuk admin panel)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT id, username, full_name, role, created_at, last_login, is_active
            FROM users
            ORDER BY created_at DESC
        """)
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except Exception as e:
        print(f"❌ Error getting all users: {e}")
        return []


def deactivate_user(username: str) -> bool:
    """Deactivate user (soft delete)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users SET is_active = 0 WHERE username = ?
        """, (username,))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error deactivating user: {e}")
        return False


def activate_user(username: str) -> bool:
    """Activate user"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users SET is_active = 1 WHERE username = ?
        """, (username,))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error activating user: {e}")
        return False


def change_password(username: str, new_password_hash: str) -> bool:
    """Change user password"""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        
        cursor.execute("""
            UPDATE users SET password_hash = ? WHERE username = ?
        """, (new_password_hash, username))
        
        conn.commit()
        conn.close()
        
        return True
        
    except Exception as e:
        print(f"❌ Error changing password: {e}")
        return False


def get_login_history(limit: int = 100) -> List[Dict]:
    """Get login history (untuk audit)"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM login_history
            ORDER BY login_time DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
        
    except Exception as e:
        print(f"❌ Error getting login history: {e}")
        return []


def backup_database(backup_dir: str = "backups"):
    """Create database backup"""
    try:
        from shutil import copy2
        
        # Create backup directory if not exists
        backup_path = Path(__file__).parent.parent / backup_dir
        backup_path.mkdir(exist_ok=True)
        
        # Generate backup filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_file = backup_path / f"users_backup_{timestamp}.db"
        
        # Copy database
        copy2(DB_PATH, backup_file)
        
        print(f"✅ Database backed up to: {backup_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error backing up database: {e}")
        return False


if __name__ == "__main__":
    # Initialize database
    print("=== Database Initialization ===\n")
    init_database()
    
    print("\n=== Database Ready ===")
    print(f"Location: {DB_PATH}")
    print(f"Size: {os.path.getsize(DB_PATH) / 1024:.2f} KB")
