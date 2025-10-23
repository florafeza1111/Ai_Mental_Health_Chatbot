#!/usr/bin/env python3
"""
Initialize the database with all required tables
"""

import sqlite3
import os
import time

# Database file path
DB_FILE = "aimhsa.db"

def init_database():
    """Initialize the database with all required tables"""
    
    print("="*60)
    print("INITIALIZING DATABASE")
    print("="*60)
    
    # Create directory if it doesn't exist (only if there's a directory)
    db_dir = os.path.dirname(DB_FILE)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    
    conn = sqlite3.connect(DB_FILE)
    
    try:
        print("Creating tables...")
        
        # Messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        print("✅ messages table created")
        
        # Attachments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS attachments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                filename TEXT NOT NULL,
                text TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        print("✅ attachments table created")
        
        # Sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                key TEXT UNIQUE NOT NULL,
                conv_id TEXT NOT NULL,
                ts REAL NOT NULL
            )
        """)
        print("✅ sessions table created")
        
        # Users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                created_ts REAL NOT NULL
            )
        """)
        print("✅ users table created")
        
        # Add additional columns to users table
        cursor = conn.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in cursor.fetchall()]
        
        if "email" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN email TEXT")
        if "fullname" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN fullname TEXT")
        if "telephone" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN telephone TEXT")
        if "province" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN province TEXT")
        if "district" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN district TEXT")
        if "created_at" not in columns:
            conn.execute("ALTER TABLE users ADD COLUMN created_at REAL")
        
        print("✅ users table columns updated")
        
        # Password resets table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                token TEXT UNIQUE NOT NULL,
                expires_ts REAL NOT NULL,
                used INTEGER DEFAULT 0
            )
        """)
        print("✅ password_resets table created")
        
        # Conversations table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                conv_id TEXT PRIMARY KEY,
                owner_key TEXT,
                preview TEXT,
                ts REAL
            )
        """)
        print("✅ conversations table created")
        
        # Add additional columns to conversations table
        try:
            cur = conn.execute("PRAGMA table_info(conversations)")
            cols = [r[1] for r in cur.fetchall()]
            if "archive_pw_hash" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN archive_pw_hash TEXT")
            if "booking_prompt_shown" not in cols:
                conn.execute("ALTER TABLE conversations ADD COLUMN booking_prompt_shown INTEGER DEFAULT 0")
        except Exception:
            pass
        
        print("✅ conversations table columns updated")
        
        # Professionals table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS professionals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                first_name TEXT NOT NULL,
                last_name TEXT NOT NULL,
                email TEXT NOT NULL,
                phone TEXT,
                license_number TEXT,
                specialization TEXT NOT NULL,
                expertise_areas TEXT NOT NULL,
                languages TEXT NOT NULL,
                qualifications TEXT NOT NULL,
                availability_schedule TEXT NOT NULL,
                location_latitude REAL,
                location_longitude REAL,
                location_address TEXT,
                district TEXT,
                max_patients_per_day INTEGER DEFAULT 10,
                consultation_fee REAL,
                experience_years INTEGER,
                bio TEXT,
                profile_picture TEXT,
                is_active BOOLEAN DEFAULT 1,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL
            )
        """)
        print("✅ professionals table created")
        
        # Risk assessments table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS risk_assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                user_query TEXT NOT NULL,
                risk_score REAL NOT NULL,
                risk_level TEXT NOT NULL,
                detected_indicators TEXT,
                assessment_timestamp REAL NOT NULL,
                processed BOOLEAN DEFAULT 0,
                booking_created BOOLEAN DEFAULT 0
            )
        """)
        print("✅ risk_assessments table created")
        
        # Automated bookings table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS automated_bookings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT UNIQUE NOT NULL,
                conv_id TEXT NOT NULL,
                user_account TEXT,
                user_ip TEXT,
                professional_id INTEGER NOT NULL,
                risk_level TEXT NOT NULL,
                risk_score REAL NOT NULL,
                detected_indicators TEXT,
                conversation_summary TEXT,
                booking_status TEXT DEFAULT 'pending',
                scheduled_datetime REAL,
                session_type TEXT DEFAULT 'routine',
                location_preference TEXT,
                notes TEXT,
                created_ts REAL NOT NULL,
                updated_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        print("✅ automated_bookings table created")
        
        # Professional notifications table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS professional_notifications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                professional_id INTEGER NOT NULL,
                booking_id TEXT NOT NULL,
                notification_type TEXT NOT NULL,
                title TEXT NOT NULL,
                message TEXT NOT NULL,
                is_read BOOLEAN DEFAULT 0,
                priority TEXT DEFAULT 'normal',
                created_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        print("✅ professional_notifications table created")
        
        # Therapy sessions table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS therapy_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT NOT NULL,
                professional_id INTEGER NOT NULL,
                conv_id TEXT NOT NULL,
                session_start REAL,
                session_end REAL,
                session_notes TEXT,
                treatment_plan TEXT,
                follow_up_required BOOLEAN DEFAULT 0,
                follow_up_date REAL,
                session_rating INTEGER,
                session_feedback TEXT,
                created_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        print("✅ therapy_sessions table created")
        
        # Session notes table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS session_notes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                booking_id TEXT NOT NULL,
                professional_id INTEGER NOT NULL,
                notes TEXT,
                treatment_plan TEXT,
                follow_up_required BOOLEAN DEFAULT 0,
                follow_up_date REAL,
                created_ts REAL NOT NULL,
                FOREIGN KEY (professional_id) REFERENCES professionals (id)
            )
        """)
        print("✅ session_notes table created")
        
        # Conversation messages table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversation_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp REAL NOT NULL
            )
        """)
        print("✅ conversation_messages table created")
        
        # Admin users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS admin_users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                email TEXT NOT NULL,
                role TEXT DEFAULT 'admin',
                permissions TEXT,
                created_ts REAL NOT NULL
            )
        """)
        print("✅ admin_users table created")
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "="*60)
        print("DATABASE INITIALIZATION COMPLETE!")
        print("="*60)
        
        # Verify tables were created
        tables = conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
        """).fetchall()
        
        print(f"\nCreated {len(tables)} tables:")
        for table in tables:
            print(f"   ✅ {table[0]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
        conn.close()
        raise

if __name__ == "__main__":
    init_database()
