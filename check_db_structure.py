#!/usr/bin/env python3
"""
Check database tables and structure
"""

import sqlite3

def check_database_structure():
    """Check database tables and structure"""
    
    DB_FILE = "aimhsa.db"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        print("="*60)
        print("DATABASE STRUCTURE CHECK")
        print("="*60)
        
        # Get all tables
        tables = conn.execute("""
            SELECT name FROM sqlite_master WHERE type='table' ORDER BY name
        """).fetchall()
        
        print(f"\nFound {len(tables)} tables:")
        for table in tables:
            print(f"   - {table[0]}")
        
        # Check automated_bookings table structure
        print("\n" + "="*40)
        print("AUTOMATED_BOOKINGS TABLE STRUCTURE:")
        print("="*40)
        columns = conn.execute("PRAGMA table_info(automated_bookings)").fetchall()
        for col in columns:
            print(f"   {col[1]} ({col[2]})")
        
        # Check if there's a user-related table
        print("\n" + "="*40)
        print("CHECKING FOR USER DATA:")
        print("="*40)
        
        # Look for any table that might contain user data
        for table in tables:
            table_name = table[0]
            if 'user' in table_name.lower() or 'account' in table_name.lower():
                print(f"\nTable: {table_name}")
                columns = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
                for col in columns:
                    print(f"   {col[1]} ({col[2]})")
                
                # Check if it has data
                count = conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
                print(f"   Records: {count}")
                
                if count > 0:
                    sample = conn.execute(f"SELECT * FROM {table_name} LIMIT 3").fetchall()
                    print(f"   Sample data: {sample}")
        
        # Check automated_bookings data
        print("\n" + "="*40)
        print("AUTOMATED_BOOKINGS DATA:")
        print("="*40)
        bookings = conn.execute("""
            SELECT booking_id, user_account, professional_id, risk_level, risk_score
            FROM automated_bookings 
            ORDER BY created_ts DESC
            LIMIT 5
        """).fetchall()
        
        if bookings:
            print(f"Found {len(bookings)} bookings:")
            for b in bookings:
                print(f"   {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]}")
        else:
            print("❌ No bookings found")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    check_database_structure()

