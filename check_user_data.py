#!/usr/bin/env python3
"""
Test script to check database for user data
"""

import sqlite3
import json

def check_user_data():
    """Check if user data exists in the database"""
    
    DB_FILE = "aimhsa.db"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        print("="*60)
        print("DATABASE USER DATA CHECK")
        print("="*60)
        
        # Check if user exists
        print("\n1. Checking if user 'Mugisha' exists:")
        user = conn.execute("""
            SELECT username, fullname, email, telephone, province, district, created_at
            FROM users 
            WHERE username = ?
        """, ('Mugisha',)).fetchone()
        
        if user:
            print("✅ User found!")
            print(f"   Username: {user[0]}")
            print(f"   Full Name: {user[1]}")
            print(f"   Email: {user[2]}")
            print(f"   Phone: {user[3]}")
            print(f"   Province: {user[4]}")
            print(f"   District: {user[5]}")
            print(f"   Created At: {user[6]}")
        else:
            print("❌ User 'Mugisha' not found in users table")
        
        # Check all users
        print("\n2. All users in database:")
        all_users = conn.execute("""
            SELECT username, fullname, email, telephone, province, district
            FROM users 
            ORDER BY created_at DESC
            LIMIT 10
        """).fetchall()
        
        if all_users:
            print(f"Found {len(all_users)} users:")
            for u in all_users:
                print(f"   {u[0]} | {u[1]} | {u[2]} | {u[3]} | {u[4]} | {u[5]}")
        else:
            print("❌ No users found in database")
        
        # Check automated bookings
        print("\n3. Checking automated bookings:")
        bookings = conn.execute("""
            SELECT booking_id, user_account, professional_id, risk_level, risk_score
            FROM automated_bookings 
            WHERE user_account = 'Mugisha'
            ORDER BY created_ts DESC
            LIMIT 5
        """).fetchall()
        
        if bookings:
            print(f"Found {len(bookings)} bookings for Mugisha:")
            for b in bookings:
                print(f"   {b[0]} | {b[1]} | {b[2]} | {b[3]} | {b[4]}")
        else:
            print("❌ No bookings found for Mugisha")
        
        # Test the exact query from the API
        print("\n4. Testing API query:")
        test_query = conn.execute("""
            SELECT ab.booking_id, ab.conv_id, ab.user_account, ab.user_ip, ab.risk_level, ab.risk_score,
                   ab.detected_indicators, ab.conversation_summary, ab.booking_status, 
                   ab.scheduled_datetime, ab.session_type, ab.created_ts, ab.updated_ts,
                   u.fullname, u.email, u.telephone, u.province, u.district, u.created_at
            FROM automated_bookings ab
            LEFT JOIN users u ON ab.user_account = u.username
            WHERE ab.booking_id = ? AND ab.professional_id = ?
        """, ('d63a7794-a89c-452c-80a6-24691e3cb848', '6')).fetchone()
        
        if test_query:
            print("✅ API query successful!")
            print(f"   Booking ID: {test_query[0]}")
            print(f"   User Account: {test_query[2]}")
            print(f"   Full Name: {test_query[13]}")
            print(f"   Email: {test_query[14]}")
            print(f"   Phone: {test_query[15]}")
            print(f"   Province: {test_query[16]}")
            print(f"   District: {test_query[17]}")
            print(f"   Created At: {test_query[18]}")
        else:
            print("❌ API query failed - no results")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Database Error: {e}")

if __name__ == "__main__":
    check_user_data()

