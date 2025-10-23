#!/usr/bin/env python3
"""
Check professional ID and fix the API query
"""

import sqlite3

def check_professional_data():
    """Check professional data and fix the query"""
    
    DB_FILE = "aimhsa.db"
    
    try:
        conn = sqlite3.connect(DB_FILE)
        
        print("="*60)
        print("CHECKING PROFESSIONAL DATA")
        print("="*60)
        
        # Check professionals
        professionals = conn.execute("""
            SELECT id, username, first_name, last_name, email
            FROM professionals 
            ORDER BY id
        """).fetchall()
        
        print(f"Found {len(professionals)} professionals:")
        for p in professionals:
            print(f"   ID: {p[0]} | Username: {p[1]} | Name: {p[2]} {p[3]} | Email: {p[4]}")
        
        # Check the specific booking
        booking = conn.execute("""
            SELECT booking_id, user_account, professional_id, risk_level, risk_score
            FROM automated_bookings 
            WHERE booking_id = 'd63a7794-a89c-452c-80a6-24691e3cb848'
        """).fetchone()
        
        if booking:
            print(f"\nBooking found:")
            print(f"   Booking ID: {booking[0]}")
            print(f"   User Account: {booking[1]}")
            print(f"   Professional ID: {booking[2]}")
            print(f"   Risk Level: {booking[3]}")
            print(f"   Risk Score: {booking[4]}")
        else:
            print("\n❌ Booking not found")
        
        # Test the exact query from the API with the correct professional ID
        if booking:
            professional_id = booking[2]
            print(f"\nTesting API query with professional ID: {professional_id}")
            
            test_query = conn.execute("""
                SELECT ab.booking_id, ab.conv_id, ab.user_account, ab.user_ip, ab.risk_level, ab.risk_score,
                       ab.detected_indicators, ab.conversation_summary, ab.booking_status, 
                       ab.scheduled_datetime, ab.session_type, ab.created_ts, ab.updated_ts,
                       u.fullname, u.email, u.telephone, u.province, u.district, u.created_at
                FROM automated_bookings ab
                LEFT JOIN users u ON ab.user_account = u.username
                WHERE ab.booking_id = ? AND ab.professional_id = ?
            """, ('d63a7794-a89c-452c-80a6-24691e3cb848', professional_id)).fetchone()
            
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
    check_professional_data()

