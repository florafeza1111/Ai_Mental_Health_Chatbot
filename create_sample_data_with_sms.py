#!/usr/bin/env python3
"""
Create sample data for AIMHSA with phone numbers for SMS testing
"""

import sqlite3
import time
import json
from werkzeug.security import generate_password_hash

DB_FILE = "storage/conversations.db"

def create_sample_users():
    """Create sample users with phone numbers"""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Sample users with phone numbers
        users = [
            {
                'username': 'testuser',
                'password': 'password123',
                'email': 'testuser@example.com',
                'fullname': 'Test User',
                'telephone': '+250788123456',
                'province': 'Kigali',
                'district': 'Gasabo'
            },
            {
                'username': 'john_doe',
                'password': 'password123',
                'email': 'john.doe@example.com',
                'fullname': 'John Doe',
                'telephone': '+250788234567',
                'province': 'Kigali',
                'district': 'Kicukiro'
            },
            {
                'username': 'jane_smith',
                'password': 'password123',
                'email': 'jane.smith@example.com',
                'fullname': 'Jane Smith',
                'telephone': '+250788345678',
                'province': 'Southern',
                'district': 'Huye'
            },
            {
                'username': 'rwanda_user',
                'password': 'password123',
                'email': 'rwanda.user@example.com',
                'fullname': 'Rwanda User',
                'telephone': '+250788456789',
                'province': 'Northern',
                'district': 'Musanze'
            }
        ]
        
        for user in users:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO users 
                    (username, password_hash, email, fullname, telephone, province, district, created_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    user['username'],
                    generate_password_hash(user['password']),
                    user['email'],
                    user['fullname'],
                    user['telephone'],
                    user['province'],
                    user['district'],
                    time.time()
                ))
                print(f"✅ Created user: {user['username']} ({user['telephone']})")
            except Exception as e:
                print(f"❌ Failed to create user {user['username']}: {e}")
        
        conn.commit()
        print(f"✅ Created {len(users)} sample users with phone numbers")
        
    finally:
        conn.close()

def create_sample_professionals():
    """Create sample professionals with phone numbers"""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Sample professionals with phone numbers
        professionals = [
            {
                'username': 'dr_mukamana',
                'password': 'password123',
                'first_name': 'Marie',
                'last_name': 'Mukamana',
                'email': 'dr.mukamana@example.com',
                'phone': '+250788111222',
                'specialization': 'psychiatrist',
                'expertise_areas': ['depression', 'anxiety', 'ptsd', 'crisis'],
                'district': 'Gasabo',
                'consultation_fee': 50000,
                'bio': 'Experienced psychiatrist specializing in trauma and crisis intervention'
            },
            {
                'username': 'counselor_ntwari',
                'password': 'password123',
                'first_name': 'Jean',
                'last_name': 'Ntwari',
                'email': 'counselor.ntwari@example.com',
                'phone': '+250788333444',
                'specialization': 'counselor',
                'expertise_areas': ['anxiety', 'stress', 'family', 'youth'],
                'district': 'Kicukiro',
                'consultation_fee': 30000,
                'bio': 'Certified counselor with expertise in family and youth mental health'
            },
            {
                'username': 'psychologist_umutoni',
                'password': 'password123',
                'first_name': 'Grace',
                'last_name': 'Umutoni',
                'email': 'psychologist.umutoni@example.com',
                'phone': '+250788555666',
                'specialization': 'psychologist',
                'expertise_areas': ['ptsd', 'trauma', 'depression', 'anxiety'],
                'district': 'Huye',
                'consultation_fee': 40000,
                'bio': 'Clinical psychologist specializing in trauma therapy and PTSD treatment'
            },
            {
                'username': 'social_worker_nyiraneza',
                'password': 'password123',
                'first_name': 'Claudine',
                'last_name': 'Nyiraneza',
                'email': 'social.worker@example.com',
                'phone': '+250788777888',
                'specialization': 'social_worker',
                'expertise_areas': ['family', 'community', 'crisis', 'youth'],
                'district': 'Musanze',
                'consultation_fee': 25000,
                'bio': 'Social worker focused on community mental health and family support'
            }
        ]
        
        for prof in professionals:
            try:
                conn.execute("""
                    INSERT OR REPLACE INTO professionals 
                    (username, password_hash, first_name, last_name, email, phone, specialization,
                     expertise_areas, district, consultation_fee, bio, created_ts, updated_ts)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    prof['username'],
                    generate_password_hash(prof['password']),
                    prof['first_name'],
                    prof['last_name'],
                    prof['email'],
                    prof['phone'],
                    prof['specialization'],
                    json.dumps(prof['expertise_areas']),
                    prof['district'],
                    prof['consultation_fee'],
                    prof['bio'],
                    time.time(),
                    time.time()
                ))
                print(f"✅ Created professional: {prof['first_name']} {prof['last_name']} ({prof['phone']})")
            except Exception as e:
                print(f"❌ Failed to create professional {prof['username']}: {e}")
        
        conn.commit()
        print(f"✅ Created {len(professionals)} sample professionals with phone numbers")
        
    finally:
        conn.close()

def create_admin_user():
    """Create admin user"""
    conn = sqlite3.connect(DB_FILE)
    try:
        conn.execute("""
            INSERT OR REPLACE INTO admin_users 
            (username, password_hash, email, full_name, created_ts)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'admin',
            generate_password_hash('admin123'),
            'admin@aimhsa.rw',
            'System Administrator',
            time.time()
        ))
        print("✅ Created admin user: admin / admin123")
        conn.commit()
    except Exception as e:
        print(f"❌ Failed to create admin user: {e}")
    finally:
        conn.close()

def verify_sms_ready_data():
    """Verify that we have users and professionals with phone numbers"""
    conn = sqlite3.connect(DB_FILE)
    try:
        # Check users with phone numbers
        users = conn.execute("""
            SELECT username, fullname, telephone 
            FROM users 
            WHERE telephone IS NOT NULL AND telephone != ''
        """).fetchall()
        
        print(f"\n📱 Users with phone numbers ({len(users)}):")
        for user in users:
            print(f"   - {user[0]} ({user[1]}): {user[2]}")
        
        # Check professionals with phone numbers
        professionals = conn.execute("""
            SELECT username, first_name, last_name, phone, specialization
            FROM professionals 
            WHERE phone IS NOT NULL AND phone != ''
        """).fetchall()
        
        print(f"\n👨‍⚕️ Professionals with phone numbers ({len(professionals)}):")
        for prof in professionals:
            print(f"   - {prof[0]} ({prof[1]} {prof[2]}): {prof[3]} - {prof[4]}")
        
        return len(users) > 0 and len(professionals) > 0
        
    finally:
        conn.close()

def main():
    """Create all sample data"""
    print("🚀 Creating AIMHSA Sample Data with SMS Support")
    print("=" * 60)
    
    # Create sample data
    create_sample_users()
    create_sample_professionals()
    create_admin_user()
    
    # Verify data
    print("\n" + "=" * 60)
    print("📊 Verification Results:")
    sms_ready = verify_sms_ready_data()
    
    if sms_ready:
        print("\n🎉 Sample data created successfully!")
        print("✅ Users and professionals have phone numbers")
        print("✅ SMS notifications are ready to work")
        print("\n📋 Test Credentials:")
        print("   Users: testuser, john_doe, jane_smith, rwanda_user (password: password123)")
        print("   Professionals: dr_mukamana, counselor_ntwari, psychologist_umutoni, social_worker_nyiraneza")
        print("   Admin: admin / admin123")
        print("\n🧪 Run 'python test_sms_integration.py' to test SMS functionality")
    else:
        print("\n❌ Sample data creation failed - check database connection")

if __name__ == "__main__":
    main()

