#!/usr/bin/env python3
"""
Add sample data to the database for testing
"""

import sqlite3
import time
import hashlib
import uuid

# Database file path
DB_FILE = "aimhsa.db"

def hash_password(password):
    """Hash a password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def add_sample_data():
    """Add sample data to the database"""
    
    print("="*60)
    print("ADDING SAMPLE DATA")
    print("="*60)
    
    conn = sqlite3.connect(DB_FILE)
    
    try:
        current_time = time.time()
        
        # Add sample user
        print("Adding sample user...")
        conn.execute("""
            INSERT OR REPLACE INTO users (
                username, password_hash, created_ts, email, fullname, 
                telephone, province, district, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'Mugisha',
            hash_password('password123'),
            current_time,
            'mugisha@example.com',
            'Mugisha DUKUZUMUREMYI',
            '0785354935',
            'Eastern',
            'Kirehe',
            current_time
        ))
        print("✅ Sample user 'Mugisha' added")
        
        # Add sample professional
        print("Adding sample professional...")
        conn.execute("""
            INSERT OR REPLACE INTO professionals (
                username, password_hash, first_name, last_name, email, phone,
                license_number, specialization, expertise_areas, languages,
                qualifications, availability_schedule, location_latitude,
                location_longitude, location_address, district, max_patients_per_day,
                consultation_fee, experience_years, bio, profile_picture,
                is_active, created_ts, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'jean.ntwari',
            hash_password('password123'),
            'Jean',
            'Ntwari',
            'jean.ntwari@example.com',
            '+250788123456',
            'LIC123456',
            'Clinical Psychologist',
            '["Depression", "Anxiety", "Trauma", "Stress Management"]',
            '["English", "Kinyarwanda", "French"]',
            '["PhD in Clinical Psychology", "Licensed Clinical Psychologist"]',
            '{"monday": "09:00-17:00", "tuesday": "09:00-17:00", "wednesday": "09:00-17:00", "thursday": "09:00-17:00", "friday": "09:00-17:00"}',
            -1.9441,
            30.0619,
            'Kigali, Rwanda',
            'Kigali',
            10,
            50000.0,
            8,
            'Experienced clinical psychologist specializing in trauma therapy and cognitive behavioral therapy.',
            None,
            1,
            current_time,
            current_time
        ))
        print("✅ Sample professional 'Jean Ntwari' added")
        
        # Get professional ID
        professional = conn.execute("SELECT id FROM professionals WHERE username = 'jean.ntwari'").fetchone()
        professional_id = professional[0]
        
        # Add sample conversation
        conv_id = str(uuid.uuid4())
        print("Adding sample conversation...")
        conn.execute("""
            INSERT OR REPLACE INTO conversations (
                conv_id, owner_key, preview, ts
            ) VALUES (?, ?, ?, ?)
        """, (
            conv_id,
            'Mugisha',
            'User is feeling overwhelmed and struggling with low mood. Needs support and resources.',
            current_time
        ))
        print("✅ Sample conversation added")
        
        # Add sample messages
        print("Adding sample messages...")
        messages = [
            (conv_id, 'user', 'I am feeling overwhelmed and tired. I need help.', current_time - 3600),
            (conv_id, 'assistant', 'I understand you\'re feeling overwhelmed and tired. That sounds really difficult. Can you tell me more about what\'s been going on?', current_time - 3500),
            (conv_id, 'user', 'I have been struggling with low mood and lack of motivation. Everything feels too much.', current_time - 3400),
            (conv_id, 'assistant', 'I hear that you\'re experiencing low mood and feeling like everything is too much. These feelings are valid and it\'s important that you\'re reaching out for support. Would you like me to help you connect with a mental health professional?', current_time - 3300),
            (conv_id, 'user', 'Yes, I think I need professional help. I am in Rwanda and not sure where to find good mental health support.', current_time - 3200),
            (conv_id, 'assistant', 'I can help you connect with a qualified mental health professional in Rwanda. Based on your needs, I\'ll create a booking for you with a professional who specializes in mood disorders and stress management.', current_time - 3100)
        ]
        
        for msg in messages:
            conn.execute("""
                INSERT OR REPLACE INTO messages (
                    conv_id, role, content, ts
                ) VALUES (?, ?, ?, ?)
            """, msg)
        
        print("✅ Sample messages added")
        
        # Add sample automated booking
        booking_id = 'd63a7794-a89c-452c-80a6-24691e3cb848'
        print("Adding sample automated booking...")
        conn.execute("""
            INSERT OR REPLACE INTO automated_bookings (
                booking_id, conv_id, user_account, user_ip, professional_id,
                risk_level, risk_score, detected_indicators, conversation_summary,
                booking_status, scheduled_datetime, session_type, location_preference,
                notes, created_ts, updated_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            booking_id,
            conv_id,
            'Mugisha',
            '127.0.0.1',
            professional_id,
            'medium',
            0.5,
            '["user_requested_booking"]',
            '**Professional Summary:**\n\n**Client:** Female, Rwandan national\n\n**Presenting Concerns:** Feeling overwhelmed, tired, and struggling with low mood.\n\n**Emotional State:** The client is expressing emotional distress, feeling not good, and acknowledging the need for support. She appears to be experiencing significant stress and is open to exploring resources for mental health support in Rwanda.\n\n**Risk Factors:** None explicitly mentioned, but the context suggests a history of challenges faced by Rwanda, which may contribute to the client\'s current emotional state.\n\n**Key Issues:**\n\n* Feeling overwhelmed and tired\n* Low mood and lack of motivation\n* Difficulty in accessing mental health resources in Rwanda\n* Interest in exploring relaxation techniques, journaling, or grounding exercises as coping mechanisms\n\nThis conversation highlights the importance of acknowledging and addressing the client\'s emotional pain, while also providing her with information and support to access necessary resources. Further guidance on managing stress and improving mental well-being is warranted.\n\n**Recommendations:**\n\n* Schedule a follow-up appointment to assess the client\'s progress and provide ongoing support.\n* Explore additional coping mechanisms, such as cognitive-behavioral therapy (CBT) or mindfulness-based interventions.\n* Connect the client with local resources, such as mental health hotlines or counseling services, for continued support.\n\n**Next Steps:**\n\n* Coordinate follow-up appointments to ensure the client\'s ongoing progress and well-being.\n* Monitor the client\'s emotional state and adjust treatment plans as needed.',
            'confirmed',
            current_time + 86400,  # Tomorrow
            'urgent',
            'Kigali',
            'Client needs immediate support for mood and stress management.',
            current_time,
            current_time + 200  # Updated 200 seconds later
        ))
        print("✅ Sample automated booking added")
        
        # Add sample conversation messages
        print("Adding conversation messages...")
        conv_messages = [
            (conv_id, 'user', 'I am feeling overwhelmed and tired. I need help.', current_time - 3600),
            (conv_id, 'assistant', 'I understand you\'re feeling overwhelmed and tired. That sounds really difficult. Can you tell me more about what\'s been going on?', current_time - 3500),
            (conv_id, 'user', 'I have been struggling with low mood and lack of motivation. Everything feels too much.', current_time - 3400),
            (conv_id, 'assistant', 'I hear that you\'re experiencing low mood and feeling like everything is too much. These feelings are valid and it\'s important that you\'re reaching out for support. Would you like me to help you connect with a mental health professional?', current_time - 3300),
            (conv_id, 'user', 'Yes, I think I need professional help. I am in Rwanda and not sure where to find good mental health support.', current_time - 3200),
            (conv_id, 'assistant', 'I can help you connect with a qualified mental health professional in Rwanda. Based on your needs, I\'ll create a booking for you with a professional who specializes in mood disorders and stress management.', current_time - 3100)
        ]
        
        for msg in conv_messages:
            conn.execute("""
                INSERT OR REPLACE INTO conversation_messages (
                    conv_id, sender, content, timestamp
                ) VALUES (?, ?, ?, ?)
            """, msg)
        
        print("✅ Conversation messages added")
        
        # Add sample professional notification
        print("Adding sample professional notification...")
        conn.execute("""
            INSERT OR REPLACE INTO professional_notifications (
                professional_id, booking_id, notification_type, title, message,
                is_read, priority, created_ts
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            professional_id,
            booking_id,
            'new_booking',
            'URGENT: MEDIUM Risk Case - Mugisha DUKUZUMUREMYI',
            'Automated booking created for medium risk case. Risk indicators: user_requested_booking\n\nUser Contact Information:\nName: Mugisha DUKUZUMUREMYI\nPhone: 0785354935\nEmail: mugisha@example.com\nLocation: Kirehe, Eastern',
            0,
            'high',
            current_time
        ))
        print("✅ Sample professional notification added")
        
        # Commit all changes
        conn.commit()
        
        print("\n" + "="*60)
        print("SAMPLE DATA ADDITION COMPLETE!")
        print("="*60)
        
        # Verify data was added
        user_count = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        professional_count = conn.execute("SELECT COUNT(*) FROM professionals").fetchone()[0]
        booking_count = conn.execute("SELECT COUNT(*) FROM automated_bookings").fetchone()[0]
        message_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        
        print(f"\nData Summary:")
        print(f"   Users: {user_count}")
        print(f"   Professionals: {professional_count}")
        print(f"   Bookings: {booking_count}")
        print(f"   Messages: {message_count}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error adding sample data: {e}")
        conn.close()
        raise

if __name__ == "__main__":
    add_sample_data()

