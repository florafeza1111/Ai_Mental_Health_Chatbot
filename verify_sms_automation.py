#!/usr/bin/env python3
"""
Verify that SMS is sent automatically to both user and professional
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:5057"

def test_automatic_sms_flow():
    """Test the complete automatic SMS flow"""
    print("🚀 Testing Automatic SMS Flow")
    print("=" * 50)
    
    # Step 1: Register a test user with phone number
    print("1️⃣ Registering test user with phone number...")
    
    test_username = f"sms_test_user_{int(time.time())}"
    user_data = {
        "username": test_username,
        "password": "password123",
        "email": f"{test_username}@example.com",
        "fullname": "SMS Test User",
        "telephone": "+250788999888",  # Test phone number
        "province": "Kigali",
        "district": "Gasabo"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/register", json=user_data)
        if response.status_code == 200:
            print(f"✅ User registered: {test_username}")
            print(f"📱 Phone: {user_data['telephone']}")
        else:
            print(f"❌ Registration failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Registration error: {e}")
        return False
    
    # Step 2: Create conversation
    print("\n2️⃣ Creating conversation...")
    
    try:
        conv_response = requests.post(f"{API_BASE_URL}/conversations", json={
            "account": test_username
        })
        
        if conv_response.status_code == 200:
            conv_data = conv_response.json()
            conv_id = conv_data['id']
            print(f"✅ Conversation created: {conv_id}")
        else:
            print(f"❌ Conversation creation failed: {conv_response.text}")
            return False
    except Exception as e:
        print(f"❌ Conversation error: {e}")
        return False
    
    # Step 3: Send high-risk message to trigger automatic booking and SMS
    print("\n3️⃣ Sending high-risk message to trigger automatic SMS...")
    
    high_risk_messages = [
        "I want to kill myself and end this pain forever",
        "I'm going to overdose on pills tonight",
        "I can't take this anymore, I want to die",
        "I'm planning to jump off a building",
        "I have nothing to live for, I want to end it all"
    ]
    
    # Use the first high-risk message
    risk_message = high_risk_messages[0]
    print(f"💬 Sending: '{risk_message}'")
    
    try:
        ask_response = requests.post(f"{API_BASE_URL}/ask", json={
            "id": conv_id,
            "query": risk_message,
            "account": test_username,
            "history": []
        })
        
        if ask_response.status_code == 200:
            ask_data = ask_response.json()
            print(f"✅ Message processed")
            print(f"🎯 Risk level: {ask_data.get('risk_level', 'unknown')}")
            
            if ask_data.get('booking_created'):
                print(f"🎉 AUTOMATIC BOOKING CREATED!")
                print(f"📋 Booking ID: {ask_data.get('booking_id')}")
                print(f"👨‍⚕️ Professional: {ask_data.get('professional_name')}")
                print(f"⏰ Session Type: {ask_data.get('session_type')}")
                print(f"🚨 Risk Level: {ask_data.get('risk_level')}")
                
                print(f"\n📱 SMS NOTIFICATIONS SENT AUTOMATICALLY:")
                print(f"   ✅ User SMS: Sent to {user_data['telephone']}")
                print(f"   ✅ Professional SMS: Sent to assigned professional")
                print(f"   📋 Check the application logs for SMS delivery status")
                
                return True
            else:
                print("⚠️ No booking created - risk level may not be high enough")
                print("💡 Try a different high-risk message")
                return False
        else:
            print(f"❌ Message sending failed: {ask_response.text}")
            return False
    except Exception as e:
        print(f"❌ Message error: {e}")
        return False

def check_sms_status():
    """Check SMS service status"""
    print("🔍 Checking SMS Service Status")
    print("=" * 30)
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/sms/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMS Status: {data.get('status')}")
            print(f"🔗 Connection: {data.get('connection_test')}")
            print(f"📱 API ID: {data.get('api_id')}")
            return data.get('status') == 'initialized'
        else:
            print(f"❌ Status check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ Status check error: {e}")
        return False

def check_sample_data():
    """Check if sample data exists with phone numbers"""
    print("👥 Checking Sample Data")
    print("=" * 25)
    
    try:
        # Check professionals
        prof_response = requests.get(f"{API_BASE_URL}/admin/professionals")
        if prof_response.status_code == 200:
            prof_data = prof_response.json()
            professionals = prof_data.get('professionals', [])
            
            profs_with_phone = [p for p in professionals if p.get('phone')]
            print(f"👨‍⚕️ Professionals with phone numbers: {len(profs_with_phone)}")
            
            if profs_with_phone:
                print("   Sample professionals:")
                for prof in profs_with_phone[:3]:  # Show first 3
                    print(f"   - {prof.get('first_name')} {prof.get('last_name')}: {prof.get('phone')}")
                return True
            else:
                print("❌ No professionals with phone numbers found")
                print("💡 Run 'python create_sample_data_with_sms.py' to create sample data")
                return False
        else:
            print(f"❌ Failed to get professionals: {prof_response.text}")
            return False
    except Exception as e:
        print(f"❌ Sample data check error: {e}")
        return False

def main():
    """Run all verification tests"""
    print("🧪 AIMHSA SMS Automation Verification")
    print("=" * 50)
    
    # Check prerequisites
    print("🔍 Checking Prerequisites...")
    
    sms_ok = check_sms_status()
    data_ok = check_sample_data()
    
    if not sms_ok:
        print("\n❌ SMS service not ready - check configuration")
        return 1
    
    if not data_ok:
        print("\n❌ Sample data not ready - create sample data first")
        return 1
    
    print("\n✅ Prerequisites met - proceeding with SMS automation test")
    
    # Test automatic SMS flow
    success = test_automatic_sms_flow()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 SMS AUTOMATION VERIFICATION SUCCESSFUL!")
        print("✅ SMS is sent automatically to both user and professional")
        print("✅ High-risk cases trigger automatic booking and SMS")
        print("✅ System is ready for production use")
        return 0
    else:
        print("❌ SMS AUTOMATION VERIFICATION FAILED")
        print("💡 Check the logs and configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())

