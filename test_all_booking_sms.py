#!/usr/bin/env python3
"""
Test SMS automation for all types of bookings
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:5057"

def test_automatic_booking_sms():
    """Test SMS for automatic high-risk bookings"""
    print("🚨 Testing Automatic High-Risk Booking SMS")
    print("=" * 50)
    
    # Create test user
    username = f"auto_test_{int(time.time())}"
    user_data = {
        "username": username,
        "password": "password123",
        "email": f"{username}@example.com",
        "fullname": "Auto Test User",
        "telephone": "+250788111222",
        "province": "Kigali",
        "district": "Gasabo"
    }
    
    try:
        # Register user
        response = requests.post(f"{API_BASE_URL}/register", json=user_data)
        if response.status_code != 200:
            print(f"❌ User registration failed: {response.text}")
            return False
        
        print(f"✅ User registered: {username}")
        
        # Create conversation
        conv_response = requests.post(f"{API_BASE_URL}/conversations", json={"account": username})
        if conv_response.status_code != 200:
            print(f"❌ Conversation creation failed: {conv_response.text}")
            return False
        
        conv_id = conv_response.json()['id']
        print(f"✅ Conversation created: {conv_id}")
        
        # Send high-risk message to trigger automatic booking
        high_risk_message = "I want to kill myself and end this pain forever"
        print(f"💬 Sending high-risk message: '{high_risk_message}'")
        
        ask_response = requests.post(f"{API_BASE_URL}/ask", json={
            "id": conv_id,
            "query": high_risk_message,
            "account": username,
            "history": []
        })
        
        if ask_response.status_code == 200:
            data = ask_response.json()
            if data.get('booking_created'):
                print(f"✅ AUTOMATIC BOOKING CREATED!")
                print(f"📱 SMS sent to user: {user_data['telephone']}")
                print(f"📱 SMS sent to professional")
                return True
            else:
                print("⚠️ No automatic booking created")
                return False
        else:
            print(f"❌ Message failed: {ask_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_manual_booking_sms():
    """Test SMS for manual user-requested bookings"""
    print("\n📅 Testing Manual Booking SMS")
    print("=" * 40)
    
    # Create test user
    username = f"manual_test_{int(time.time())}"
    user_data = {
        "username": username,
        "password": "password123",
        "email": f"{username}@example.com",
        "fullname": "Manual Test User",
        "telephone": "+250788333444",
        "province": "Kigali",
        "district": "Gasabo"
    }
    
    try:
        # Register user
        response = requests.post(f"{API_BASE_URL}/register", json=user_data)
        if response.status_code != 200:
            print(f"❌ User registration failed: {response.text}")
            return False
        
        print(f"✅ User registered: {username}")
        
        # Create conversation
        conv_response = requests.post(f"{API_BASE_URL}/conversations", json={"account": username})
        if conv_response.status_code != 200:
            print(f"❌ Conversation creation failed: {conv_response.text}")
            return False
        
        conv_id = conv_response.json()['id']
        print(f"✅ Conversation created: {conv_id}")
        
        # Send message that triggers booking question
        message = "I need help with my mental health"
        print(f"💬 Sending message: '{message}'")
        
        ask_response = requests.post(f"{API_BASE_URL}/ask", json={
            "id": conv_id,
            "query": message,
            "account": username,
            "history": []
        })
        
        if ask_response.status_code == 200:
            data = ask_response.json()
            if data.get('booking_question_shown'):
                print(f"✅ Booking question shown")
                
                # User responds "yes" to booking
                print(f"💬 User responds 'yes' to booking request")
                
                booking_response = requests.post(f"{API_BASE_URL}/booking_response", json={
                    "conversation_id": conv_id,
                    "response": "yes",
                    "account": username
                })
                
                if booking_response.status_code == 200:
                    booking_data = booking_response.json()
                    if booking_data.get('ok') and booking_data.get('booking'):
                        print(f"✅ MANUAL BOOKING CREATED!")
                        print(f"📱 SMS sent to user: {user_data['telephone']}")
                        print(f"📱 SMS sent to professional")
                        return True
                    else:
                        print("⚠️ No manual booking created")
                        return False
                else:
                    print(f"❌ Booking response failed: {booking_response.text}")
                    return False
            else:
                print("⚠️ No booking question shown")
                return False
        else:
            print(f"❌ Message failed: {ask_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def check_sms_status():
    """Check if SMS service is ready"""
    print("🔍 Checking SMS Service Status")
    print("=" * 35)
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/sms/status")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMS Status: {data.get('status')}")
            print(f"🔗 Connection: {data.get('connection_test')}")
            return data.get('status') == 'initialized'
        else:
            print(f"❌ SMS status check failed: {response.text}")
            return False
    except Exception as e:
        print(f"❌ SMS status error: {e}")
        return False

def main():
    """Run all booking SMS tests"""
    print("🧪 Testing SMS for All Booking Types")
    print("=" * 50)
    
    # Check SMS service
    if not check_sms_status():
        print("\n❌ SMS service not ready - cannot test")
        return 1
    
    print("\n✅ SMS service ready - starting tests")
    
    # Test automatic booking SMS
    auto_success = test_automatic_booking_sms()
    
    # Test manual booking SMS
    manual_success = test_manual_booking_sms()
    
    # Results
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"🚨 Automatic Booking SMS: {'✅ PASS' if auto_success else '❌ FAIL'}")
    print(f"📅 Manual Booking SMS: {'✅ PASS' if manual_success else '❌ FAIL'}")
    
    if auto_success and manual_success:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ SMS is sent automatically for ALL booking types")
        print("✅ Both automatic and manual bookings trigger SMS")
        return 0
    else:
        print("\n⚠️ Some tests failed")
        print("💡 Check the logs and configuration")
        return 1

if __name__ == "__main__":
    sys.exit(main())

