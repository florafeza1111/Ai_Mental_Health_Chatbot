#!/usr/bin/env python3
"""
Example usage of SMS integration in AIMHSA
Demonstrates how to send SMS notifications for bookings
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5057"

def example_booking_with_sms():
    """
    Example: Complete booking flow with SMS notifications
    """
    print("🚀 AIMHSA SMS Integration Example")
    print("=" * 50)
    
    # Step 1: Register a user with phone number
    print("1️⃣ Registering user with phone number...")
    
    user_data = {
        "username": f"demo_user_{int(time.time())}",
        "password": "password123",
        "email": f"demo_{int(time.time())}@example.com",
        "fullname": "Demo User",
        "telephone": "+250788123456",  # Rwanda phone number
        "province": "Kigali",
        "district": "Gasabo"
    }
    
    response = requests.post(f"{API_BASE_URL}/register", json=user_data)
    
    if response.status_code == 200:
        print(f"✅ User registered: {user_data['username']}")
        print(f"📱 Phone: {user_data['telephone']}")
    else:
        print(f"❌ Registration failed: {response.text}")
        return
    
    # Step 2: Create conversation
    print("\n2️⃣ Creating conversation...")
    
    conv_response = requests.post(f"{API_BASE_URL}/conversations", json={
        "account": user_data['username']
    })
    
    if conv_response.status_code == 200:
        conv_data = conv_response.json()
        conv_id = conv_data['id']
        print(f"✅ Conversation created: {conv_id}")
    else:
        print(f"❌ Conversation creation failed: {conv_response.text}")
        return
    
    # Step 3: Send high-risk message to trigger booking
    print("\n3️⃣ Sending high-risk message...")
    
    high_risk_message = "I feel completely hopeless and want to end my life. I can't take this pain anymore."
    
    ask_response = requests.post(f"{API_BASE_URL}/ask", json={
        "id": conv_id,
        "query": high_risk_message,
        "account": user_data['username'],
        "history": []
    })
    
    if ask_response.status_code == 200:
        ask_data = ask_response.json()
        print(f"✅ Message processed")
        print(f"🎯 Risk level: {ask_data.get('risk_level', 'unknown')}")
        
        if ask_data.get('booking_created'):
            print(f"🎉 Automated booking created!")
            print(f"📋 Booking ID: {ask_data.get('booking_id')}")
            print(f"👨‍⚕️ Professional: {ask_data.get('professional_name')}")
            print(f"⏰ Session Type: {ask_data.get('session_type')}")
            print(f"📱 SMS notifications sent to user and professional!")
        else:
            print("⚠️ No booking created - risk level may not be high enough")
    else:
        print(f"❌ Message sending failed: {ask_response.text}")

def example_manual_sms_test():
    """
    Example: Manual SMS testing
    """
    print("\n" + "=" * 50)
    print("📱 Manual SMS Testing")
    print("=" * 50)
    
    # Test SMS service status
    print("1️⃣ Checking SMS service status...")
    
    status_response = requests.get(f"{API_BASE_URL}/admin/sms/status")
    
    if status_response.status_code == 200:
        status_data = status_response.json()
        print(f"✅ SMS Status: {status_data.get('status')}")
        print(f"🔗 Connection Test: {status_data.get('connection_test')}")
    else:
        print(f"❌ Status check failed: {status_response.text}")
        return
    
    # Test sending SMS
    print("\n2️⃣ Testing SMS send...")
    
    test_phone = input("Enter test phone number (e.g., +250788123456): ").strip()
    if not test_phone:
        test_phone = "+250788123456"
    
    sms_response = requests.post(f"{API_BASE_URL}/admin/sms/test", json={
        "phone": test_phone,
        "message": "AIMHSA SMS Test - Integration is working! 🎉"
    })
    
    if sms_response.status_code == 200:
        sms_data = sms_response.json()
        print(f"✅ SMS Test: {sms_data.get('success')}")
        print(f"📱 Phone: {sms_data.get('result', {}).get('phone', 'N/A')}")
    else:
        print(f"❌ SMS test failed: {sms_response.text}")

def example_professional_setup():
    """
    Example: Setting up professional with phone number
    """
    print("\n" + "=" * 50)
    print("👨‍⚕️ Professional Setup Example")
    print("=" * 50)
    
    # This would typically be done through the admin dashboard
    # Here we show the data structure needed
    
    professional_data = {
        "username": "dr_mukamana",
        "password": "password123",
        "first_name": "Marie",
        "last_name": "Mukamana",
        "email": "dr.mukamana@example.com",
        "phone": "+250788111222",  # Professional phone number
        "specialization": "psychiatrist",
        "expertise_areas": ["depression", "anxiety", "ptsd", "crisis"],
        "district": "Gasabo",
        "consultation_fee": 50000,
        "bio": "Experienced psychiatrist specializing in trauma and crisis intervention"
    }
    
    print("Professional data structure for SMS notifications:")
    print(json.dumps(professional_data, indent=2))
    print("\nNote: This would be created via the admin dashboard")

if __name__ == "__main__":
    print("Choose an example to run:")
    print("1. Complete booking flow with SMS")
    print("2. Manual SMS testing")
    print("3. Professional setup example")
    print("4. Run all examples")
    
    choice = input("\nEnter choice (1-4): ").strip()
    
    if choice == "1":
        example_booking_with_sms()
    elif choice == "2":
        example_manual_sms_test()
    elif choice == "3":
        example_professional_setup()
    elif choice == "4":
        example_booking_with_sms()
        example_manual_sms_test()
        example_professional_setup()
    else:
        print("Invalid choice. Running complete booking flow...")
        example_booking_with_sms()
    
    print("\n🎉 Example completed!")
    print("Check the SMS_INTEGRATION_README.md for more details.")

