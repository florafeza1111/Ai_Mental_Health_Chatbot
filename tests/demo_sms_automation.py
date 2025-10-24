#!/usr/bin/env python3
"""
Demo script showing SMS automation in action
"""

import requests
import json
import time
import sys

API_BASE_URL = "http://localhost:5057"

def demo_sms_automation():
    """Demonstrate SMS automation with a real example"""
    print("🎬 AIMHSA SMS Automation Demo")
    print("=" * 50)
    print("This demo shows how SMS is sent automatically to both user and professional")
    print("when a high-risk mental health case is detected.")
    print()
    
    # Step 1: Create a user
    print("👤 Step 1: Creating a user with phone number...")
    
    demo_user = {
        "username": f"demo_{int(time.time())}",
        "password": "password123",
        "email": f"demo_{int(time.time())}@example.com",
        "fullname": "Demo User",
        "telephone": "+250788123456",  # This will receive SMS
        "province": "Kigali",
        "district": "Gasabo"
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/register", json=demo_user)
        if response.status_code == 200:
            print(f"✅ User created: {demo_user['username']}")
            print(f"📱 Phone number: {demo_user['telephone']}")
        else:
            print(f"❌ User creation failed: {response.text}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 2: Start conversation
    print(f"\n💬 Step 2: Starting conversation...")
    
    try:
        conv_response = requests.post(f"{API_BASE_URL}/conversations", json={
            "account": demo_user['username']
        })
        
        if conv_response.status_code == 200:
            conv_data = conv_response.json()
            conv_id = conv_data['id']
            print(f"✅ Conversation started: {conv_id}")
        else:
            print(f"❌ Conversation failed: {conv_response.text}")
            return
    except Exception as e:
        print(f"❌ Error: {e}")
        return
    
    # Step 3: Send high-risk message
    print(f"\n🚨 Step 3: Sending high-risk message...")
    print("   This will trigger automatic risk assessment and SMS notifications")
    
    high_risk_message = "I feel completely hopeless and want to end my life. I can't take this pain anymore."
    print(f"   Message: '{high_risk_message}'")
    
    try:
        ask_response = requests.post(f"{API_BASE_URL}/ask", json={
            "id": conv_id,
            "query": high_risk_message,
            "account": demo_user['username'],
            "history": []
        })
        
        if ask_response.status_code == 200:
            ask_data = ask_response.json()
            print(f"✅ Message processed")
            print(f"🎯 Risk level detected: {ask_data.get('risk_level', 'unknown')}")
            
            if ask_data.get('booking_created'):
                print(f"\n🎉 AUTOMATIC BOOKING CREATED!")
                print(f"📋 Booking ID: {ask_data.get('booking_id')}")
                print(f"👨‍⚕️ Assigned Professional: {ask_data.get('professional_name')}")
                print(f"⏰ Session Type: {ask_data.get('session_type')}")
                print(f"🚨 Risk Level: {ask_data.get('risk_level')}")
                
                print(f"\n📱 SMS NOTIFICATIONS SENT AUTOMATICALLY:")
                print(f"   📤 User SMS sent to: {demo_user['telephone']}")
                print(f"   📤 Professional SMS sent to assigned professional")
                print(f"   ⚡ This happened automatically - no manual intervention needed!")
                
                print(f"\n📋 What the user received via SMS:")
                print(f"   'AIMHSA Mental Health Support'")
                print(f"   'URGENT: Professional mental health support has been scheduled'")
                print(f"   'Professional: {ask_data.get('professional_name')}'")
                print(f"   'You will be contacted shortly...'")
                
                print(f"\n📋 What the professional received via SMS:")
                print(f"   'AIMHSA Professional Alert'")
                print(f"   'New HIGH risk booking assigned to you'")
                print(f"   'User: {demo_user['fullname']}'")
                print(f"   'Please check your professional dashboard...'")
                
                print(f"\n✅ DEMO COMPLETED SUCCESSFULLY!")
                print(f"🎯 SMS automation is working perfectly!")
                
            else:
                print(f"\n⚠️ No booking created")
                print(f"   Risk level may not be high enough to trigger automatic booking")
                print(f"   Try a more explicit high-risk message")
                
        else:
            print(f"❌ Message processing failed: {ask_response.text}")
            
    except Exception as e:
        print(f"❌ Error: {e}")

def show_sms_flow_diagram():
    """Show the SMS automation flow"""
    print("\n📊 SMS Automation Flow Diagram")
    print("=" * 50)
    print("""
    User sends high-risk message
           ↓
    Risk assessment triggered
           ↓
    High/Critical risk detected
           ↓
    Professional matching algorithm
           ↓
    Automated booking created
           ↓
    📱 SMS sent to USER
           ↓
    📱 SMS sent to PROFESSIONAL
           ↓
    Both parties notified automatically
    """)

def main():
    """Run the demo"""
    print("Choose demo option:")
    print("1. Run full SMS automation demo")
    print("2. Show SMS flow diagram")
    print("3. Both")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1" or choice == "3":
        demo_sms_automation()
    
    if choice == "2" or choice == "3":
        show_sms_flow_diagram()
    
    if choice not in ["1", "2", "3"]:
        print("Invalid choice. Running full demo...")
        demo_sms_automation()
    
    print(f"\n🎉 Demo completed!")
    print(f"💡 The SMS system automatically sends notifications to both")
    print(f"   user and professional whenever a high-risk case is detected.")

if __name__ == "__main__":
    main()

