#!/usr/bin/env python3
"""
Test script for SMS integration with AIMHSA
Tests the HDEV SMS API integration and booking notifications
"""

import requests
import json
import time
import sys

# Configuration
API_BASE_URL = "http://localhost:5057"
SMS_API_ID = "HDEV-23fb1b59-aec0-4aef-a351-bfc1c3aa3c52-ID"
SMS_API_KEY = "HDEV-6e36c286-19bb-4b45-838e-8b5cd0240857-KEY"

def test_sms_status():
    """Test SMS service status"""
    print("🔍 Testing SMS service status...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/sms/status")
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMS Status: {data.get('status')}")
            print(f"📱 API ID: {data.get('api_id')}")
            print(f"🔑 API Key: {data.get('api_key_masked')}")
            print(f"🔗 Connection Test: {data.get('connection_test')}")
            print(f"💬 Message: {data.get('message')}")
            return True
        else:
            print(f"❌ SMS status check failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SMS status: {e}")
        return False

def test_sms_send():
    """Test sending SMS message"""
    print("\n📤 Testing SMS send functionality...")
    
    # Get test phone number from user
    test_phone = input("Enter test phone number (Rwanda format, e.g., +250788123456): ").strip()
    if not test_phone:
        test_phone = "+250000000000"  # Dummy number for testing
    
    test_message = "AIMHSA SMS Integration Test - Service is working correctly! 🎉"
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/sms/test", json={
            "phone": test_phone,
            "message": test_message
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ SMS Test Result: {data.get('success')}")
            print(f"📱 Phone: {data.get('result', {}).get('phone', 'N/A')}")
            print(f"💬 Message: {data.get('message')}")
            
            if data.get('success'):
                print("🎉 SMS sent successfully!")
            else:
                print("⚠️ SMS sending failed - check API credentials and phone number format")
            
            return data.get('success', False)
        else:
            print(f"❌ SMS test failed: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing SMS send: {e}")
        return False

def test_user_registration():
    """Test user registration with phone number"""
    print("\n👤 Testing user registration with phone number...")
    
    test_username = f"testuser_{int(time.time())}"
    test_phone = input("Enter phone number for test user (e.g., +250788123456): ").strip()
    if not test_phone:
        test_phone = "+250788123456"
    
    try:
        # Register user
        response = requests.post(f"{API_BASE_URL}/register", json={
            "username": test_username,
            "password": "password123",
            "email": f"{test_username}@example.com",
            "fullname": "Test User",
            "telephone": test_phone,
            "province": "Kigali",
            "district": "Gasabo"
        })
        
        if response.status_code == 200:
            data = response.json()
            print(f"✅ User registered: {data.get('ok')}")
            print(f"👤 Username: {test_username}")
            print(f"📱 Phone: {test_phone}")
            return test_username
        else:
            print(f"❌ User registration failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error registering user: {e}")
        return None

def test_booking_sms():
    """Test booking SMS notification"""
    print("\n📅 Testing booking SMS notification...")
    
    # First, register a test user
    test_username = test_user_registration()
    if not test_username:
        print("❌ Cannot test booking SMS without user registration")
        return False
    
    # Create a test conversation and trigger risk assessment
    print("💬 Creating test conversation...")
    
    try:
        # Create conversation
        conv_response = requests.post(f"{API_BASE_URL}/conversations", json={
            "account": test_username
        })
        
        if conv_response.status_code != 200:
            print(f"❌ Failed to create conversation: {conv_response.status_code}")
            return False
        
        conv_data = conv_response.json()
        conv_id = conv_data.get('id')
        print(f"✅ Conversation created: {conv_id}")
        
        # Send a high-risk message to trigger booking
        print("🚨 Sending high-risk message to trigger booking...")
        
        risk_message = "I feel hopeless and want to end it all. I can't take this pain anymore."
        
        ask_response = requests.post(f"{API_BASE_URL}/ask", json={
            "id": conv_id,
            "query": risk_message,
            "account": test_username,
            "history": []
        })
        
        if ask_response.status_code == 200:
            ask_data = ask_response.json()
            print(f"✅ Risk assessment completed")
            print(f"🎯 Risk level: {ask_data.get('risk_level', 'unknown')}")
            
            if ask_data.get('booking_created'):
                print("🎉 Automated booking created!")
                print(f"📋 Booking ID: {ask_data.get('booking_id')}")
                print(f"👨‍⚕️ Professional: {ask_data.get('professional_name')}")
                print(f"⏰ Session Type: {ask_data.get('session_type')}")
                
                # Check if SMS was sent
                print("📱 SMS notifications should have been sent automatically")
                return True
            else:
                print("⚠️ No booking was created - risk level may not be high enough")
                return False
        else:
            print(f"❌ Failed to send message: {ask_response.status_code}")
            print(f"Response: {ask_response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error testing booking SMS: {e}")
        return False

def main():
    """Run all SMS integration tests"""
    print("🚀 AIMHSA SMS Integration Test Suite")
    print("=" * 50)
    
    # Test 1: SMS Service Status
    status_ok = test_sms_status()
    
    # Test 2: SMS Send Test
    if status_ok:
        sms_ok = test_sms_send()
    else:
        print("⚠️ Skipping SMS send test due to status check failure")
        sms_ok = False
    
    # Test 3: User Registration with Phone
    user_ok = test_user_registration()
    
    # Test 4: Booking SMS Notification
    if user_ok and sms_ok:
        booking_ok = test_booking_sms()
    else:
        print("⚠️ Skipping booking SMS test due to previous failures")
        booking_ok = False
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    print(f"🔍 SMS Status: {'✅ PASS' if status_ok else '❌ FAIL'}")
    print(f"📤 SMS Send: {'✅ PASS' if sms_ok else '❌ FAIL'}")
    print(f"👤 User Registration: {'✅ PASS' if user_ok else '❌ FAIL'}")
    print(f"📅 Booking SMS: {'✅ PASS' if booking_ok else '❌ FAIL'}")
    
    if all([status_ok, sms_ok, user_ok, booking_ok]):
        print("\n🎉 All tests passed! SMS integration is working correctly.")
        return 0
    else:
        print("\n⚠️ Some tests failed. Check the output above for details.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

