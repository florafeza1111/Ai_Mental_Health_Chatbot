#!/usr/bin/env python3
"""
Comprehensive Test Script for User Information Flow in AIMHSA
Tests the complete flow from user registration to professional dashboard display
"""

import requests
import json
import time
import uuid
from typing import Dict, Optional

# Configuration
API_BASE = "http://localhost:5057"

def test_api_endpoint(method: str, endpoint: str, data: Optional[Dict] = None, expected_status: int = 200) -> Optional[Dict]:
    """Test an API endpoint and return the response"""
    url = f"{API_BASE}{endpoint}"
    
    try:
        if method.upper() == "GET":
            response = requests.get(url)
        elif method.upper() == "POST":
            response = requests.post(url, json=data)
        elif method.upper() == "PUT":
            response = requests.put(url, json=data)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        print(f"{method} {endpoint} -> {response.status_code}")
        
        if response.status_code == expected_status:
            print(f"✅ SUCCESS: {method} {endpoint}")
            return response.json() if response.content else {}
        else:
            print(f"❌ FAILED: {method} {endpoint}")
            print(f"   Expected: {expected_status}, Got: {response.status_code}")
            try:
                error_data = response.json()
                print(f"   Error: {error_data.get('error', 'Unknown error')}")
            except:
                print(f"   Response: {response.text}")
            return None
            
    except requests.exceptions.ConnectionError:
        print(f"❌ CONNECTION ERROR: Could not connect to {API_BASE}")
        print("   Make sure the Flask app is running on port 5057")
        return None
    except Exception as e:
        print(f"❌ ERROR: {method} {endpoint} - {str(e)}")
        return None

def create_test_user() -> Optional[Dict]:
    """Create a test user with complete information"""
    print("\n🧪 Creating Test User with Complete Information")
    print("=" * 60)
    
    test_user = {
        "username": f"testuser_{int(time.time())}",
        "email": f"testuser_{int(time.time())}@example.com",
        "fullname": "Test User Complete",
        "telephone": "+250788123456",
        "province": "Kigali City",
        "district": "Gasabo",
        "password": "testpassword123"
    }
    
    result = test_api_endpoint("POST", "/register", test_user, 200)
    if result:
        print(f"✅ User created: {test_user['username']}")
        print(f"   Full Name: {test_user['fullname']}")
        print(f"   Phone: {test_user['telephone']}")
        print(f"   Email: {test_user['email']}")
        print(f"   Location: {test_user['district']}, {test_user['province']}")
        return test_user
    else:
        print("❌ Failed to create test user")
        return None

def simulate_high_risk_conversation(user_account: str) -> Optional[str]:
    """Simulate a high-risk conversation that triggers booking"""
    print(f"\n🚨 Simulating High-Risk Conversation for {user_account}")
    print("=" * 60)
    
    # Create a conversation
    conv_id = str(uuid.uuid4())
    
    # Simulate high-risk messages
    high_risk_messages = [
        "I've been feeling really hopeless lately",
        "Sometimes I think everyone would be better off without me",
        "I don't see any point in continuing",
        "I've been having thoughts of hurting myself"
    ]
    
    for i, message in enumerate(high_risk_messages):
        print(f"   Message {i+1}: {message}")
        
        # Send message to trigger risk assessment
        message_data = {
            "query": message,
            "id": conv_id,
            "account": user_account,
            "history": []
        }
        
        result = test_api_endpoint("POST", "/ask", message_data, 200)
        if result and result.get('risk_assessment'):
            risk_level = result['risk_assessment'].get('risk_level', 'unknown')
            risk_score = result['risk_assessment'].get('risk_score', 0)
            print(f"   Risk Assessment: {risk_level} (Score: {risk_score:.2f})")
            
            if risk_level in ['high', 'critical']:
                print(f"✅ High-risk conversation detected! Booking should be created.")
                return conv_id
    
    print("⚠️ No high-risk assessment triggered")
    return None

def check_professional_sessions(professional_id: str = "6") -> Optional[Dict]:
    """Check professional sessions for user information"""
    print(f"\n👨‍⚕️ Checking Professional Sessions (ID: {professional_id})")
    print("=" * 60)
    
    headers = {"X-Professional-ID": professional_id}
    
    # Get sessions
    sessions_result = test_api_endpoint("GET", "/professional/sessions", headers=headers)
    if not sessions_result:
        print("❌ Failed to get professional sessions")
        return None
    
    sessions = sessions_result if isinstance(sessions_result, list) else []
    print(f"✅ Found {len(sessions)} sessions")
    
    # Check for user contact information in sessions
    for i, session in enumerate(sessions[:3]):  # Check first 3 sessions
        print(f"\n   Session {i+1}:")
        print(f"   Booking ID: {session.get('bookingId', 'N/A')}")
        print(f"   User: {session.get('userName', 'N/A')}")
        print(f"   Risk Level: {session.get('riskLevel', 'N/A')}")
        print(f"   Phone: {session.get('userPhone', 'Not provided')}")
        print(f"   Email: {session.get('userEmail', 'Not provided')}")
        print(f"   Location: {session.get('userLocation', 'Not provided')}")
        
        # Check if contact info is present
        has_contact = any([
            session.get('userPhone'),
            session.get('userEmail'),
            session.get('userLocation')
        ])
        
        if has_contact:
            print("   ✅ Contact information available")
        else:
            print("   ⚠️ No contact information")
    
    return sessions_result

def check_professional_notifications(professional_id: str = "6") -> Optional[Dict]:
    """Check professional notifications for user information"""
    print(f"\n🔔 Checking Professional Notifications (ID: {professional_id})")
    print("=" * 60)
    
    headers = {"X-Professional-ID": professional_id}
    
    notifications_result = test_api_endpoint("GET", "/professional/notifications", headers=headers)
    if not notifications_result:
        print("❌ Failed to get professional notifications")
        return None
    
    notifications = notifications_result if isinstance(notifications_result, list) else []
    print(f"✅ Found {len(notifications)} notifications")
    
    # Check recent notifications for user contact info
    for i, notification in enumerate(notifications[:3]):  # Check first 3 notifications
        print(f"\n   Notification {i+1}:")
        print(f"   Title: {notification.get('title', 'N/A')}")
        print(f"   Message: {notification.get('message', 'N/A')[:100]}...")
        
        # Check if message contains contact information
        message = notification.get('message', '')
        has_contact_info = any([
            'Phone:' in message,
            'Email:' in message,
            'Location:' in message,
            'Contact Information:' in message
        ])
        
        if has_contact_info:
            print("   ✅ Contains user contact information")
        else:
            print("   ⚠️ No contact information in notification")
    
    return notifications_result

def check_booked_users(professional_id: str = "6") -> Optional[Dict]:
    """Check booked users for comprehensive user information"""
    print(f"\n👥 Checking Booked Users (ID: {professional_id})")
    print("=" * 60)
    
    headers = {"X-Professional-ID": professional_id}
    
    users_result = test_api_endpoint("GET", "/professional/booked-users", headers=headers)
    if not users_result:
        print("❌ Failed to get booked users")
        return None
    
    users = users_result.get('users', [])
    print(f"✅ Found {len(users)} booked users")
    
    # Check user information completeness
    for i, user in enumerate(users[:3]):  # Check first 3 users
        print(f"\n   User {i+1}:")
        print(f"   Name: {user.get('fullName', 'N/A')}")
        print(f"   Account: {user.get('userAccount', 'N/A')}")
        print(f"   Phone: {user.get('telephone', 'Not provided')}")
        print(f"   Email: {user.get('email', 'Not provided')}")
        print(f"   Location: {user.get('district', 'N/A')}, {user.get('province', 'N/A')}")
        print(f"   Total Bookings: {user.get('totalBookings', 0)}")
        print(f"   Highest Risk: {user.get('highestRiskLevel', 'N/A')}")
        
        # Check information completeness
        info_score = 0
        if user.get('fullName') and user.get('fullName') != 'Not provided':
            info_score += 1
        if user.get('telephone') and user.get('telephone') != 'Not provided':
            info_score += 1
        if user.get('email') and user.get('email') != 'Not provided':
            info_score += 1
        if user.get('district') and user.get('district') != 'Not provided':
            info_score += 1
        
        print(f"   Information Completeness: {info_score}/4")
        
        if info_score >= 3:
            print("   ✅ Complete user information")
        elif info_score >= 2:
            print("   ⚠️ Partial user information")
        else:
            print("   ❌ Incomplete user information")
    
    return users_result

def test_sms_capability() -> bool:
    """Test SMS service capability"""
    print(f"\n📱 Testing SMS Service Capability")
    print("=" * 60)
    
    # Test SMS service status
    sms_status = test_api_endpoint("GET", "/admin/sms/status")
    if sms_status:
        print("✅ SMS service is configured and ready")
        return True
    else:
        print("❌ SMS service not available")
        return False

def main():
    print("🧪 AIMHSA User Information Flow Test")
    print("=" * 80)
    print("This test verifies the complete flow of user information from registration")
    print("through booking creation to professional dashboard display.")
    print("=" * 80)
    
    # Step 1: Create test user
    test_user = create_test_user()
    if not test_user:
        print("❌ Cannot proceed without test user")
        return
    
    # Step 2: Simulate high-risk conversation
    conv_id = simulate_high_risk_conversation(test_user['username'])
    if not conv_id:
        print("⚠️ No high-risk conversation detected, but continuing with tests...")
    
    # Step 3: Check professional sessions
    sessions = check_professional_sessions()
    
    # Step 4: Check professional notifications
    notifications = check_professional_notifications()
    
    # Step 5: Check booked users
    booked_users = check_booked_users()
    
    # Step 6: Test SMS capability
    sms_available = test_sms_capability()
    
    # Summary
    print("\n" + "=" * 80)
    print("📋 TEST SUMMARY")
    print("=" * 80)
    
    print(f"✅ User Registration: {'PASS' if test_user else 'FAIL'}")
    print(f"✅ Professional Sessions: {'PASS' if sessions else 'FAIL'}")
    print(f"✅ Professional Notifications: {'PASS' if notifications else 'FAIL'}")
    print(f"✅ Booked Users: {'PASS' if booked_users else 'FAIL'}")
    print(f"✅ SMS Service: {'PASS' if sms_available else 'FAIL'}")
    
    print("\n🎯 KEY FINDINGS:")
    
    if sessions:
        sessions_list = sessions if isinstance(sessions, list) else []
        sessions_with_contact = sum(1 for s in sessions_list if s.get('userPhone') or s.get('userEmail'))
        print(f"   - {sessions_with_contact}/{len(sessions_list)} sessions have contact information")
    
    if booked_users:
        users = booked_users.get('users', [])
        complete_users = sum(1 for u in users if all([
            u.get('telephone') and u.get('telephone') != 'Not provided',
            u.get('email') and u.get('email') != 'Not provided',
            u.get('district') and u.get('district') != 'Not provided'
        ]))
        print(f"   - {complete_users}/{len(users)} users have complete contact information")
    
    print(f"   - SMS notifications: {'Available' if sms_available else 'Not configured'}")
    
    print("\n💡 RECOMMENDATIONS:")
    if not sms_available:
        print("   - Configure SMS service for complete notification flow")
    if sessions and not all(s.get('userPhone') for s in sessions):
        print("   - Ensure all users provide phone numbers during registration")
    if booked_users and not all(u.get('telephone') for u in booked_users.get('users', [])):
        print("   - Encourage users to update their contact information")
    
    print("\n🎉 User Information Flow Test Complete!")

if __name__ == "__main__":
    main()

