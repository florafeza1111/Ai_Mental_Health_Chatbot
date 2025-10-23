#!/usr/bin/env python3
"""
Test script to verify the professional session details API endpoint
"""

import requests
import json

def test_session_details_api():
    """Test the session details API endpoint"""
    
    # Test data - using the booking ID from the user's example
    booking_id = "d63a7794-a89c-452c-80a6-24691e3cb848"
    professional_id = "1"  # Jean Ntwari
    
    # API endpoint
    url = f"http://localhost:5057/professional/sessions/{booking_id}"
    
    # Headers
    headers = {
        'X-Professional-ID': professional_id,
        'Content-Type': 'application/json'
    }
    
    try:
        print(f"Testing API endpoint: {url}")
        print(f"Headers: {headers}")
        print("-" * 50)
        
        # Make the request
        response = requests.get(url, headers=headers)
        
        print(f"Status Code: {response.status_code}")
        print(f"Response Headers: {dict(response.headers)}")
        print("-" * 50)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ API Response Success!")
            print(f"Response Data: {json.dumps(data, indent=2)}")
            
            # Check specific fields
            print("\n" + "="*50)
            print("USER DATA VERIFICATION:")
            print("="*50)
            print(f"User Account: {data.get('userAccount', 'NOT FOUND')}")
            print(f"User Name: {data.get('userName', 'NOT FOUND')}")
            print(f"User Full Name: {data.get('userFullName', 'NOT FOUND')}")
            print(f"User Email: {data.get('userEmail', 'NOT FOUND')}")
            print(f"User Phone: {data.get('userPhone', 'NOT FOUND')}")
            print(f"User Province: {data.get('userProvince', 'NOT FOUND')}")
            print(f"User District: {data.get('userDistrict', 'NOT FOUND')}")
            print(f"User Created At: {data.get('userCreatedAt', 'NOT FOUND')}")
            print(f"User Location: {data.get('userLocation', 'NOT FOUND')}")
            
            print("\n" + "="*50)
            print("SESSION DATA:")
            print("="*50)
            print(f"Booking ID: {data.get('bookingId', 'NOT FOUND')}")
            print(f"Risk Level: {data.get('riskLevel', 'NOT FOUND')}")
            print(f"Risk Score: {data.get('riskScore', 'NOT FOUND')}")
            print(f"Session Type: {data.get('sessionType', 'NOT FOUND')}")
            print(f"Booking Status: {data.get('bookingStatus', 'NOT FOUND')}")
            
            print("\n" + "="*50)
            print("ADDITIONAL DATA:")
            print("="*50)
            print(f"Sessions Count: {len(data.get('sessions', []))}")
            print(f"Risk Assessments Count: {len(data.get('riskAssessments', []))}")
            print(f"Conversation History Count: {len(data.get('conversationHistory', []))}")
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"Error Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Could not connect to the API server")
        print("Make sure the Flask app is running on http://localhost:5057")
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")

if __name__ == "__main__":
    test_session_details_api()
