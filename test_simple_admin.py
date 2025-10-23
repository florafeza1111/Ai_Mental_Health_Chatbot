#!/usr/bin/env python3
"""
Simple Admin Dashboard Test
Tests the professional management functionality
"""

import requests
import json

# Configuration
API_BASE_URL = "http://localhost:5057"
ADMIN_EMAIL = "eliasfeza@gmail.com"
ADMIN_PASSWORD = "EliasFeza@12301"

def test_admin_login():
    """Test admin login"""
    print("Testing admin login...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("SUCCESS: Admin login successful")
                return True
            else:
                print(f"FAILED: Admin login failed: {data.get('error')}")
                return False
        else:
            print(f"FAILED: Admin login failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Admin login error: {e}")
        return False

def test_add_professional():
    """Test adding a new professional"""
    print("Testing add professional...")
    
    professional_data = {
        "username": "test_professional",
        "password": "password123",
        "first_name": "Test",
        "last_name": "Professional",
        "email": "test.professional@example.com",
        "phone": "+250788123456",
        "specialization": "counselor",
        "expertise_areas": ["depression", "anxiety"],
        "experience_years": 5,
        "district": "Gasabo",
        "consultation_fee": 50000,
        "bio": "Test professional",
        "languages": ["english"],
        "qualifications": [],
        "availability_schedule": {}
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/professionals", json=professional_data)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("SUCCESS: Add professional successful")
                return data.get('professional', {}).get('id')
            else:
                print(f"FAILED: Add professional failed: {data.get('error')}")
                return None
        else:
            print(f"FAILED: Add professional failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"ERROR: Add professional error: {e}")
        return None

def test_get_professionals():
    """Test getting all professionals"""
    print("Testing get professionals...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/professionals")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('professionals'):
                print(f"SUCCESS: Get professionals successful - found {len(data['professionals'])} professionals")
                return True
            else:
                print("SUCCESS: Get professionals successful - no professionals found")
                return True
        else:
            print(f"FAILED: Get professionals failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"ERROR: Get professionals error: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("Cleaning up test data...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/professionals")
        if response.status_code == 200:
            data = response.json()
            professionals = data.get('professionals', [])
            
            for prof in professionals:
                if prof.get('username') == 'test_professional':
                    delete_response = requests.delete(f"{API_BASE_URL}/admin/professionals/{prof['id']}")
                    if delete_response.status_code == 200:
                        print(f"SUCCESS: Cleaned up {prof['username']}")
                    else:
                        print(f"FAILED: Failed to clean up {prof['username']}")
                        
    except Exception as e:
        print(f"ERROR: Cleanup error: {e}")

def main():
    """Run simple admin dashboard test"""
    print("=" * 50)
    print("ADMIN DASHBOARD TEST")
    print("=" * 50)
    
    # Test admin login
    login_success = test_admin_login()
    
    # Test get professionals
    get_success = test_get_professionals()
    
    # Test add professional
    professional_id = test_add_professional()
    
    # Cleanup
    cleanup_test_data()
    
    print("\n" + "=" * 50)
    print("TEST RESULTS:")
    print(f"Login: {'PASS' if login_success else 'FAIL'}")
    print(f"Get Professionals: {'PASS' if get_success else 'FAIL'}")
    print(f"Add Professional: {'PASS' if professional_id else 'FAIL'}")
    
    if all([login_success, get_success, professional_id]):
        print("\nALL TESTS PASSED! Backend is working correctly.")
    else:
        print("\nSOME TESTS FAILED. Check the backend API.")

if __name__ == "__main__":
    main()
