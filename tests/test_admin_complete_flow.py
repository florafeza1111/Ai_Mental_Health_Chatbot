#!/usr/bin/env python3
"""
Complete Admin Dashboard Flow Test
Tests the entire professional management workflow
"""

import requests
import json
import time

# Configuration
API_BASE_URL = "http://localhost:5057"
ADMIN_EMAIL = "eliasfeza@gmail.com"
ADMIN_PASSWORD = "EliasFeza@12301"

def test_admin_login():
    """Test admin login"""
    print("🔐 Testing admin login...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/login", json={
            "email": ADMIN_EMAIL,
            "password": ADMIN_PASSWORD
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Admin login successful")
                return data.get('token')
            else:
                print(f"❌ Admin login failed: {data.get('error')}")
                return None
        else:
            print(f"❌ Admin login failed: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"❌ Admin login error: {e}")
        return None

def test_add_professional():
    """Test adding a new professional"""
    print("\n➕ Testing add professional...")
    
    professional_data = {
        "username": "test_add_professional",
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
        "bio": "Test professional for add functionality",
        "languages": ["english"],
        "qualifications": [],
        "availability_schedule": {}
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/professionals", json=professional_data)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Add professional successful")
                return data.get('professional', {}).get('id')
            else:
                print(f"❌ Add professional failed: {data.get('error')}")
                return None
        else:
            print(f"❌ Add professional failed: {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Add professional error: {e}")
        return None

def test_edit_professional(professional_id):
    """Test editing a professional"""
    print(f"\n✏️ Testing edit professional {professional_id}...")
    
    update_data = {
        "first_name": "Updated",
        "last_name": "Professional",
        "email": "updated.professional@example.com",
        "phone": "+250788654321",
        "specialization": "psychologist",
        "expertise_areas": ["ptsd", "trauma"],
        "experience_years": 7,
        "district": "Kicukiro",
        "consultation_fee": 75000,
        "bio": "Updated professional for testing"
    }
    
    try:
        response = requests.put(f"{API_BASE_URL}/admin/professionals/{professional_id}", json=update_data)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Edit professional successful")
                return True
            else:
                print(f"❌ Edit professional failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Edit professional failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Edit professional error: {e}")
        return False

def test_get_professionals():
    """Test getting all professionals"""
    print("\n📋 Testing get professionals...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/admin/professionals")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('professionals'):
                print(f"✅ Get professionals successful - found {len(data['professionals'])} professionals")
                return data.get('professionals')
            else:
                print("❌ Get professionals failed - no professionals found")
                return []
        else:
            print(f"❌ Get professionals failed: {response.status_code}")
            return []
            
    except Exception as e:
        print(f"❌ Get professionals error: {e}")
        return []

def test_toggle_professional_status(professional_id):
    """Test toggling professional status"""
    print(f"\n🔄 Testing toggle professional status {professional_id}...")
    
    try:
        response = requests.post(f"{API_BASE_URL}/admin/professionals/{professional_id}/status", json={
            "is_active": False
        })
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Toggle professional status successful")
                return True
            else:
                print(f"❌ Toggle professional status failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Toggle professional status failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Toggle professional status error: {e}")
        return False

def test_delete_professional(professional_id):
    """Test deleting a professional"""
    print(f"\n🗑️ Testing delete professional {professional_id}...")
    
    try:
        response = requests.delete(f"{API_BASE_URL}/admin/professionals/{professional_id}")
        
        if response.status_code == 200:
            data = response.json()
            if data.get('success'):
                print("✅ Delete professional successful")
                return True
            else:
                print(f"❌ Delete professional failed: {data.get('error')}")
                return False
        else:
            print(f"❌ Delete professional failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"❌ Delete professional error: {e}")
        return False

def cleanup_test_data():
    """Clean up test data"""
    print("\n🧹 Cleaning up test data...")
    
    try:
        # Get all professionals
        response = requests.get(f"{API_BASE_URL}/admin/professionals")
        if response.status_code == 200:
            data = response.json()
            professionals = data.get('professionals', [])
            
            for prof in professionals:
                if prof.get('username') == 'test_add_professional':
                    delete_response = requests.delete(f"{API_BASE_URL}/admin/professionals/{prof['id']}")
                    if delete_response.status_code == 200:
                        print(f"✅ Cleaned up {prof['username']}")
                    else:
                        print(f"❌ Failed to clean up {prof['username']}")
                        
    except Exception as e:
        print(f"❌ Cleanup error: {e}")

def main():
    """Run complete admin dashboard flow test"""
    print("🧪 Testing Complete Admin Dashboard Flow")
    print("=" * 60)
    
    # Test admin login
    token = test_admin_login()
    if not token:
        print("❌ Cannot proceed without admin authentication")
        return
    
    # Test get professionals (should work even if empty)
    professionals = test_get_professionals()
    
    # Test add professional
    professional_id = test_add_professional()
    if not professional_id:
        print("❌ Cannot proceed without successful professional creation")
        return
    
    # Test edit professional
    edit_success = test_edit_professional(professional_id)
    
    # Test toggle status
    toggle_success = test_toggle_professional_status(professional_id)
    
    # Test delete professional
    delete_success = test_delete_professional(professional_id)
    
    # Cleanup
    cleanup_test_data()
    
    print("\n" + "=" * 60)
    print("🎉 Admin Dashboard Flow Test Complete!")
    print("\n📋 Results:")
    print(f"✅ Login: {'PASS' if token else 'FAIL'}")
    print(f"✅ Get Professionals: {'PASS' if professionals is not None else 'FAIL'}")
    print(f"✅ Add Professional: {'PASS' if professional_id else 'FAIL'}")
    print(f"✅ Edit Professional: {'PASS' if edit_success else 'FAIL'}")
    print(f"✅ Toggle Status: {'PASS' if toggle_success else 'FAIL'}")
    print(f"✅ Delete Professional: {'PASS' if delete_success else 'FAIL'}")
    
    if all([token, professional_id, edit_success, toggle_success, delete_success]):
        print("\n🚀 All tests passed! The admin dashboard backend is working perfectly!")
    else:
        print("\n⚠️ Some tests failed. Check the backend API endpoints.")

if __name__ == "__main__":
    main()
