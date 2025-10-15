import requests
import json

# Test the professional sessions endpoint
try:
    response = requests.get('http://localhost:5057/professional/sessions')
    print('=== PROFESSIONAL SESSIONS ===')
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Number of sessions: {len(data)}')
        for session in data:
            print(f'User: {session.get("userAccount", "N/A")}, Status: {session.get("bookingStatus", "N/A")}, Risk: {session.get("riskLevel", "N/A")}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Error connecting to API: {e}')

print('\n=== PROFESSIONAL USERS ===')
try:
    response = requests.get('http://localhost:5057/professional/users')
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Number of users: {len(data)}')
        for user in data:
            print(f'User: {user.get("username", "N/A")}, Sessions: {user.get("totalSessions", 0)}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Error connecting to API: {e}')

print('\n=== DASHBOARD STATS ===')
try:
    response = requests.get('http://localhost:5057/professional/dashboard-stats')
    print(f'Status: {response.status_code}')
    if response.status_code == 200:
        data = response.json()
        print(f'Dashboard Stats: {data}')
    else:
        print(f'Error: {response.text}')
except Exception as e:
    print(f'Error connecting to API: {e}')

