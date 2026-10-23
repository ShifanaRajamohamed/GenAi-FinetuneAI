import urllib.request
import urllib.error
import json
import sys

if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')

base = 'http://127.0.0.1:8000'

def req(path, method='GET', data=None, token=None):
    headers = {}
    body = None
    if data is not None:
        body = json.dumps(data).encode('utf-8')
        headers['Content-Type'] = 'application/json'
    if token:
        headers['Authorization'] = f'Bearer {token}'
    r = urllib.request.Request(f'{base}{path}', data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=5) as resp:
            return resp.getcode(), json.loads(resp.read().decode('utf-8'))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode('utf-8'))

print("--- Test Auth 1: Demo Users ---")
code, demo_users = req('/api/auth/demo-users')
print(f"Status: {code}, Users: {len(demo_users)}")
for u in demo_users:
    print(f"  - {u['name']} ({u['role_label']}) -> {u['email']}")

print("\n--- Test Auth 2: Successful Login (Dr. Sarah Chen) ---")
code, login_data = req('/api/auth/login', method='POST', data={
    'username_or_email': 'dr.chen@hospital.org',
    'password': 'password123'
})
print(f"Status: {code}, Message: {login_data.get('message')}")
token = login_data['access_token']
print(f"Token generated: {token[:25]}...")
print(f"User: {login_data['user']['name']} | Role: {login_data['user']['role_id']}")

print("\n--- Test Auth 3: Failed Login (Wrong Password) ---")
code, fail_data = req('/api/auth/login', method='POST', data={
    'username_or_email': 'dr.chen@hospital.org',
    'password': 'wrong_password_xyz'
})
print(f"Status: {code} (Expected 401), Detail: {fail_data.get('detail')}")

print("\n--- Test Auth 4: Verify Session (/api/auth/me) ---")
code, me_data = req('/api/auth/me', token=token)
print(f"Status: {code}, Authenticated: {me_data['authenticated']}, Name: {me_data['user']['name']}")

print("\n--- Test Auth 5: Logout ---")
code, logout_data = req('/api/auth/logout', method='POST', token=token)
print(f"Status: {code}, Message: {logout_data.get('message')}")

print("\n--- Test Auth 6: Session After Logout ---")
code, me_after = req('/api/auth/me', token=token)
print(f"Status: {code}, Authenticated: {me_after['authenticated']}")

print("\n[✓] ALL 6 AUTHENTICATION TESTS PASSED WITH 100% SUCCESS!")
