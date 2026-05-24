import requests

BASE = "http://localhost:8000"


def test_register():
    res = requests.post(f"{BASE}/auth/register", json={
        "email": "testuser@example.com",
        "password": "SecurePass123",
        "full_name": "Test User"
    })
    print(f"Register: {res.status_code} → {res.json()}")


def test_duplicate_register():
    res = requests.post(f"{BASE}/auth/register", json={
        "email": "testuser@example.com",
        "password": "SecurePass123",
        "full_name": "Test User"
    })
    print(f"Duplicate register: {res.status_code} → {res.json()}")


def test_login():
    res = requests.post(f"{BASE}/auth/login", data={
        "username": "testuser@example.com",
        "password": "SecurePass123"
    })
    print(f"Login: {res.status_code}")
    return res.json().get("access_token")


def test_wrong_password():
    res = requests.post(f"{BASE}/auth/login", data={
        "username": "testuser@example.com",
        "password": "WrongPassword"
    })
    print(f"Wrong password: {res.status_code} → {res.json()}")


def test_get_me(token):
    res = requests.get(f"{BASE}/auth/me", headers={"Authorization": f"Bearer {token}"})
    print(f"Me: {res.status_code} → {res.json()}")


def test_me_no_token():
    res = requests.get(f"{BASE}/auth/me")
    print(f"Me no token: {res.status_code} → {res.json()}")


if __name__ == "__main__":
    print("\n=== Auth Tests ===\n")
    test_register()
    test_duplicate_register()
    token = test_login()
    test_wrong_password()
    test_get_me(token)
    test_me_no_token()
    print("\n=== Done ===")
