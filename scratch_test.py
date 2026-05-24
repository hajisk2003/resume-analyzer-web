import requests

BASE_URL = "http://localhost:8000"

def test_api():
    print("Testing Registration...")
    res = requests.post(f"{BASE_URL}/auth/register", json={
        "full_name": "API Test",
        "email": "apitest@example.com",
        "password": "Password123!"
    })
    print(res.status_code, res.text)
    
    print("\nTesting Login...")
    res = requests.post(f"{BASE_URL}/auth/login", data={
        "username": "apitest@example.com",
        "password": "Password123!"
    })
    print(res.status_code, res.text)
    
    if res.status_code == 200:
        token = res.json().get("access_token")
        print("Got token:", token)
        
        # Test file upload (mock PDF)
        print("\nTesting Upload...")
        with open("test.pdf", "wb") as f:
            f.write(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
            
        with open("test.pdf", "rb") as f:
            files = {"file": ("test.pdf", f, "application/pdf")}
            data = {
                "job_title": "Software Engineer",
                "job_description": "We need a software engineer with Python experience. Must have 5 years experience." * 2
            }
            headers = {"Authorization": f"Bearer {token}"}
            res = requests.post(f"{BASE_URL}/resume/upload", headers=headers, data=data, files=files)
            print(res.status_code, res.text)
            
            if res.status_code == 200:
                resume_id = res.json().get("resume_id")
                print(f"Got resume_id: {resume_id}")
                
                print("\nTesting Analysis...")
                res = requests.post(f"{BASE_URL}/resume/{resume_id}/full-analysis", headers=headers)
                print(res.status_code, res.text)

if __name__ == "__main__":
    test_api()
