import pytest
import sys
import os
import asyncio

from motor.motor_asyncio import AsyncIOMotorClient

from fastapi import status
from fastapi.testclient import TestClient

# Add API directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.main import app
from api.core.config import settings

client = TestClient(app)

def test_available_username():
    response = client.get("/available", params={ "username" : "0XNone"})
    assert response.status_code == 200
    assert response.json().get("data", False)

@pytest.mark.asyncio
async def test_signup_username():
    username = "NoneType"
    
    # Create UserRegister compatible data
    user_data = {
        "username": username,
        "password": "68656c6c6f776f726c6"
    }
    
    # First signup attempt
    response = client.post(
        "/signup",
        json=user_data  # Send as JSON to match FastAPI's automatic parsing
    )
    assert response.status_code == status.HTTP_200_OK, response.json()
    response = response.json()
    assert response.get("data"), "User was not created"

    # Test duplicate username (capitalized)
    duplicate_data = {
        "username": username.capitalize(),
        "password": "68656c6c6f776f726c6"
    }
    
    response = client.post(
        "/signup",
        json=duplicate_data
    )
    assert response.status_code == status.HTTP_400_BAD_REQUEST, "Expected duplicate username to fail"
    assert "already exists" in response.json().get("detail", ""), "Expected proper error message"

    # Cleanup
    client_db = AsyncIOMotorClient(settings.MONGODB_DATABASE_URI)
    db = client_db[settings.MONGODB_NAME]
    db.users.delete_one({"username": username})
    client_db.close()

@pytest.mark.asyncio
async def test_login_username():
    username = "User"
    password = "68656c6c6f776f726c64"
    
    # First register the user
    response = client.post(
        "/signup",
        json={
            "username": username,
            "password": password
        }
    )

    assert response.status_code == 200, response.json()
    response = response.json()
    assert response.get("data"), "User was not created"
    
    # Login to get access token
    # Note: Using data parameter for form-urlencoded, not json
    response = client.post(
        "/login/access-token",
        data={
            "username": username,
            "password": password,
            "grant_type": "password"  # Often required for OAuth2 form login
        },
        headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    token = response.json()
    assert response.status_code == 200, token
    assert token.get("token_type") is not None, "Token type should be present"
    assert token.get("access_token") is not None, "Access token should be present"
    
    # Verify the token
    response = client.get(
        "/verify-token",
        headers={
            "Authorization": f"Bearer {token['access_token']}"
        }
    )
    assert response.status_code == 200
    assert response.json().get("data", False)
    
    # Cleanup
    client_db = AsyncIOMotorClient(settings.MONGODB_DATABASE_URI)
    db = client_db[settings.MONGODB_NAME]
    await db.users.delete_one({"username": username})
