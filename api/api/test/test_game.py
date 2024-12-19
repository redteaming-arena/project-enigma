import pytest
import sys
import os

from motor.motor_asyncio import AsyncIOMotorClient

from fastapi import status
from fastapi.testclient import TestClient

# Add API directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from api.main import app
from api.core.config import settings

client = TestClient(app)


def test_game_list_query():
    response = client.get('/game/')
    assert response.status_code == 200
    