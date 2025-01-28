from typing import Optional
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from api.core.config import settings


class DatabaseManager:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        """singleton builder pattern"""
        if cls.client is None:
            cls.client = AsyncIOMotorClient(settings.MONGODB_DATABASE_URI)
        return cls.client

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        """access collection"""
        if cls.db is None:
            cls.db = cls.get_client()[settings.MONGODB_NAME]
        return cls.db

    @classmethod
    def close_db(cls) -> None:
        """close db connection"""
        if cls.db is not None:
            cls.get_client().close()
        return None
