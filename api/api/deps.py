import jwt

from datetime import datetime, UTC

from bson.errors import InvalidId
from bson import ObjectId
from typing import Annotated, Optional

from jwt.exceptions import InvalidTokenError

from pydantic import ValidationError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

from api.core.config import settings
from api.core.security import ALGORITHM, verify_expired
from api.models import User
from api.utils import logger

from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

class DatabaseManager:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    def get_client(cls) -> AsyncIOMotorClient:
        if cls.client is None:
            cls.client = AsyncIOMotorClient(settings.MONGODB_DATABASE_URI)
        return cls.client

    @classmethod
    def get_db(cls) -> AsyncIOMotorDatabase:
        if cls.db is None:
            cls.db = cls.get_client()[settings.MONGODB_NAME]
        return cls.db
    
    @classmethod
    def close_db(cls) -> None:
        if cls.db is not None:
            cls.get_client().close()
        return None

async def get_database() -> AsyncIOMotorDatabase:
    return DatabaseManager.get_db()

async def close_database() -> None:
    return DatabaseManager.close_db()

Database = Annotated[AsyncIOMotorDatabase, Depends(get_database)]

async def get_current_user(
        token: Annotated[str, Depends(oauth2_scheme)],
        db: Database
    ) -> User:
    """
    Validate JWT token and return current user.
    
    Args:
        token (str): JWT token from authorization header
        db (Database): Database session
        
    Returns:
        User: Current authenticated user
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    user_id = None
    try:

        # check if the token has expired
        if verify_expired(token):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials due to expired token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        payload = jwt.decode(
            token, settings.SECRET_KEY, algorithms=[ALGORITHM]
        )
        
        if (user := payload.get("sub", None)) is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )

        user_id = ObjectId(user)

    except (InvalidTokenError, ValidationError, InvalidId, Exception):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await db.users.find_one({"_id": user_id})

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if user["access_token"] != token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The token you enter is has not been updated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return User.model_validate(user)

async def clear_user_token(db: Database, user_id: ObjectId) -> None:
    """
    Clear a user's access token when it's no longer valid.
    
    Args:
        db : Database session
        user_id: User's ObjectId
    """
    try:
        await db.users.update_one(
            {"_id": user_id},
            {"$set": {"access_token": None, "last_signout" : datetime.now(UTC)}}
        )
    except Exception as e:
        # Log error but don't raise - this is a cleanup operation
        logger.error(f"Interpret: {e}")
        pass

# Annotation type that used in context we desire the user information
CurrentUser = Annotated[User, Depends(get_current_user)]