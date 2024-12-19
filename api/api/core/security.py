from datetime import datetime, timedelta, timezone
from typing import Any

import jwt

from passlib.context import CryptContext

from api.core.config import settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


ALGORITHM = "HS256"


def create_access_token(subject: str | Any, expires_delta: timedelta) -> str:
    """
    Creating an access token

    Args:
        subject (str | Any): extra information that could be serialized json or just an email.
        expires_delta (timedelta): expiry date to when the token is not valid

    Returns:
        str: _description_
    """
    expire = datetime.now(timezone.utc) + expires_delta
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    verify plaintext with cyphered-text

    Args:
        plain_password (str): un-hashed string
        hashed_password (str): hashed string

    Returns:
        bool: True if the hash-password is equal to the un-hashed password, else False
    """
    return pwd_context.verify(plain_password, hashed_password)


def verify_expired(token: str | None, _id : str = None) -> bool:
    """
    Verify if a JWT token has expired by checking its expiration timestamp.
    
    Args:
        token (str): The JWT token to verify
        secret_key (str): Secret key used to decode the token
        algorithm (str): Algorithm used for token encoding/decoding
    
    Returns:
        Dict[str, Any]: The decoded payload if valid
        
    Raises:
        jwt.ExpiredSignatureError: If token has expired
        jwt.InvalidTokenError: If token is invalid
        ValueError: If token doesn't contain expiration claim
    """
    try:
        if token is None:
            return True

        # Decode the token without verification first to check exp claim
        payload = jwt.decode(
            token,
            options={"verify_signature": False}  # First pass without verification
        )
        
        # Check if expiration claim exists
        if (exp_timestamp := payload.get("exp")) is None:
            raise ValueError("Token has no expiration claim")
            
        # Get current timestamp
        current_timestamp = datetime.utcnow().timestamp()
        
        # Check if token has expired
        if current_timestamp > exp_timestamp:
            return True
            
        # If we get here then it is not expired
        return False
        
    except jwt.InvalidTokenError as e:
        raise jwt.InvalidTokenError(f"Invalid token: {str(e)}")
    except Exception as e:
        raise ValueError(f"Token verification failed: {str(e)}")


def get_password_hash(password: str) -> str:
    """ return hash password """
    return pwd_context.hash(password)