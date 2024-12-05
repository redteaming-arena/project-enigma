"""
Authentication Routes Module
===========================

This module handles all authentication-related endpoints including login and
token validation. It implements OAuth2 password flow with JWT tokens for
secure authentication.

Routes:
- POST /login/: Login endpoint returning JWT token
- GET /verify-token: verify if token is valid

Dependencies:
- FastAPI OAuth2 for authentication
- JWT for token generation
- MongoDB for user storage
"""
import requests
import base64
from urllib.parse import quote
from datetime import datetime, timedelta, UTC
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.security import OAuth2PasswordRequestForm

from api import crud
from api.deps import CurrentUser, Database
from api.core import security
from api.core.security import verify_expired
from api.core.config import settings
from api.models import Token, Message, UserPublic, UserRegister, User
from api.utils import logger, generate



router = APIRouter()

@router.post("/login", 
             response_description="login authorization", 
             response_model=Token,
             tags=["Auth"])
async def login_access_token(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db : Database
) -> Token:
    """
    OAuth2 compatible token login, get an access token for future requests
    and update the user's stored token.
    
    Args:
        db : Database session
        form_data: OAuth2 password request form
        
    Returns:
        Token: Access token for authentication
        
    Raises:
        HTTPException: If authentication fails
    """
    try:

        # check if username exist, as well if the passwords match else return None
        user = await crud.authenticate(
            db=db, username=form_data.username, password=form_data.password
        )

        # raise exception if user does not exist or validate credential
        if not user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, 
                detail="Incorrect email or password"
            )
        access_token = user.access_token 
        
            # check if the the token has been expired
        if verify_expired(token=user.access_token):
            access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
            access_token = security.create_access_token(
                str(user.id),  # Convert ObjectId to string for token
                expires_delta=access_token_expires
            )
        
            # NOTE: Move this to crud.py
            update_result = await db.users.update_one(
                    {"_id": user.id},
                    {"$set": {"access_token": access_token, "last_login" : datetime.now(UTC) }}
                )
            
            if update_result.modified_count == 0:
                    raise HTTPException(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="Failed to update user token"
                )
            
        return Token(
            access_token=access_token
        )


    except HTTPException as e:
        raise e
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server side error, please try again later"
        )

@router.get("/token", response_model=Message, tags=["Auth"])
async def test_token(user: CurrentUser) -> Message:
    """test the clients token

    Args:
        current_user (CurrentUser): dependency on the current client auth token

    Returns:
        UserPublic: public information of the user 
    """
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return Message(
        status=status.HTTP_200_OK,
        message="Valid session token",
        data=dict(
            username=user.username,
            image=f"/avatar/{user.id}"
        )
    )
@router.get("/auth/{provider}", tags=["Auth"])
def oauth_provider_sign_in(
    provider: Literal["google"],
    popup: bool = False
) -> RedirectResponse:
    if provider == "google":
        # Create the callback URL with popup parameter
        base_redirect_uri = settings.GOOGLE_REDIRECT_URI.rstrip('/')
        # Use quote to properly encode the parameters
        redirect_url = f"{base_redirect_uri}?popup={str(popup).lower()}"
        
        logger.info(f"Redirect URL: {redirect_url}")
        
        # Build the authorization URL
        auth_url = (
            "https://accounts.google.com/o/oauth2/auth"
            "?response_type=code"
            f"&client_id={settings.GOOGLE_CLIENT_ID}"
            f"&redirect_uri={quote(redirect_url)}"
            "&scope=openid%20profile%20email"
            "&access_type=offline"
            "&prompt=consent"
        )
        
        return RedirectResponse(
            url=auth_url,
            status_code=status.HTTP_307_TEMPORARY_REDIRECT
        )
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="That provider does not exist"
        )

@router.get("/callback/oauth/{provider}", tags=["Callback"])
async def callback(
    request: Request,
    provider : str,
    db: Database,
) -> HTMLResponse:
    
    html_content = """
            <html>
            <body>
            <script>
                if (window.opener) {
                    window.opener.postMessage('%s', '*');
                    window.close();
                } else {
                    window.location.href = '/';
                }
            </script>
            </body>
            </html>""".replace(' ', '')
    popup = request.query_params.get("popup", "").lower() == "true"
    user = None
    if provider == "google":
        
        # NOTE: handel access_denied
        if _ := request.query_params.get("error", None) is not None:
            html_content = html_content % ("authFailed")
            response = HTMLResponse(content=html_content, status_code=200)
            return response
        code = request.query_params.get("code")
        if not code:
            raise HTTPException(status_code=400, detail="Authorization code not found")
        
        try:
            # Reconstruct the exact same redirect_uri used in the initial request
            base_redirect_uri = settings.GOOGLE_REDIRECT_URI.rstrip('/')
            redirect_uri = f"{base_redirect_uri}?popup={str(popup).lower()}"
            
            logger.info(f"Using redirect URI for token exchange: {redirect_uri}")
            
            # Exchange the authorization code for an access token
            token_response = requests.post(
                "https://oauth2.googleapis.com/token",  # Updated endpoint
                data={
                    "code": code,
                    "client_id": settings.GOOGLE_CLIENT_ID,
                    "client_secret": settings.GOOGLE_CLIENT_SECRET,
                    "redirect_uri": redirect_uri,  # Use the same redirect_uri as in the initial request
                    "grant_type": "authorization_code",
                },
            )
            
            logger.info(f"Token response status: {token_response.status_code}")
            logger.debug(f"Token response content: {token_response.content}")
            
            token_response.raise_for_status()
            token_data = token_response.json()
            
            access_token = token_data["access_token"]
            _ = token_data.get("refresh_token")  # Use .get() in case refresh_token is not present
            
            # Fetch user information from Google
            userinfo_response = requests.get(
                "https://www.googleapis.com/oauth2/v3/userinfo",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            userinfo_response.raise_for_status()
            userinfo = userinfo_response.json()
            
            logger.info(f"Received user info: {userinfo}")
            
            picture_url = userinfo.get("picture")

            if picture_url:
                image_response = requests.get(picture_url)
                image_response.raise_for_status()  # Ensure the image was fetched successfully
                
                # Extract MIME type (e.g., "image/jpeg")
                mime_type = image_response.headers.get("Content-Type")
                if not mime_type or not mime_type.startswith("image/"):
                    raise ValueError("Invalid or missing image content type")
                
                image_data = image_response.content
                base64_image = f"data:{mime_type};base64," + base64.b64encode(image_data).decode('utf-8')
            else:
                base64_image =  generate(userinfo["email"]) # Handle cases where the picture is not provided

            # user does not exist create user
            if (user := await crud.get_user_by_email(db=db, email=userinfo["email"])) is None:
                user = UserRegister(
                    email=userinfo["email"],
                    image=base64_image,
                    provider=provider)
                
                user : User = await crud.create_user(
                        db=db,
                        user_create=user
                    )

            if verify_expired(token=user.access_token):
                access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
                new_access_token = security.create_access_token(
                    str(user.id),  # Convert ObjectId to string for token
                    expires_delta=access_token_expires
                )
            
                update_result = await db.users.update_one(
                        {"_id": user.id},
                        {"$set": {"access_token": new_access_token, "last_login" : datetime.now(UTC) }}
                    )
                
                if update_result.modified_count == 0:
                        raise HTTPException(
                            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail="Failed to update user token"
                        )
                user.access_token = new_access_token
            
        
        except (requests.exceptions.RequestException, Exception) as e:
            logger.error(f"Error: {str(e)}")
            html_content = html_content % ("authError")
            response = HTMLResponse(content=html_content, status_code=500)
            return response

    if user.access_token is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No access token was provided"
        )
    
    html_content = html_content % ("authCompleted" if user.username else "authUsername")
    response = HTMLResponse(content=html_content, status_code=200)
    response.set_cookie(
                key="sessionKey",
                value=user.access_token,
                httponly=False,  # NOTE: Keep as False if you need to access it from JavaScript
                samesite="lax",
                secure=settings.ENVIRONMENT == "production",  # Set secure flag based on environment
            )
    return response