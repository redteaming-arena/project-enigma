"""
User Management Routes Module
============================

This module handles all user-related API endpoints including user registration,
retrieval, and logout functionality. It provides the core user management
functionality for the application.

Routes:
- POST /signup: User registration
- POST /logout: User logout
- GET /{id}: Retrieve user details
- GET /available: check if username is available

Dependencies:
- FastAPI for route handling
- MongoDB for user storage
- Pydantic for data validation
"""
import io
import httpx
import base64


from bson import ObjectId
from bson.errors import InvalidId

from pydantic import ValidationError

from fastapi import APIRouter, Query, HTTPException, status
from fastapi.responses import StreamingResponse

from api import crud
from api.deps import (
    Database, 
    CurrentUser,
    clear_user_token,
)
from api.models import (
    UserStats,
    UserRegister, 
    UserPublic,
    Message
)

router = APIRouter()

@router.post("/signup", response_model=Message)
async def register_user(db : Database, user_in : UserRegister) -> Message:
    """
    Register a new user in the system.
    
    This endpoint allows for user registration without requiring authentication.
    It checks for username uniqueness before creating the new user account.
    
    Args:
        db (Database): Database session for performing database operations
        user_in (UserRegister): User registration data including username and password
        
    Returns:
        UserPublic: Public user information of the newly created user
        
    Raises:
        HTTPException (400): If the username is already taken
    
    """
    try:
        user = await crud.find_user(db=db, 
                                    username=user_in.username)
        if user:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Interrupt the user with this username already exists in the system.",
            )
        
        await crud.create_user(db=db, user_create=user_in)
        return Message(
            status=status.HTTP_201_CREATED,
            message=f"{user_in.username} was successfully created",
            data=True
        )
    except HTTPException as h:
        raise h
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server Error: {str(e)}"
        )

@router.post("/logout")
async def logout(
    db: Database,
    user: CurrentUser
) -> Message:
    """
    Logout the current user and invalidate their access token.
    
    This endpoint handles user logout by clearing the user's active token
    from the system and ending their current session.
    
    Args:
        current_user (CurrentUser): The currently authenticated user
        db (Database): Database session for performing database operations
        
    Returns:
        Message: Success message indicating successful logout
        
    Raises:
        HTTPException (500): If an unexpected error occurs during logout
    """
    try:
        await clear_user_token(db, user.id)
        
    except HTTPException as h:
        raise h
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An unexpected error occurred: {str(e)}"
        )
    
    return Message(
        message=f"{user.username} was successfully logged out",
        status=status.HTTP_200_OK,
        data=True
    )

@router.post('/update-username', response_model=Message)
async def update_username(db: Database, user: CurrentUser, username: str):
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not authenticated"
        )
    
    if user.username == username:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="You can not update your username to the same username"
        )
        

    # Check if username already exists
    if await crud.get_user_by_username(db=db, username=username) is not None:
        raise HTTPException(
            detail="Username already exists",
            status_code=status.HTTP_401_UNAUTHORIZED
        )

    try:
        # Update the username
        results = await db.users.update_one(
            {"_id": user.id},
            {"$set": {"username": username}}
        )

        if results.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update user token"
            )
        
        # Return success response
        return Message(
            status=200,
            message="Username updated successfully",
            data={
                "id": str(user.id),
                "username": username
            }
        )
    
    except HTTPException as h:
        raise h
    except Exception:
        # Handle database errors
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: Failed to update username"
        )



@router.get("/available-username")
async def is_available_username(db : Database,
                                username : str = Query(..., min_length=1, max_length=50, pattern="^[a-zA-Z0-9_-]+$")) -> Message:
    """
    Check if a username is available for registration.
    
    This endpoint performs a case-insensitive check to determine if the
    requested username is already taken in the system.
    
    Args:
        db (Database): Database session for performing database operations
        username (str): The username to check for availability
        
    Returns:
        Message: Success message if username is available
        
    Raises:
        HTTPException (400): If the username is already taken
    """

    try:


        user = await crud.find_user(db=db, username=username)

        if user is not None and user.provider is not None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="You must login via your provider."
            )

        message = f"{username} is currently available" if user is None\
                    else f"Username is currently taken"
        
        return Message(
            message=message,
            status=status.HTTP_200_OK,
            data=user is None
        )
    except HTTPException as h:
        raise h
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: Failed check if username is available"
        )

@router.get("/user", response_model=UserPublic)
async def read_user_by_id(
    user : CurrentUser,
    db: Database
) -> UserPublic:
    """
    Retrieve user information by their unique identifier.
    This endpoint fetches and returns public user information for the specified
    user ID. It includes validation of the ID format and handles various error cases.

    Args:
        id (str): The unique identifier of the user to retrieve
        db (Database): Database session for performing database operations

    Returns:
        UserPublic: Public user information for the requested user

    Raises:
        HTTPException (400): If the user ID format is invalid
        HTTPException (404): If no user is found with the specified ID
        HTTPException (422): If the retrieved user data fails validation
    """
    try:
        # NOTE: Move this to crud function
        # Combined query using aggregation
        pipeline = [
            {"$match": {"_id": user.id}},
            
            # Lookup recent completed sessions
            {
            "$lookup": {
                "from": "sessions",
                "let": { "user_id": "$_id" },
                "pipeline": [
                {
                    "$match": {
                    "$expr": { "$eq": ["$user_id", "$$user_id"] },
                    "completed": True,
                    "visible" : True,
                    "$and": [
                        { "history": { "$exists": True } }, 
                        { "history": { "$not": { "$size": 0 } } }
                        ],
                    }
                },
                {
                    "$sort": { "completed_time": -1 }
                },
                {
                    "$project": {
                    "_id": { "$toString": "$_id" },
                    "title": 1,
                    "completed_time": 1
                    }
                },
                { "$limit": 10 }
                ],
                "as": "history"
            }},
            # Lookup games from pinned array
            {
                "$lookup": {
                "from": "games",
                "let": { "pinned_ids": "$pinned" },
                "pipeline": [
                    {
                    "$match": {
                        "$expr": {
                        "$in": ["$_id", "$$pinned_ids"]
                        }
                    }
                    },
                    {
                    "$project": {
                        "id": { "$toString": "$_id" },
                        "title": 1,
                        "image": 1
                    }
                    }
                ],
                "as": "pinned"
                }
            },
            
            # Project only needed fields
            {"$project": {
                "id": {"$toString": "$_id"},
                "username": 1,
                "history": 1,
                "pinned" : {
                    "id" : 1,
                    "title" : 1,
                    "image" : 1
                }
            }}
        ]

        result = await db.users.aggregate(pipeline).to_list(None)
        
        if not result or len(result) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interrupt user with id {id} not found"
            )

        user_dict = result[0]
        
        # Convert to UserPublic for response
        return UserPublic(
            id=user_dict.get("id", None),
            username=user_dict.get("username", None),
            history=user_dict.get("history", []) ,
            pinned=user_dict.get("pinned", [])
        )

    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{id} is an invalid user ID"
        )
    except ValidationError as e:
        print
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e)
        )

@router.post("/pinned/{game_id}")
async def update_pinned_games(
    game_id: str,
    user: CurrentUser,
    db: Database
):
    try:
        # Validate game_id format and convert to ObjectId
        if not ObjectId.is_valid(game_id):
            raise HTTPException(status_code=400, detail="Invalid game ID format")
        game_object_id = ObjectId(game_id)
        
        # Check if game exists
        game = await db.games.find_one({"_id": game_object_id})
        if not game:
            raise HTTPException(status_code=404, detail="Game not found")
            
        # Check if game is already pinned
        user_data = await db.users.find_one({"_id": user.id})
        current_pins = user_data.get("pinned", [])
        
        # Convert existing pins to ObjectId if they aren't already
        current_pins = [ObjectId(pin) if isinstance(pin, str) else pin for pin in current_pins]
        
        # Don't add if already pinned
        if game_object_id in current_pins:
            return {"message": "Game already pinned", "pinned": True}
            
        # Add new pin
        updated = await db.users.update_one(
            {"_id": user.id},
            {
                "$addToSet": {
                    "pinned": game_object_id
                }
            }
        )
        
        if updated.modified_count == 0:
            raise HTTPException(status_code=400, detail="Failed to pin game")
            
        return {
            "message": "Game pinned successfully",
            "pinned": True,
            "game_id": str(game_object_id)
        }
    
    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid game ID format")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/pinned/{game_id}")
async def unpin_game(
    game_id: str,
    user: CurrentUser,
    db: Database
):
    try:
        # Validate game_id format and convert to ObjectId
        game_object_id = ObjectId(game_id)
        
        # Remove pin
        updated = await db.users.update_one(
            {"_id": user.id},
            {
                "$pull": {
                    "pinned": game_object_id
                }
            }
        )
        
        if updated.modified_count == 0:
            return {"message": "Game was not pinned", "pinned": False}
            
        return {
            "message": "Game unpinned successfully",
            "pinned": False,
            "game_id": str(game_object_id)
        }
    
    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(status_code=400, detail=f"{game_id} Invalid game ID format")
    except Exception:
        raise HTTPException(status_code=500,
                            detail=f"Server Error: unable to unpin game, try again later")


@router.get('/stats')
async def user_stats(
        current_user : CurrentUser,
        db : Database
) -> UserStats:
    try:
        stats = await crud.user_stats(db=db, user_id=current_user.id)
        return stats
    except HTTPException as h:
        raise h
    except Exception:
        raise HTTPException(status_code=500,
                            detail=f"Server Error: unable retrieve stats")

@router.get('/avatar/{id}')
async def get_avatar(db: Database, id: str):
    try:
        # Retrieve the user from the database
        user = await db.users.find_one({"_id": ObjectId(id)})

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User does not exist"
            )

        # Get the image path or URL
        image : str = user.get("image")
        if image is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Image not found"
            )

        # Check if the image is a URL
        if image.startswith("https://"):
            # Fetch the image content from the URL
            async with httpx.AsyncClient() as client:
                response = await client.get(image)
                if response.status_code != 200:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail="Image could not be retrieved from the URL"
                    )
                
                # Get the Content-Type from the response headers
                content_type = response.headers.get("Content-Type", "")
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="URL does not point to a valid image"
                    )

                # Stream the image response back to the client with the detected MIME type
                return StreamingResponse(
                    response.iter_bytes(),
                    media_type=content_type
                )
        elif image.startswith("http://"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Image could not be retrieved from non secure URLs"
            )
       
        base64_data = None
        if image.startswith("data:image"):
            base64_data = image.split(",")[-1]
        
        image = base64.b64decode(base64_data)

        # If image is a local path, serve it as a file
        return StreamingResponse(
            io.BytesIO(image),
            media_type='image/webp'
        )
    
    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{id} is an invalid user ID"
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: failed to retrieving the avatar"
        )