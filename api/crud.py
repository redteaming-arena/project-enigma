"""
Database CRUD Operations Helper Module
====================================

This module provides helper functions for database CRUD (Create, Read, Update, Delete) operations,
primarily used by route handlers. It serves as a layer of abstraction between the API routes
and direct database operations, promoting code reusability and maintainability.

Key Features:
- User management operations (create, retrieve, authenticate)
- Game-related operations (TBD)

Usage:
    from api import crud
    
    # In your route handler:
    user = await crud.get_user_by_username(db=db, username="example")

Dependencies:
- FastAPI for HTTP exception handling
- MongoDB for database operations
- Custom security utils for password hashing
"""
import re


from typing import Optional, List, Union, AsyncGenerator
from datetime import datetime, timedelta, UTC

from fastapi import HTTPException, status

from api.deps import Database
from api.core.security import get_password_hash, verify_password
from api.models import User,\
                       UserRegister,\
                       Game,\
                       GameSession,\
                       GameSessionPublic,\
                       Model,\
                       Judge,\
                       Leaderboard
from api.judge import registry
from api.utils import generate, to_object_id, logger
from bson import ObjectId
from bson.errors import InvalidId
import random


# = User ==============================================================

async def get_user_by_username(*, db: Database, username: str) -> Optional[User]:
    """Retrieve user from database by username.
    
    Args:
        db (Database): Database session
        username (str): Username to look up
        
    Returns:
        Optional[User]: User if found, None otherwise
    """
    user_data = await db.users.find_one({"username": re.compile(f"^{username}$", re.IGNORECASE)})
    if user_data:
        return User(**user_data)
    return None

async def get_user_by_email(*, db: Database, email: str) -> Optional[User]:
    """Retrieve user from database by email.
    
    Args:
        db (Database): Database session
        email (str): Email to look up
        
    Returns:
        Optional[User]: User if found, None otherwise
    """
    user_data = await db.users.find_one({"email": email})
    if user_data:
        return User(**user_data)
    return None


async def find_user(*, db: Database, username: str) -> Union[User, None]:
    """
    Find a user by username, checking both case-sensitive and case-insensitive matches
    Args:
        session: MongoDB session instance
        username: Username to search for
    Returns:
        User document if found, None otherwise
    """
    try:
        user = await db.users.find_one({
            "$or": [
                {"username": username},  # Exact case-sensitive match
                {"username": re.compile(f"^{username}$", re.IGNORECASE)}  # Case-insensitive match
            ]
        })

        if user is None:
            return None
        
        return User(**user)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error finding user: {str(e)}"
        )

async def update_username(*, db : Database, username : str):
    existing_user = await get_user_by_username(
        db=db, username=username
    )
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Interrupt username already registered"
        )
    
    db["users"].update_one({
        "_id" : existing_user.id
    })


async def create_user(*, db: Database, user_create: UserRegister) -> User:
    """Create new user in database.
    
    Args:
        db (Database): Database session
        user_create (UserRegister): User registration data
        
    Returns:
        User: Created user object
        
    Raises:
        HTTPException: If username already exists or if there's a database error
    """
    try:

        if user_create.provider == "google":
            user_data = user_create.model_dump()
            user_data["created_at"]   = datetime.now(UTC)
            user_data["password"]     = None
            user_data["last_login"]   = None
            user_data["last_signout"] = None
            user_data["access_token"] = None
            user_data["pinned"] = []
        else:
            existing_user = await get_user_by_username(
                db=db, username=user_create.username
            )
            if existing_user:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Interrupt username already registered"
                )


            hashed_password = get_password_hash(user_create.password)
            user_data = user_create.model_dump()
            user_data["password"] = hashed_password
            user_data["created_at"] = datetime.now(UTC)
            user_data["last_login"] = None
            user_data["last_signout"] = None
            user_data["access_token"] = None
            user_data["image"] = generate(user_create.username)
            user_data["pinned"] = []

        result = await db.users.insert_one(user_data)

        user_data["id"] = str(result.inserted_id)
        return User(**user_data)

    except Exception as e:
        logger.info(f"Error occurred {str(e)}")
        # Catch any unexpected errors
        # Log the error here if you have logging configured
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interrupt an unexpected error occurred"
        ) from e

async def authenticate(*, db : Database, username : str, password : str) -> Optional[User]:
    """
    

    Args:
        session (Annotated[AsyncIOMotorDatabase, Depends(get_db)]): client session to the backend
        username (str): username of the client
        password (str): un hashed password of the client

    Returns:
        Optional[User]: if the user does not exist or the password does not match return None else User
    """
    user = await find_user(db=db, username=username)
    if user is None:
        return None
    if not verify_password(password, user.password):
        return None
    return user

async def user_stats(*, db : Database, user_id : str | ObjectId):
    """ user states such as games played list of sessions """
    try :
        
        if isinstance(user_id, str):
            user_id = ObjectId(user_id) 
        
        user = await db.users.aggregate([
            {
                "$match": {
                    "_id": user_id 
                }
            },
            {
                "$project": {
                    "_id": 0, 
                    "games_played": {
                        "$ifNull": ["$games_played", -1]
                    }
                }
            }
        ]).to_list(length=1)

        user = user[0] if user else None

        pipeline = [
            {
                "$match": { 
                    "user_id": user_id,
                    "completed": True
                }
            },
            {
                "$lookup": {
                    "from": "models",
                    "localField": "agent_id",
                    "foreignField": "_id",
                    "as": "model"
                }
            },
                        {
                "$lookup": {
                    "from": "users",
                    "localField": "user_id",
                    "foreignField": "_id",
                    "as": "user"
                }
            },
            {
                "$unwind": {
                    "path": "$model",
                    "preserveNullAndEmptyArrays": True
                }
            },{
                "$unwind": {
                    "path": "$user",
                    "preserveNullAndEmptyArrays": True
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "completed_time": 1,
                    "outcome": 1,
                    "provider" : "$model.provider",
                }
            }
        ]

        sessions = await db.sessions.aggregate(pipeline).to_list(length=None)

        # In case where games_played does not exist early users...
        games_played = max(0, len(sessions))
        if user["games_played"] < 0:
            await db.users.update_one(
                {"_id": user_id},
                {"$set": {"games_played": games_played}}
            )
        
        result = dict(games_played=games_played, sessions=sessions)
        return result

    except Exception as e:
        raise e
# =====================================================================

# NOTE add more 

# = Game ==============================================================

async def get_game_from_id(*, db: Database, id: str) -> Game:
    """
    Fetch Game object by its game_id including associated judge information
    
    Args:
        session: MongoDB session instance
        game_id (str): Unique ID of Game collection
        
    Returns:
        Game: Game object with judge information, if found
        
    Raises:
        HTTPException: If game_id is invalid or game is not found
    """
    try:
        id = to_object_id(id)
        game_data = await db.games.find_one({
            "_id": id
        })

    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Interrupt not a valid ID"
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Interrupt exception thrown trying to find game form ID"
        )
    
    if not game_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Interrupt game not found from ID: {id}"
        )
    
    game_data["id"] = game_data["_id"]; del game_data["_id"] 

    return Game(**game_data)

async def get_games(*, db: Database, skip : int = 0, limit : int | None= None ) -> AsyncGenerator[Game, None]:
    """
    Fetch all games from the database
    
    Args:
        session: MongoDB session instance
        
    Returns:
        List[Game]: List of all games
        
    Raises:
        HTTPException: If there's an error retrieving games
    """
    try:
        # Convert cursor to list since find() returns a cursor
        query = db.games.find({}).skip(skip=skip)
        if limit is not None:
            query = query.limit(limit=limit)
        
        game_data = await query.to_list(length=None)

        if not game_data:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Interrupt something went wrong when finding the games"
            )
        
        for game in game_data:
            game["id"] = game["_id"]
            del game["_id"]
            yield Game(**game)
    
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Interrupt retrieving games: {str(e)}"
        )

# =====================================================================


# = Session ==============================================================

async def create_game_session(*, 
                              db: Database, 
                              game : Game,
                              judge : Judge,
                              user_id:  Union[str, ObjectId], 
                              model_id: Union[str, ObjectId]) -> GameSession:
    """
    Creates new game session in database

    Args:
        session: MongoDB session instance
        user_id (str): ID of user creating the session
        game_id (str): ID of the game for the session
        model_id (str): ID of the model for the session
        target (str): Randomly chosen prompt from the game's contexts
        session: MonogDB session instance
    
    Returns:
        GameSession: The created GameSession object
    """

    try:

        user_id  = to_object_id(user_id)
        model_id = to_object_id(model_id)
        game_id = to_object_id(game.id)
        judge_id = to_object_id(judge.id)

        if (sample_fn := judge.sampler.function) is None:
            raise HTTPException(
                status_code=500,
                detail="Missing sampler function"
            )

        # we need to sample from out game distribution sample function.
        # Properties a sampler should hold if game needs to change model
        # configurator then we need to add a meta data object to which it will be deleted
        # after it's been allocated to the game session 
        sample = registry.get_sampler(sample_fn.name)()

        description=None
        if game.metadata.game_rules.get("deterministic", False):
            description = f"{game.session_description} {sample.get("kwargs", {}).get("target", "")}"
        else:
            description = f"{game.session_description}"

        metadata = game.metadata.model_dump(exclude_none=True)
        if "kwargs" in sample:
            metadata["kwargs"] = sample["kwargs"]
        if "model_config" in sample:
            metadata["model_config"] |= sample["model_config"]

        new_session = GameSession(
            user_id=user_id,
            game_id=game_id,
            judge_id=judge_id,
            agent_id=model_id,
            description=description,
            history=list(),
            completed=False,
            visible=True,
            create_time=datetime.now(UTC),
            completed_time=None,
            outcome=None,
            shared=None,
            metadata=metadata)

        
        result = await db.sessions.insert_one(new_session\
                                              .create_session()) 

        new_session.id = result.inserted_id
        if game:
            await db.users.update_one(
                {"_id": user_id},  # Match the game by its _id
                {"$inc": {"games_played": 1}}  # Increment games_played by 1
            )

    except InvalidId:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid ID format"
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Something else went wrong"
        )
    return new_session


async def delete_game_session(db : Database, session_id: str) -> None:
    """ delete session for testing """
    await db.sessions.delete_one({"_id": ObjectId(session_id)})


async def get_session_from_shared_id(*, shared_id: str, db: Database):
    try:
        shared_id = ObjectId(shared_id)
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format"
        )
    
    pipeline = [
        {
            "$match": {
                "shared": shared_id,
                "completed" : True
            }
        },
        {
            "$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user"
            }
        },
         {
            "$lookup": {
                "from": "models",
                "localField": "agent_id",
                "foreignField": "_id",
                "as": "model"
            }
        },
        {
            "$unwind": {
                "path": "$user",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$unwind": {
                "path": "$model",
                "preserveNullAndEmptyArrays": True
            }
        },
        {
            "$project": {
                "id": "$_id",
                "_id": 0,
                "title": 1,
                "history": 1,
                "user_id": 1,
                "completed": 1,
                "outcome": 1,
                "completed_time": 1,
                "create_time": 1,
                "description" : 1,
                "user": {
                    "username" : 1,
                },
                "model" : 1

            }
        }
    ]

    cursor = db.sessions.aggregate(pipeline)
    game_sessions = await cursor.to_list(length=None)
    print(game_sessions)

    if not game_sessions:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,  # Changed from 204 to 404
            detail="Game session not found"
        )

    return GameSessionPublic(**game_sessions[0])


async def get_session(*, 
                      session_id : str, 
                      user_id : ObjectId, 
                      completed : bool | None = None,
                      db: Database) -> GameSessionPublic:
    """
    Get session object by session_id

    Args:
        session_id (str): Unique session ID of the GameSession
        session: MonogDB session instance
    Returns:
        GameSession: The GameSession object, if found
    """
    query = {}
    try: 
        query["_id"] = ObjectId(session_id)
        if user_id is not None:
            query["user_id"] = user_id
            query["visible"] = True

        if completed is not None:    
            query["completed"] = completed


    except Exception:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST, 
            detail="Invalid session_id format"
        )

    session_data = db.sessions.aggregate([
        {"$match": query},
        # Join with related collections
        {"$lookup": {
            "from": "users",
            "localField": "user_id",
            "foreignField": "_id",
            "as": "user"
        }},
        {"$lookup": {
            "from": "models",
            "localField": "agent_id",
            "foreignField": "_id",
            "as": "model"
        }},
        {"$lookup": {
            "from": "judges",
            "localField": "judge_id",
            "foreignField": "_id",
            "as": "judge"
        }},
        # Unwind arrays (with null preservation)
        {"$unwind": {
            "path": "$user",
            "preserveNullAndEmptyArrays": True
        }},
        {"$unwind": {
            "path": "$model",
            "preserveNullAndEmptyArrays": True
        }},
        {"$unwind": {
            "path": "$judge",
            "preserveNullAndEmptyArrays": True
        }},
        # Project only needed fields
       {
        "$project": {
            "id": "$_id", 
            "_id": 0, 
            "title": 1,
            "history": 1,
            "user_id": 1,
            "completed": 1,
            "outcome": 1,
            "completed_time": 1,
            "create_time": 1,
            "description" : 1,
            "user": {
            "username": 1
            },
            "shared": {
            "$toString": "$shared"
            },
            "model": {
            "name": 1,
            "provider": 1,
            "image": 1,
            "metadata": 1
            },
            "judge": {
            "active": 1,
            "sampler": 1,
            "validator": 1
            },
            "metadata": 1
        }
        },
        {"$limit": 1}   
    ])
    
    game_session = await session_data.to_list(length=None)

    if len(game_session) == 0 or game_session is None:
        raise HTTPException(
            status_code=status.HTTP_204_NO_CONTENT,
            detail="game session is not found"
        )
    game = game_session.pop(0)

    return GameSessionPublic(**game)

async def update_game_session(*, 
                              db : Database, 
                              session_id: str,
                              updated_session: GameSession,
                              updates : List[str]):
    """
    Update an existing GameSession in the database.

    Args:
        session: MongoDB session instance
        session_id (str): ID of the session to be updated
        updated_session: The GameSession object containing updated fields
    """
    try:
        session_id_obj = ObjectId(session_id)
    except :
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid session_id format"
        )

    result = await db.sessions.update_one(
        {"_id": session_id_obj},
        {"$set": { name : getattr(updated_session, name) for name in updates }}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or no changes made"
        )

    return {"message": "Session updated successfully"}

async def get_sessions_for_user(user_id: str | ObjectId, db: Database, skip : int = 0, limit : int | None= None) -> List[GameSession]:
    """
    Get all game sessions for a specific user.

    Args:
        user_id (str): ID of the user sessions to retrieve
        session (Session): MongoDB session instance

    Returns:
        List[GameSession]: List of GameSession objects for the user.
    """
    try:

        user_id = to_object_id(user_id)

    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid user_id format"
        )

    query = db.sessions.find({
        "user_id": user_id,
        "completed": True,
        "visible" : True
    }).skip(skip=skip)
    
    if limit is not None:
        query = query.limit(limit=limit)
        
    sessions_data = await query.to_list(length=None)

    if not sessions_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No history found"
        )
    

    response = [0] * len(sessions_data)  
    for i, session_data in enumerate(sessions_data):
        session_data["id"] = session_data["_id"]
        del session_data["_id"]
        response[i] = GameSession(**session_data) 

    return response

# =====================================================================

# = Model ==============================================================


async def get_models(db : Database) -> List[Model]:
    models = await db.models.find({
        "available" : True
    }).to_list()

    if models is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No available models found in the database"
        )
    

    response = []
    for model in models:
        print(model)
        model["id"] = model['_id']
        del model['_id']
        response.append(Model(**model))
    return response

async def get_models_dependent_on_game(db : Database,
                                       game : Game) -> List[Model]:
     
     # Build the query
    query = {"available": True}
    if game.metadata.game_rules.get("tools_enabled", False):
        query["tools"] = True  

    models = await db.models.find(query).to_list(None)
    
    if models is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No available models found in the database"
        )

    response = []
    for model in models:
        model["id"] = model['_id']
        del model['_id']
        response.append(Model(**model))
    return response


async def get_random_model_id(db : Database, game : Game) -> ObjectId:
    """
    Fetch a random model ID from the Models collection.

    Args:
        session: MongoDB session instance
    Returns:
        str: The ID of the randomly selected model
    """

    models : List[Model] = await get_models_dependent_on_game(db=db, 
                                                              game=game)

    if not models:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No models found in the database"
        )
    

    selected_model = random.choice(models)

    if (id := selected_model.id) is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Not a valid model"
        )
    return id


def get_model_client_from_provider(provider : str):
    ...
# =====================================================================


# Judge ===============================================================

async def get_judge_from_id(*, 
                            db :Database,
                            id : str | ObjectId) -> Judge:
    """
    Retrieves a Judge from the database by its unique ID.

    Parameters:
        db (Database): The database instance to query.
        id (str | ObjectId): The ID of the judge to retrieve, either as a string or ObjectId.

    Returns:
        Judge: An instance of the Judge model populated with the retrieved document's data.

    Raises:
        HTTPException: Raises 404 if no judge is found, 400 if the ID is invalid or another error occurs.
    """
    try:
        response = await db.judges.find_one({
            "_id" : ObjectId(id)
        })

        if response is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Interupt there exist no judge with id: {id}"
            )
        
        response["_id"] = str(response["_id"])

        return Judge(**response)

    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interupt {id} is not a valid id."
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Interupt {e}."
        )

# =====================================================================

# Leaderboard

async def fetch_and_process_leaderboard_data(
    *,
    db: Database,
    last_snapshot: datetime,
    current_snapshot: timedelta,
):
    try:


        # Time range for sessions
        time_range = {
            "$gte": last_snapshot,
            "$lt":  current_snapshot,
        }

        # Step 2: Aggregate session data
        pipeline = [
            {"$match": {"completed": True, "completed_time": time_range}},
            {"$addFields": {
                "target": {
                    "$cond": {
                        "if": {"$eq": ["$metadata.game_rules.deterministic", True]},
                        "then": "$metadata.kwargs.target",
                        "else": None
                    }
                },
                "system_prompt": {
                    "$cond": {
                        "if": {
                            "$and": [
                                {"$ne": ["$metadata.models_config.system_prompt", None]},
                                {"$ne": ["$metadata.models_config.system_prompt", ""]}
                            ]
                        },
                        "then": "$metadata.models_config.system_prompt",
                        "else": None
                    }
                }
            }},
            {"$lookup": {
                "from": "models",
                "localField": "agent_id",
                "foreignField": "_id",
                "as": "model_info"
            }},
            {"$unwind": {"path": "$model_info", "preserveNullAndEmptyArrays": True}},
            {"$group": {
                "_id": "$game_id",
                "sessions": {
                    "$push": {
                        "session_id": {"$toString": "$_id"},
                        "user_id": "$user_id",
                        "outcome": "$outcome",
                        "target": "$target",
                        "system_prompt": "$system_prompt",
                        "model": "$model_info._id"
                    }
                }
            }},
            {
                "$project": {
                    "_id": 0,
                    "game_id": "$_id",
                    "sessions": 1
            }}
        ]
        sessions = await db.sessions.aggregate(pipeline=pipeline)\
                                    .to_list(length=None)

        game_ids = [ s["game_id"] for s in sessions ]

        # Step 1: Fetch leaderboard data
        leaderboards = await db.leaderboards.find({
            "game_id" : { "$in" : game_ids } 
        }).to_list(None)

        leaderboard = { l["game_id"] : l for l in leaderboards }

        if not leaderboards and len(leaderboards) == 0:
            yield None
        

        # NOTE: rename me to a better function name
        def unique_objects(game : dict):
            unique_users =   {user['user_id'] for user in game['sessions']}
            unique_models =  {user['model']   for user in game['sessions']}
            unique_targets = {user['target']  for user in game['sessions']}
            return unique_users, unique_models, unique_targets
        
    
        for game in sessions:
            users, models, targets = unique_objects(game)
            yield (list(targets), list(users), list(models), game, leaderboard[game["game_id"]])

    except Exception as e:
        logger.error(e)
        raise e

# NOTE: Deprecate this when Abstract games 
async def get_leaderboards(db : Database):
    
    try:
        leaderboards = await db.leaderboards.find({}).to_list(None)
        if leaderboards is None or len(leaderboards) == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No leaderboard found"    
            )
        
        return leaderboards
    except HTTPException as h:
        raise h
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something occurred"
        )



async def get_leaderboard_from_game_id(game_id : ObjectId, 
                                       db : Database,
                                       skip : int =0,
                                       limit: int = 0) -> dict:
    """get leaderboard latest snapshot from mongodb"""
    try :
        leaderboard = await db.leaderboards.aggregate([
            { 
                "$match": { 
                    "game_id": game_id 
                }
            },
            # {
            #     "$unwind" : "$models"
            # },
            {
                "$unwind": "$players"  # Deconstruct the players array to prepare for lookup
            },
            {
                "$lookup": {
                    "from": "users",  # Collection to join with
                    "localField": "players.id",  # Field in players to match with users
                    "foreignField": "_id",  # Field in users to match against
                    "as": "user_details"  # Name for the joined data
                }
            },
            {
                "$unwind": "$user_details"  # Flatten the user_details array
            },
            {
                "$project": {
                    "_id": 0,
                    "id": { "$toString": "$_id" },  # Convert leaderboard _id to string
                    "game_id": { "$toString": "$game_id" },  # Convert game_id to string
                    "last_snapshot": 1,
                    "mean" : 1,
                    "player": {
                        "id": { "$toString": "$players.id" },  # Convert ObjectId to string
                        "elo": "$players.elo",
                        "username": "$user_details.username",
                    }
                }
            },
            {
                "$group": {
                    "_id": {
                        "id": { "$toString" : "$id" },
                        "game_id": { "$toString" : "$game_id"},
                        "last_snapshot": "$last_snapshot",
                        "mean" : "$mean",
                    },
                    "players": { "$push": "$player" } 
                }
            },
            {
                "$project": {
                    "_id": 0,
                    "id": "$_id.id",
                    "game_id": "$_id.game_id",
                    "mean": "$_id.mean",
                    "last_snapshot": "$_id.last_snapshot",
                    "players": { "$slice" : [ "$players", max(0, skip), max(skip + 1, limit) ] } ,
                }
            }
        ]).next()
        if leaderboard is None or len(leaderboard) == 0:
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"No leaderboard found under {str(game_id)}"    
            )



        return leaderboard
    except HTTPException as h:
        raise h
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something on server end could not complete this call"
        )
    


# cron

# 1. load games session into object structure [{ game_id : str, session [ outcome : "win" | "loss", user_id : str, target :  ] }]
# 2. load leaderboard into memory
# 3. take time snapshot
# 4. set users elo to 0 prepare for new elo
# 5. use linear regression model to define new elo
# 
     

# Leaderboard state
# id
# game_id one to one relationship
# last_snapshot
# players: object holds { username : str, beta : float64 }

    