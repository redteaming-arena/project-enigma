import re
import json
from typing import Any, List, Dict, Literal
from bson import ObjectId
from fastapi import APIRouter, \
                    HTTPException,\
                    status,\
                    BackgroundTasks
from fastapi.responses import StreamingResponse
from pymongo.results import UpdateResult

from datetime import datetime, UTC
from zoneinfo import ZoneInfo

from api import crud
from api.deps import Database, CurrentUser
from api.models import (Request,
                        ClientMessage,
                        GameSessionCreateResponse,
                        GameSessionPublic,  
                        GameReadOnly,
                        GameSessionTitleRequest,
                        Message)
from api.generative import Registry as Models
from api.judge import registry
from api.utils import logger

router = APIRouter()

@router.post("/create-chat", tags=["Game Session"])
async def create_session(
    game_id: str, 
    user: CurrentUser, 
    db: Database
) -> GameSessionCreateResponse:
    """
    Creates a new game session for the current user with the specified game ID

    Args:
        game_id (str): ID of the game for the session
        current_user (CurrentUser): Currently authenticated user
        session: MongoDB session instance
    
    Returns:
        GameSessionCreateResponse: Response model with session_id and target
    """
    try:
        game = await crud.get_game_from_id(db=db, id=game_id)

        if not game:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, 
                detail="No contexts found for the specified game"
            )

        model_id = await crud.get_random_model_id(db=db, 
                                                  game=game)
        
        judge = await crud.get_judge_from_id(db=db, 
                                             id=game.judge_id)
        
        logger.info(f"sample model id: {model_id}")
        new_session = await crud.create_game_session(
            db=db,
            user_id=user.id,
            game=game,
            judge=judge,
            model_id=model_id
        )

    except HTTPException as h:
        raise h
    except Exception:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="Server Error: Something went wrong creating session")
        

    return GameSessionCreateResponse.from_game(new_session)


@router.get('/{id}/chat_conversation', tags=["Game Session"])
async def get_chat_session(
    id : str,
    user : CurrentUser,
    db : Database
) -> GameSessionPublic:
    
    try:
        session = await crud.get_session(
            user_id=ObjectId(user.id),
            session_id=id,
            db=db
        )

        timed = session.metadata.game_rules.get("timed", False)

        if timed and not session.completed:
            timed_limit = session.metadata.game_rules.get("timed_limit", 5400) // 60
            wiggle_room = 30  # 30 seconds buffer
            total_time_limit = timed_limit + wiggle_room

            # Calculate elapsed time
            start = session.create_time
            if start.tzinfo is None:
                start = start.replace(tzinfo=UTC)
            end = datetime.now(ZoneInfo("UTC"))  # Get the current time in UTC
            elapsed_time = (end - start).total_seconds()
            if elapsed_time > total_time_limit:
                # Time is over, mark session as completed or lost
                session.outcome = "loss"
                session.completed_time = end
                session.completed = True

                # Update the session in the database
                await crud.update_game_session(session_id=id, 
                                               db=db, 
                                               updated_session=session,
                                               updates=["outcome", 
                                                        "completed_time",
                                                        "completed"])



    except HTTPException as h:
        raise h
    except Exception as e:
        print(e)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, 
                            detail="Server Error: Something happened when trying to retrieve session")

    return session

@router.delete("/{user_id}/chat_conversation", tags=["Game Session"])
async def deleted_sessions(
    user_id : str,
    session_ids : List[str],
    user: CurrentUser, 
    db: Database,
) -> Message:
    try:
        if user_id != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="You don't have authorization to delete these sessions."
            )

        # Validate session IDs
        session_object_ids = [ObjectId(id) for id in session_ids if ObjectId.is_valid(id)]
        if not session_object_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="None of the provided session IDs are valid."
            )

        # Update sessions to mark as invisible
        response: UpdateResult = await db.sessions.update_many(
            {
                "_id": {"$in": session_object_ids},
                "user_id": user.id,
                "visible": True
            },
            {
                "$set": {"visible": False},
            },
        )

        # Handle no modifications
        if response.modified_count == 0:
            return Message(
                status=status.HTTP_200_OK,
                message="No sessions were modified.",
            )

    except HTTPException as h:
        raise h
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Server Error: Unable to delete sessions. {str(e)}",
        )

    # Success response
    return Message(
        status=status.HTTP_200_OK,
        message=f"{response.modified_count} session(s) successfully deleted.",
    )


@router.post("/{id}/chat_conversation/{user_id}/title", tags=["Game Session"])
async def title_completion(
    current_user: CurrentUser,
    db: Database,
    content: GameSessionTitleRequest,
    id: str,
    user_id: str = None,
) -> Message:
    """
    Generate and update a session title using minimal tokens.
    Args:
        current_user (CurrentUser): The authenticated user
        db (Database): Database connection
        content (GameSessionTitleRequest): Request containing prompt/history
        id (str): Session ID
        user_id (str): User ID
    Returns:
        dict: Updated session information
    """

    
    try:
        # 

        # Either completed or not we can change the title
        session = await crud.get_session(
            session_id=id,
            user_id=current_user.id,
            db=db
        )
        
        # and session.title is None
        if str(session.user_id) != str(current_user.id) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="You do not have permission to access this session"
            )

        if content.generate and session.title is None and len(session.history) >= 2:
            # Create minimal context for title generation with modified format
            title_prompt = [
                {
                    "role": "system", 
                    "content": "Analyze the input message model assistants response and generate a concise,\
                                \n\rengaging, and relevant title that captures the essence of the content or theme. \
                                \n\rEnsure the title is clear and compelling for its intended audience and in xml \
                                \n\rstarting with <3c7469746c653e> end with</3c7469746c653e>."
                },
                {
                    "role": "user", 
                    "content": session.history[1].content
                }
            ]

            client = Models.get_client(session.model.name)
            
            # Generate title with max_tokens limit
            title_response = client.generate(
                title_prompt,
                session.model.name,
                stream=False,
                max_tokens=100
            )
            
            response = title_response.get_text().strip()

            title = re.sub('</?3c7469746c653e>', '', response)
        else:

            title = "Untitled Game"
        
        # Update title in database
        result = await db.sessions.update_one(
            {"_id": session.id},
            {"$set": {"title": title}}
        )
        
        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or no changes made"
            )
            
        return Message(
            status=status.HTTP_200_OK,
            message="Title completion was successful.",
            data=title
        )
        
    except HTTPException as h:
        print(h)
        raise h
    except Exception as e:
        print(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: something happen while title completion"
        )

def model_generate_generator(
        history : List[ClientMessage],
        id: str,
        user_id: str,
        current_user: CurrentUser,
        db : Database,
        session : Dict[str, Any],
        background : BackgroundTasks):

    try:
        if str(session.user_id) != str(current_user.id) != user_id or session.completed:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this session"
            )

        # NOTE: where assuming the user does not have access 
        session.history = history
        model = session.model.name
        metadata = session.metadata
        validator = registry.get_validator(session.judge.validator.function.name)

        client = Models.get_client(model)

        updates = {"history"}
        calmative_token = ""
        
        tools_config = metadata.models_config.get("tools_config", {})
        system_prompt = metadata.models_config.get("system_prompt", "")
        history = session.history
        if system_prompt:
            history = [{"role": "system", "content": system_prompt}] + history

        if tool_enabled := tools_config.get("enabled", False):
            stream = client.generate(history, model, tools=tools_config.get("tools", []))
        else:
            stream = client.generate(history, model)
        

        for token in stream.iter_tokens():
            calmative_token += token
            
            if not tool_enabled and not session.outcome and validator(**{"source" : calmative_token} | metadata.kwargs):
                session.outcome = "win"
                session.completed = True
                session.completed_time = datetime.now(UTC)
                updates = updates | {"completed", "outcome", "completed_time"}

            yield "{payload}\n".format(
                payload=json.dumps(dict(event="message", content=token))
            )


        functions_called = stream.get_function_call()
        if functions_called and any(
            validator(
                **{"source": calmative_token} |
                metadata.get("kwargs", {}) |
                {"function_call_name": func.name, "function_call_arguments": func.argument}
            ) for func in functions_called
        ):
            session.outcome = "win"
            session.completed = True
            session.completed_time = datetime.now(UTC)
            updates = updates | {"completed", "outcome", "completed_time"}


    finally:
        session.history.append({"role": "assistant", "content": calmative_token }) 
        background.add_task(crud.update_game_session,
            session_id=id,
            updated_session=session,
            db=db,
            updates=list(updates)
        )


@router.post("/{id}/chat_conversation/{user_id}/conversation", tags=["Game Session"])
async def completion(
    request : Request,
    id: str,
    user_id: str,
    current_user: CurrentUser,
    db: Database,
    background : BackgroundTasks
) -> StreamingResponse:
    """
    Handle prompt from the user.

    Args:
        session_id (str): ID of the current game session
        current_user (CurrentUser): The currently authenticated user
        session: MongoDB session instance

    Returns:
        StreamingResponse: Streaming response with model output
    """
    session = await crud.get_session(
        session_id=id,
        user_id=current_user.id,
        completed=False,
        db=db
    )
    
    response = StreamingResponse(
        model_generate_generator(history=request.encode, 
                                 id=id, 
                                 user_id=user_id, 
                                 current_user=current_user, 
                                 db=db, 
                                 session=session,
                                 background=background),
    )
    response.headers['x-vercel-ai-data-stream'] = 'v1'
    return response

@router.post("/{id}/chat_conversation/{user_id}/conclude", tags=["Game Session"])
async def end(
    id: str, 
    user_id : str,
    current_user: CurrentUser, 
    db: Database,
    outcome : Literal['loss'],
    history : List[ClientMessage]
) -> Message:
    """
    Ends current game session for user
    Args:
        session_id (str): The ID of the session to forfeit.
        current_user (CurrentUser): The currently authenticated user.
        session: MongoDB session instance.
    Returns:
        dict: response containing outcome {"outcome": "loss"}
    """

    session = await crud.get_session(session_id=id,
                                     user_id=ObjectId(user_id),
                                     completed=False,
                                     db=db)

    if user_id != str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You do not have permission to access this session"
        )
    
    
    session.completed = True
    session.outcome = outcome
    session.completed_time = datetime.now(UTC)
    session.history = [msg.model_dump(exclude_none=True) for msg in history]


    try:

        await crud.update_game_session(session_id=id, 
                                       db=db, 
                                       updated_session=session,
                                       updates=[
                                            "completed",
                                            "outcome",
                                            "completed_time",
                                            "history"])

    except Exception as e:
        raise e


    return Message(
        status=status.HTTP_200_OK,
        message="Session written successful",
        data={"outcome": session.outcome}
    )



@router.get("/history", tags=["History"])
async def get_history(
    current_user: CurrentUser,
    db: Database,
    s: int,
    l: int = None
) -> List[GameReadOnly]:
    """
    Retrieves the chat history of all game sessions of the current user.

    Args:
        current_user (CurrentUser): The currently authenticated user.
        db (Database): The database instance.
        s (int): The skip value for pagination.
        l (int): The limit value for pagination (optional).
    Returns:
        List[GameSessionHistoryResponse]: List of completed game sessions with session_id, target, outcome, duration.
    """
    try:
        # Fetch the sessions from the database
        sessions = await crud.get_sessions_for_user(user_id=current_user.id, db=db, skip=s, limit=l)

        # Prepare the session history data
        session_history = [
            GameReadOnly.from_game_session(session)
            for session in sessions
        ]
    except Exception as e:
        # Raise an error if something goes wrong
        raise e
    
    # Return the session history as a JSON response
    return session_history

@router.get("/chat_conversation/{shared_id}", tags=["Shared"])
async def get_shared_conversation(
    shared_id: str,
    db : Database
) -> GameReadOnly:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (CurrentUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """


    session = await crud.get_session_from_shared_id(shared_id=shared_id,
                                                    db=db)
    if not session.completed:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You do not have permission to access this session"
        )
    
    duration = (session.completed_time - session.create_time).total_seconds()
    

    response_data = GameReadOnly(
        username=session.user.username,
        session_id=str(session.id),
        title=session.title,
        outcome=session.outcome,
        duration=duration,
        last_message=session.completed_time,
        history=session.history,
        model=session.model,
        description=session.description
    )

    return response_data

@router.post("/{id}/chat_conversation/{user_id}/share", tags=["Shared"])
async def post_shared_conversation(
    id : str,
    user_id : str,
    user : CurrentUser,
    db : Database
) -> Message:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (CurrentUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """

    if user_id != str(user.id):
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You do not write access to access this session"
        )


    session = await crud.get_session(session_id=id,
                                     user_id=user.id,
                                     completed=True,
                                     db=db)
    
    if not session.completed:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You do not have permission to access this session"
        )
    

    session_id = ObjectId()
    result = await db.sessions.update_one(
            {"_id": session.id},
            {"$set": {"shared": session_id}}
        )
        
    if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or no changes made"
            )

    return Message(
        status=200,
        message="Session was successfully shared",
        data=str(session_id)
    )


@router.get("/{id}/chat_conversation/{user_id}/history", tags=["History"])
async def get_session_history(
    id: str,
    user_id: str,
    db : Database
) -> GameReadOnly:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (CurrentUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """


    session = await crud.get_session(session_id=id, 
                                     user_id=ObjectId(user_id),
                                     completed=True,
                                     db=db)
    
    if not session.completed:
        raise HTTPException(
            status_code = status.HTTP_403_FORBIDDEN,
            detail = "You do not have permission to access this session"
        )
    
    duration = (session.completed_time - session.create_time).total_seconds()
    
    response_data = GameReadOnly(
        session_id=str(session.id),
        title=session.title,
        outcome=session.outcome,
        duration=duration,
        last_message=session.completed_time,
        history=session.history,
        description=session.description
    )

    return response_data



@router.delete('/all')
async def all_session(db : Database):
    await db.sessions.delete_many({})
