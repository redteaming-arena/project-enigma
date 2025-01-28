import re
import json
from typing import Any, List, Dict, Literal
from bson import ObjectId
from fastapi import APIRouter, HTTPException, status, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pymongo.results import UpdateResult

from datetime import datetime, UTC
from zoneinfo import ZoneInfo

from api import crud
from api.deps import Database, AuthUser
from api.models import (
    Request,
    ClientMessage,
    GameSessionCreateResponse,
    GameSessionPublic,
    GameReadOnly,
    GameSessionTitleRequest,
    Message,
)
from api.generative import Registry as Models
from api.judge import registry
from api.utils import logger

__all__ = ["router"]

router = APIRouter()


@router.post("/create-chat", tags=["Session", "Creation"])
async def create_session(
    game_id: str, user: AuthUser, db: Database
) -> GameSessionCreateResponse:
    """
    Creates a new game session for the current user with the specified game ID

    Args:
        game_id (str): ID of the game for the session
        current_user (AuthUser): Currently authenticated user
        session: MongoDB session instance

    Returns:
        GameSessionCreateResponse: Response model with session_id and target
    """
    try:
        game = await crud.get_game_from_id(db=db, id=game_id)

        if not game:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No contexts found for the specified game",
            )

        model_id = await crud.get_random_model_id(db=db, game=game)

        judge = await crud.get_judge_from_id(db=db, id=game.judge_id)

        # DEBUG:
        logger.debug(f"sample model id: {model_id}")

        new_session = await crud.create_game_session(
            db=db, user_id=user.id, game=game, judge=judge, model_id=model_id
        )

    # handle the http request of the child calls
    except HTTPException as h:
        raise h
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: Something went wrong creating session",
        )

    return GameSessionCreateResponse.from_game(new_session)


@router.get("/{session_id}/chat_conversation", tags=["Session", "Progress"])
async def get_chat_session(
    session_id: str, user: AuthUser, db: Database
) -> GameSessionPublic:

    try:
        assert ObjectId.is_valid(session_id), "not a valid ID"
        session_id = ObjectId(session_id)

        session = await crud.get_session(user_id=user.id, session_id=session_id, db=db)

        timed = session.metadata.game_rules.get("timed", False)

        if timed and not session.completed:
            timed_limit = session.metadata.game_rules.get("timed_limit", 5400) // 60

            # Calculate elapsed time
            if session.start_time:
                # Sessions was stated but never completed
                start = session.start_time
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)

            else:
                # session is never started
                # then were comparing the game was created
                # if it exceeds triple the time then the session
                # will be deleted
                start = session.create_time
                if start.tzinfo is None:
                    start = start.replace(tzinfo=UTC)

            end = datetime.now(ZoneInfo("UTC"))  # Get the current time in UTC
            elapsed_time = (end - start).total_seconds()

            if elapsed_time > timed_limit:
                # Time is over, mark session as completed or lost
                session.outcome = "loss"
                session.completed_time = end
                session.completed = True

                # don't show this to the user it's not needed
                if session.start_time is None:
                    session.visible = False

                # Update the session in the database
                await crud.update_game_session(
                    session_id=session_id,
                    db=db,
                    updated_session=session,
                    updates=["outcome", "completed_time", "completed", "visible"],
                )

            else:
                # set the game to the elapse timeleft so the game does not give
                # a false sense of time
                if session.start_time:
                    session.metadata.game_rules["time_limit"] = (
                        timed_limit - elapsed_time
                    ) * 60

    except HTTPException as h:
        raise h
    except AssertionError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Server Error: Something happened when trying to retrieve session",
        )
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: Something happened when trying to retrieve session",
        )

    return session


@router.post(
    "/{session_id}/chat_conversation/{user_id}/start", tags=["Session", "Progress"]
)
async def start_game(session_id: str, user_id: str, user: AuthUser, db: Database):

    try:
        assert ObjectId.is_valid(session_id), "Session ID is not valid"
        assert ObjectId.is_valid(user_id), "User ID is not valid"

        session_id = ObjectId(session_id)
        user_id = ObjectId(user_id)

        start = datetime.now(UTC)
        session = await crud.get_session(user_id=user_id, session_id=session_id, db=db)
        if not session.start_time:
            session.start_time = start

            if str(session.user_id) != str(user.id) != str(user_id):
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="You do not have permission to access this session",
                )

            await crud.update_game_session(
                db=db,
                session_id=session_id,
                updated_session=session,
                updates=["start_time"],
            )
        else:
            return Message(
                status=status.HTTP_200_OK, message="sessions has already been started"
            )
    except HTTPException as h:
        logger.error(h)
        raise h
    except AssertionError as a:
        logger.error(a)
        raise HTTPException(detail=str(a), status_code=status.HTTP_400_BAD_REQUEST)


@router.delete("/{user_id}/chat_conversation", tags=["Session", "Completed"])
async def deleted_sessions(
    user_id: str,
    session_ids: List[str],
    user: AuthUser,
    db: Database,
) -> Message:
    try:
        assert ObjectId.is_valid(user_id), "Invalid user id"
        if user_id != str(user.id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="You don't have authorization to delete these sessions.",
            )

        # Validate session IDs
        session_object_ids = [
            ObjectId(id) for id in session_ids if ObjectId.is_valid(id)
        ]
        if not session_object_ids:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="None of the provided session IDs are valid.",
            )

        # Update sessions to mark as invisible
        response: UpdateResult = await db.sessions.update_many(
            {"_id": {"$in": session_object_ids}, "user_id": user.id, "visible": True},
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


@router.post("/{session_id}/chat_conversation/{user_id}/title", tags=["Session", "End"])
async def title_completion(
    current_user: AuthUser,
    db: Database,
    content: GameSessionTitleRequest,
    session_id: str,
    user_id: str,
) -> Message:
    """
    Generate and update a session title using minimal tokens.
    Args:
        current_user (AuthUser): The authenticated user
        db (Database): Database connection
        content (GameSessionTitleRequest): Request containing prompt/history
        id (str): Session ID
        user_id (str): User ID
    Returns:
        dict: Updated session information
    """

    try:
        assert ObjectId.is_valid(user_id), "Invalid user user id"
        assert ObjectId.is_valid(session_id), "Invalid user session id"

        if str(current_user.id) != str(user_id):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="You do not have permission to access this session",
            )

        # Either completed or not we can change the title
        user_id = ObjectId(user_id)
        session_id = ObjectId(session_id)
        session = await crud.get_session(session_id=session_id, user_id=user_id, db=db)

        # and session.title is None
        if content.generate and len(session.history) >= 1:
            # Create minimal context for title generation with modified format
            title_prompt = [
                {
                    "role": "system",
                    "content": "Analyze the input message model assistants response and generate a concise, and short,\
                                \n\rengaging, and relevant title that captures the essence of the content or theme. \
                                \n\rEnsure the title is clear and compelling for its intended audience and in xml \
                                \n\rstarting with <title> end with </title>.",
                },
                {"role": "user", "content": session.history[0].content},
            ]

            # let he model of the session determine the title
            client = Models.get_client(session.model.namespace)

            # Generate title with max_tokens limit
            title_response = client.generate(
                title_prompt, session.model.name, stream=False, max_tokens=1000
            )

            response = title_response.get_text().strip()
            title = re.sub("</?title>", "", response)  # remove <title> tags
            if title == "":
                title = "Untitled Game"
        else:

            title = "Untitled Game"

        # Update title in database
        result = await db.sessions.update_one(
            {"_id": session.id}, {"$set": {"title": title}}
        )

        if result.modified_count == 0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found or no changes made",
            )

        return Message(
            status=status.HTTP_200_OK,
            message="Title completion was successful.",
            data=title,
        )

    except HTTPException as h:
        raise h
    except Exception as e:
        logger.error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server Error: something happen while title completion",
        )


def model_generate_generator(
    history: List[ClientMessage],
    id: ObjectId,
    user_id: ObjectId,
    current_user: AuthUser,
    db: Database,
    session: Dict[str, Any],
    background: BackgroundTasks,
):

    # calmative tokens accumulated from stream
    calmative_token = ""
    updates = {"history"}

    try:
        # session id should equal user token id, and user_id and or session is completed
        # then raise a forbidden status
        if (
            str(session.user_id) != str(current_user.id) != str(user_id)
            or session.completed
        ):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to access this session",
            )
        # NOTE: where assuming the user does not have access
        session.history = history
        model_client_name = session.model.namespace
        model = session.model.name
        metadata = session.metadata
        validator = registry.get_validator(session.judge.validator.function.name)

        client = Models.get_client(model_client_name)

        tools_config = metadata.models_config.get("tools_config", {})
        system_prompt = metadata.models_config.get("system_prompt", "")
        history = session.history
        if system_prompt:
            history = [{"role": "system", "content": system_prompt}] + history

        if tool_enabled := tools_config.get("enabled", False):
            stream = client.generate(
                history, model, tools=tools_config.get("tools", [])
            )
        else:
            stream = client.generate(history, model)
        for token in stream.iter_chunks():
            calmative_token += token

            # NOTE:
            #       Target Base comparison allows for determinism output the models
            #       want to be produced.
            if (
                not tool_enabled
                and not session.outcome
                and validator(**{"source": calmative_token} | metadata.kwargs)
            ):
                session.outcome = "win"
                session.completed = True
                session.completed_time = datetime.now(UTC)
                updates = updates | {"completed", "outcome", "completed_time"}

            yield "{payload}\n".format(
                payload=json.dumps(dict(event="message", content=token))
            )

        functions_called = stream.get_function_call()

        # TODO
        # NOTE: this needs to be abstracted out to execute and validated
        #       possible more intricate function calling...
        if functions_called is not None and any(
            validator(
                **(
                    {
                        "source": calmative_token,
                        "function_call_name": func["name"],
                        "function_call_arguments": func["arguments"],
                    }
                    | metadata.kwargs
                    if metadata.kwargs is not None
                    else {}
                )
            )
            for func in list(functions_called)
        ):
            session.outcome = "win"
            session.completed = True
            session.completed_time = datetime.now(UTC)
            updates = updates | {"completed", "outcome", "completed_time"}

        if session.outcome == "win":
            yield "{payload}\n".format(
                payload=json.dumps(
                    dict(
                        event="end",
                        outcome="win",
                        content=calmative_token,
                    )
                )
            )
    except Exception as e:
        logger.error(e)
    finally:
        session.history.append({"role": "assistant", "content": calmative_token})
        background.add_task(
            crud.update_game_session,
            session_id=id,
            updated_session=session,
            db=db,
            updates=list(updates),
        )


@router.post(
    "/{session_id}/chat_conversation/{user_id}/conversation",
    tags=["Session", "Progress"],
)
async def completion(
    request: Request,
    session_id: str,
    user_id: str,
    current_user: AuthUser,
    db: Database,
    background: BackgroundTasks,
) -> StreamingResponse:
    """
    Handle prompt from the user.

    Args:
        session_id (str): ID of the current game session
        current_user (AuthUser): The currently authenticated user
        session: MongoDB session instance

    Returns:
        StreamingResponse: Streaming response with model output
    """
    assert ObjectId.is_valid(session_id), "not a valid ID"
    assert ObjectId.is_valid(user_id), "not a valid ID"

    session_id = ObjectId(session_id)
    user_id = ObjectId(user_id)
    session = await crud.get_session(
        session_id=session_id, user_id=current_user.id, completed=False, db=db
    )

    response = StreamingResponse(
        model_generate_generator(
            history=request.encode,
            id=session_id,
            user_id=user_id,
            current_user=current_user,
            db=db,
            session=session,
            background=background,
        ),
    )
    response.headers["x-vercel-ai-data-stream"] = "v1"
    return response


@router.post(
    "/{session_id}/chat_conversation/{user_id}/conclude", tags=["Session", "End"]
)
async def end(
    session_id: str,
    user_id: str,
    current_user: AuthUser,
    db: Database,
    outcome: Literal["loss"],
    history: List[ClientMessage],
) -> Message:
    """
    Ends current game session for user
    Args:
        session_id (str): The ID of the session to forfeit.
        current_user (AuthUser): The currently authenticated user.
        session: MongoDB session instance.
    Returns:
        dict: response containing outcome {"outcome": "loss"}
    """

    session_id = ObjectId(session_id)
    user_id = ObjectId(user_id)
    session = await crud.get_session(
        session_id=session_id, user_id=user_id, completed=False, db=db
    )

    if str(user_id) != str(session.user_id) != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this session",
        )

    session.completed = True
    session.outcome = outcome
    session.completed_time = datetime.now(UTC)
    session.history = [msg.model_dump(exclude_none=True) for msg in history]

    try:

        await crud.update_game_session(
            session_id=session_id,
            db=db,
            updated_session=session,
            updates=["completed", "outcome", "completed_time", "history"],
        )

    except Exception as e:
        raise e

    return Message(
        status=status.HTTP_200_OK,
        message="Session written successful",
        data={"outcome": session.outcome},
    )


@router.get("/chat_conversation/{shared_id}", tags=["Session", "Completed"])
async def get_shared_conversation(shared_id: str, db: Database) -> GameReadOnly:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (AuthUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """

    session = await crud.get_session_from_shared_id(shared_id=shared_id, db=db)
    if not session.completed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this session",
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
        description=session.description,
    )

    return response_data


@router.post(
    "/{session_id}/chat_conversation/{user_id}/share", tags=["Session", "Completed"]
)
async def post_shared_conversation(
    session_id: str, user_id: str, user: AuthUser, db: Database
) -> Message:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (AuthUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """
    assert ObjectId.is_valid(user_id)
    assert ObjectId.is_valid(session_id)

    if user_id != str(user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not write access to access this session",
        )

    session_id = ObjectId(session_id)
    session = await crud.get_session(
        session_id=session_id, user_id=user.id, completed=True, db=db
    )

    if not session.completed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this session",
        )

    session_id = ObjectId()
    result = await db.sessions.update_one(
        {"_id": session.id}, {"$set": {"shared": session_id}}
    )

    if result.modified_count == 0:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Session not found or no changes made",
        )

    return Message(
        status=200, message="Session was successfully shared", data=str(session_id)
    )


@router.get("/history", tags=["Session", "Completed"])
async def get_history(
    current_user: AuthUser, db: Database, s: int = 0, l: int = None
) -> List[GameReadOnly]:
    """
    Retrieves the chat history of all game sessions of the current user.

    Args:
        current_user (AuthUser): The currently authenticated user.
        db (Database): The database instance.
        s (int): The skip value for pagination.
        l (int): The limit value for pagination (optional).
    Returns:
        List[GameSessionHistoryResponse]: List of completed game sessions with session_id, target, outcome, duration.
    """
    try:

        sessions = await crud.get_sessions_for_user(
            user_id=current_user.id, db=db, skip=s, limit=l
        )

        session_history = [
            GameReadOnly.from_game_session(session) for session in sessions
        ]

    except AssertionError as a:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=a)
    except Exception as e:
        raise e

    return session_history


@router.get(
    "/{session_id}/chat_conversation/{user_id}/history", tags=["Session", "Completed"]
)
async def get_session_history(
    session_id: str, user_id: str, db: Database
) -> GameReadOnly:
    """
    Retrieves the chat history of session_id of current user

    Args:
        session_id (str): The ID of the session to retrieve history for
        current_user (AuthUser): The currently authenticated user
        session: MongoDB session instance
    Returns:
        GameSessionHistoryResponse: Database history details
    """
    assert ObjectId.is_valid(user_id)
    assert ObjectId.is_valid(session_id)

    session = await crud.get_session(
        session_id=session_id, user_id=user_id, completed=True, db=db
    )

    if not session.completed:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="You do not have permission to access this session",
        )

    duration = (session.completed_time - session.create_time).total_seconds()

    response_data = GameReadOnly(
        session_id=str(session.id),
        title=session.title,
        outcome=session.outcome,
        duration=duration,
        last_message=session.completed_time,
        history=session.history,
        description=session.description,
    )

    return response_data
