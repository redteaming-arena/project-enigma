from __future__ import annotations
import json
import asyncio
from typing import List, Any, Optional, Dict

from fastapi import APIRouter, HTTPException
from sse_starlette.sse import EventSourceResponse

from api.deps import Database
from api.crud import get_game_from_id, get_games
from api.models import Game, StreamResponse
from api.utils import handleStreamResponse



router = APIRouter()

@router.get("/game")
async def get_all_games(
    db : Database,
    s : int,
    l : Optional[int] = None
) -> List[Game]:
    """
    Get all games and convert them to GamePublic model.
    Accessible via both /game/ and /game
    """
    try:
        response = []
        async for game in get_games(db=db, skip=s, limit=l):
                response.append(
                     Game(
                          id=str(game.id),
                          title=game.title,
                          image=game.image
                    ).to_dict()
                )
                
        return response
    except HTTPException as e:
        raise e
    
@router.get("/game/stream")
async def get_all_games_stream(
    db : Database,
    s : int,
    l : Optional[int] = None
):
    """
    Get all games and convert them to GamePublic model.
    Accessible via both /game/ and /game
    """

    async def stream_games():
        async for game in get_games(db=db, skip=s, limit=l):
            yield "{payload}".format(
                    payload=json.dumps(dict(event="message", content=Game(
                        id=str(game.id),
                        title=game.title,
                        image=str(game.image)
                        ).model_dump(mode="json", exclude_none=True))
                    )
                )
            await asyncio.sleep(2.0)
            
    return EventSourceResponse(stream_games())



@router.get("/game/{id}")
async def game_from_id(db : Database, id : str) -> Any:
    response = await get_game_from_id(db=db, id=id)
    return response


