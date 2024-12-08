from fastapi import APIRouter, HTTPException, status

from typing import Dict, Any

from bson import ObjectId
from bson.errors import InvalidId
from api.core.config import settings



# from datetime import datetime, timedelta, UTC

from api import crud
from api.deps import Database
from api.function import sigmoid


router = APIRouter()

@router.get("/leaderboard/{game_id}", tags=["stats"])
async def get_leaderboard(s : int,
                          l: int,
                          game_id : str,
                          db : Database) -> Dict[str, Any]:
    
    if s < 0 or l < 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="`skip` and `limit` must be non-negative integers."
        )
    
    try:
        id = ObjectId(game_id)
        leaderboard = await crud.get_leaderboard_from_game_id(skip=s,
                                                              limit=l,
                                                              game_id=id,
                                                              db=db)
        leaderboard["players"] = list(map(lambda x : dict(
            id=x["id"],
            username=x["username"],
            elo=int(sigmoid(x["elo"], 
                            k=settings.K, 
                            center=leaderboard["mean"]["player"]) * settings.ELO_SCALE)), 
            leaderboard["players"]))
        return leaderboard
    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{game_id} is not a valid ID."
        )
    