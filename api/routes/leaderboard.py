from fastapi import APIRouter, HTTPException, status

from typing import Dict, Any

from bson import ObjectId
from bson.errors import InvalidId

from api import crud
from api.deps import Database


__all__ = ["router"]
router = APIRouter()


@router.get("/leaderboard/{game_id}", tags=["Game"])
async def get_leaderboard(game_id: str, db: Database) -> Dict[str, Any]:
    """access leaderboard score of game"""
    try:
        assert ObjectId.is_valid(game_id), "Not valid Id"
        game_id = ObjectId(game_id)
        leaderboard = await crud.get_game_leaderboard(game_id=game_id, db=db)
        return leaderboard
    except HTTPException as h:
        raise h
    except InvalidId:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"{game_id} is not a valid ID.",
        )
