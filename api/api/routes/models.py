from typing import List

from fastapi import APIRouter, status, HTTPException

from api.deps import Database
from api.crud import get_models
from api.models import ModelPublic, Model
router = APIRouter()


@router.get("/")
async def get_public_models(db : Database) -> List[ModelPublic]:
    try:
        models : List[Model]= await get_models(db=db)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error: {str(e)}"

        )
    return [ModelPublic.from_model(model) for model in models]