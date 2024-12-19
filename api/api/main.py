"""Entrance file """
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.routing import APIRoute
from starlette.middleware.cors import CORSMiddleware

from api.core.config import settings
from api.routes import (login,\
                        user,\
                        game,\
                        models,\
                        game_session,\
                        health)
from api.backend_ping_test import db_ping_server
from api.cron import compute_leaderboard, history_garbage_collection
from api.deps import close_database
from api.utils import logger

def custom_generate_unique_id(route: APIRoute) -> str:
    if len(route.tags) > 0:
        return f"{route.tags[0]}-{route.name}"
    else:
        return f'{route.name}'


@asynccontextmanager
async def lifespan(_app : FastAPI):
    """ check if the server is up """
    # NOTE: before the server starts run these functions
    try :
        await db_ping_server()
        await history_garbage_collection()

    except Exception as e:
        raise e
    
    # NOTE Server has started
    yield
    
    logger.info(f"Closing Database Connection")
    await close_database()
    # NOTE: if anything needs to be cleaned up put below here


app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url="/openapi.json",
    generate_unique_id_function=custom_generate_unique_id,
    lifespan=lifespan,
    docs_url="/docs",
)

if settings.all_cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.all_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


app.include_router(login.router)
app.include_router(user.router,  tags=["User"])
app.include_router(game.router, tags=["Game"])
app.include_router(models.router, prefix="/model", tags=["Model"])
app.include_router(game_session.router)
app.include_router(health.router)