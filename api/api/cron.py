from api.utils import repeat_every, logger
from api.deps import get_database

from pymongo.results import DeleteResult

@repeat_every(seconds=60, logger=logger)
def compute_leaderboard():
    # { "visible": { "$exists": False } },
    # { "visible": False },
    ...
    # spawn threads loop though all games in threads
    # run job
    

# NOTE: this function will deprecated as cron 
#       and move to computer leaderboards
#       for clean up after the elo is scored
#       for the given snapshot.
@repeat_every(seconds=60, logger=logger)
async def history_garbage_collection():
    """
        function just run a query within the backend 
        to clean up any forfeited games or game user 
        was afk
    """
    db = await get_database()
    result: DeleteResult = await db.sessions.delete_many({
        "$or": [
            {
                # game session completed 
                "completed": True, 
                "visible": False
            },
            {
                # game session is not visible
                "visible": False
            },
            {
                # quote on quote faild game sessions
                "completed" : True,
                "$expr": {
                    "$lt": [{ "$size": "$history" }, 2]
                }
            }
        ]
    })
    if result.deleted_count == 0:
        logger.info("Currently no empty sessions")
