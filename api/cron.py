import numpy as np

from datetime import datetime, timedelta, timezone, UTC

from typing import List, AsyncGenerator, Any


from api import crud
from api.deps import get_database, Database
from api.function import sigmoid
from api.utils import repeat_every,\
                      logger,\
                      convert_list_to_one_hot_vector

DEFAULT_CRON_JOB_CALL = 60

# FIXME: I don't like how I coded this
#        ideally this is handled in one query, and where I'm 
#        under utilizing mongodb queries
@repeat_every(seconds=DEFAULT_CRON_JOB_CALL, 
              logger=logger)
async def compute_leaderboard():

    # cap games counting
    snapshot = datetime.now(UTC)

    # get a AsyncIOMotorDatabase client to access mongoDB
    # collections.
    db : Database = await get_database()

    # NOTE: I don't like this solution 
    leaderboard = await db.leaderboards.find_one({})

    if leaderboard is None:
        logger.error("Session not recorded due to not finding leaderboard")
        return
    
    default_delta = snapshot - timedelta(minutes=DEFAULT_CRON_JOB_CALL)
    last_snapshot = leaderboard.get("last_snapshot", default_delta)
    
    last_snapshot = last_snapshot.replace(tzinfo=timezone.utc)

    sessions = crud.fetch_and_process_leaderboard_data(db=db, 
                                                       last_snapshot=last_snapshot,
                                                       current_snapshot=snapshot)

    logger.info(f"obtain future for processing: {sessions}")
    update_leaderboards = await process_sessions(sessions)

    # update leaderboards 
    for l in update_leaderboards:
        update = {
            "$set" : {
                "players" : l["players"],
                "models"  : l["models"],
                "targets" : l["targets"],
                "mean"    : l["mean"],
                "last_snapshot" : snapshot
            },
            "$inc" : { "step" : 1 }
        }
        await db.leaderboards.update_one({ "_id" : l["_id"] }, update)

    logger.info(f"Updated {len(update_leaderboards)} leaderboards.")

        
    # if len(update_leaderboards) != 0:
        # NOTE: possible best not to delete for now just keep
        #       invisible for user not to see them to count 
        #       personal stat(s).
        # time_range = {
        #         "$gte": last_snapshot,
        #         "$lt":  snapshot,
        # }
        # delete_query = {
        #     "$and": [
        #         {"game_id": {"$in": [l["_id"] for l in update_leaderboards]},
        #         "completed": True, 
        #         "completed_time": time_range },
        #         {"$or": [
        #             {"visible": False},
        #             {"$expr": {"$lt": [{"$size": "$history"}, 2]}}
        #         ]}
        #     ]
        # }

        # # delete all the session user have discarded or are garbage
        # result = await db.sessions.delete_many(delete_query)
        # logger.info(f"Deleted {result.deleted_count} sessions from this snapshot.")
    
    # always update leaderboard even if there are no sessions 
    snapshots = await db.leaderboards.update_many({}, { "$set" : { "last_snapshot" : snapshot } })
    
    logger.info(f"Updated {snapshots.modified_count} leaderboards snapshot")


async def process_sessions(sessions : AsyncGenerator[tuple[List[Any], List[Any], List[Any], Any, Any] | None, Any]):
    """ take the AsyncGenerator and process there sessions within a fixed snapshot """
    updated_leaderboards = []
    async for game in sessions:
        if game is None:
            continue
            
        target, users, models, game, leaderboard = game
        step = 0.1


        user_elo   = { l["id"] : [l["elo"], 0] for l in leaderboard["players"] }
        model_elo  = { l["id"] : [l["elo"], 0] for l in leaderboard["models"] }
        target_elo = { l["id"] : [l["elo"], 0] for l in leaderboard["targets"] }

        game_sessions = game["sessions"]

        utoi = {user  : i for i, user in enumerate(users)}
        mtoi = {model : i for i, model in enumerate(models)}
        ttoi = {target: i for i, target in enumerate(target)}
        
        
        # user index
        user_session_list   : List[int]  = list(map(lambda s : utoi[s["user_id"]], game_sessions))
        model_session_list  : List[int]  = list(map(lambda s : mtoi[s["model"]], game_sessions))
        target_session_list : List[int]  = list(map(lambda s : ttoi[s["target"]], game_sessions))
        
        # one hot vector of users, models, and targets
        u_m = convert_list_to_one_hot_vector(*user_session_list)
        m_m = convert_list_to_one_hot_vector(*model_session_list)
        t_m = convert_list_to_one_hot_vector(*target_session_list)

        # outcomes of the game sessions
        Y =  list(map(lambda s : 1 if s["outcome"] == "win" else 0, game_sessions))

        # concatenated vector user, model, target
        m = np.concat([u_m, m_m, t_m], axis=1)
        n, c = m.shape
        beta = np.zeros((c))

        # NOTE: This scales poorly, the more people who play games 
        #       the larger the vector becomes so the space complexity
        #       can get out of hand as where converting these users into 
        #       one hot vector.
   
        # train linear regression on sessions 
        for i in range(n):
            x = m[i, :]           # vector of sessions
            y = Y[i]              # target
            h = sigmoid(x @ beta)
            grad = x * ( h - y )  # grad of the BCELoss

            beta -= step * grad  # update new beta

        # join new elo with previous leaderboard elo's
        user_elo |= {
            users[j]: [
                float(beta[j]), 
                1 if users[j] in user_elo and float(beta[j]) > user_elo[users[j]][0]  # Increased
                else -1 if users[j] in user_elo and float(beta[j]) < user_elo[users[j]][0]  # Decreased
                else 0  # New entry or no change
            ] 
            for j in range(len(users))
        }
        model_elo |= {
            models[j]: [
                -float(beta[len(users) + j]),
                1 if models[j] in model_elo and -float(beta[len(users) + j]) > model_elo[models[j]][0]  # Increased
                else -1 if models[j] in model_elo and -float(beta[len(users) + j]) < model_elo[models[j]][0]  # Decreased
                else 0  # New entry or no change
            ]
            for j in range(len(models))
        }

        target_elo |= {
            target[j]: [
                -float(beta[len(users) + len(models) + j]),
                1 if target[j] in target_elo and -float(beta[len(users) + len(models) + j]) > target_elo[target[j]][0]  # Increased
                else -1 if target[j] in target_elo and -float(beta[len(users) + len(models) + j]) < target_elo[target[j]][0]  # Decreased
                else 0  # New entry or no change
            ]
            for j in range(len(target))
        }


        # FIXME: this is static, we can only take P(Y = 1 | X^{model}, X^{target}, X^{player}).
        # item[0]    user id
        # item[1][0] elo score
        # 
        leaderboard["players"] = list(map(lambda item: dict(id=item[0], elo=item[1][0], delta=item[1][1]), 
                                     user_elo.items()))
        leaderboard["models"]  = list(map(lambda item: dict(id=item[0], elo=item[1][0], delta=item[1][1]), 
                                     model_elo.items()))
        leaderboard["targets"]  = list(map(lambda item: dict(id=item[0], elo=item[1][0], delta=item[1][1]), 
                                     target_elo.items()))

        
        leaderboard["mean"] = {
            "players" : float(np.array(list(map(lambda x : x[0], user_elo.values()))).mean()),
            "models"  : float(np.array(list(map(lambda x : x[0], model_elo.values()))).mean()),
            "targets" : float(np.array(list(map(lambda x : x[0], target_elo.values()))).mean()), 
        }

        updated_leaderboards.append(leaderboard)
    
    return updated_leaderboards