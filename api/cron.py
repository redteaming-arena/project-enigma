import numpy as np

from datetime import datetime, timedelta, timezone, UTC

from typing import Dict, Any, List

from bson import ObjectId

from api import crud
from api.deps import get_database, Database
from api.function import sigmoid
from api.models import GameSession, Leaderboard, Score
from api.utils import repeat_every, logger, convert_list_to_one_hot_vector

from bson import ObjectId


DEFAULT_CRON_JOB_CALL = 60


def update_leaderboard(
    leaderboard: Leaderboard,
    sessions: List[GameSession],
    include_targets: bool = False,
    includes_prior: bool = False,
):
    # Create dictionaries for players, models, and optionally targets
    utos: Dict[ObjectId, Score] = {player.id: player for player in leaderboard.players}
    mtos: Dict[ObjectId, Score] = {model.id: model for model in leaderboard.models}
    ttos: Dict[str, Score] = (
        {target.id: target for target in leaderboard.targets} if include_targets else {}
    )

    for game in sessions:
        # Update user scores
        if not game.user_id in utos:
            utos[game.user_id] = Score(_id=game.user_id)
        # Update model scores
        if not game.agent_id in mtos:
            mtos[game.agent_id] = Score(_id=game.agent_id)
        # Update target scores (if applicable)
        if include_targets and not game.target in ttos:
            ttos[game.target] = Score(_id=game.target)

    # Create index mappings
    itou: Dict[ObjectId, int] = {index: user for index, user in enumerate(utos.keys())}
    itom: Dict[ObjectId, int] = {
        index: model for index, model in enumerate(mtos.keys())
    }
    itot: Dict[str, int] = (
        {index: target for index, target in enumerate(ttos.keys())}
        if include_targets
        else {}
    )

    utoi: Dict[int, ObjectId] = {user: index for index, user in itou.items()}
    mtoi: Dict[int, ObjectId] = {model: index for index, model in itom.items()}
    ttoi: Dict[int, str] = (
        {target: index for index, target in itot.items()} if include_targets else {}
    )

    # Create session lists
    user_session_list = list(map(lambda s: utoi[s.user_id], sessions))
    model_session_list = list(map(lambda s: mtoi[s.agent_id], sessions))
    target_session_list = (
        list(map(lambda s: ttoi[s.target], sessions)) if include_targets else []
    )

    step = 0.01

    # Convert to one-hot vectors
    u_m = convert_list_to_one_hot_vector(*user_session_list, min_size=len(utoi))
    m_m = convert_list_to_one_hot_vector(*model_session_list, min_size=len(mtoi))
    t_m = convert_list_to_one_hot_vector(*target_session_list, min_size=len(ttoi))
    # Get prior ELO scores if includes_prior is True
    if includes_prior:
        user_elos = np.array([utos[itou[i]].elo for i in range(len(utos))])
        model_elos = np.array([mtos[itom[i]].elo for i in range(len(mtos))])
        target_elos = (
            np.array([ttos[itot[i]].elo for i in range(len(ttos))])
            if include_targets
            else None
        )

        # Initialize prior weights based on ELO scores
        prior_weights = []
        prior_weights.extend(user_elos)
        prior_weights.extend(model_elos)

        if target_elos is not None:
            prior_weights.extend(target_elos)
        prior_weights = np.array(prior_weights)

    # Outcomes (labels)
    Y = np.array([1 if s.outcome == "win" else 0 for s in sessions])

    # Concatenate feature vectors
    cat = [vec for vec in [u_m, m_m, t_m] if vec is not None]
    m = np.concatenate(cat, axis=1)
    n, c = m.shape

    beta = np.zeros(c)

    if includes_prior:
        beta += prior_weights

    print("beta", beta)

    # Gradient descent for binary logistic regression
    for i in range(n):
        x = m[i, :]  # Feature vector
        y = Y[i]  # Target value
        h = sigmoid(np.dot(x, beta))  # Prediction using sigmoid
        grad = x * (h - y)  # Gradient of binary cross-entropy loss

        # # Add L2 regularization if using prior
        # if includes_prior:
        #     grad += 0.01 * (beta - prior_weights)  # Regularization term

        beta -= step * grad  # Update weights

    # Update scores based on beta vector
    num_users = u_m.shape[1]
    num_models = m_m.shape[1]

    # Update user scores
    for i in range(num_users):
        user_id = itou[i]
        utos[user_id].update(float(beta[i]))

    # Update model scores
    for i in range(num_models):
        model_id = itom[i]
        mtos[model_id].update(-float(beta[i + num_users]))

    # Update target scores if included
    if include_targets:
        num_targets = t_m.shape[1]
        for i in range(num_targets):
            target_id = itot[i]
            ttos[target_id].update(-float(beta[i + num_users + num_models]))

    leaderboard.players = list(utos.values())
    leaderboard.models = list(mtos.values())
    leaderboard.targets = list(ttos.values()) if include_targets else None
    return leaderboard


@repeat_every(seconds=DEFAULT_CRON_JOB_CALL, logger=logger)
async def compute_leaderboard():
    # take capacity of collected session(s)
    snapshot = datetime.now(UTC)

    # # get a AsyncIOMotorDatabase client to access mongoDB
    # # collections.
    db: Database = await get_database()

    # # NOTE: I don't like this solution
    # collect leaderboards
    leaderboards = list(
        map(lambda x: Leaderboard(**x), await db.leaderboards.find({}).to_list())
    )

    if leaderboards is None or len(leaderboards) == 0:
        logger.error("Session not recorded due to not finding leaderboard")
        return

    # they should all have the same snapshots so we can take the first snapshot
    prev_snapshot = leaderboards[0].last_snapshot
    print(leaderboards[0].last_snapshot)
    # DEBUG: random value form month ago to collect all the data till today
    # prev_snapshot = datetime(year=2025, day=25, month=12)

    for leaderboard in leaderboards:
        # obtain game session metadata
        sessions = await db.sessions.find(
            {
                "game_id": leaderboard.game_id,
                "completed": True,
                "completed_time": {"$gt": prev_snapshot, "$lte": snapshot},
            }
        ).to_list()

        sessions: List[GameSession] = list(map(lambda x: GameSession(**x), sessions))

        if len(sessions) == 0:
            logger.info(f"No session to game {leaderboard.game_id}")

        else:
            # check if session contain target
            strategy = sessions[0].target is not None
            leaderboard = update_leaderboard(
                leaderboard, sessions, include_targets=strategy, includes_prior=True
            )

        # save the new snapshot, increment the step
        # break
        leaderboard.last_snapshot = snapshot
        await db.leaderboards.update_one(
            {"_id": leaderboard.id}, {"$set": leaderboard.serde()}
        )
