import { cookies } from "next/headers";
import { HandleErrorResponse, handleResponse } from "./utils";


export interface LeaderboardModelItem 
    { id : string; delta : number; elo : number; image : string; name : string }

export interface LeaderboardPlayerItem
    { id : string; username : string; delta : number; elo : number }

export interface LeaderboardTargetItem
    { id : string; delta : number; elo : number }

export type LeaderboardItem = LeaderboardModelItem | LeaderboardPlayerItem | LeaderboardTargetItem;

interface Leaderboard {
    id? : string
    game_id? : string;
    mean? : { [key :string] : any };
    last_snapshot? : Date;
    models? :  LeaderboardModelItem[];
    players? : LeaderboardPlayerItem[];
    targets? : LeaderboardTargetItem[];

}

type LeaderboardResponse = Leaderboard & HandleErrorResponse;

export async function getGameLeaderboard(game_id : string, 
                                         skip    : number = 0, 
                                         limit   : number = 10) : Promise<LeaderboardResponse>  {
 
    const cookieStore = await cookies();
    const authToken = cookieStore.get("sessionKey")?.value;
  
    if (!authToken) {
      return { ok: false, status: 401, message: "Token does not exist" };
    }

    try {

    const response = await fetch(
        `${process.env.FRONTEND_HOST}/api/leaderboard/${game_id}?s=${skip}&l=${limit}`,
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${authToken}`,
            "Content-Type": "application/json",
            accept: "application/json",
          },
        }
      );
  
      if (!response.ok || response.status == 204) {
        return {
          ok: false,
          status: response.status,
          message: `HTTP error! status: ${response.status}`,
        };
      }

      const data = await handleResponse<LeaderboardResponse>(response);
      console.log(data)
      return {ok : true, ...data};
    } catch(error) {
        return error as HandleErrorResponse;
    }

    
}