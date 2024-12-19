"use server";
import { cookies } from "next/headers";
import { HandleErrorResponse, handleResponse } from "./utils";
import { Message } from "ai/react";

const CACHE_DURATION = 60 * 60 * 24; // 24 hours
const HISTORY_CACHE = new Map<
  string,
  { data: GameSessionPublic; timestamp: number }
>();

export interface MessageType {
  data?: any;
}

export type MessageTypeResponse = MessageType & HandleErrorResponse;

export interface GameSessionReadOnly {
  username?: string;
  session_id?: ObjectId;
  title?: string;
  outcome?: "win" | "loss" | undefined;
  duration?: number;
  last_message?: string;
  history?: any[];
  model?: Model | null;
  description?: string;
}

export type GameSessionReadOnlyResponse =
  | GameSessionReadOnly & HandleErrorResponse;

export interface GameSessionPublic {
  id?: string | ObjectId;
  user_id?: ObjectId | null;
  game_id?: ObjectId | null;
  judge_id?: ObjectId | null;
  agent_id?: ObjectId | null;
  user?: UserPublic | null;
  model?: Model | null;
  judge?: Judge | null;
  history?: any[];
  title?: string | null;
  description?: string | null;
  create_time?: Date | null;
  completed_time?: Date | null;
  start_time?: Date | null;
  outcome?: "win" | "loss" | "forfeit" | null;
  metadata?: GameSessionMetadata | null;
  completed?: boolean;
  shared?: string | null;
}

export type GameSessionPublicResponse = GameSessionPublic & HandleErrorResponse;

type ObjectId = string; // Replace with an actual ObjectId type if available
type UserPublic = any; // Define UserPublic type appropriately
type Model = { image?: string; name?: string }; // Define Model type appropriately
type Judge = any; // Define Judge type appropriately
type GameSessionMetadata = {
  model_config: any;
  game_rules: { timed?: boolean; time_limit?: number; deterministic?: boolean };
}; // Define GameSessionMetadata type appropriately

export interface GameSessionRecentItem {
  session_id?: string;
  title?: string;
  outcome?: string;
  duration?: number;
  last_message?: number;
}

export type GameSessionRecentItemResponse = {
  content?: GameSessionRecentItem[];
} & HandleErrorResponse;

interface ChatCreation {
  data?: any;
  session_id?: string;
}

export type ChatCreationResponse = ChatCreation & HandleErrorResponse;

export async function getSession(
  session_id: string
): Promise<GameSessionPublicResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, status: 401, message: "Token does not exist" };
  }

  try {
    // const cache = HISTORY_CACHE.get(key);
    // if (cache && (now - cache.timestamp) / 1000 < CACHE_DURATION) {
    //   console.log("cache")
    //   return { ok: true, ...cache.data };
    // }

    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation`,
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

    const data = await handleResponse<GameSessionPublic>(response);

    // console.log("session", data);
    // if (data.outcome === "win" || data.outcome === "loss") {
    //   HISTORY_CACHE.set(key, {
    //     data,
    //     timestamp: Date.now(),
    //   });
    // }

    return { ok: true, ...data };
  } catch (error) {
    console.error("Error in getting session:", error);
    return error as HandleErrorResponse;
  }
}

export async function startGame(
  session_id: string,
  user_id: string
): Promise<MessageTypeResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, error: "Session token does not exist" };
  }

  try {
    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation/${user_id}/start`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }
    const data = await response.json();
    return {
      ...data,
      ok: true,
    };
  } catch (err) {
    return { ok: false, error: "An unexpected error occurred" };
  }
}

// Create Chat Session
export async function createChat(
  game_id: string
): Promise<ChatCreationResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, error: "Session token does not exist" };
  }

  try {
    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/create-chat?game_id=${game_id}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    if (!response.ok) {
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }

    const result = await handleResponse<ChatCreation>(response);
    return { ok: true, ...result };
  } catch (err) {
    console.error("Error in createChat:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

// Create Title
export async function createTitle(
  session_id: string,
  user_id: string,
  generate: boolean = true
): Promise<MessageTypeResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, error: "Token does not exist" };
  }

  try {
    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation/${user_id}/title`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json", // Ensure Content-Type is set to application/json
        },
        body: JSON.stringify({
          generate,
        }),
      }
    );

    // Check if the response is OK, then handle the response
    if (!response.ok) {
      return {
        ok: false,
        status: response.status,
        error: `HTTP error! status: ${response.status}`,
      };
    }

    // Parse the response body and return the result
    const data = await response.json();
    return {
      ...data,
      ok: true,
    };
  } catch (err) {
    console.error("Error in createTitle:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

// End Game
export async function concludeSessionGame(
  session_id: string,
  user_id: string,
  outcome: "loss",
  message: Message[]
): Promise<MessageTypeResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, error: "Token does not exist" };
  }

  try {
    // fetch game response
    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation/${user_id}/conclude?outcome=${outcome}`,
      {
        method: "POST",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
        body: JSON.stringify(message),
      }
    );

    const data = await handleResponse<MessageType>(response);
    return {
      ...data,
      ok: true,
    };
  } catch (err) {
    return { ok: false, error: "An unexpected error occurred" };
  }
}

// Get History
export async function getRecents(
  skip: number,
  limit?: number
): Promise<GameSessionRecentItemResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, error: "Token does not exist" };
  }

  try {
    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/history?s=${skip}${
        limit ? `&l=${limit}` : ""
      }`,
      {
        method: "GET",
        headers: {
          Authorization: `Bearer ${authToken}`,
          "Content-Type": "application/json",
        },
      }
    );

    const data = {
      ok: true,
      content: await handleResponse<GameSessionRecentItem[]>(response),
    };

    return data;
  } catch (err) {
    console.error("Error in getRecents:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

// Get Session History
export async function getSessionHistory(
  session_id: string,
  user_id: string | null = null
): Promise<GameSessionReadOnlyResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, status: 401, error: "Token does not exist" };
  }

  const uri = `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation/${user_id}/history`;
  try {
    const response = await fetch(uri, {
      method: "GET",
      headers: {
        Authorization: `Bearer ${authToken}`,
        "Content-Type": "application/json",
      },
    });

    const data = {
      ok: true,
      readonly: true,
      ...(await handleResponse<GameSessionReadOnlyResponse>(response)),
    };

    return data;
  } catch (err) {
    console.error("Error in getSessionHistory:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

export async function getSharedSession(
  shared_id: string
): Promise<GameSessionReadOnlyResponse> {
  // obtain cache if exist

  const uri = `${process.env.FRONTEND_HOST}/api/chat_conversation/${shared_id}`;
  try {
    const response = await fetch(uri, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
      },
      // next: {
      //   revalidate: CACHE_DURATION, // Next.js 13+ cache configuration
      // },
    });

    if (!response.ok) {
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }

    const res = await handleResponse<GameSessionReadOnlyResponse>(response);

    const data = {
      ok: true,
      ...res,
    };

    return data;
  } catch (err) {
    console.error("Error in getSessionHistory:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

export async function postSharedSession(
  session_id: string,
  user_id: string
): Promise<MessageTypeResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, status: 401, error: "Token does not exist" };
  }

  const uri = `${process.env.FRONTEND_HOST}/api/${session_id}/chat_conversation/${user_id}/share`;
  try {
    const response = await fetch(uri, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }

    const res = await handleResponse<MessageType>(response);

    return { ok: true, ...res };
  } catch (err) {
    console.error("Error in getSessionHistory:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

export async function deleteSession(
  user_id: string,
  session_ids: string[]
): Promise<MessageTypeResponse> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey")?.value;

  if (!authToken) {
    return { ok: false, status: 401, error: "Token does not exist" };
  }

  const uri = `${process.env.FRONTEND_HOST}/api/${user_id}/chat_conversation`;
  try {
    const response = await fetch(uri, {
      method: "DELETE",
      headers: {
        Authorization: `Bearer ${authToken}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(session_ids),
    });

    if (!response.ok) {
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }

    const res = await handleResponse<MessageType>(response);

    return { ok: true, ...res };
  } catch (err) {
    console.error("Error in getSessionHistory:", err);
    return { ok: false, error: "An unexpected error occurred" };
  }
}
