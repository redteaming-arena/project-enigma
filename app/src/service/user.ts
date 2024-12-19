"use server";
import { cookies } from "next/headers";
import { usernameSchema } from "@/lib/definition";
import { HandleErrorResponse, handleResponse } from "./utils";
import { API_CONFIG } from "@/lib/config/api";

export interface GameStats {
  games_played: number;
  sessions: Array<{
    completed_time: string;
    outcome: "win" | "loss" | "forfeit";
    provider: string;
  }>;
}

export type GameResponse = GameStats & HandleErrorResponse;

export async function createUsername(username: string): Promise<{
  ok: boolean;
  status?: number;
  message?: string;
}> {
  // Validate the username
  const result = usernameSchema.safeParse(username);
  if (!result.success) {
    return {
      ok: false,
      status: 400,
      message: result.error.errors[0].message,
    };
  }

  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("sessionKey")?.value;

    if (!token) {
      return {
        ok: false,
        status: 401,
        message: "Authentication required",
      };
    }

    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/update-username?username=${username}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      }
    );

    const res: { status: number; message: string; data: any } =
      await handleResponse(response);

    return {
      ok: true,
      status: res.status,
    };
  } catch (error: any) {
    console.error("Username Update Error:", error);
    return {
      ok: false,
      status: error.status || 500,
      message: error.message || "Failed to update username",
    };
  }
}

export async function getUser(): Promise<{
  ok: boolean;
  status?: number;
  message?: string;
  id?: string;
  username?: string;
  image?: string;
  history?: { title: string; _id: string }[];
  pinned?: { _id: string }[];
}> {
  const cookieStore = await cookies();
  const token = cookieStore.get("sessionKey")?.value;

  if (!token) {
    return {
      ok: false,
      status: 401,
      message: "Authentication required",
    };
  }

  try {
    const response = await fetch(`${process.env.FRONTEND_HOST}/api/user`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
    });

    const res: { status: number; message: string; data: any } =
      await handleResponse(response);
    return {
      ok: true,
      ...res,
    };
  } catch (error) {
    // Handle specific error types
    if (error instanceof TypeError) {
      // Network errors, CORS issues, etc.
      return {
        ok: false,
        status: 503,
        message: "Service unavailable. Please check your connection.",
      };
    }

    if (error instanceof Error) {
      // Generic error handling
      return {
        ok: false,
        status: 500,
        message: error.message || "An unexpected error occurred",
      };
    }

    // Fallback for unknown error types
    return {
      ok: false,
      status: 500,
      message: "An unknown error occurred",
    };
  }
}

export async function pinGame(gameId: string): Promise<{
  ok: boolean;
  status?: number;
  message?: string;
}> {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("sessionKey")?.value;
    if (!token) {
      return {
        ok: false,
        status: 401,
        message: "Authentication required",
      };
    }

    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/pinned/${gameId}`,
      {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      }
    );

    const res: { status: number; message: string; data: any } =
      await handleResponse(response);
    return {
      ok: true,
      status: res.status,
      message: res.message,
    };
  } catch (error: any) {
    console.error("Pin Game Error:", error);
    return {
      ok: false,
      status: error.status || 500,
      message: error.message || "Failed to pin game",
    };
  }
}

export async function unpinGame(gameId: string): Promise<{
  ok: boolean;
  status?: number;
  message?: string;
}> {
  try {
    const cookieStore = await cookies();
    const token = cookieStore.get("sessionKey")?.value;
    if (!token) {
      return {
        ok: false,
        status: 401,
        message: "Authentication required",
      };
    }

    const response = await fetch(
      `${process.env.FRONTEND_HOST}/api/pinned/${gameId}`,
      {
        method: "DELETE",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
      }
    );

    const res: { status: number; message: string; data: any } =
      await handleResponse(response);
    return {
      ok: true,
      status: res.status,
      message: res.message,
    };
  } catch (error: any) {
    console.error("Unpin Game Error:", error);
    return {
      ok: false,
      status: error.status || 500,
      message: error.message || "Failed to unpin game",
    };
  }
}

export async function logout(): Promise<{ ok: boolean; error?: string }> {
  const cookieStore = await cookies();
  const authToken = cookieStore.get("sessionKey");

  if (!authToken) {
    return { ok: false, error: "Token does not exist" }; // Indicate missing token
  }

  const url = new URL(
    API_CONFIG.ENDPOINTS.USER.LOGOUT,
    process.env.FRONTEND_HOST
  );

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        Authorization: `Bearer ${authToken.value}`,
        "Content-Type": "application/json",
      },
    });

    if (!response.ok) {
      // Custom error message based on response status
      return {
        ok: false,
        error:
          response.status === 401
            ? "Unauthorized: Invalid token"
            : `Error: ${response.statusText} (${response.status})`,
      };
    }

    await handleResponse(response);
    cookieStore.delete("sessionKey"); // Remove the token from cookies

    return { ok: true }; // Indicate successful logout
  } catch (error) {
    if (error instanceof TypeError) {
      console.error("Network error or invalid request:", error.message);
      return { ok: false, error: "Network error or invalid request" };
    } else if (error instanceof Error) {
      console.error("Logout error:", error.message);
      return { ok: false, error: error.message };
    }

    // Catch-all for other types of errors
    console.error("Unexpected error:", error);
    return { ok: false, error: "An unexpected error occurred" };
  }
}

export async function userStats() {
  const cookieStore = await cookies();
  const token = cookieStore.get("sessionKey")?.value;

  if (!token) {
    return {
      ok: false,
      status: 401,
      message: "Authentication required",
    };
  }

  try {
    const response = await fetch(`${process.env.FRONTEND_HOST}/api/stats`, {
      method: "GET",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token}`,
      },
      next : {
        revalidate : 60
      }
    });

    const res: GameStats = await handleResponse(response);
    return {
      ok: true,
      ...res,
    };
  } catch (error: any) {
    return {
      ok: false,
      status: error.status || 500,
      message: error.message || "Failed to unpin game",
    };
  }
}
