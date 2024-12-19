import { API_CONFIG } from "@/lib/config/api";
import { createStream } from "@/lib/sse";
import { Game, GameErrorResponse } from "@/types/game";

// Cache duration in seconds
export const CACHE_DURATION = 5 * 60; // 5 minutes
export const GAMES_CACHE = new Map<string, { data: Game[] | Game; timestamp: number }>();

export async function getGameStream(skip: number = 0, include: number = 0): Promise<ReadableStream<any> | Game[]> {
  const cacheKey = `games-${skip}-${include}`;
  const now = Date.now();
  const cached = GAMES_CACHE.get(cacheKey);
  // Return cached data if it's still valid
  if (cached && (now - cached.timestamp) / 1000 < CACHE_DURATION) {
    // console.log("return cache games")
    return cached.data as Game[];
  }
  
  const response = await fetch(`${process.env.FRONTEND_HOST}${API_CONFIG.ENDPOINTS.GAME}stream?s=${skip}`);
  
  const stream = createStream(response, {
    eventTypes: ["message", "error"],
  });
  return stream as ReadableStream<any>;
}

export async function getGames(skip: number = 0, include: number = 0): Promise<Game[] | GameErrorResponse> {
  const cacheKey = `games-${skip}-${include}`;
  const now = Date.now();
  const cached = GAMES_CACHE.get(cacheKey);

  // Return cached data if it's still valid
  if (cached && (now - cached.timestamp) / 1000 < CACHE_DURATION) {
    return cached.data as Game[];
  }
  try {
    const response = await fetch(
      `${API_CONFIG.ENDPOINTS.GAME}?s=${skip}`,
      {
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        next: {
          revalidate: CACHE_DURATION, // Next.js 13+ cache configuration
        },
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      return {
        status: response.status,
        message: "Failed to retrieve games",
        details: errorData.message || "An unknown error occurred",
      } as GameErrorResponse;
    }

    const data = await response.json();
    
    // Update cache
    GAMES_CACHE.set(cacheKey, {
      data,
      timestamp: now,
    });

    return data;
  } catch (error: any) {
    console.error("Error checking games:", error);
    return {
      status: error?.response?.status,
      message: "Failed to retrieve games",
      details: error?.message || "An unknown error occurred",
    } as GameErrorResponse;
  }
}

export async function getGamesFromId(id: string): Promise<Game | GameErrorResponse | Game[]> {
  const cacheKey = `game-${id}`;
  const now = Date.now();
  const cached = GAMES_CACHE.get(cacheKey);

  // Return cached data if it's still valid
  if (cached && (now - cached.timestamp) / 1000 < CACHE_DURATION) {
    return cached.data as Game;
  }

  try {
    const response = await fetch(
      `${process.env.FRONTEND_HOST}${API_CONFIG.ENDPOINTS.GAME}/${id}`,
      {
        headers: {
          "Content-Type": "application/json",
          "Accept": "application/json",
        },
        next: {
          revalidate: CACHE_DURATION, // Next.js 13+ cache configuration
        },
      }
    );

    if (!response.ok) {
      const errorData = await response.json();
      return {
        status: response.status,
        message: "Failed to retrieve game",
        details: errorData.message || "An unknown error occurred",
      } as GameErrorResponse;
    }

    const data = await response.json();
    
    // Update cache
    GAMES_CACHE.set(cacheKey, {
      data,
      timestamp: now,
    });

    return data;
  } catch (error: any) {
    console.error("Error checking games:", error);
    throw new Error(error);
  }
}

// Optional: Cache cleanup function
export function clearGamesCache(): void {
  GAMES_CACHE.clear();
}

// Optional: Function to invalidate specific cache entries
export function invalidateGameCache(id?: string): void {
  if (id) {
    GAMES_CACHE.delete(`game-${id}`);
  } else {
    // Clear all entries that start with 'games-'
    for (const key of GAMES_CACHE.keys()) {
      if (key.startsWith('games-')) {
        GAMES_CACHE.delete(key);
      }
    }
  }
}