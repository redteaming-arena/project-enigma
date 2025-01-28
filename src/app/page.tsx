"use client";
import { useState, useEffect } from "react";
import { getGames } from "@/service/game";
import { GameErrorResponse, type Game } from "@/types/game";
import GameCardGallery from "@/components/game/gallery";
import { AppSidebar, SideBarCloseButton } from "@/components/sidebar";
export default function Game() {
  const [games, setGames] = useState<Game[] | null>(null);

  useEffect(() => {
    async function fetchGames() {
      try {
        const data: Game[] | GameErrorResponse = await getGames();
        setGames(data as Game[]);
      } catch (error) {
        console.error("Failed to fetch games:", error);
      }
    }
    fetchGames();
  }, []);

  return (
    <>
      <AppSidebar />
      <SideBarCloseButton />
      <main className="md:mt-20 mt-10 flex-1">
        <div className="mx-auto max-w-3xl px-4">
          <div className="mb-6">
            <div className="my-2 text-center text-3xl font-bold md:my-4 md:text-5xl">
              RedArena Games
            </div>
            <div className="mx-auto w-full text-center text-sm text-token-text-secondary md:text-lg md:leading-tight">
              Test your Jailbreaking skills on current games that combine unique
              blue team system prompt.
            </div>
          </div>
          <GameCardGallery games={games ?? []} />
        </div>
      </main>
    </>
  );
}
