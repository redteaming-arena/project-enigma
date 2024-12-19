import { getGamesFromId } from "@/service/game";
import { redirect } from "next/navigation";
import type { Game } from "@/types/game";
import { GameHeader } from "@/components/game/header";
import { GameDetails } from "@/components/game/details";
import { GameSettings } from "@/components/game/setting";
import {
  GameDetailsSkeleton,
  GameHeaderSkeleton,
  GameSettingsSkeleton,
} from "@/components/skeleton/game";
import { Metadata } from "next";

interface GamePageProps {
  params: Promise<{
    id: string;
  }>;
}

export async function generateMetadata({
  params,
}: GamePageProps): Promise<Metadata> {
  const { id } = await params;

  try {
    const response = await getGamesFromId(id);
    if ("error" in response || !response) {
      return {
        title: "Game Not Found - RedArena",
        description: "The requested game does not exist on RedArena.",
      };
    }

    const game = response as Game;

    return {
      metadataBase: new URL(
        "https://project-enigma-620119407459.us-central1.run.app/"
      ),
      title: `${game.title} - RedArena`,
      description: game.description,
      keywords: [
        "AI",
        "redteaming",
        "games",
        "RedArena",
        game.title,
        "artificial intelligence",
        "community-driven",
      ],
      openGraph: {
        type: "article",
        title: `${game.title} - RedArena`,
        description: game.description,
        url: `https://project-enigma-620119407459.us-central1.run.app/game/${id}`,
        images: [
          {
            url: game.image || "/og-image.png", // Provide a fallback image if none exists
            width: 796,
            height: 460,
            alt: `${game.title} - RedArena`,
          },
        ],
      },
      twitter: {
        card: "summary_large_image",
        title: `${game.title} - RedArena`,
        description: `Discover ${game.title} and join the RedArena community to explore more.`,
        images: [
          {
            url: game.image || "/og-image.png",
            alt: `${game.title} - RedArena`,
          },
        ],
      },
    };
  } catch (error) {
    console.error("Error generating metadata:", error);
    return {
      title: "Error - RedArena",
      description: "An error occurred while fetching the game details.",
    };
  }
}

export default async function GamePage({ params }: GamePageProps) {
  const { id } = await params;
  let game: Game | null = null;

  try {
    const response = await getGamesFromId(id);
    if ("error" in response) {
      console.error("API returned an error:", response.error);
      redirect("/");
      return;
    }
    game = response as Game;
  } catch (error) {
    console.error("Error fetching game data:", error);
    redirect("/");
    return;
  }

  if (!game) {
    return (
      <div className="max-w-4xl mx-auto p-6 space-y-6">
        <GameHeaderSkeleton />
        <GameDetailsSkeleton />
        <GameSettingsSkeleton />
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto p-6 space-y-6">
      <GameHeader game={game} />
      <GameDetails game={game} />
      <GameSettings game={game} />
    </div>
  );
}
