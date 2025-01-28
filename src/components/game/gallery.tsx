import React, { useState, useMemo } from "react";
import { Card, CardDescription, CardFooter, CardTitle } from "@/components/ui/card";
import { Game } from "@/types/game";
import Link from "next/link";
import { Input } from "../ui/input";
import { Search } from "lucide-react";
import Loading from "../loading";
import Image from "next/image";

function GameCardComponent({ game }: { game: Game }) {
  console.log(game)
  return (
    <Link href={`/games/${game._id}`} key={game._id} prefetch={true}>
      <Card className="overflow-hidden hover:shadow-lg transition-all duration-300 hover:scale-[1.02] bg-zinc-950 text-white">
        <div className="p-6 flex items-start space-x-4">
          <div className="relative h-16 w-16 flex-shrink-0">
            {game.image && (
              <Image
                src={game.image}
                alt={game.title}
                className="object-cover rounded-lg"
                fill={true}
              />
            )}
          </div>
          
          <div className="flex-1 min-w-0">
            <CardTitle className="text-xl font-bold text-white mb-2">
              {game.title}
            </CardTitle>
            
            <CardDescription className="text-zinc-400 line-clamp-2 text-sm mb-2">
              {game.description}
            </CardDescription>
            
            <div className="text-zinc-500 line-clamp-1 text-xs">
              By {game.author?.join(", ")}
            </div>
          </div>
        </div>
      </Card>
    </Link>
  );
}

const GameCardGallery = ({ games }: { games: Game[] }) => {
  const [searchQuery, setSearchQuery] = useState<string>("");
  const validGames = Array.isArray(games) ? games : [];
  
  const filteredGames = useMemo(() => {
    return validGames.filter((game) =>
      game.title.toLowerCase().includes(searchQuery.toLowerCase())
    );
  }, [searchQuery, validGames]);

  if (!validGames || validGames.length === 0) {
    return (
      <Loading
        fullScreen
        className="flex items-center justify-center w-full h-screen text-center relative"
      />
    );
  }

  return (
    <div className="container mx-auto p-4 space-y-6">
      <div className="relative mx-auto">
        <Search className="absolute left-2 top-2.5 h-4 w-4 text-muted-foreground" />
        <Input
          placeholder="Search games..."
          className="pl-8"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
        />
      </div>

      {filteredGames.length === 0 ? (
        <div className="text-center text-muted-foreground py-8">
          No games found matching "{searchQuery}"
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredGames.map((game) => (
            <GameCardComponent key={game._id} game={game} />
          ))}
        </div>
      )}
    </div>
  );
};

export default GameCardGallery;