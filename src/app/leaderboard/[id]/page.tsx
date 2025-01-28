// Server Component (app/games/[id]/leaderboard/page.tsx)
"use server";
import { Suspense } from "react";
import { LeaderboardSkeleton } from "@/components/skeleton/leaderboard";
import { LeaderboardComponent } from "@/components/leaderboard";
import { getGameLeaderboard } from "@/service/leaderboard";
import { redirect } from "next/navigation";
import { cn } from "@/lib/utils";

interface LeaderboardProps {
  params: Promise<{
    id: string;
  }>;
}

export default async function Leaderboard({ params }: LeaderboardProps) {
  const { id } = await params;
  const response = await getGameLeaderboard(id);
  
  if (!response.ok) {
    redirect(`/games/${id}`);
  }

  return (
    <div className="space-y-5 flex flex-col h-full mt-4">
      <Suspense
        fallback={
          <div className="flex w-full space-x-5">
            <LeaderboardSkeleton />
            <LeaderboardSkeleton />
          </div>
        }
      >
        <div className={cn("grid grid-cols-1 gap-6 p-6", (response.targets?.length ?? 0) !== 0 ? "lg:grid-cols-3" : "lg:grid-col-2" )}>
          {/* Players Leaderboard */}
          <div className="w-full">
            <LeaderboardComponent
              title="Players"
              data={response.players ?? []}
              variant="players"
            />
          </div>

          {/* Models Leaderboard */}
          <div className="w-full">
            <LeaderboardComponent
              title="Models"
              data={response.models ?? []}
              variant="models"
            />
          </div>

          {/* Targets Leaderboard */}
          {(response.targets?.length ?? 0) > 0 && (
            <div className="w-full">
              <LeaderboardComponent
                title="Targets"
                data={response.targets ?? []}
                variant="targets"
              />
            </div>
          )}
        </div>
      </Suspense>
    </div>
  );
}