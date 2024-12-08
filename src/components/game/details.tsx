"use client";
import React, { useEffect, useState } from "react";
import { Card, CardContent } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { Button } from "../ui/button";
import { ChartColumn, Pin, PlayCircle } from "lucide-react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { redirect, useRouter } from "next/navigation";
import type { Game } from "@/types/game";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "../ui/tooltip";
import { useUser } from "@/context/user";
import { createChat } from "@/service/session";
import { useNotification } from "../toast";
import Link from "next/link";

interface GameDetailsProps {
  game: Game;
  isAuthenticated?: boolean; // Add this prop to check auth status
}

export function GameDetails({ game }: GameDetailsProps) {
  const { state: user, handlePin, handleUnpin, isLoading } = useUser();
  const notification = useNotification();
  const [pinned, setPinned] = useState(
    user.pinned.filter((e: { id: string }) => e.id == game.id).length != 0
  );
  const [showAuthDialog, setShowAuthDialog] = useState(false);
  const router = useRouter();

  useEffect(() => {
    setPinned(
      user.pinned.filter((e: { id: string }) => e.id == game.id).length != 0
    );
  }, [user]);

  const handlePlayNow = async () => {
    const chat = await createChat(game.id);
    if (!chat.ok) {
      notification.showWarning(chat.error ?? "Could not create new Chat ");
      return;
    }
    notification.showSuccess("Game session created");
    router.push(`/c/${chat.session_id}`);
  };

  const handleLogin = () => {
    router.push("/login"); // Adjust this to your login route
  };

  return (
    <>
      <div className="flex space-x-2">
        <TooltipProvider>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={handlePlayNow}
                variant="default"
                className="gap-2 items-center"
              >
                Play Game
                <PlayCircle />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Play Now</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Link href={`/leaderboard/${game.id}`} prefetch={true} >
                <Button
                  size="icon"
                  variant="ghost"
                  className="gap-2 items-center"
                >
                  <ChartColumn />
                </Button>
              </Link>
            </TooltipTrigger>
            <TooltipContent>Leaderboard</TooltipContent>
          </Tooltip>
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                onClick={async () => {
                  if (!pinned) {
                    await handlePin(game.id, game.image ?? "", game.title);
                  } else {
                    await handleUnpin(game.id);
                  }
                }}
                size="icon"
                variant="ghost"
                className="gap-2 items-center"
                disabled={isLoading}
              >
                <Pin fill={pinned ? "white" : ""} />
              </Button>
            </TooltipTrigger>
            <TooltipContent>Pin to sidebar</TooltipContent>
          </Tooltip>
        </TooltipProvider>
      </div>
      <Dialog open={showAuthDialog} onOpenChange={setShowAuthDialog}>
        <DialogContent className="sm:max-w-[425px]">
          <DialogHeader>
            <DialogTitle>Login Required</DialogTitle>
            <DialogDescription>
              You need to be logged in to play {game?.title}. Please login or go
              back.
            </DialogDescription>
          </DialogHeader>
          <DialogFooter className="flex flex-row gap-1 w-full">
            <div className="w-full">
              <Button
                variant="outline"
                onClick={() => setShowAuthDialog(false)}
                className="w-full"
              >
                Go Back
              </Button>
            </div>
            <div className="w-full">
              <Button onClick={handleLogin} className="w-full">
                Login
              </Button>
            </div>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Card>
        <CardContent className="pt-6">
          <div className="space-y-6">
            <div>
              <h2 className="text-xl font-semibold mb-2">Description</h2>
              <p className="text-gray-300 leading-relaxed">
                {game.description}
              </p>
            </div>
          </div>
        </CardContent>
      </Card>
    </>
  );
}
