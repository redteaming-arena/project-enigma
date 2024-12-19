"use client";
import { createUsername, getUser } from "@/service/user";
import React, { useEffect, useState } from "react";
import { useRouter } from "next/navigation";
import {
  Card,
  CardContent,
  CardFooter,
  CardHeader,
} from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { DeleteSessionKey } from "@/service/auth";
import { useNotification } from "@/components/toast";
import { useUser } from "@/context/user";

export default function UsernamePage() {
  

  const [username, setUsername] = useState<string>("");
  const [error, setError] = useState<string>("");
  const [isLoading, setIsLoading] = useState(false);
  const { dispatch, setIsLoading : setIsWaitingForUsername } = useUser();
  const router = useRouter();
  const notification = useNotification();

  useEffect(() => {

    setIsWaitingForUsername(true)
    
  }, [setIsWaitingForUsername]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError("");
    setIsLoading(true);

    try {
      const result = await createUsername(username);

      if (result.ok) {
        const user = await getUser();
        dispatch({
          type: "SET_USER",
          payload: {
            id: user.id ?? null,
            username: user.username ?? null,
            history: user.history ?? [],
            pinned: user.pinned ?? [],
          },
        });
        notification.showSuccess(`Welcome to RedArena ${username}`);
        setIsWaitingForUsername(false)
        router.push("/");
      } else {
        setError(result.message || "Failed to update username");
      }
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred");
    } finally {
      setIsLoading(false);
      
    }
  }

  return (
    <div className="fixed min-h-screen w-full flex flex-col items-center justify-center p-4 z-20 bg-black">
      <Card className="w-full max-w-md bg-black border-zinc-800">
        <CardHeader>
          <h1 className="text-3xl font-medium text-white text-center">
            Welcome User
          </h1>
          <p className="text-gray-200 text-center">
            What would you like your username to be?
          </p>
        </CardHeader>

        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            <input
              type="text"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              placeholder="username"
              required
              disabled={isLoading}
              className="w-full bg-zinc-900 border-none text-white h-10 text-md rounded-md px-3 focus:outline-none focus:ring-2 focus:ring-white/20 disabled:opacity-50"
            />
            {error && <p className="text-red-500 text-sm">{error}</p>}
            <Button
              variant="default"
              disabled={isLoading || !username.trim()}
              className="w-full bg-white text-black py-2 rounded-md text-md font-medium hover:bg-gray-100 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Updating..." : "Continue"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="w-full text-zinc-400 hover:text-white"
              onClick={async () => {
                await DeleteSessionKey();
                if (window.history?.length && window.history.length > 1) {
                  router.back();
                } else {
                  router.push("/login");
                }
              }}
            >
              Go back to login
            </Button>
          </form>
        </CardContent>

        <CardFooter>
          <p className="text-gray-400 text-sm text-center w-full">
            By clicking continue, you agree to our{" "}
            <a href="#" className="text-gray-400 underline hover:text-white">
              Terms of Service
            </a>
            .
          </p>
        </CardFooter>
      </Card>
    </div>
  );
}
