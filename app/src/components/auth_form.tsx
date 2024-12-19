"use client";

import React, { useEffect, useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Label } from "@/components/ui/label";
import { SignupFormSchema } from "@/lib/definition";
import { z } from "zod";
import { useRouter } from "next/navigation";
import { GoogleProvider } from "./oauth/google";
import { useNotification } from "./toast";
import { useUser } from "@/context/user";
import { getUser } from "@/service/user";

interface AuthFormsProps {
  onUsernameSubmit: (username: string) => Promise<any>;
  onSignIn: (username: string, password: string) => Promise<any>;
  onSignUp: (formData: FormData) => Promise<any>;
  error: string[];
}

export const AuthForms = ({
  onUsernameSubmit,
  onSignIn,
  onSignUp,
}: AuthFormsProps) => {
  const router = useRouter();
  const notification = useNotification();
  const { dispatch } = useUser();
  const [state, setState] = useState("username");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [username, setUsername] = useState("");
  const [error, setError] = useState<string[]>([]);

  const handleUsernameSubmit = async () => {
    if (!username) {
      return;
    }

    setIsLoading(true);
    try {
      const response = await onUsernameSubmit(username);
      setState(response.step);
      if (response.error){
        setError([response.error ?? "You must sign in via the provider"])
      } else {
        setUsername(response.username);
      }
      
      
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignIn = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!password) return;

    setIsLoading(true);
    try {
      const formData = new FormData();
      formData.append("username", username);
      formData.append("password", password);
      const response = await onSignIn(username, password);
      if (!response.ok) {
        if (typeof response.message === "string") {
          setError([response.message]);
        } else {
          setError(response.message.map((m: any) => m.msg));
        }
      } else {
        notification.showSuccess("Successfully Logged In");
        const user = await getUser();
        dispatch({
          type : "SET_USER",
          payload : {
            id : user.id ?? null, 
            username : user.username ?? null,
            history : user.history ?? [],
            pinned : user.pinned ?? []
          }
        })
        router.push("/");
      }
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignUp = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();

    // First check password match
    if (password !== confirmPassword) {
      setError(["Passwords do not match"]);
      return;
    }

    setIsLoading(true);
    try {
      // Validate the form data using Zod schema
      SignupFormSchema.parse({
        username,
        password,
      });

      // If validation passes, create and send form data
      const formData = new FormData();
      formData.append("username", username);
      formData.append("password", password);

      const response = await onSignUp(formData);

      // Handle server response
      if (!response.ok && response.message) {
        if (typeof response.message === "string") {
          setError([response.message]);
        } else {
          setError(response.message.map((m: any) => m.msg));
        }
        return;
      }

      notification.showSuccess("Successfully Signed Up");

      // sign in
      await handleSignIn(e);
    } catch (err) {
      // Handle Zod validation errors
      if (err instanceof z.ZodError) {
        setError(
          err.errors.map((error) => {
            return `• ${error.message}`;
          })
        );
      } else {
        // Handle unexpected errors
        setError(["An unexpected error occurred during sign up"]);
      }
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      <div className="space-y-4">
        {error.length > 0 && (
          <Alert
            variant="destructive"
            className="bg-red-900/20 border-red-900/50 text-red-400"
          >
            <AlertDescription>
              <ul>
                {error.map((err: string, index: number) => (
                  <li key={`error-${index}`}>{err}</li>
                ))}
              </ul>
            </AlertDescription>
          </Alert>
        )}

        {state === "username" && (
          <>
            <Input
              type="text"
              placeholder="username"
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              className="bg-zinc-900 border-zinc-800 text-white"
              disabled={isLoading}
            />
            <Button
              className="w-full bg-white text-black hover:bg-gray-100"
              onClick={handleUsernameSubmit}
              disabled={isLoading}
            >
              {isLoading ? "Loading..." : "Continue"}
            </Button>
          </>
        )}

        {state === "signin" && (
          <form onSubmit={handleSignIn} className="space-y-4">
            <div className="space-y-2">
              <div className="text-sm text-zinc-400">
                Signing in as {username}
              </div>
              <Input
                type="password"
                placeholder="Password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="bg-zinc-900 border-zinc-800 text-white"
                disabled={isLoading}
              />
            </div>
            <Button
              type="submit"
              className="w-full bg-white text-black hover:bg-gray-100"
              disabled={isLoading}
            >
              {isLoading ? "Signing in..." : "Sign in"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="w-full text-zinc-400 hover:text-white"
              onClick={() => {
                setState("username");
                setPassword("");
                setConfirmPassword("");
                setUsername("");
                setError([]);
              }}
            >
              Use a different username
            </Button>
          </form>
        )}

        {state === "signup" && (
          <form onSubmit={handleSignUp} className="space-y-4">
            <div className="space-y-2">
              <div className="text-sm text-zinc-400">
                Creating account for {username}
              </div>
              <Label className="block mb-2 text-sm font-medium text-white">
                Password
                <Input
                  type="password"
                  placeholder="**************"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="mt-1 block w-full bg-zinc-900 border-zinc-800 text-white"
                  disabled={isLoading}
                />
              </Label>

              <Label className="block mb-2 text-sm font-medium text-white">
                Confirm Password
                <Input
                  type="password"
                  placeholder="**************"
                  value={confirmPassword}
                  onChange={(e) => setConfirmPassword(e.target.value)}
                  className="mt-1 block w-full bg-zinc-900 border-zinc-800 text-white"
                  disabled={isLoading}
                />
              </Label>
            </div>
            <Button
              type="submit"
              className="w-full bg-white text-black hover:bg-gray-100"
              disabled={isLoading}
            >
              {isLoading ? "Creating account..." : "Create account"}
            </Button>
            <Button
              type="button"
              variant="ghost"
              className="w-full text-zinc-400 hover:text-white"
              onClick={() => {
                setState("username");
                setPassword("");
                setConfirmPassword("");
                setUsername("");
                setError([]);
              }}
            >
              Use a different username
            </Button>
          </form>
        )}
      </div>
      {state == "username" && (
        <>
          <div className="relative">
            <div className="absolute inset-0 flex items-center">
              <span className="w-full border-t border-zinc-800"></span>
            </div>
            <div className="relative flex justify-center text-xs uppercase">
              <span className="bg-zinc-950 px-2 text-zinc-400">
                Or continue with
              </span>
            </div>
          </div>

          <GoogleProvider loading={isLoading} setIsLoading={setIsLoading} />
        </>
      )}
    </>
  );
};
