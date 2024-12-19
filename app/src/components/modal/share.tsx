"use client";
import { useState } from "react";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import Image from "next/image";
import { postSharedSession } from "@/service/session";
import { useUser } from "@/context/user";
import { useNotification } from "../toast";
import Link from "next/link";

export function ShareDialog({
  session_id,
  share_id,
  children,
}: {
  session_id?: string;
  share_id?: string;
  children: React.ReactNode;
}) {
  const [link, setLink] = useState<string | undefined>(
    share_id !== undefined
      ? `${window.location.protocol}//${window.location.host}/share/${share_id}`
      : undefined
  );
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const notification = useNotification();
  const { state: user } = useUser();

  const createLink = async () => {
    setLoading(true);
    setError(null);

    try {
      if (!user.id) {
        throw { message: "Authentication required" };
      }
      if (!session_id) {
        throw { message: "No Session selected" };
      }

      const response = await postSharedSession(session_id, user.id);
      setLink(
        `${window.location.protocol}//${window.location.host}/share/${response.data}`
      );
      notification.showSuccess("Successfully Created Link");
    } catch (err: any) {
      setError(err.message || "An unexpected error occurred.");
    } finally {
      setLoading(false);
    }
  };

  const handleShareToX = () => {
    const text = encodeURIComponent("Check out this jailbreak!");
    const url = encodeURIComponent(link ?? "");
    window.open(
      `https://twitter.com/intent/tweet?text=${text}&url=${url}`,
      "_blank"
    );
  };

  const handleShareToReddit = () => {
    const title = encodeURIComponent("Check out this jailbreak!");
    const url = encodeURIComponent(link ?? "");
    window.open(
      `https://www.reddit.com/submit?title=${title}&url=${url}`,
      "_blank"
    );
  };

  return (
    <Dialog>
      <DialogTrigger asChild>{children}</DialogTrigger>
      <DialogContent className="w-full">
        <DialogHeader>
          <DialogTitle>
            {link ? "Public Link Created" : "Share Public Link to Chat"}
          </DialogTitle>
          <DialogDescription>
            {link
              ? "Your chat is now publicly accessible. The entire conversation, including your username, messages, and objectives, will be visible to anyone with this link. Be mindful of the content you've shared."
              : "Your username, messages, and objective will become public for all users to see. By creating a public link, you'll allow anyone with the link to view the entire conversation without authentication."}
          </DialogDescription>
        </DialogHeader>

        <div className="grid grid-cols-1 py-4 items-center gap-4 w-full">
          <div className="col-span-1 flex items-center justify-center relative w-full">
            <Input
              disabled
              id="generated-link"
              value={link || "Generate a link first"}
              className=" w-full" // Add padding to make room for the button
            />
            <Button
              onClick={createLink}
              disabled={loading}
              variant={"secondary"}
              className="absolute right-1 top-1/2 -translate-y-1/2 rounded-3xl"
              size="sm"
            >
              {loading ? "Loading..." : link ? "New Link" : "Create Link"}
            </Button>
          </div>
        </div>
        {error && <p className="text-red-500 text-sm">{error}</p>}
        {link && (
          <div className="  flex flex-row gap-4 items-center justify-center w-full">
            <div
              className="border-[1px] rounded-2xl p-2 border-gray-400 hover:border-gray-300 flex items-center justify-center"
              onClick={handleShareToX}
            >
              <Image
                src={"/X.svg"}
                alt="X"
                className="dark:invert"
                width={15}
                height={15}
              />
            </div>
            <div
              className="border-[1px] rounded-2xl p-2 border-gray-400 hover:border-gray-300 flex items-center justify-center"
              onClick={handleShareToReddit}
            >
              <Image
                src={"/reddit.svg"}
                alt="Reddit"
                className="dark:invert"
                width={15}
                height={15}
              />
            </div>

            <Link
              href={`https://www.linkedin.com/shareArticle?url=${encodeURIComponent(
                link ?? ""
              )}&text=${encodeURIComponent("Check out this jailbreak!")}`}
              target=""
              rel="noopener noreferrer"
              className="border-[1px] rounded-2xl p-2 border-gray-400 hover:border-gray-300 flex items-center justify-center"
            >
               <Image
                src={"/linkedin.svg"}
                alt="LinkedIn"
                className="dark:invert"
                width={13}
                height={13}
              />
            </Link>
          </div>
        )}
      </DialogContent>
    </Dialog>
  );
}
