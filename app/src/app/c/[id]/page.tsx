"use server";

import { ChatComponent } from "@/components/chat";
import AuthMonitor from "@/hooks/useCookieCheck";
import {
  GameSessionPublic,
  GameSessionPublicResponse,
  getSession,
} from "@/service/session";
import { cookies } from "next/headers";
import { Suspense } from "react";
import Loading from "@/components/loading";
import { Metadata } from "next";
import { redirect } from "next/navigation";

interface ChatPageProps {
  params: Promise<{
    id: string;
  }>;
}


export async function generateMetadata({
  params,
}: ChatPageProps): Promise<Metadata> {
  const { id } = await params;

  try {
    const session = (await getSession(id)) as GameSessionPublicResponse;

    if (!session.ok) {
      return {
        title: "Game Not Found - RedArena",
        description: "The requested game does not exist on RedArena.",
      };
    }
    const title = session.title ?? "New Game";
    const description = `Game session: ${title}`;

    return {
      metadataBase: new URL(
        "https://project-enigma-620119407459.us-central1.run.app/"
      ),
      title,
      description,
      openGraph: {
        title,
        description,
        images: [
          {
            url: "/og-card.png",
            width: 796,
            height: 468,
            alt: "Red Arena Game",
          },
        ],
      },
      twitter: {
        card: "summary_large_image",
        title,
        description,
        images: [
          {
            url: "/og-card.png",
            alt: "Red Arena Game",
          },
        ],
      },
    };
  } catch (error) {
    console.error("Error fetching session data:", error);

    // Fallback metadata in case of an error
    return {
      title: "Error - RedArena",
      description: "Unable to load game session details.",
      openGraph: {
        title: "Error - RedArena",
        description: "Unable to load game session details.",
        images: [
          {
            url: "/og-card.png",
            width: 796,
            height: 468,
            alt: "Red Arena Game",
          },
        ],
      },
      twitter: {
        card: "summary_large_image",
        title: "Error - RedArena",
        description: "Unable to load game session details.",
        images: [
          {
            url: "/og-card.png",
            alt: "Red Arena Game",
          },
        ],
      },
    };
  }
}

export default async function Page({ params }: ChatPageProps) {
  const { id } = await params;
  const session: GameSessionPublicResponse = await getSession(id);

  const cookeStore = cookies();
  const authorization = (await cookeStore).get("sessionKey")?.value;

  return (
    <AuthMonitor>
      <Suspense
        fallback={
          <Loading
            fullScreen
            className="flex items-center justify-center w-full h-screen text-center relative"
          />
        }
      >
        <ChatComponent
          session={session as GameSessionPublic}
          authorization={authorization}
        />
      </Suspense>
    </AuthMonitor>
  );
}
