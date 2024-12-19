"use server";

import { SharedConversation } from "@/components/chat";
import { getSharedSession } from "@/service/session";
import { Metadata } from "next";
import { redirect } from "next/navigation";

interface SharedChatPageProps { params: Promise<{ id: string }> };

export async function generateMetadata({ params }: SharedChatPageProps): Promise<Metadata> {
  const { id } = await params;

  // Fetch the session using the provided id
  try {
    const session = await getSharedSession(id);

    if (!session.ok) {
      return {
        title: "Shared Game Not Found - RedArena",
        description: "The shared game does not exist on RedArena.",
      };
    }
    const title = session.title ?? "New Game";
    const description = `Game session: ${title}`;

    return {
      metadataBase: new URL("https://project-enigma-620119407459.us-central1.run.app/"),
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

export default async function Share({ params }: SharedChatPageProps) {
  const { id } = await params;

  const sharedGame = await getSharedSession(id);
  if (!sharedGame.ok) {
    redirect("/");
  }

  console.log(sharedGame.description)

  return (
    <SharedConversation
      outcome={sharedGame.outcome ?? undefined}
      username={sharedGame.username}
      title={sharedGame.title}
      history={sharedGame.history ?? []}
      modelImage={sharedGame.model?.image}
      modelName={sharedGame.model?.name}
      description={sharedGame.description ?? "No Objective"}
    />
  );
}
