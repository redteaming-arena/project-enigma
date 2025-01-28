// types/chat.ts
import { Message } from "ai/react";

export interface GameSessionPublicResponse {
  id: string;
  ok: boolean;
  completed: boolean;
  description?: string;
  start_time: string | null;
  model?: {
    name: string;
    image: string;
  };
  metadata?: {
    game_rules: {
      time_limit: number;
    };
  };
  history: Message[];
  outcome: "win" | "loss" | null;
  shared?: string;
}

export interface SharedGameProps {
  title?: string;
  outcome?: "win" | "loss";
  duration?: string;
  username?: string;
  history: Message[];
  modelName?: string;
  modelImage?: string;
  userImage?: string;
  description?: string;
}

export interface ChatComponentProps {
  session: GameSessionPublicResponse;
  authorization?: string;
}