// Interface for the game metadata model configuration
interface ModelConfig {
    system_prompt: string;
    temperature: number;
    max_tokens: number;
    tools_config: {
      enabled: boolean;
      tools: string[];
    };
}
  
  // Interface for game rules metadata
interface GameRules {
    timed: boolean;
    time_limit: number;
}
  
  // Interface for game metadata
interface Metadata {
    model_config: ModelConfig;
    game_rules: GameRules;
}
  
  // Main interface for game data
interface GameData {
    id: string;
    title: string;
    author?: string[];
    description?: string;
    image?: string;
    created_at?: string; // ISO timestamp
    updated_at?: string | null; // Nullable for cases where updated_at might not be set
    stars?: number;
    metadata?: Metadata;
}
  
// Interface for error response
interface ErrorResponse {
    status: number; // HTTP status code
    message: string; // Short description of the error
    details?: string; // Optional details for additional context
}
  
// Example of a response that might include either GameData or ErrorResponse
export type Game = GameData; 
export type GameErrorResponse = ErrorResponse;

  
  