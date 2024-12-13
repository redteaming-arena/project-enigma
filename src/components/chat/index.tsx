"use client";
import { ChangeEvent, KeyboardEvent, useEffect, useRef, useState } from "react";
import { useUser } from "@/context/user";
import { useNotification } from "../toast";
import {
  createTitle,
  concludeSessionGame,
  GameSessionPublicResponse,
  startGame
} from "@/service/session";
import { Message as MessageComponent } from "./message";
import { ScrollArea } from "../ui/scroll-area";
import { Textarea } from "../ui/textarea";
import { Card } from "../ui/card";
import { Button } from "../ui/button";
import { ArrowRight, Target, Disc, Lock, Share2, Unlock } from "lucide-react";
import { cn } from "@/lib/utils";
import { Message, useChat } from "ai/react";
import useTimer from "@/hooks/use-timer";
import { useRouter } from "next/navigation";
import {
  Tooltip,
  TooltipProvider,
  TooltipTrigger,
  TooltipContent,
} from "../ui/tooltip";
import { ShareDialog } from "../modal/share";
import { useSidebar } from "../ui/sidebar";
import Loading from "../loading";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "../ui/dialog";
import Image from "next/image";

interface SharedGameProps {
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

interface ChatComponentProps {
  session: GameSessionPublicResponse;
  authorization?: string;
}

export const SharedConversation = ({
  outcome,
  username,
  history,
  modelName,
  modelImage,
  description,
}: SharedGameProps) => {
  const [showTargetModal, setShowTargetModal] = useState(false);

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen mx-auto relative w-full max-w-3xl">
        {/* Top gradient */}
        <div className="absolute top-0 w-full bg-gradient-to-t from-transparent to-zinc-950 z-10 rounded-b-xl h-[50px]" />

        {/* Model info header */}
        <div className="absolute top-4 right-4 flex items-center space-x-3 z-20">
          {modelName && modelImage && (
            <div className="flex items-center space-x-2">
              <img
                src={modelImage}
                alt={modelName}
                className="h-6 w-6 object-cover rounded-xl border-[1px] border-zinc-500"
              />
              <span className="text-md font-normal text-white select-none cursor-default">
                {modelName}
              </span>
            </div>
          )}
          {outcome && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="px-2 py-1 rounded-lg bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 text-sm select-none">
                  {outcome === "win" ? (
                    <Unlock className="text-green-400 h-4 w-4" />
                  ) : (
                    <Lock className="text-red-400 h-4 w-4" />
                  )}
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>{outcome === "win" ? "Jailbroken" : "Unbroken"}</p>
              </TooltipContent>
            </Tooltip>
          )}
          <Tooltip>
            <TooltipTrigger asChild>
              <Button
                variant="ghost"
                size="icon"
                className="h-8 w-8 rounded-xl bg-zinc-800/50 border border-zinc-700/50 hover:bg-zinc-700/50"
                onClick={() => {
                  setShowTargetModal(true);
                }}
              >
                <Target className="h-4 w-4 text-zinc-200" />
              </Button>
            </TooltipTrigger>
            <TooltipContent>
              <p>Objective</p>
            </TooltipContent>
          </Tooltip>
        </div>

        {/* Messages area */}
        <ScrollArea className="flex-1 w-full">
          <div className="relative flex-1 overflow-hidden mt-2 mx-auto p-6 pt-8">
            {history.length === 0 ? (
              <div className="text-center text-gray-500 py-8">
                No messages yet
              </div>
            ) : (
              <div className="flex flex-col space-y-4 px-4 pb-4">
                {history.map((message, idx) => (
                  <MessageComponent
                    className="rel"
                    key={`${message.role}-message-${idx}`}
                    variant={message.role}
                  >
                    {message.role === "user" && (
                      <MessageComponent.Avatar
                        className="mt-[.65rem] ml-[.30rem]"
                        src={undefined}
                        fallback={username ? username[0] : "U"}
                      />
                    )}
                    <MessageComponent.Content
                      className={cn(
                        message.role === "user" && "items-end font-light pr-2",
                        message.role === "assistant" && "px-2 font-medium"
                      )}
                    >
                      {message.content}
                    </MessageComponent.Content>
                  </MessageComponent>
                ))}
              </div>
            )}
          </div>
        </ScrollArea>
        <Dialog
          open={showTargetModal}
          onOpenChange={() => {
            // Do nothing when trying to close via outside click or escape
            return;
          }}
        >
          <DialogContent
            className="sm:max-w-[425px]"
            // Prevent closing by clicking outside
            onPointerDownOutside={(e) => e.preventDefault()}
            // Prevent closing by pressing escape
            onEscapeKeyDown={(e) => e.preventDefault()}
          >
            <DialogHeader className="text-center">
              <DialogTitle>Game Objective</DialogTitle>
              <DialogDescription>
                <br className="mb-4" />
                <span className=" font-bold text-white mt-3">
                  Objective
                </span>: {description}
              </DialogDescription>
            </DialogHeader>
            <DialogFooter>
              <Button
                variant="secondary"
                onClick={() => {
                  setShowTargetModal(false);
                }}
              >
                Close
              </Button>
            </DialogFooter>
          </DialogContent>
        </Dialog>
        {/* Bottom gradient */}
        <div className="absolute bottom-0 w-full bg-gradient-to-b from-transparent to-zinc-950 z-10 h-[70px]" />
      </div>
    </TooltipProvider>
  );
};

export const ChatComponent = ({
  session,
  authorization,
}: ChatComponentProps) => {
  const router = useRouter();
  const {
    state: user,
    isLoading: isUserLoading,
    handleSessionPush,
  } = useUser();
  // State
  console.log(!session.completed && session.start_time !== null,  session.start_time !== null, !session.completed, session.start_time)

  // session is not completed, or if the time has not started
  console.log(session.completed ? false : session.start_time === null)
  const [showTargetModal, setShowTargetModal] = useState(session.completed ? false : session.start_time === null);
  const [loadingTargetModal, setLoadingTargetModal] = useState<boolean>(false);
  const [hasText, setHasText] = useState(false);
  const [outcome, setOutcome] = useState<null | "win" | "loss">(
    null
  );

  // Ref
  const scrollAreaRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);

  // Hook
  const { isMobile } = useSidebar();
  const notification = useNotification();
  const { seconds, isComplete, start, pause, formatTime } = useTimer(session.metadata?.game_rules.time_limit ?? 10);

  const {
    messages,
    setMessages,
    handleSubmit,
    handleInputChange,
    stop,
    isLoading,
  } = useChat({
    api: `/api/${session.id}/chat_conversation/${user.id}/conversation`,
    headers: {
      Authorization: `Bearer ${authorization}`,
    },
    initialMessages: session.history,
    streamProtocol: "text",

    onResponse: async (response) => {
      if (!response.ok) {
        notification.showError(
          "An error occurred while connecting to the server."
        );
        return;
      }

      if (response.status === 204 && outcome !== "win") {
        setMessages([]);
        notification.showError("There is currently no live session.");
        router.refresh();
        return;
      }

      notification.showSuccess("Stream started");

      let reader;
      try {
        reader = response.body?.getReader();
        if (!reader) return;

        const decoder = new TextDecoder();
        let buffer = "";

        // indicate to the user the session is running
        setMessages((prevMessages: any) => {
          return [...prevMessages, { role: "assistant", content: "" }];
        });

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          // Decode and append the chunk to the buffer
          buffer += decoder.decode(value, { stream: true });
          console.log(buffer);
          const lines = buffer.split("\n");
          let content = "";

          // Process all complete lines
          for (let i = 0; i < lines.length - 1; i++) {
            const line = lines[i].trim();
            if (!line) continue;

            try {
              const chunk = JSON.parse(line);

              switch (chunk.event) {
                case "message":
                  content = content + chunk.content;
                  handleNewMessage(chunk.content);

                  break;

                case "error":
                  notification.showError(chunk.message);
                  break;

                case "end":
                  if (chunk.outcome === "win") {
                    pause();
                    const message =
                      messages.length === 0 ? [ { content : chunk.content ?? "<empty>" }, { content }] : messages;
                    await handleSessionEnd(message, "win");
                  }
                  break;

                default:
                  console.warn("Unhandled event type:", chunk.event);
              }
            } catch (e) {
              console.error("Error parsing chunk:", line, e);
            }
          }

          // Keep the last incomplete line in the buffer
          buffer = lines[lines.length - 1];
        }

        // Process any remaining buffer content
        processRemainingBuffer(buffer);
      } catch (error) {
        console.warn(error);
      } finally {
        if (reader) reader.releaseLock();
      }
    },

    onError: (error) => {
      notification.showError(error.message);
      console.error(error);
    },

    onFinish: () => {
      notification.showSuccess("Stream finished successfully");
    },
  });

  // Helper to handle a new message chunk
  const handleNewMessage = (content: any) => {
    setMessages((prevMessages: any) => {
      const lastMessage = prevMessages[prevMessages.length - 1];
      if (lastMessage?.role === "assistant") {
        return [
          ...prevMessages.slice(0, -1),
          { ...lastMessage, content: lastMessage.content + content },
        ];
      }
      return [...prevMessages, { role: "assistant", content }];
    });
  };

  const handleSessionEnd = async (messages: Message[] | { content : string }[], outcome: string) => {
    if (!session.id || !user.id) {
      notification.showError("Authorization Error Occurred");
      router.push("/");
      return;
    }
    console.log(messages, outcome)

    try {

      // Handle game outcome
      if (outcome === "loss" && session.outcome == null) {
        const endResponse = await concludeSessionGame(
          session.id,
          user.id,
          outcome,
          messages as Message[]
        );
        if (!endResponse.ok) {
          throw new Error(`Failed to process game end: ${endResponse.message}`);
        }
      }

      let title = "";
      if (messages.length >= 2) {
        // Construct message content for title
        const response = await createTitle(session.id, user.id, true);
        if (!response.ok) {
          throw new Error(
            `Failed to create session title: ${response.message}`
          );
        }

        // title = response.data.title as string;
        title = response.data;
      } else {
        // Handle empty or single message case
        const response = await createTitle(
          session.id,
          user.id,
          false
        );
        if (!response.ok) {
          throw new Error(
            `Failed to create empty session title: ${response.message}`
          );
        }
        notification.showWarning("No content within session");
        title = response.data ?? "Untitled Game"
      }

      // push the page
      if (messages.length > 1)
        handleSessionPush({ title, _id: session.id });
      router.refresh();
    } catch (error) {
      notification.showError(`Error: ${(error as Error).message}`);
    }
  };

  // Helper to process the remaining buffer
  const processRemainingBuffer = (buffer: any) => {
    if (buffer.trim()) {
      try {
        const chunk = JSON.parse(buffer);
        if (chunk.event === "message") {
          handleNewMessage(chunk.content);
        }
      } catch (e) {
        notification.showError(`Error parsing final chunk: ${buffer}, ${e}`);
      }
    }
  };


  useEffect(() => {
    if(!session.ok){
      router.push("/")
    }
  }, [])

  // show model targe if the session at the start of the game
  useEffect(() => {
    // Check if session.description is available and not empty
    if (!session.completed && session.description) {
      // Show the target modal
      setShowTargetModal(session.start_time === null);
      if (session.start_time !== null){
        start()
      }
    }
  }, [session.description, setShowTargetModal]);

  // On message stream scroll to the bottom
  useEffect(() => {
    if (scrollAreaRef.current) {
      scrollAreaRef.current.scrollIntoView(false);
    }
  }, [messages]);

  // handle input
  const handleInput = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const textarea = textareaRef.current;

    if (textarea) {
      setHasText(e.target.value.length > 0);
      // Reset height to auto to get the correct scrollHeight
      textarea.style.height = "auto";
      // Set new height based on scrollHeight, with max lines limit
      const lineHeight = 24; // Assuming 24px line height
      const maxHeight = lineHeight * 6;
      textarea.style.height = `${Math.min(textarea.scrollHeight, maxHeight)}px`;
      handleInputChange(e);
    }
  };

  // handle enter key
  const handleKeyDown = async (e: KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      const message = textareaRef.current?.value.trim();
      if (message) {
        if (textareaRef.current && user.id) {
          setHasText(false);
          textareaRef.current.value = "";
          // Reset height after sending
          textareaRef.current.style.height = "auto";
          handleSubmit(e);
        }
      }
    }
  };

  // on
  useEffect(() => {
    if (outcome === "loss" && !session.completed) {
      pause(); // Pause timer to indicate the end of the game
      handleSessionEnd(messages, "loss")
    }
  }, [outcome, session.id, user.id]);

  // Loss Case 1: where timer runs out
  useEffect(() => {
    if (isComplete && outcome === null) {
      // Use functional update to ensure state is set correctly
      // set the outcome
      pause(); // pause timer
      stop();
      setOutcome("loss");
    }
  }, [isComplete, outcome, messages]);

  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.focus();
    }
  }, [session, textareaRef]);

  // NOTE: uncomment when fix loss
  // Replace your existing useEffect forfeit logic with this:
  useEffect(() => {
    // State to track the forfeit status
    const forfeitState = {
      shouldForfeit: false,
      isForfeitInProgress: false,
    };

    // Function to handle forfeiting the game

    // Listener for the `beforeunload` event (warn user before leaving)
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!session.completed && outcome === null && !isComplete) {
        console.log("user leaving");
        forfeitState.shouldForfeit = true; // Mark forfeit as required
        event.returnValue = ""; // Trigger browser's default confirmation dialog
      }
    };

    // Listener for the `unload` event (ensure API call on page exit)
    const handleUnload = async (event: any) => {
      if (session.id && user.id) {
        // await concludeSessionGame(session.id, user.id, "forfeit");
        console.log("finished");
      }
    };

    // Add event listeners
    // document.addEventListener("visibilitychange", handleVisibilityChange);

    if (!session.completed && !isComplete) {
      window.addEventListener("beforeunload", handleBeforeUnload);
      window.addEventListener("unload", handleUnload);
    }

    // Cleanup event listeners on component unmount
    return () => {
      // document.removeEventListener("visibilitychange", handleVisibilityChange);
      if (isComplete) {
        window.removeEventListener("beforeunload", handleBeforeUnload);
        window.removeEventListener("unload", handleUnload);
      }
    };
  }, [session.id, user.id, outcome, session.completed, isComplete]);

  if (isUserLoading || !session.ok) {
    return (
      <Loading
        fullScreen
        className="flex items-center justify-center w-full h-screen text-center relative"
      />
    );
  }

  return (
    <TooltipProvider>
      <div className="flex flex-col h-screen mx-auto relative w-full max-w-4xl">
        {/* Messages Container */}
        <div
          className={cn(
            "absolute top-0 w-full bg-gradient-to-t from-transparent to-zinc-950 z-10 rounded-b-xl",
            !session.completed ? "h-[100px]" : "h-[50px]"
          )}
        />

        {/* Model info for completed session */}
        <div className="absolute top-4 right-4 flex items-center space-x-3 z-20">
          {!session.completed && (
            <Tooltip>
              <TooltipTrigger asChild>
                <div className="px-3 py-1 rounded-lg bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 text-sm select-none">
                  <span
                    className={cn(
                      "font-mono",
                      isComplete || (seconds < 10 && "text-red-400")
                    )}
                  >
                    {formatTime()}
                  </span>
                </div>
              </TooltipTrigger>
              <TooltipContent>
                <p>Time remaining</p>
              </TooltipContent>
            </Tooltip>
          )}
          {session.completed && (
            <>
              <div className="flex items-center space-x-2">
                <img
                  src={session.model?.image}
                  alt={session.model?.name}
                  className="h-6 w-6 object-cover rounded-xl border-[1px] border-zinc-500"
                />
                <span className="text-md font-normal text-white select-none cursor-default">
                  {session.model?.name}
                </span>
              </div>
              <Tooltip>
                <TooltipTrigger asChild>
                  <div className="px-2 py-1 rounded-lg bg-zinc-800/50 border border-zinc-700/50 text-zinc-200 text-sm select-none">
                    {session.outcome === "win" ? (
                      <Unlock className="text-green-400 h-4 w-4" />
                    ) : (
                      <Lock className="text-red-400 h-4 w-4" />
                    )}
                  </div>
                </TooltipTrigger>
                <TooltipContent>
                  <p>{session.outcome === "win" ? "Jailbroken" : "Unbroken"}</p>
                </TooltipContent>
              </Tooltip>
              {session.history?.length !== 0 && (
                <Tooltip>
                  <TooltipTrigger asChild>
                    <ShareDialog
                      session_id={session.id}
                      share_id={session.shared ?? undefined}
                    >
                      <Button
                        variant="ghost"
                        size="icon"
                        className="h-8 w-8 rounded-xl bg-zinc-800/50 border border-zinc-700/50 hover:bg-zinc-700/50"
                      >
                        <Share2 className="h-4 w-4 text-zinc-200" />
                      </Button>
                    </ShareDialog>
                  </TooltipTrigger>
                  <TooltipContent>
                    <p>Share conversation</p>
                  </TooltipContent>
                </Tooltip>
              )}
              <Tooltip>
                <TooltipTrigger asChild>
                  <Button
                    variant="ghost"
                    size="icon"
                    className="h-8 w-8 rounded-xl bg-zinc-800/50 border border-zinc-700/50 hover:bg-zinc-700/50"
                    onClick={() => {
                      setShowTargetModal(true);
                    }}
                  >
                    <Target className="h-4 w-4 text-zinc-200" />
                  </Button>
                </TooltipTrigger>
                <TooltipContent>
                  <p>Objective</p>
                </TooltipContent>
              </Tooltip>
            </>
          )}
        </div>

        {/* Chat Area */}
        <ScrollArea className="flex-1 w-full">
          <div
            className={cn(
              "relative flex-1 overflow-hidden mt-2 mx-auto p-6",
              session.completed ? "pt-8" : "pt-12",
              isMobile && "pt-14"
            )}
          >
            {session.completed && messages.length === 0 ? (
              <div className="inset-0 flex items-center text-center justify-center my-auto mt-10">
                <span className="text-gray-500">
                  There are no messages here.
                  <p className="text-gray-500">
                    This chat will be removed shortly.
                  </p>
                </span>
              </div>
            ) : (
              <>
                <div
                  className="flex flex-col space-y-4 px-2 sm:px-4 pb-4 overflow-auto max-w-full"
                  ref={scrollAreaRef}
                >
                  {messages.map((message, idx) =>
                    message.content.length !== 0 ? (
                      <MessageComponent
                        className="relative"
                        key={`${message.role}-message-${idx}`}
                        variant={message.role}
                      >
                        {message.role === "user" && (
                          <MessageComponent.Avatar
                            className="mt-[.65rem] ml-[.30rem]"
                            src={user.id ? `/api/avatar/${user.id}` : undefined}
                            fallback={user.username ? user.username[0] : "U"}
                          />
                        )}
                        <MessageComponent.Content
                          className={cn(
                            "break-words", // Ensure text wraps properly
                            message.role === "user" &&
                              "items-end font-light pr-2",
                            message.role === "assistant" && "px-2 font-medium"
                          )}
                        >
                          {message.content}
                        </MessageComponent.Content>
                      </MessageComponent>
                    ) : (
                      <div
                        key={`${message.role}-message-${idx}`}
                        className=" italic text-lg mt-5"
                      >
                        <span className="animate-pulse">Thinking...</span>
                      </div>
                    )
                  )}
                </div>
              </>
            )}
          </div>
        </ScrollArea>
        <div
          className={cn(
            "absolute bottom-0 w-full bg-gradient-to-b from-transparent to-zinc-950 z-10",
            !session.completed ? "h-[200px]" : "h-[10px]"
          )}
        />

        {/* Input Container */}
        {!session.completed && (
          <div className="w-full relative z-10">
            <Card
              className="border-zinc-700/50 flex flex-col gap-1.5 pl-4 pt-2.5 pr-2.5 pb-2.5 items-stretch transition-all duration-200 relative shadow-[0_0.25rem_1.25rem_rgba(0,0,0,0.035)] focus-within:shadow-[0_0.25rem_1.25rem_rgba(0,0,0,0.075)] hover:border-zinc-600 focus-within:border-zinc-600 cursor-text z-10 rounded-t-2xl rounded-b-none border-b-0"
              title={
                session.description || "Hover to view the session objective"
              }
            >
              <div className="relative flex">
                <Textarea
                  ref={textareaRef}
                  placeholder="Let's Start Cracking..."
                  className="resize-none text-md overflow-y-auto bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 text-white placeholder:text-zinc-400 min-h-[24px] max-h-[144px] pb-8"
                  onInput={handleInput}
                  onKeyDown={handleKeyDown}
                  disabled={isComplete || outcome === "win"}
                  rows={1}
                />
                <div
                  className={cn(
                    "absolute top-1 right-2 flex gap-2 transition-all duration-200 transform",
                    (hasText && !isLoading) || isLoading
                      ? "opacity-100 translate-x-0 scale-100"
                      : "opacity-0 translate-x-4 scale-95 pointer-events-none"
                  )}
                >
                  {hasText && !isLoading && (
                    <Button
                      onClick={(e) => {
                        const message = textareaRef.current?.value.trim();
                        if (message) {
                          if (textareaRef.current && user.id) {
                            setHasText(false);
                            textareaRef.current.value = "";
                            textareaRef.current.style.height = "auto";
                            handleSubmit(e);
                          }
                        }
                      }}
                      className="rounded-xl transition-colors duration-200 hover:scale-105"
                      variant="ghost"
                      size="icon"
                      disabled={isComplete}
                    >
                      <ArrowRight className="h-5 w-5 text-zinc-400" />
                    </Button>
                  )}
                  {isLoading && (
                    <Button
                      onClick={() => {
                        notification.showSuccess("Stop stream");
                        stop(); // stops stream
                      }}
                      className="rounded-xl transition-colors duration-200 hover:scale-105"
                      variant="ghost"
                      size="icon"
                    >
                      <Disc className="h-5 w-5 text-zinc-400" />
                    </Button>
                  )}
                </div>
              </div>
              <div className="flex flex-row space-x-2 cursor-default select-none">
                {session.model && session.model.image && session.model.name && (
                  <Image
                    src={session.model?.image}
                    alt={session.model?.name}
                    width={50}
                    height={50}
                    className={cn(
                      `h-6 w-6 object-cover rounded-xl border-[1px] border-zinc-500 cursor-default`,
                      isLoading && "animate-spin",
                      "hover:animate-spin"
                    )}
                  />
                )}
                <span
                  className={cn(
                    `text-md font-normal text-white`,
                    isLoading && "animate-pulse"
                  )}
                >
                  {session.model?.name}
                </span>
              </div>
              <div className="absolute right-3 bottom-2 text-xs text-zinc-500 cursor-default select-none">
                Press{" "}
                <code className="text-xxs p-[2px] border-[1px] rounded-lg">
                  Shift + Enter
                </code>{" "}
                for new line
              </div>
            </Card>
          </div>
        )}
      </div>

      <Dialog
        open={showTargetModal}
        onOpenChange={() => {
          // Do nothing when trying to close via outside click or escape
          return;
        }}
      >
        <DialogContent
          className="sm:max-w-[425px]"
          // Prevent closing by clicking outside
          onPointerDownOutside={(e) => e.preventDefault()}
          // Prevent closing by pressing escape
          onEscapeKeyDown={(e) => e.preventDefault()}
        >
          <DialogHeader className="text-center">
            <DialogTitle>Game Objective</DialogTitle>
            <DialogDescription>
              Get ready to challenge your skills! Dive into this session where
              every move counts.
              <br className="mb-4" />
              <span className=" font-bold text-white mt-3">
                Objective
              </span>: {session.description}
            </DialogDescription>
          </DialogHeader>
          <DialogFooter>
            <Button
              disabled={loadingTargetModal}
              variant="secondary"
              onClick={async () => {
                if (textareaRef.current) {
                  textareaRef.current.focus();
                }
                if (session.id && user.id){
                  setLoadingTargetModal(true)
                  const response = await startGame(session.id, user.id)
                  if (!response.ok){
                    notification.showError("Could not start sessions")
                    router.push("/")
                  }
                  // if session is not completed we don't need to start
                  if (!session.completed) {
                    start(); // the timer
                  }
                } else {
                  router.push("/")
                }
                setLoadingTargetModal(false)
                setShowTargetModal(false);
              }}
            >
              {loadingTargetModal && <Loading/>}
              {session.completed ? "Close" : "Start Cracking"}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </TooltipProvider>
  );
};
