"use client";
import React, { useState } from "react";
import { Avatar, AvatarFallback, AvatarImage } from "../ui/avatar";
import { Card } from "../ui/card";
import { cn } from "@/lib/utils";
import ReactMarkdown from "react-markdown";
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/cjs/styles/prism'
import { Check, Copy } from "lucide-react";
import { Button } from "../ui/button";

// Types
interface BaseProps extends React.HTMLAttributes<HTMLDivElement> {
  children?: React.ReactNode;
  className?: string;
}

interface MessageProps extends BaseProps {
  variant?: "system" | "user" | "assistant" | "data";
}

interface MessageAvatarProps extends Omit<BaseProps, "children"> {
  src?: string;
  fallback: string;
}

interface MessageContentProps extends BaseProps {
  isStreaming?: boolean;
}

interface MessageActionsProps extends BaseProps {}

interface MessageTimeProps extends BaseProps {
  date: Date;
}

// Define the compound component type
type MessageComponent = React.ForwardRefExoticComponent<
  MessageProps & React.RefAttributes<HTMLDivElement>
> & {
  Avatar: React.ForwardRefExoticComponent<
    MessageAvatarProps & React.RefAttributes<HTMLDivElement>
  >;
  Content: React.ForwardRefExoticComponent<
    MessageContentProps & React.RefAttributes<HTMLDivElement>
  >;
};

// Root Message component
const MessageRoot = React.forwardRef<HTMLDivElement, MessageProps>(
  ({ children, className, variant = "assistant", ...props }, ref) => {
    return (
      <Card
        ref={ref}
        className={cn(
          "flex gap-3 p-1",
          "flex-row",
          variant === "assistant"
            ? "bg-zinc-950 bg-[radial-gradient(ellipse_at_center,_var(--tw-gradient-stops))] from-zinc-900 via-zinc-900/70 to-zinc-900/30 border-[0.1rem] border-zinc-900"
            : "w-fit bg-black border-0",
          "max-w-full overflow-hidden",
          className
        )}
        {...props}
      >
        {children}
      </Card>
    );
  }
);
MessageRoot.displayName = "Message";

// Message Avatar component
const MessageAvatar = React.forwardRef<HTMLDivElement, MessageAvatarProps>(
  ({ src, fallback, className, ...props }, ref) => {
    return (
      <Avatar ref={ref} className={cn("h-7 w-7 mt-2", className)} {...props}>
        <AvatarImage src={src} alt={fallback} />
        <AvatarFallback className="text-xs">{fallback}</AvatarFallback>
      </Avatar>
    );
  }
);
MessageAvatar.displayName = "MessageAvatar";

// Message Content component
const MessageContent = React.forwardRef<HTMLDivElement, MessageContentProps>(
  ({ children, className, ...props }, ref) => {
    const [copied, setCopied] = useState<boolean>(false);

    const copyToClipboard = (text: string) => {
      if (!copied) {
        setCopied(true);
        navigator.clipboard.writeText(text);
        setTimeout(() => setCopied(false), 3000);
      }
    };

    return (
      <div className="flex flex-col flex-1 min-w-0">
        <div
          ref={ref}
          className={cn(
            "px-0 py-3 text-md w-full",
            "text-white",
            "break-words max-w-full overflow-hidden", // Ensure content stays within bounds
            className
          )}
          {...props}
        >
          <div className="break-words overflow-hidden">
            {typeof children === "string" ? (
              <ReactMarkdown
                components={{
                  code({ node, className, children, style, ...props }) {
                    const inline = undefined;
                    const match = /language-(\w+)/.exec(className || "");
                    const language = match ? match[1] : "";
                    const codeString = String(children).replace(/\n$/, "");

                    return !inline ? (
                      <div className="relative group w-full">
                        <div className="absolute right-2 top-6 opacity-0 group-hover:opacity-100 transition-opacity z-10">
                          <Button
                            size="icon"
                            variant="ghost"
                            className="h-4 w-4 bg-zinc-800/50 hover:bg-zinc-800"
                            onClick={() => copyToClipboard(codeString)}
                          >
                            {copied ? (
                              <Check className="h-2 w-2" />
                            ) : (
                              <Copy className="h-2 w-2" />
                            )}
                          </Button>
                        </div>
                        
                        
                        <div className="max-w-full overflow-x-auto">
                          <SyntaxHighlighter
                            style={oneDark}
                            language={language}
                            PreTag="div"
                            className="rounded-xl my-0 text-sm shadow-lg"
                            customStyle={{
                              margin: "1rem 0",
                              padding: "1rem",
                              backgroundColor: "rgb(40, 44, 52)",
                              width: "100%",
                              maxWidth: "100%",
                            }}
                          >
                            {codeString}
                          </SyntaxHighlighter>
                        </div>
                      </div>
                    ) : (
                      <span className="relative group inline-block">
                        <code
                          className="bg-zinc-800 rounded-lg px-1.5 py-0.5 text-sm font-mono"
                          {...props}
                        >
                          {children}
                        </code>
                        <Button
                          size="icon"
                          variant="ghost"
                          className="absolute -right-8 top-1/2 -translate-y-1/2 h-6 w-6 opacity-0 group-hover:opacity-100 transition-opacity bg-zinc-800/50 hover:bg-zinc-800"
                          onClick={() => copyToClipboard(String(children))}
                        >
                          {copied ? (
                            <Check className="h-2 w-2" />
                          ) : (
                            <Copy className="h-2 w-2" />
                          )}
                        </Button>
                      </span>
                    );
                  },
                  p: ({ children }) => (
                    <div className="mb-0 last:mb-0">{children}</div>
                  ),
                  ul: ({ children }) => (
                    <ul className="list-disc pl-4 mb-4 last:mb-0">
                      {children}
                    </ul>
                  ),
                  ol: ({ children }) => (
                    <ol className="list-decimal pl-4 mb-4 last:mb-0">
                      {children}
                    </ol>
                  ),
                  li: ({ children }) => (
                    <li className="mb-1 last:mb-0">{children}</li>
                  ),
                  blockquote: ({ children }) => (
                    <blockquote className="border-l-2 border-zinc-600 pl-4 my-4 italic">
                      {children}
                    </blockquote>
                  ),
                  strong: ({ children }) => (
                    <strong className="font-semibold">{children}</strong>
                  ),
                  em: ({ children }) => <em className="italic">{children}</em>,
                }}
              >
                {children}
              </ReactMarkdown>
            ) : (
              children
            )}
          </div>
        </div>
      </div>
    );
  }
);
MessageContent.displayName = "MessageContent";
// Create the compound component
const Message = MessageRoot as MessageComponent;
Message.Avatar = MessageAvatar;
Message.Content = MessageContent;

export { Message };
export type {
  MessageProps,
  MessageAvatarProps,
  MessageContentProps,
  MessageActionsProps,
  MessageTimeProps,
};
