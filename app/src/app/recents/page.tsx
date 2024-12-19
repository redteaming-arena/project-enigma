"use client";
import Loading from "@/components/loading";
import { SearchComponent } from "@/components/searchbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardFooter } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { useUser } from "@/context/user";
import { cn } from "@/lib/utils";
import {
  GameSessionRecentItem,
  GameSessionRecentItemResponse,
  getRecents,
} from "@/service/session";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { CheckCheck } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useState, useCallback } from "react";

export default function Recent() {
  const router = useRouter();
  const { isLoading, handlePopSessions } = useUser();
  const [loading, setLoading] = useState<boolean>(true);
  const [select, setSelect] = useState<boolean>(false);
  const [selectedChats, setSelectedChats] = useState<string[]>([]);
  const [history, setHistory] = useState<GameSessionRecentItem[]>([]);
  const [query, setQuery] = useState<string>("");

  const handleHistory = useCallback(async () => {
    const recent: GameSessionRecentItemResponse = await getRecents(
      0,
      undefined
    );

    if (recent.ok) {
      setHistory(recent.content ?? []);
      setLoading(false);
    } else {
      router.push("/");
    }
  }, [setHistory, setLoading, router]);

  useEffect(() => {
    handleHistory()
  }, [])

  if (loading || isLoading) {
    return (
      <Loading
        fullScreen
        className="flex items-center justify-center w-full h-screen text-center relative"
      />
    );
  }

  // Handle checkbox toggle
  const toggleSelection = (sessionId: string) => {
    setSelectedChats((prevSelected) =>
      prevSelected.includes(sessionId)
        ? prevSelected.filter((id) => id !== sessionId)
        : [...prevSelected, sessionId]
    );
  };

  return (
    <div className="h-screen bg-background md:p-20 p-10">
      {/* Search Bar */}
      <SearchComponent onSearch={setQuery} />

      {!select && selectedChats.length === 0 ? (
        <p className="mt-2 text-center md:text-lg text-md font-medium text-gray-200">
          You have {history.length} previous chats within RedArena{" "}
          <span
            onClick={() => {
              setSelect(true);
            }}
            className=" hover:underline text-blue-500 "
          >
            Select
          </span>
        </p>
      ) : (
        <div className="flex items-center justify-between px-4 py-3 bg-background rounded-lg shadow-md">
          {/* Selected Count */}
          <div className="flex items-center space-x-2">
            <CheckCheck />
            <span className="text-sm text-primary">
              {selectedChats.length} selected{" "}
              {selectedChats.length === 1 ? "chat" : "chats"}
            </span>
          </div>

          {/* Action Buttons */}
          <div className="flex items-center space-x-3">
            {selectedChats.length !== history.length && (
              <button
                onClick={() => {
                  setSelectedChats(history.map((item: any) => item.session_id));
                }}
                className="text-sm text-primary hover:underline"
              >
                Select all
              </button>
            )}
            <Button
              variant="secondary"
              onClick={() => {
                setSelect(false);
                setSelectedChats([]);
              }}
              className="px-4 py-2"
            >
              Cancel
            </Button>
            <Button
              variant="destructive"
              onClick={async () => {
                const successfully = await handlePopSessions(selectedChats);
                if (successfully) {
                  setHistory((prevHistory) =>
                    prevHistory.filter(
                      (hst: GameSessionRecentItem) =>
                        !selectedChats.includes(hst.session_id ?? "")
                    )
                  );
                  setSelect(false);
                  setSelectedChats([]);
                }
              }}
              className="px-4 py-2 bg-red-500 hover:bg-red-600 text-white"
            >
              Delete Selected
            </Button>
          </div>
        </div>
      )}

      {/* History List */}
      <div className="mt-4 space-y-5">
        <ScrollArea className="space-y-5">
          {history.map((item) => {
            if (
              (query.length === 0 || item.title?.includes(query)) &&
              item.session_id
            ) {
              return (
                <div
                  key={item.session_id}
                  className="group flex items-center space-x-3 relative"
                >
                  {/* Checkbox */}
                  <Checkbox
                    id={`${item.session_id}`}
                    checked={selectedChats.includes(item.session_id)}
                    onClick={() => toggleSelection(item.session_id ?? "")}
                    className={cn(
                      "form-checkbox h-7 w-7 text-primary absolute transition-opacity duration-200",
                      !select && !selectedChats.includes(item.session_id)
                        ? "opacity-0 group-hover:opacity-100"
                        : "opacity-100"
                    )}
                  />

                  {/* Chat Item */}
                  <Link href={`/c/${item.session_id}`} className="w-full">
                    <Card className="p-3 bg-card rounded-lg shadow-md hover:shadow-lg transition-shadow hover:border-gray-500 duration-300">
                      <CardContent>
                        <div className="text-lg font-semibold text-primary">
                          {item.title}
                        </div>
                      </CardContent>
                      <CardFooter className="-mb-5">
                        Last message 1 day ago
                      </CardFooter>
                    </Card>
                  </Link>
                </div>
              );
            }
          })}
        </ScrollArea>
      </div>
    </div>
  );
}
