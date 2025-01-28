"use client";
import Loading from "@/components/loading";
import { SearchComponent } from "@/components/searchbar";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import { Checkbox } from "@/components/ui/checkbox";
import { useUser } from "@/context/user";
import { cn } from "@/lib/utils";
import { ScrollArea } from "@radix-ui/react-scroll-area";
import { CheckCheck } from "lucide-react";
import Link from "next/link";
import { useState } from "react";

export default function Recent() {
  const { state : user , isLoading, handlePopSessions } = useUser();
  const [select, setSelect] = useState<boolean>(false);
  const [selectedChats, setSelectedChats] = useState<string[]>([]);
  const [query, setQuery] = useState<string>("");


  if (isLoading) {
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
          You have {user.history.length} previous chats within RedArena{" "}
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
                  setSelectedChats(user.history.map((item: any) => item._id));
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
          {user.history.map((item) => {
            console.log(item.title)
            console.log(item._id)

            if (
              (query.length === 0 || item.title?.includes(query)) &&
              item._id
            ) {
              return (
                <div
                  key={item._id}
                  className="group flex items-center space-x-3 relative"
                >
                  {/* Checkbox */}
                  <Checkbox
                    id={`${item._id}`}
                    checked={selectedChats.includes(item._id)}
                    onClick={() => toggleSelection(item._id ?? "")}
                    className={cn(
                      "form-checkbox h-7 w-7 text-primary absolute transition-opacity duration-200",
                      !select && !selectedChats.includes(item._id)
                        ? "opacity-0 group-hover:opacity-100"
                        : "opacity-100"
                    )}
                  />

                  {/* Chat Item */}
                  <Link href={`/c/${item._id}`} className="w-full">
                    <Card className="p-3 bg-card rounded-lg shadow-md hover:shadow-lg transition-shadow hover:border-gray-500 duration-300">
                      <CardContent>
                        <div className="text-lg font-semibold text-primary">
                          {item.title}
                        </div>
                      </CardContent>
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
