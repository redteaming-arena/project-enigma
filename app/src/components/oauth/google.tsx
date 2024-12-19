"use client";
import React, { useCallback, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button } from "../ui/button";
import { useNotification } from "../toast";
import { validateToken } from "@/service/auth";
import { useUser } from "@/context/user";
import { getUser } from "@/service/user";

// Helper function to check if a window is a popup
const isPopupWindow = (win: Window): boolean => {
  try {
    return Boolean(
      // Must have an opener
      win.opener &&
        // Must not be the main window
        win !== window.top &&
        // Should have typical popup dimensions
        win.outerWidth < screen.width &&
        win.outerHeight < screen.height &&
        // Should have been opened with window.open
        !win.location.href.includes("noopener") &&
        // Additional checks for popup-like behavior
        typeof win.opener.location.href === "string"
    );
  } catch (e) {
    // If we can't access these properties, assume it's not a popup
    return false;
  }
};

export const GoogleProvider = ({
  loading,
  setIsLoading,
}: {
  loading: boolean;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
}) => {

  
  const router = useRouter();
  const notification = useNotification();
  const { dispatch } = useUser();
  // Handle messages from popup window
  const handleMessage = useCallback(
    async (event: MessageEvent) => {
      if (event.data === "authUsername"){
        router.push("/username")
      } else if (event.data === "authCompleted") {
        const user = await getUser();
        if (user.username === null) {
          router.push('/username')
        }
        dispatch({
          type : "SET_USER",
          payload : {
            id : user.id ?? null, 
            username : user.username ?? null,
            history : user.history ?? [],
            pinned : user.pinned ?? []
          }
        })
        router.push('/')
      } else if (event.data === "authFailed") {
        notification.showWarning(
          "Something went wrong when signing into Google"
        );
      } else if (event.data === "authError") {
        notification.showError("Error within the backend occurred");
      }
    },
    [router]
  );

  useEffect(() => {
    window.addEventListener("message", handleMessage);
    return () => {
      window.removeEventListener("message", handleMessage);
    };
  }, [handleMessage]);

  const handleGoogleLogin = async () => {
    try {
      // Server-side fallback
      if (typeof window === "undefined") {
        router.push("/api/auth/google");
        return;
      }

      // Configure popup window
      const width = 500;
      const height = 600;
      const left = window.screenX + (window.outerWidth - width) / 2;
      const top = window.screenY + (window.outerHeight - height) / 2;
      const popupFeatures = `width=${width},height=${height},left=${left},top=${top},scrollbars=yes,status=yes,resizable=yes`;

      setIsLoading(true);

      // Open popup with blank page initially
      const popup = window.open("about:blank", "GoogleLogin", popupFeatures);

      if (!popup || popup.closed || typeof popup.closed === "undefined") {
        console.warn("Popup blocked - falling back to redirect");
        router.push("/api/auth/google");
        setIsLoading(false);
        return;
      }

      // Determine if it's a proper popup window
      const isPopup = isPopupWindow(popup);

      // Set the appropriate URL based on popup detection
      if (isPopup) {
        popup.location.href = `/api/auth/google?popup=true`;
      } else {
        // If not a proper popup, handle differently based on context
        if (popup.location.href === "about:blank") {
          popup.location.href = `/api/auth/google?popup=false`;
        } else {
          window.location.href = "/api/auth/google?popup=false";
          popup.close(); // Close the unnecessary window
        }
      }

      // Monitor popup status
      const checkPopup = setInterval(() => {
        try {
          if (!popup || popup.closed) {
            clearInterval(checkPopup);
            setIsLoading(false);

            // Check for session cookie
            const sessionCookie = document.cookie
              .split("; ")
              .find((row) => row.startsWith("sessionKey="));

            if (sessionCookie) {
              notification.showSuccess("Successfully Logged In");
              router.push('/')
            } else {
              notification.showWarning(
                "Something went wrong when signing into Google"
              );
            }
          }
        } catch (e) {
          clearInterval(checkPopup);
          setIsLoading(false);
          console.error("Error checking popup status:", e);
        }
      }, 500);

      // Cleanup interval after 5 minutes
      setTimeout(() => {
        clearInterval(checkPopup);
        setIsLoading(false);
      }, 300000);
    } catch (error) {
      console.error("Error handling Google login:", error);
      setIsLoading(false);
      router.push("/api/auth/google");
    }
  };

  return (
    <Button
      variant="outline"
      type="button"
      disabled={loading}
      onClick={handleGoogleLogin}
      className="w-full bg-transparent border-zinc-800 text-white hover:bg-zinc-900 relative"
    >
      {loading ? (
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="w-5 h-5 border-2 border-white border-t-transparent rounded-full animate-spin" />
        </div>
      ) : (
        <>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            height="24"
            viewBox="0 0 24 24"
            width="24"
            className="mr-2 h-4 w-4"
          >
            <path
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92c-.26 1.37-1.04 2.53-2.21 3.31v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.09z"
              fill="#4285F4"
            />
            <path
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z"
              fill="#34A853"
            />
            <path
              d="M5.84 14.09c-.22-.66-.35-1.36-.35-2.09s.13-1.43.35-2.09V7.07H2.18C1.43 8.55 1 10.22 1 12s.43 3.45 1.18 4.93l2.85-2.22.81-.62z"
              fill="#FBBC05"
            />
            <path
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z"
              fill="#EA4335"
            />
            <path d="M1 1h22v22H1z" fill="none" />
          </svg>
          Continue with Google
        </>
      )}
    </Button>
  );
};
