"use client";

import React, { useEffect, useState, useReducer, useContext } from "react";
import { useNotification } from "@/components/toast";
import { useRouter } from "next/navigation";
import {
  getUser,
  pinGame,
  unpinGame,
  logout as fetchLogout,
} from "@/service/user";
import { deleteSession } from "@/service/session";

// Types
interface UserState {
  id: string | null;
  username: string | null;
  history: { _id: string; title: string }[];
  pinned: any[];
}

type UserAction =
  | { type: "SET_USER"; payload: UserState }
  | { type: "PIN_GAME"; payload: { id: string; image: string; title: string } }
  | { type: "UNPIN_GAME"; payload: { id: string } }
  | { type: "PUSH_SESSION"; payload: { _id: string; title: string } }
  | { type: "POP_SESSION"; payload: { _id: string } }
  | { type: "POP_SESSIONS"; payload: string[] }
  | { type: "UPDATE_USERNAME"; payload: string | null }
  | { type: "UPDATE_IMAGE"; payload: string | null }
  | { type: "CLEAR_USER" };

interface UserContextType {
  state: UserState;
  dispatch: React.Dispatch<UserAction>;
  isLoading: boolean;
  setIsLoading: React.Dispatch<boolean>;
  handlePin: (id: string, image: string, title: string) => Promise<void>;
  handleUnpin: (id: string) => Promise<void>;
  logout: () => Promise<void>;
  handleSessionPush: (session: { title: string; _id: string }) => void;
  handleSessionPop: (_id: string) => Promise<void>;
  handlePopSessions: (id: string[]) => Promise<boolean>;
}

// Initial State
const initialState: UserState = {
  id: null,
  username: null,
  pinned: [],
  history: [],
};

// Reducer
function userReducer(state: UserState, action: UserAction): UserState {
  switch (action.type) {
    case "SET_USER":
      return { ...action.payload };
    case "PUSH_SESSION":
      return {
        ...state,
        history: [
          action.payload as { title: string; _id: string },
          ...state.history,
        ],
      };
    case "POP_SESSION":
      return {
        ...state,
        history: state.history.filter(
          (item) => item._id !== action.payload._id
        ),
      };
    case "POP_SESSIONS":
      return {
        ...state,
        history : state.history.filter(
          (item) => !action.payload.includes(item._id)
        )
      }
    case "UPDATE_USERNAME":
      return { ...state, username: action.payload };
    case "UPDATE_IMAGE":
      return { ...state };
    case "CLEAR_USER":
      return { ...initialState };
    case "PIN_GAME":
      return { ...state, pinned: [...state.pinned, action.payload] };
    case "UNPIN_GAME":
      return {
        ...state,
        pinned: state.pinned.filter((item) => item.id !== action.payload.id),
      };
    default:
      return state;
  }
}

// Context
const UserContext = React.createContext<UserContextType | undefined>(undefined);

// Provider
function UserProvider({ children }: { children: React.ReactNode }) {
  const notification = useNotification();
  const router = useRouter();
  const [isLoading, setIsLoading] = useState(true);
  const [state, dispatch] = useReducer(userReducer, initialState);

  useEffect(() => {
    const fetchUserData = async () => {
      try {
        const response = await getUser();
        dispatch({
          type: response.ok ? "SET_USER" : "CLEAR_USER",
          payload: response.ok
            ? {
                id: response.id ?? null,
                username: response.username ?? null,
                history: (response.history ?? []).map((item: any) => {
                  return { title: item.title, _id: item._id };
                }),
                pinned: response.pinned ?? [],
              }
            : initialState,
        });
      } catch (error) {
        notification.showError("Failed to fetch user data");
        dispatch({ type: "CLEAR_USER" });
      } finally {
        setIsLoading(false);
      }
    };

    fetchUserData();
  }, []);

  const checkAuthorization = () => {
    const hasAuthToken = document.cookie
      .split(";")
      .some((cookie) => cookie.trim().startsWith("sessionKey="));

    if (!hasAuthToken) {
      notification.showWarning("Authorization required");
      return false;
    }
    return true;
  };

  const handlePin = async (id: string, image: string, title: string) => {
    if (!checkAuthorization()) return;

    const response = await pinGame(id);

    if (response.ok) {
      notification.showSuccess(`${title} was pinned to sidebar`);
      dispatch({
        type: "PIN_GAME",
        payload: { id, image, title },
      });
    } else {
      notification.showError(
        response.status === 401
          ? response.message ?? "Authorization required"
          : `Something went wrong, ${title} was not pinned to sidebar`
      );
    }
  };

  const handleUnpin = async (id: string) => {
    if (!checkAuthorization()) return;

    const response = await unpinGame(id);

    if (response.ok) {
      notification.showSuccess("Game was successfully un-pinned");
      dispatch({ type: "UNPIN_GAME", payload: { id } });
    } else {
      notification.showError(
        response.status === 401
          ? response.message ?? "Authorization required"
          : "Something went wrong when un-pinning game"
      );
    }
  };

  const logout = async () => {
    if (!checkAuthorization()) {
      router.push("/login");
      return;
    }

    setIsLoading(true);
    const response = await fetchLogout();

    if (response.ok) {
      notification.showSuccess(`${state.username} has successfully signed out`);
      dispatch({ type: "CLEAR_USER" });
    } else {
      notification.showError(
        `Something went wrong, ${state.username} was not signed out`
      );
      setIsLoading(false);
      return;
    }
    setIsLoading(false);
    router.push("/login")
  };

  const handleSessionPop = async (_id: string) => {
    if (!state.id){
      notification.showError("Unauthorized to process this.")
      return;
    }

    try {

      const result = await deleteSession(state.id, [_id]);
      if (result.ok) {
        dispatch({ type: "POP_SESSION", payload: { _id } });
      } else {
        notification.showError("Could not remove session, Try Again");
      }
    } catch (error) {
      notification.showError("Could not remove session, Try Again");
    }
  };

  const handlePopSessions = async (ids: string[]) => {
    if (!state.id){
      notification.showError("Unauthorized to process this.")
      return false;
    }

    try {
      const result = await deleteSession(state.id, ids);
      console.log(result)
      if (result.ok) {
        dispatch({ type: "POP_SESSIONS", payload: ids });
        notification.showSuccess(result.message ?? "")
        return true;
      } else {
        notification.showError("Could not remove session, Try Again");
        return false
      }
    } catch (error) {
      notification.showError("Could not remove session, Try Again");
      return false
    }
  };

  const value = {
    state,
    dispatch,
    isLoading,
    setIsLoading,
    handlePin,
    handleUnpin,
    logout,
    handleSessionPush: (session: { title: string; _id: string }) => {
      console.log(session, state.history);
      dispatch({ type: "PUSH_SESSION", payload: session });
    },
    handleSessionPop,
    handlePopSessions
  };

  return <UserContext.Provider value={value}>{children}</UserContext.Provider>;
}

// Custom hook
function useUser() {
  const context = useContext(UserContext);
  if (context === undefined) {
    throw new Error("useUser must be used within a UserProvider");
  }
  return context;
}

export { UserProvider, useUser };
