import { useState, useEffect, useCallback } from "react";

interface TimerHookReturn {
  seconds: number;
  isRunning: boolean;
  isComplete: boolean;
  start: () => void;
  pause: () => void;
  reset: (time: number) => void;
  formatTime: () => string;
}

/**
 * Custom hook for managing a countdown timer
 * @param initialSeconds - The initial number of seconds in ms to count down from
 * @returns Object containing timer state and control functions
 */
const useTimer = (initialSeconds: number): TimerHookReturn => {
  const [seconds, setSeconds] = useState<number>(
    Math.floor(initialSeconds / 60)
  );
  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [isComplete, setIsComplete] = useState<boolean>(false);

  // Reset timer to initial value
  const reset = useCallback((time: number): void => {
    setSeconds(time);
    setIsRunning(false);
    setIsComplete(false);
  }, []);

  // Start the timer
  const start = useCallback((): void => {
    if (seconds > 0) {
      setIsRunning(true);
      setIsComplete(false);
    }
  }, [seconds]);

  // Pause the timer
  const pause = useCallback((): void => {
    setIsRunning(false);
  }, []);

  // Format time as mm:ss
  const formatTime = useCallback((): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, "0")}:${remainingSeconds
      .toString()
      .padStart(2, "0")}`;
  }, [seconds]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout | undefined;

    if (isRunning && seconds > 0) {
      intervalId = setInterval(() => {
        setSeconds((prevSeconds) => {
          if (prevSeconds <= 1) {
            setIsRunning(false);
            setIsComplete(true);
            return 0;
          }
          return prevSeconds - 1;
        });
      }, 1000);
    }

    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
    };
  }, [isRunning, seconds]);

  return {
    seconds,
    isRunning,
    isComplete,
    start,
    pause,
    reset,
    formatTime,
  };
};

export default useTimer;
