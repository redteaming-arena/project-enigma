'use client';
import * as LoadingComponent from "@/components/loading";


export default function Loading() {
  return (
    <LoadingComponent.default
      fullScreen
      className="flex items-center justify-center w-full h-screen text-center relative"
    />
  );
}
