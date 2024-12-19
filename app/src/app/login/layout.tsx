import type { Metadata } from "next";
import "../globals.css";

export const metadata: Metadata = {
  metadataBase: new URL("https://project-enigma-620119407459.us-central1.run.app/"),
  title: "RedArena Login",
  description:
    "A community-driven redteaming platform",
  keywords: [
      "AI",
      "machine learning",
      "deep learning",
      "RedArena",
      "artificial intelligence",
      "developer tools",
      "AI research",
    ],
  authors: [{ name: "RedArena Team" }, { name: "LLMSYS", url: "https://lmsys.org/" }, { name : "Luca Vivona" }] ,
  openGraph: {
    type: "website",
    url : "https://project-enigma-620119407459.us-central1.run.app",
    title: "RedArena Login",
    description:
      "A community-driven redteaming platform",
    images: [
      {
        url: "/og-card.png",
        width: 796,
        height: 460,
        alt: "Redarena Logo",
      },
    ],
  },
  twitter: {
    card: "summary_large_image",
    site: "@lmarena_ai", // Replace with your actual Twitter handle
    title: "RedArena Login",
    description:
      "A community-driven redteaming platform",
    images: [
      {
        url: "/og-card.png",
        width: 796,
        height: 460,
        alt: "Redarena Logo",
      },
    ],
  },
  // Add more metadata as needed
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return <>{children}</>;
}
