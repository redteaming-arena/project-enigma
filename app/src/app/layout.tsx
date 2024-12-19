import type { Metadata } from "next";
import localFont from "next/font/local";
import "./globals.css";
import { Toaster } from "@/components/ui/toaster";
import { UserProvider } from "@/context/user";
import { SidebarProvider } from "@/components/ui/sidebar";


const geistSans = localFont({
  src: "./fonts/GeistVF.woff",
  variable: "--font-geist-sans",
  weight: "100 900",
});
const geistMono = localFont({
  src: "./fonts/GeistMonoVF.woff",
  variable: "--font-geist-mono",
  weight: "100 900",
});


export const metadata: Metadata = {
  metadataBase: new URL("https://project-enigma-620119407459.us-central1.run.app/"),
  title: "RedArena",
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
    title: "RedArena",
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
    title: "RedArena",
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
  return (
    <html lang="en" className="dark">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased bg-zinc-950`}
      >
        <SidebarProvider>
          <UserProvider>
            <div className="flex min-h-screen w-full">
              {children}
            </div>
          </UserProvider>
        </SidebarProvider>
        <Toaster />
      </body>
    </html>
  );
}
