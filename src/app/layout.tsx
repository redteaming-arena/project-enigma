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
  title: "RedArena",
  authors: { name: "LLMSYS", url: "https://lmsys.org/" },
  applicationName: "",
  description: "Generated by create next app",
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
