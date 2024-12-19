import { AppSidebar, SideBarCloseButton } from "@/components/sidebar";

export default function SubGameRootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <>
      <AppSidebar />
      <main className="flex-1 flex flex-col max-w-full ">
        <SideBarCloseButton />
        <div className="flex-1 pt-14 md:pt-0 ">{children}</div>
      </main>
    </>
  );
}
