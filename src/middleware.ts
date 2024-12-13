import { NextResponse } from "next/server";
import type { NextRequest } from "next/server";
import { validateToken } from "@/service/auth";

const PUBLIC_ROUTES = ["/", "/login", "/api", "/static"]; // Add all public routes
const PROTECTED_ROUTES = ["/c"]; // Add all protected routes that require auth

export async function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get("sessionKey");
  
  // Validate token for all routes
  const tokenValidation = await validateToken();
  const isAuthenticated = tokenValidation.ok;
  const hasUsername = tokenValidation.username !== null;


  // If authenticated but no username, force stay on username page
  if (isAuthenticated && !hasUsername) {
    // Allow access to the username page itself
    if (pathname === "/username") {
      return NextResponse.next();
    }
    // Redirect to username page for all other routes
    return NextResponse.redirect(new URL("/username", request.url));
  }

  // Handle login/username routes for other cases
  if (pathname.startsWith("/login") || pathname.startsWith("/username")) {
    // If fully authenticated with username, redirect to home
    if (isAuthenticated && hasUsername) {
      return NextResponse.redirect(new URL("/", request.url));
    }
    
    // If not authenticated and trying to access username page, redirect to login
    if (!isAuthenticated && pathname.startsWith("/username")) {
      return NextResponse.redirect(new URL("/login", request.url));
    }
  }

  // Handle protected routes
  if (PROTECTED_ROUTES.some(route => pathname.startsWith(route))) {
    // Not authenticated - redirect to login
    if (!isAuthenticated) {
      if (token) {
        request.cookies.delete("sessionKey");
      }
      return NextResponse.redirect(new URL("/login", request.url));
    }
    
    // This check might be redundant now but keeping for safety
    if (!hasUsername) {
      return NextResponse.redirect(new URL("/username", request.url));
    }
  }

  if (!isAuthenticated && !pathname.startsWith("/login")){
    return NextResponse.redirect(new URL("/login", request.url));
  }

  // Allow the request to continue
  return NextResponse.next();
}

export const config = {
  matcher: [
    "/((?!api|_next/static|_next/image|favicon.ico|sitemap.xml|robots.txt|file.svg|globe.svg|next.svg|vercel.svg|window.svg).*)",
  ],
};