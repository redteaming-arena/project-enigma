import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'standalone',
  devIndicators: {
    appIsrStatus: false,
  },
  images: {
    remotePatterns: [
        {
            protocol: 'https',
            hostname: 'avatars.githubusercontent.com',
            port: '',
            pathname: '/u/**',
        },
    ],
  },
  rewrites: async () => {
    const backendHost = process.env.NEXT_PUBLIC_BACKEND_HOST || 'https://backend-620119407459.us-central1.run.app/'; // Fallback to localhost if not set
    return [
      {
        source: "/api/:path*",
        destination:
          process.env.NODE_ENV === "development"
            ? `${backendHost}/:path*`
            : `${backendHost}/:path*`,
      },
    ];
  },

};

export default nextConfig;