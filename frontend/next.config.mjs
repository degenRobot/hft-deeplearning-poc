/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // Keep Turbopack inside this app instead of inferring a parent lockfile.
  turbopack: { root: process.cwd() },
  // Next can generate repo instruction files on first dev run; this repo owns its instructions.
  agentRules: false,
};

export default nextConfig;
