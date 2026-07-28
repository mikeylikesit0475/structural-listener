/** @type {import('next').NextConfig} */
const nextConfig = {
  // One page, no routing, no SSR optimisation needed (ADR-007).
  reactStrictMode: true,
};
module.exports = nextConfig;