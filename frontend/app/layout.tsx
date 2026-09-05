import type { Metadata } from "next";
import "./globals.css";
import "./pageBackdrop.css";

export const metadata: Metadata = {
  title: "Market Gate Lab",
  description: "A read-only educational simulation of a learned strategy gate.",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
