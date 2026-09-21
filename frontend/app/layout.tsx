import type { Metadata } from "next";
import "./globals.css";
import { Nav, Footer } from "../components/Nav";

export const metadata: Metadata = { title: "NeuroVision AI" };

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className="bg-[#f4f8ff] text-slate-800 min-h-screen">
        <Nav />
        <main className="max-w-6xl mx-auto px-6 py-8">{children}</main>
        <Footer />
      </body>
    </html>
  );
}
