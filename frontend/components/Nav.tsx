import Link from "next/link";
import { Bell, User } from "lucide-react";

export function Nav() {
  return (
    <header className="flex items-center justify-between px-6 py-4 bg-white shadow-sm">
      <Link href="/" className="font-bold text-xl text-blue-700">
        NeuroVision AI
      </Link>
      <nav className="hidden md:flex gap-6 text-sm text-slate-600">
        <span>Dashboard</span>
        <span>Patient Scans</span>
        <span>Analysis Reports</span>
        <span>Neural Insights</span>
      </nav>
      <div className="flex items-center gap-3">
        <Bell size={18} />
        <User size={18} />
        <Link
          href="/upload"
          className="bg-blue-700 text-white text-sm px-4 py-2 rounded-xl"
        >
          Detect Stroke
        </Link>
      </div>
    </header>
  );
}

export function Footer() {
  return (
    <footer className="mt-12 px-6 py-6 text-center text-xs text-slate-500">
      <div className="flex justify-center gap-4 mb-2">
        <span>Research Ethics</span>
        <span>Privacy Protocol</span>
        <span>Medical Support</span>
      </div>
      <p>© 2026 NeuroVision AI — Research / education only, not a diagnostic device.</p>
    </footer>
  );
}
