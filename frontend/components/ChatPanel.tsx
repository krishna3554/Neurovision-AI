"use client";
import { useState } from "react";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function ChatPanel({ studyId, opening }: { studyId: string; opening: string }) {
  const [messages, setMessages] = useState<{ role: string; text: string }[]>([
    { role: "assistant", text: opening },
  ]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  async function ask(q: string) {
    if (!q.trim() || loading) return;
    setMessages((m) => [...m, { role: "user", text: q }]);
    setInput("");
    setLoading(true);
    try {
      const r = await fetch(`${API}/api/chat/${studyId}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: q }),
      });
      const j = await r.json();
      setMessages((m) => [...m, { role: "assistant", text: j.answer ?? "No answer." }]);
    } catch {
      setMessages((m) => [...m, { role: "assistant", text: "Chat request failed." }]);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="bg-white rounded-2xl shadow p-4 flex flex-col h-[480px]">
      <div className="flex items-center justify-between mb-2">
        <h3 className="font-semibold">NeuroAssist AI</h3>
        <span className="text-xs text-green-600">Active &amp; Analyzing · RAG Active</span>
      </div>
      <div className="flex-1 overflow-y-auto space-y-2 text-sm">
        {messages.map((m, i) => (
          <div
            key={i}
            className={`p-2 rounded-xl max-w-[90%] ${
              m.role === "user" ? "bg-blue-700 text-white ml-auto" : "bg-slate-100"
            }`}
          >
            {m.text}
          </div>
        ))}
        {loading && <div className="text-slate-400 text-xs">● ● ● typing…</div>}
      </div>
      <div className="flex gap-2 my-2">
        {["Show precise location", "Compare with previous"].map((c) => (
          <button
            key={c}
            onClick={() => ask(c)}
            className="text-xs border rounded-full px-3 py-1 text-blue-700"
          >
            {c}
          </button>
        ))}
      </div>
      <form
        onSubmit={(e) => {
          e.preventDefault();
          ask(input);
        }}
        className="flex gap-2"
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Ask NeuroAssist about this scan…"
          className="flex-1 border rounded-xl px-3 py-2 text-sm"
        />
        <button className="bg-blue-700 text-white px-4 rounded-xl text-sm">Send</button>
      </form>
      <p className="text-[11px] text-slate-500 mt-2">
        AI insights are for supplementary analysis only. Final diagnosis rests with the
        attending physician.
      </p>
    </div>
  );
}
