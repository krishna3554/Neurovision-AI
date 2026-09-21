"use client";
import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const STEPS = ["VALIDATED", "PREPROCESSING", "INFERENCE", "POST-PROCESSING", "VISUALIZATION", "RESPONSE GENERATION"];

export default function ProcessingPage() {
  const { id } = useParams() as { id: string };
  const router = useRouter();
  const [state, setState] = useState("UPLOADED");
  const [error, setError] = useState("");

  useEffect(() => {
    const t = setInterval(async () => {
      const r = await fetch(`${API}/api/status/${id}`);
      const j = await r.json();
      setState(j.state);
      setError(j.error ?? "");
      if (j.state === "COMPLETED") {
        clearInterval(t);
        router.push(`/dashboard/${id}`);
      }
    }, 1500);
    return () => clearInterval(t);
  }, [id, router]);

  async function retry() {
    await fetch(`${API}/api/retry/${id}`, { method: "POST" });
    setError("");
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Processing…</h1>
      <ol className="space-y-2">
        {STEPS.map((s) => (
          <li
            key={s}
            className={`p-3 rounded-xl ${state === s ? "bg-blue-700 text-white" : "bg-white"}`}
          >
            {s} {state === s && "← current"}
          </li>
        ))}
      </ol>
      {state === "ERROR HANDLING" && (
        <div className="mt-4 bg-red-50 border border-red-200 p-4 rounded-xl">
          <p className="text-sm text-red-700 whitespace-pre-wrap">{error}</p>
          <div className="flex gap-2 mt-3">
            <button onClick={retry} className="bg-red-600 text-white px-4 py-2 rounded-xl text-sm">
              Retry
            </button>
            <button
              onClick={() => router.push("/upload")}
              className="border px-4 py-2 rounded-xl text-sm"
            >
              Return to idle
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
