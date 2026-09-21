"use client";
import { useState } from "react";
import { useRouter } from "next/navigation";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";
const MODS = ["dwi", "adc", "flair"] as const;

export default function UploadPage() {
  const [files, setFiles] = useState<Record<string, File | null>>({ dwi: null, adc: null, flair: null });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const router = useRouter();

  async function start() {
    setError("");
    if (!files.dwi || !files.adc || !files.flair) {
      setError("Please attach DWI, ADC and FLAIR (.nii.gz) files.");
      return;
    }
    setBusy(true);
    try {
      const fd = new FormData();
      fd.append("dwi", files.dwi);
      fd.append("adc", files.adc);
      fd.append("flair", files.flair);
      const up = await fetch(`${API}/api/upload`, { method: "POST", body: fd });
      if (!up.ok) {
        const j = await up.json().catch(() => ({}));
        throw new Error(JSON.stringify(j.detail ?? j));
      }
      const { study_id } = await up.json();
      await fetch(`${API}/api/analyze/${study_id}`, { method: "POST" });
      router.push(`/processing/${study_id}`);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div>
      <h1 className="text-2xl font-bold mb-4">Upload MRI Study</h1>
      <div className="grid md:grid-cols-3 gap-4">
        {MODS.map((m) => (
          <label key={m} className="bg-white rounded-2xl shadow p-6 block cursor-pointer">
            <p className="uppercase text-xs text-slate-500">{m}</p>
            <p className="mt-2 text-sm">{files[m]?.name ?? "Drop .nii.gz here"}</p>
            <p className="text-xs mt-2 text-green-600">
              {files[m] ? "✓ attached" : "pending"}
            </p>
            <input
              type="file"
              accept=".nii.gz,.nii"
              className="hidden"
              onChange={(e) =>
                setFiles((f) => ({ ...f, [m]: e.target.files?.[0] ?? null }))
              }
            />
          </label>
        ))}
      </div>
      {error && <p className="text-red-600 text-sm mt-4">{error}</p>}
      <button
        onClick={start}
        disabled={busy}
        className="mt-6 bg-blue-700 text-white px-6 py-3 rounded-xl disabled:opacity-50"
      >
        {busy ? "Uploading…" : "Start analysis"}
      </button>
    </div>
  );
}
