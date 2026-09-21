"use client";
import { useEffect, useState } from "react";
import { useParams } from "next/navigation";
import BrainViewer3D, { Mesh } from "../../../components/BrainViewer3D";
import ChatPanel from "../../../components/ChatPanel";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export default function DashboardPage() {
  const { id } = useParams() as { id: string };
  const [results, setResults] = useState<any>(null);
  const [meshes, setMeshes] = useState<{ brain: Mesh; lesion: Mesh } | null>(null);

  useEffect(() => {
    (async () => {
      const r = await fetch(`${API}/api/results/${id}`);
      if (r.ok) setResults(await r.json());
      const m = await fetch(`${API}/api/mesh/${id}`);
      if (m.ok) setMeshes(await m.json());
    })();
  }, [id]);

  if (!results) return <p>Loading results…</p>;

  return (
    <div>
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h1 className="text-2xl font-bold">Neural Insights Dashboard</h1>
        <div className="flex gap-2 text-xs">
          <span className="bg-green-100 text-green-700 px-3 py-1 rounded-full">ACTIVE ANALYSIS</span>
          <span className="bg-slate-200 px-3 py-1 rounded-full">Patient ID: {id}</span>
          <button className="border px-3 py-1 rounded-full">Export Data</button>
          <button className="border px-3 py-1 rounded-full">Share Findings</button>
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4 mt-6">
        <div className="bg-white rounded-2xl shadow p-4">
          <div className="flex justify-between text-xs text-slate-500 mb-2">
            <h3 className="font-semibold text-slate-800">2D MRI Detection Views</h3>
            <span>MODEL OVERLAY</span>
          </div>
          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-900 text-white rounded-xl p-2">
              <p>Top View — Slice {results.slice_z} | Max P {Number(results.max_prob).toFixed(3)}</p>
              <div className="h-40 flex items-center justify-center text-pink-400">[FLAIR + lesion overlay]</div>
              <p>{results.slice_pixels} highlighted lesion pixels</p>
            </div>
            <div className="bg-slate-900 text-white rounded-xl p-2">
              <p>Side View</p>
              <div className="h-40 flex items-center justify-center text-pink-400">[DWI + lesion overlay]</div>
              <p>{results.slice_pixels} highlighted lesion pixels</p>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-2xl shadow p-4">
          <h3 className="font-semibold mb-1">Volumetric Brain Scan</h3>
          <span className="text-xs bg-red-100 text-red-600 px-2 py-1 rounded-full">
            Lesion Detected (Ischemic)
          </span>
          <div className="h-64 mt-2">
            {meshes ? (
              <BrainViewer3D brain={meshes.brain} lesion={meshes.lesion} />
            ) : (
              <p className="text-sm">Loading 3D mesh…</p>
            )}
          </div>
          <p className="text-xs text-slate-500 mt-1">Slice Position Z: 42.5mm · drag to rotate, scroll to zoom</p>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-4">
        <div className="bg-white rounded-2xl shadow p-4">
          <p className="text-xs text-slate-500">Dice Confidence*</p>
          <p className="text-2xl font-bold">{(results.confidence * 100).toFixed(1)}%</p>
          <div className="h-2 bg-slate-200 rounded mt-2">
            <div className="h-2 bg-blue-700 rounded" style={{ width: `${results.confidence * 100}%` }} />
          </div>
        </div>
        <div className="bg-white rounded-2xl shadow p-4">
          <p className="text-xs text-slate-500">Lesion Volume</p>
          <p className="text-2xl font-bold">{Number(results.volume_cm3).toFixed(1)} cm³</p>
          <span className="text-xs bg-amber-100 px-2 py-0.5 rounded-full">{results.severity?.label}</span>
        </div>
        <div className="bg-white rounded-2xl shadow p-4">
          <p className="text-xs text-slate-500">Hemisphere</p>
          <p className="text-2xl font-bold">{results.hemisphere} {results.territory}</p>
          <p className="text-xs">Territory Involved</p>
        </div>
        <div className="bg-white rounded-2xl shadow p-4 border-l-4 border-red-500">
          <p className="text-xs text-slate-500">Severity Estimate*</p>
          <p className="text-2xl font-bold">Level {results.severity?.level}</p>
          <p className="text-xs text-red-600">{results.severity?.note}</p>
        </div>
      </div>
      <p className="text-[11px] text-slate-500 mt-2">
        *Confidence = mean predicted probability in mask (not a Dice score — Dice needs ground
        truth). Severity = unvalidated volume-band heuristic, supplementary only.
      </p>

      {results.regions?.length > 0 && (
        <div className="bg-white rounded-2xl shadow p-4 mt-4 text-sm">
          <h3 className="font-semibold mb-2">Atlas regions</h3>
          <ul className="list-disc ml-5">
            {results.regions.map((r: any) => (
              <li key={r.region}>{r.region} — {r.overlap_pct}%</li>
            ))}
          </ul>
        </div>
      )}

      <div className="mt-4">
        <ChatPanel studyId={id} opening={results.opening ?? "Analysis complete."} />
      </div>
    </div>
  );
}
