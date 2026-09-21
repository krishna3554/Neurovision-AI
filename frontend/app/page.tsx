import Link from "next/link";

export default function Landing() {
  return (
    <div className="grid md:grid-cols-2 gap-10 items-center">
      <div>
        <span className="text-xs uppercase tracking-widest text-blue-700 bg-blue-100 px-3 py-1 rounded-full">
          Clinical grade AI analysis
        </span>
        <h1 className="text-4xl font-bold mt-4">
          AI-Powered Stroke Detection &amp;{" "}
          <span className="text-blue-700">Neuro Analysis Platform</span>
        </h1>
        <p className="text-slate-600 mt-4">
          Advanced 3D MRI segmentation and intelligent neuro-assistance powered by deep
          learning, atlas localization, and retrieval-based clinical reasoning.
        </p>
        <Link
          href="/upload"
          className="inline-block mt-6 bg-blue-700 text-white px-6 py-3 rounded-xl"
        >
          Detect Stroke →
        </Link>
      </div>
      <div className="bg-[#0b1220] rounded-2xl p-8 text-cyan-200 relative min-h-[320px]">
        <p className="text-sm text-slate-400">3D brain preview</p>
        <div className="text-6xl mt-10">🧠</div>
        <span className="absolute top-6 right-6 text-xs bg-cyan-900 px-3 py-1 rounded-full">
          Validation Dice 80.6%
        </span>
        <span className="absolute bottom-6 right-6 text-xs bg-cyan-900 px-3 py-1 rounded-full">
          AI Agent: RAG Active
        </span>
      </div>
    </div>
  );
}
