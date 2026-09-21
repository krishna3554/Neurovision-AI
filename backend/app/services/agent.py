"""Neuro-Agent with retrieval (FR-8).

[REPORT] conversational, retrieval-based ('RAG Active' badge), structured
explanations from lesion characteristics + contextual medical knowledge.
"""
import glob
import os

SYSTEM = (
    "You are NeuroAssist, an educational assistant for brain MRI stroke analysis. "
    "Use ONLY the supplied findings and context. Give structured answers (Finding, "
    "Anatomical context, Why it matters, Limitations). You are not a doctor; findings "
    "need review by a clinician. Do not give a diagnosis or treatment orders."
)


class NeuroAgent:
    def __init__(self, kb_dir="backend/knowledge", lazy=True):
        self.kb_dir = kb_dir
        self.chunks = []
        for p in glob.glob(f"{kb_dir}/*.md"):
            with open(p, encoding="utf-8") as f:
                self.chunks += [c.strip() for c in f.read().split("\n\n") if c.strip()]
        self._emb = None
        self._index = None
        self._client = None
        if not lazy:
            self._ensure_index()

    def _ensure_index(self):
        if self._index is not None:
            return
        import numpy as np
        from sentence_transformers import SentenceTransformer
        import faiss

        self._emb = SentenceTransformer("all-MiniLM-L6-v2")
        X = self._emb.encode(self.chunks, normalize_embeddings=True).astype("float32")
        self._index = faiss.IndexFlatIP(X.shape[1])
        self._index.add(X)

    def retrieve(self, q, k=4):
        if not self.chunks:
            return []
        try:
            self._ensure_index()
        except Exception:
            return self.chunks[:k]  # offline fallback: keyword-agnostic prefix
        import numpy as np

        qv = self._emb.encode([q], normalize_embeddings=True).astype("float32")
        _, I = self._index.search(qv, min(k, len(self.chunks)))
        return [self.chunks[i] for i in I[0] if 0 <= i < len(self.chunks)]

    def _findings_text(self, result: dict) -> str:
        regions = result.get("regions", [])
        top = regions[0].get("region", "unknown") if regions else "unknown"
        return (
            f"Region(s): {regions}\nHemisphere: {result.get('hemisphere', '?')} "
            f"({result.get('territory', '')})\n"
            f"Lesion volume: {result.get('volume_cm3', 0):.1f} cm3\n"
            f"Model confidence: {result.get('confidence', 0):.3f}\n"
            f"Top region: {top}"
        )

    def fallback_answer(self, result: dict, question: str) -> str:
        ctx = "\n---\n".join(self.chunks[:2]) if self.chunks else ""
        return (
            f"**Finding:** {self._findings_text(result)}\n\n"
            f"**Anatomical context:** {ctx[:600]}\n\n"
            f"**Why it matters:** Lesion location and volume inform downstream "
            f"clinical review; correlate with the neurological exam.\n\n"
            f"**Limitations:** Educational output only (offline mode — no LLM key). "
            f"Final diagnosis rests with the attending physician.\n\n"
            f"(Question was: {question})"
        )

    def generate(self, result: dict, question: str, history=()) -> str:
        from backend.app.core.config import settings

        if not settings.llm_api_key or not os.environ.get("LLM_API_KEY", settings.llm_api_key):
            return self.fallback_answer(result, question)
        import anthropic

        regions = result.get("regions", [])
        seed = f"{question} {(regions[0]['region'] if regions else '')}"
        ctx = "\n---\n".join(self.retrieve(seed))
        findings = self._findings_text(result)
        msgs = list(history) + [
            {"role": "user", "content": f"FINDINGS:\n{findings}\n\nCONTEXT:\n{ctx}\n\nQUESTION: {question}"}
        ]
        client = anthropic.Anthropic(api_key=settings.llm_api_key)
        r = client.messages.create(
            model=settings.llm_model, max_tokens=800, system=SYSTEM, messages=msgs
        )
        return r.content[0].text
