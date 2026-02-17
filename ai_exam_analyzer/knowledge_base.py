"""Knowledge-base ingestion and retrieval for question evidence."""

from __future__ import annotations

import json
import math
import re
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

_TOKEN_RE = re.compile(r"[A-Za-zÄÖÜäöüß0-9]{2,}")


@dataclass
class Chunk:
    chunk_id: str
    source: str
    page: int
    text: str
    tokens: set[str]


@dataclass
class KnowledgeImage:
    image_id: str
    source: str
    page: int
    perceptual_hash: str


class KnowledgeBase:
    def __init__(self, chunks: List[Chunk], images: Optional[List[KnowledgeImage]] = None):
        self.chunks = chunks
        self.images = images or []

    def retrieve(
        self,
        query_text: str,
        *,
        top_k: int,
        min_score: float,
        max_chars: int,
    ) -> Tuple[List[Dict[str, Any]], float]:
        q_tokens = _tokenize(query_text)
        if not q_tokens:
            return [], 0.0

        scored: List[Tuple[float, Chunk]] = []
        for chunk in self.chunks:
            overlap = len(q_tokens & chunk.tokens)
            if overlap == 0:
                continue
            score = overlap / math.sqrt(len(q_tokens) * max(1, len(chunk.tokens)))
            if score >= min_score:
                scored.append((score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)

        selected: List[Dict[str, Any]] = []
        used_chars = 0
        total_score = 0.0

        for score, chunk in scored[: max(1, top_k * 3)]:
            snippet = chunk.text.strip()
            if not snippet:
                continue
            remaining = max_chars - used_chars
            if remaining <= 0:
                break
            if len(snippet) > remaining:
                if remaining < 220:
                    break
                snippet = snippet[:remaining].rsplit(" ", 1)[0] + " …"
            selected.append(
                {
                    "chunkId": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": round(score, 4),
                    "text": snippet,
                }
            )
            used_chars += len(snippet)
            total_score += score
            if len(selected) >= top_k:
                break

        retrieval_quality = round(total_score / max(1, len(selected)), 4) if selected else 0.0
        return selected, retrieval_quality

    def find_similar_images(self, perceptual_hash: str, *, max_hamming_distance: int) -> List[Dict[str, Any]]:
        hits: List[Tuple[int, KnowledgeImage]] = []
        for img in self.images:
            dist = _hamming_distance_hex(perceptual_hash, img.perceptual_hash)
            if dist <= max_hamming_distance:
                hits.append((dist, img))
        hits.sort(key=lambda row: row[0])
        return [
            {
                "imageId": img.image_id,
                "source": img.source,
                "page": img.page,
                "hammingDistance": dist,
            }
            for dist, img in hits[:8]
        ]


def _tokenize(text: str) -> set[str]:
    return {m.group(0).lower() for m in _TOKEN_RE.finditer(text or "")}


def _chunk_text(text: str, *, max_chars: int) -> List[str]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]
    if not paragraphs:
        paragraphs = [line.strip() for line in (text or "").splitlines() if line.strip()]

    chunks: List[str] = []
    buf = ""
    for part in paragraphs:
        if len(part) > max_chars:
            # hard split for very long paragraph
            for i in range(0, len(part), max_chars):
                segment = part[i : i + max_chars].strip()
                if segment:
                    if buf:
                        chunks.append(buf)
                        buf = ""
                    chunks.append(segment)
            continue

        candidate = f"{buf}\n\n{part}".strip() if buf else part
        if len(candidate) <= max_chars:
            buf = candidate
        else:
            if buf:
                chunks.append(buf)
            buf = part

    if buf:
        chunks.append(buf)

    return chunks


def _extract_pdf_chunks_from_bytes(raw_pdf: bytes, source_name: str, max_chunk_chars: int) -> List[Chunk]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Knowledge ZIP contains PDFs, but dependency `pypdf` is missing. Install with: pip install pypdf"
        ) from exc

    from io import BytesIO

    reader = PdfReader(BytesIO(raw_pdf))
    result: List[Chunk] = []

    for p_idx, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue

        for c_idx, chunk_text in enumerate(_chunk_text(text, max_chars=max_chunk_chars), start=1):
            chunk_id = f"{source_name}#p{p_idx}c{c_idx}"
            tokens = _tokenize(chunk_text)
            if not tokens:
                continue
            result.append(Chunk(chunk_id=chunk_id, source=source_name, page=p_idx, text=chunk_text, tokens=tokens))

    return result


def _extract_pdf_images_from_bytes(raw_pdf: bytes, source_name: str) -> List[KnowledgeImage]:
    try:
        from pypdf import PdfReader  # type: ignore
    except ModuleNotFoundError:
        return []

    reader = PdfReader(BytesIO(raw_pdf))
    result: List[KnowledgeImage] = []
    for p_idx, page in enumerate(reader.pages, start=1):
        images = getattr(page, "images", None)
        if not images:
            continue
        for i, image in enumerate(images, start=1):
            raw = getattr(image, "data", b"")
            if not raw:
                continue
            result.append(KnowledgeImage(
                image_id=f"{source_name}#p{p_idx}i{i}",
                source=source_name,
                page=p_idx,
                perceptual_hash=_compute_perceptual_hash(raw),
            ))
    return result


def build_knowledge_base_from_zip(
    zip_path: str,
    *,
    max_chunk_chars: int,
    subject_hint: Optional[str] = None,
) -> KnowledgeBase:
    zpath = Path(zip_path)
    if not zpath.exists():
        raise FileNotFoundError(f"Knowledge ZIP not found: {zip_path}")

    subject_tokens = _tokenize(subject_hint or "")
    chunks: List[Chunk] = []
    images: List[KnowledgeImage] = []

    with zipfile.ZipFile(zpath, "r") as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename
            lower = name.lower()
            if not (lower.endswith(".pdf") or lower.endswith(".txt") or lower.endswith(".md")):
                continue

            basename = Path(name).name
            if subject_tokens:
                name_tokens = _tokenize(name)
                if subject_tokens.isdisjoint(name_tokens):
                    # keep file, but with lower priority by allowing all; do not skip hard
                    pass

            raw = zf.read(info)

            if lower.endswith(".pdf"):
                chunks.extend(_extract_pdf_chunks_from_bytes(raw, basename, max_chunk_chars))
                images.extend(_extract_pdf_images_from_bytes(raw, basename))
            else:
                text = raw.decode("utf-8", errors="ignore")
                for i, chunk_text in enumerate(_chunk_text(text, max_chars=max_chunk_chars), start=1):
                    chunk_id = f"{basename}#t{i}"
                    tokens = _tokenize(chunk_text)
                    if tokens:
                        chunks.append(Chunk(chunk_id=chunk_id, source=basename, page=0, text=chunk_text, tokens=tokens))

    if not chunks:
        raise RuntimeError("No extractable knowledge chunks found in ZIP (supported: PDF/TXT/MD).")

    return KnowledgeBase(chunks, images=images)


def save_index_json(path: str, kb: KnowledgeBase) -> None:
    serializable = [
        {
            "chunkId": c.chunk_id,
            "source": c.source,
            "page": c.page,
            "text": c.text,
            "tokens": sorted(c.tokens),
        }
        for c in kb.chunks
    ]
    payload = {
        "chunks": serializable,
        "images": [
            {
                "imageId": img.image_id,
                "source": img.source,
                "page": img.page,
                "perceptualHash": img.perceptual_hash,
            }
            for img in kb.images
        ],
    }
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_index_json(path: str) -> KnowledgeBase:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        data = {"chunks": data, "images": []}
    chunks: List[Chunk] = []
    for row in data.get("chunks", []):
        chunks.append(
            Chunk(
                chunk_id=row["chunkId"],
                source=row.get("source", "unknown"),
                page=int(row.get("page", 0)),
                text=row.get("text", ""),
                tokens=set(row.get("tokens") or _tokenize(row.get("text", ""))),
            )
        )
    images = [
        KnowledgeImage(
            image_id=row.get("imageId", ""),
            source=row.get("source", "unknown"),
            page=int(row.get("page", 0)),
            perceptual_hash=row.get("perceptualHash", "0" * 16),
        )
        for row in data.get("images", [])
    ]
    return KnowledgeBase(chunks, images=images)


def build_query_text(payload: Dict[str, Any]) -> str:
    parts = [payload.get("questionText", ""), payload.get("explanationText", "")]
    for a in payload.get("answers", []):
        parts.append(a.get("text", ""))
    return "\n".join(x for x in parts if x)


def _compute_perceptual_hash(raw: bytes) -> str:
    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError:
        return raw[:8].hex().ljust(16, "0")
    try:
        with Image.open(BytesIO(raw)) as img:
            img = img.convert("L").resize((9, 8))
            px = list(img.getdata())
    except Exception:
        return raw[:8].hex().ljust(16, "0")

    bits = []
    for y in range(8):
        for x in range(8):
            bits.append("1" if px[y * 9 + x] > px[y * 9 + x + 1] else "0")
    return f"{int(''.join(bits), 2):016x}"


def _hamming_distance_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 64
