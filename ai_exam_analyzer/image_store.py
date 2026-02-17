"""Image loading helpers for question-linked image ZIP archives."""

from __future__ import annotations

import base64
import binascii
import mimetypes
import os
import zipfile
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class _ImageEntry:
    archive_path: str
    stem: str
    question_id: str
    mime_type: str
    data_url: str
    perceptual_hash: str


class QuestionImageStore:
    """In-memory index of question images from a ZIP file."""

    def __init__(self, zip_path: str, entries: List[_ImageEntry]):
        self.zip_path = zip_path
        self._entries = entries
        self._by_stem: Dict[str, _ImageEntry] = {entry.stem: entry for entry in entries}
        self._by_question_id: Dict[str, List[_ImageEntry]] = {}
        for entry in entries:
            self._by_question_id.setdefault(entry.question_id, []).append(entry)

        for values in self._by_question_id.values():
            values.sort(key=lambda item: item.stem)

    @classmethod
    def from_zip(cls, zip_path: str) -> "QuestionImageStore":
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Images ZIP not found: {zip_path}")

        entries: List[_ImageEntry] = []
        with zipfile.ZipFile(zip_path) as zf:
            for name in zf.namelist():
                if name.endswith("/"):
                    continue

                filename = os.path.basename(name)
                stem, ext = os.path.splitext(filename)
                if not ext:
                    continue

                question_id = _extract_question_id(stem)
                if not question_id:
                    continue

                mime_type = mimetypes.types_map.get(ext.lower(), "application/octet-stream")
                raw_bytes = zf.read(name)
                b64_data = base64.b64encode(raw_bytes).decode("ascii")
                data_url = f"data:{mime_type};base64,{b64_data}"
                perceptual_hash = _compute_perceptual_hash(raw_bytes)
                entries.append(_ImageEntry(
                    archive_path=name,
                    stem=stem,
                    question_id=question_id,
                    mime_type=mime_type,
                    data_url=data_url,
                    perceptual_hash=perceptual_hash,
                ))

        return cls(zip_path=zip_path, entries=entries)

    def prepare_question_images(self, q: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        question_id = str(q.get("id") or "").strip()
        expected_refs = [str(ref).strip() for ref in (q.get("imageFiles") or []) if str(ref).strip()]

        selected: Dict[str, _ImageEntry] = {}
        missing_expected_refs: List[str] = []

        for ref in expected_refs:
            entry = self._by_stem.get(ref)
            if entry is None:
                missing_expected_refs.append(ref)
                continue
            selected[entry.archive_path] = entry

        if question_id:
            for entry in self._by_question_id.get(question_id, []):
                selected.setdefault(entry.archive_path, entry)

        ordered_entries = sorted(selected.values(), key=lambda item: item.stem)
        input_images = [{"type": "input_image", "image_url": entry.data_url} for entry in ordered_entries]

        payload_image_context = {
            "questionHasImageReference": bool(expected_refs or q.get("imageUrls")),
            "imageZipConfigured": True,
            "imageZipPath": self.zip_path,
            "expectedImageRefs": expected_refs,
            "missingExpectedImageRefs": missing_expected_refs,
            "providedImageCount": len(ordered_entries),
            "providedImageRefs": [entry.stem for entry in ordered_entries],
            "providedImageArchivePaths": [entry.archive_path for entry in ordered_entries],
        }

        return input_images, payload_image_context

    def build_image_clusters(self, questions: List[Dict[str, Any]], max_hamming_distance: int = 8) -> Dict[str, Any]:
        question_to_images: Dict[str, List[_ImageEntry]] = {}
        for q in questions:
            qid = str(q.get("id") or "").strip()
            if not qid:
                continue
            question_to_images[qid] = list(self._by_question_id.get(qid, []))

        cluster_counter = 1
        clusters: List[Dict[str, Any]] = []
        question_cluster_refs: Dict[str, List[str]] = {}

        seen: set[str] = set()
        for qid, entries in question_to_images.items():
            assigned: List[str] = []
            for entry in entries:
                if entry.archive_path in seen:
                    continue
                similar = [e for e in self._entries if _hamming_distance_hex(entry.perceptual_hash, e.perceptual_hash) <= max_hamming_distance]
                cluster_id = f"img-cluster-{cluster_counter}"
                cluster_counter += 1
                clusters.append({
                    "clusterId": cluster_id,
                    "representativeRef": entry.stem,
                    "members": [e.stem for e in similar],
                    "memberArchivePaths": [e.archive_path for e in similar],
                })
                for sim in similar:
                    seen.add(sim.archive_path)
                    question_cluster_refs.setdefault(sim.question_id, []).append(cluster_id)
                assigned.append(cluster_id)
            question_cluster_refs.setdefault(qid, assigned)

        return {
            "clusters": clusters,
            "questionToClusters": question_cluster_refs,
        }

    def match_knowledge_images(self, questions: List[Dict[str, Any]], knowledge_base: Any, max_hamming_distance: int = 10) -> Dict[str, Any]:
        if not hasattr(knowledge_base, "find_similar_images"):
            return {}

        out: Dict[str, Any] = {}
        for q in questions:
            qid = str(q.get("id") or "").strip()
            if not qid:
                continue
            matches: List[Dict[str, Any]] = []
            for entry in self._by_question_id.get(qid, []):
                for hit in knowledge_base.find_similar_images(entry.perceptual_hash, max_hamming_distance=max_hamming_distance):
                    matches.append({
                        "questionImageRef": entry.stem,
                        "questionImageArchivePath": entry.archive_path,
                        "knowledgeImageId": hit.get("imageId"),
                        "knowledgeSource": hit.get("source"),
                        "knowledgePage": hit.get("page"),
                        "hammingDistance": hit.get("hammingDistance"),
                    })
            out[qid] = matches
        return out


def _extract_question_id(stem: str) -> str:
    # expected pattern in sample: img_<question_id>_<index>
    if stem.startswith("img_"):
        remainder = stem[4:]
        if "_" in remainder:
            return remainder.rsplit("_", 1)[0]
    return ""


def _compute_perceptual_hash(raw: bytes) -> str:
    try:
        from PIL import Image  # type: ignore
    except ModuleNotFoundError:
        return binascii.hexlify(raw[:8]).decode("ascii").ljust(16, "0")

    try:
        with Image.open(BytesIO(raw)) as img:
            img = img.convert("L").resize((9, 8))
            px = list(img.getdata())
    except Exception:
        return binascii.hexlify(raw[:8]).decode("ascii").ljust(16, "0")

    bits = []
    for y in range(8):
        for x in range(8):
            left = px[y * 9 + x]
            right = px[y * 9 + x + 1]
            bits.append("1" if left > right else "0")

    return f"{int(''.join(bits), 2):016x}"


def _hamming_distance_hex(a: str, b: str) -> int:
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except Exception:
        return 64
