"""Image loading helpers for question-linked image ZIP archives."""

from __future__ import annotations

import base64
import mimetypes
import os
import zipfile
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple


@dataclass(frozen=True)
class _ImageEntry:
    archive_path: str
    stem: str
    question_id: str
    mime_type: str
    data_url: str


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
                b64_data = base64.b64encode(zf.read(name)).decode("ascii")
                data_url = f"data:{mime_type};base64,{b64_data}"
                entries.append(_ImageEntry(
                    archive_path=name,
                    stem=stem,
                    question_id=question_id,
                    mime_type=mime_type,
                    data_url=data_url,
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


def _extract_question_id(stem: str) -> str:
    # expected pattern in sample: img_<question_id>_<index>
    if stem.startswith("img_"):
        remainder = stem[4:]
        if "_" in remainder:
            return remainder.rsplit("_", 1)[0]
    return ""
