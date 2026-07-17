import argparse
import json
import os
import sqlite3
from collections import defaultdict
from pathlib import Path


def _sanitize_filename(name: str) -> str:
    illegal_chars = r'<>:"/\|?*'
    for ch in illegal_chars:
        name = name.replace(ch, "_")
    return name.strip().strip(".")


def _build_doc_text(pages: dict[int, list[str]]) -> str:
    page_texts = []
    for page_num in sorted(pages.keys()):
        chunks = [c.strip() for c in pages[page_num] if isinstance(c, str) and c.strip()]
        if not chunks:
            continue
        page_body = "\n\n".join(chunks)
        page_texts.append(f"Page {page_num}:\n{page_body}")
    return "\n\n--- Page Break ---\n\n".join(page_texts).strip()


def export_texts(db_path: str, output_dir: str, only_pdf: bool = True) -> dict:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    if only_pdf:
        cursor.execute(
            """
            SELECT doc_name, page_num, chunk_text, chunk_id
            FROM chunks
            WHERE lower(doc_name) LIKE '%.pdf'
            ORDER BY doc_name ASC, page_num ASC, chunk_id ASC
            """
        )
    else:
        cursor.execute(
            """
            SELECT doc_name, page_num, chunk_text, chunk_id
            FROM chunks
            ORDER BY doc_name ASC, page_num ASC, chunk_id ASC
            """
        )

    doc_pages: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
    for doc_name, page_num, chunk_text, _chunk_id in cursor.fetchall():
        if not isinstance(doc_name, str) or not doc_name.strip():
            continue
        if page_num is None:
            page_num = -1
        try:
            page_num = int(page_num)
        except Exception:
            page_num = -1
        doc_pages[doc_name][page_num].append(chunk_text or "")

    written = 0
    skipped = 0
    for doc_name, pages in doc_pages.items():
        full_text = _build_doc_text(pages)
        if not full_text:
            skipped += 1
            continue
        stem = Path(doc_name).stem
        out_name = _sanitize_filename(f"{stem}.txt")
        if not out_name:
            skipped += 1
            continue
        out_path = out_dir / out_name
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(full_text)
        written += 1

    conn.close()
    return {"written": written, "skipped": skipped, "output_dir": str(out_dir)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", default="./knowledge.db")
    parser.add_argument("--out", default="./extracted_text")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    result = export_texts(args.db, args.out, only_pdf=not args.all)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()

