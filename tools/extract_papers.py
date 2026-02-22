#!/usr/bin/env python3
"""
extract_papers.py — извлечение текста из PDF-статей Franz German.

Требует: poppler-utils (pdftotext) или pymupdf (pip install pymupdf).

Использование:
    python3 tools/extract_papers.py [--out docs/papers/]
"""
import os
import sys
import subprocess
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_DIR = ROOT
OUT_DIR = ROOT / "docs" / "papers"


def extract_with_pdftotext(pdf_path: Path, out_path: Path) -> bool:
    """Извлечь текст через pdftotext (poppler-utils)."""
    try:
        result = subprocess.run(
            ["pdftotext", "-layout", str(pdf_path), str(out_path)],
            capture_output=True, timeout=30
        )
        return result.returncode == 0 and out_path.exists()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def extract_with_pymupdf(pdf_path: Path, out_path: Path) -> bool:
    """Извлечь текст через pymupdf (pip install pymupdf)."""
    try:
        import fitz  # pymupdf
        doc = fitz.open(str(pdf_path))
        text = ""
        for page in doc:
            text += page.get_text()
        out_path.write_text(text, encoding="utf-8")
        doc.close()
        return bool(text.strip())
    except ImportError:
        return False


def guess_title(text: str, filename: str) -> str:
    """Попытаться извлечь заголовок из первых строк текста."""
    lines = [l.strip() for l in text.split("\n") if l.strip()]
    # Первая непустая строка длиной 10-150 символов
    for line in lines[:10]:
        if 10 < len(line) < 150 and not line.startswith("http"):
            return line
    return filename


def main():
    parser = argparse.ArgumentParser(description="Извлечь текст из PDF-статей")
    parser.add_argument("--out", default=str(OUT_DIR), help="Каталог для .txt файлов")
    parser.add_argument("--pdf-dir", default=str(PDF_DIR), help="Каталог с PDF")
    parser.add_argument("--method", choices=["auto", "pdftotext", "pymupdf"], default="auto")
    args = parser.parse_args()

    out_dir = Path(args.out)
    pdf_dir = Path(args.pdf_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(pdf_dir.glob("2cb3bf_*.pdf"))
    if not pdfs:
        print(f"PDF-файлы не найдены в {pdf_dir}")
        sys.exit(1)

    print(f"Найдено {len(pdfs)} PDF-файлов")
    print(f"Вывод: {out_dir}")
    print()

    success = 0
    index = []

    for i, pdf in enumerate(pdfs, 1):
        out_txt = out_dir / (pdf.stem + ".txt")
        print(f"[{i:02d}/{len(pdfs)}] {pdf.name} ...", end=" ")

        ok = False
        if args.method in ("auto", "pdftotext"):
            ok = extract_with_pdftotext(pdf, out_txt)
        if not ok and args.method in ("auto", "pymupdf"):
            ok = extract_with_pymupdf(pdf, out_txt)

        if ok:
            text = out_txt.read_text(encoding="utf-8", errors="replace")
            title = guess_title(text, pdf.name)
            words = len(text.split())
            print(f"OK  ({words} слов)")
            index.append((i, pdf.name, title, words))
            success += 1
        else:
            print("FAIL")
            index.append((i, pdf.name, "(не извлечено)", 0))

    # Обновить индекс
    index_md = out_dir / "index.md"
    with open(index_md, "w", encoding="utf-8") as f:
        f.write("# Извлечённые статьи\n\n")
        f.write(f"Обработано: {success}/{len(pdfs)}\n\n")
        f.write("| № | Файл | Заголовок | Слов |\n")
        f.write("|---|------|-----------|------|\n")
        for num, fname, title, words in index:
            f.write(f"| {num:02d} | `{fname}` | {title[:60]} | {words:,} |\n")

    print()
    print(f"Готово: {success}/{len(pdfs)} извлечено")
    print(f"Индекс: {index_md}")


if __name__ == "__main__":
    main()
