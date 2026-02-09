#!/usr/bin/env python3
"""Export `REPORT.md` to a clean PDF.

This script prefers a Chrome/Chromium-based export, because it doesn't require
LaTeX and is widely available on macOS.

It works in two stages:
1) Convert REPORT.md -> REPORT.html using Python Markdown.
2) Print REPORT.html -> REPORT.pdf via a headless browser (Chrome/Chromium).

If no supported browser is found, the script will still generate REPORT.html and
tell you how to print it to PDF manually.

Usage:
    python3 scripts/export_report_pdf.py

Outputs:
    Task0/REPORT.html
    Task0/REPORT.pdf (if a supported browser is found)
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _find_browser() -> list[str] | None:
    """Return argv prefix for a PDF-capable headless browser, or None."""
    # macOS app bundle paths first
    chrome_app = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if chrome_app.exists():
        return [str(chrome_app)]

    chromium_app = Path("/Applications/Chromium.app/Contents/MacOS/Chromium")
    if chromium_app.exists():
        return [str(chromium_app)]

    # PATH fallbacks
    for name in ("google-chrome", "chrome", "chromium", "chromium-browser"):
        p = shutil.which(name)
        if p:
            return [p]

    return None


def _md_to_html(md: str) -> str:
    try:
        import markdown  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "Missing dependency 'markdown'. Install it with: pip install markdown"
        ) from e

    body = markdown.markdown(
        md,
        extensions=[
            "extra",
            "tables",
            "toc",
            "sane_lists",
        ],
        output_format="html5",
    )

    # KaTeX isn't required for PDF export; equations render as plain text.
    return body


def main() -> None:
    task0_dir = Path(__file__).resolve().parents[1]
    md_path = task0_dir / "REPORT.md"
    html_tpl_path = task0_dir / "report_template.html"

    if not md_path.exists():
        raise FileNotFoundError(f"Could not find {md_path}")
    if not html_tpl_path.exists():
        raise FileNotFoundError(f"Could not find {html_tpl_path}")

    md = _read_text(md_path)
    body_html = _md_to_html(md)

    html_tpl = _read_text(html_tpl_path)
    out_html = html_tpl.replace("{{CONTENT}}", body_html)

    html_out_path = task0_dir / "REPORT.html"
    _write_text(html_out_path, out_html)

    # Attempt PDF export via headless browser
    browser = _find_browser()
    pdf_out_path = task0_dir / "REPORT.pdf"

    if not browser:
        print("[INFO] No Chrome/Chromium browser found for automated PDF export.")
        print(f"[OK] Generated HTML: {html_out_path}")
        print("\nManual fallback:")
        print("- Open REPORT.html in a browser")
        print("- File -> Print -> Save as PDF")
        return

    html_file_url = html_out_path.resolve().as_uri()

    # Chrome/Chromium headless printing
    cmd = (
        browser
        + [
            "--headless=new",
            "--disable-gpu",
            f"--print-to-pdf={str(pdf_out_path)}",
            "--no-margins",
            html_file_url,
        ]
    )

    # Some Chrome versions don't support --headless=new; try fallback.
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        cmd2 = (
            browser
            + [
                "--headless",
                "--disable-gpu",
                f"--print-to-pdf={str(pdf_out_path)}",
                "--no-margins",
                html_file_url,
            ]
        )
        subprocess.run(cmd2, check=True)

    print(f"[OK] Generated HTML: {html_out_path}")
    print(f"[OK] Generated PDF : {pdf_out_path}")


if __name__ == "__main__":
    main()
