"""Build the real corpus: State Bank of Pakistan BPRD circulars (2024).

For each circular the script:
  1. fetches the circular's page on sbp.org.pk and saves the circular text as
     data/docs/<id>.md  (title, date, subject + body),
  2. downloads any annexure PDFs we know of into data/docs/.

Then drop any extra PDFs you like (bank / telecom annual reports) into
data/docs/ and run  python scripts/ingest.py

Usage:  python scripts/download_docs.py [--out data/docs]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BASE = "https://www.sbp.org.pk/circulars/"
ASSETS = "https://www.sbp.org.pk/assets/documents/circulars/"

# (file id, page slug, title)  — the substantive 2024 BPRD circulars.
# Holiday notices are deliberately excluded (no useful content).
CIRCULARS = [
    ("BPRD-2024-C01", "bprd-circular-no-01-of-2024", "Facilitation Framework - BISP Sahulat Account"),
    ("BPRD-2024-C02", "bprd-circular-no-02-of-2024", "Charging-off of Loans, Advances and Finances"),
    ("BPRD-2024-C03", "bprd-circular-no-03-of-2024", "Guidelines - Non-performing Loans (NPLs) Management Strategy"),
    ("BPRD-2024-C04", "bprd-circular-no-04-of-2024", "Minimum Capital Requirements: Basel Framework - Regulatory Retail Portfolio"),
    ("BPRD-2024-C05", "bprd-circular-no-05-of-2024", "Minimum Rate of Return on Saving Deposits"),
    ("BPRD-2024-CL03", "bprd-circular-letter-no-03-of-2024", "Deduction of Zakat at Source"),
    ("BPRD-2024-CL05", "bprd-circular-letter-no-05-of-2024", "Office and Business Hours during Ramadan 1445 AH"),
    ("BPRD-2024-CL07", "bprd-circular-letter-no-07-of-2024", "Temporary Waiver of OTP/Callback Confirmation Requirement on AMA"),
    ("BPRD-2024-CL13", "bprd-circular-letter-no-13-of-2024", "Format of Annual Financial Statements of Banks/DFIs"),
    ("BPRD-2024-CL15", "bprd-circular-letter-no-15-of-2024", "Transfer and Assignment of NPAs to Corporate Restructuring Companies"),
    ("BPRD-2024-CL16", "bprd-circular-letter-no-16-of-2024", "Implementation of IFRS 9"),
    ("BPRD-2024-CL19", "bprd-circular-letter-no-19-of-2024", "Change in Nomenclature of Firm on Panel of Auditors"),
    ("BPRD-2024-CL20", "bprd-circular-letter-no-20-of-2024", "Improvement in Operations of CCTVs"),
    ("BPRD-2024-CL22", "bprd-circular-letter-no-22-of-2024", "Implementation of Clean Note Policy by Banks"),
]

# Annexure PDFs referenced by those circulars (the meaty content lives here).
ANNEXES = [
    "BPRD-2024-C3-Annex.pdf",        # NPL Management Strategy guidelines
    "BPRD-2024-CL15-Annex-A.pdf",    # Revised CRC transfer guidelines
    "BPRD-2024-CL16-Annex-A.pdf",    # IFRS 9 amendments
    "BPRD-2024-CL3-Annex.pdf",       # Zakat nisab notification (scanned - may need OCR)
    "C2-Annex_6.pdf",                # Charged-off loans disclosure template
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; rag-assistant/1.0; portfolio project)"}


def extract_text(html: str) -> str:
    """Main-content extraction with trafilatura, falling back to BeautifulSoup."""
    try:
        import trafilatura

        txt = trafilatura.extract(html, include_tables=True, include_links=False, favor_recall=True)
        if txt and len(txt) > 200:
            return txt
    except Exception:  # noqa: BLE001
        pass
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "nav", "header", "footer", "noscript"]):
        t.decompose()
    text = soup.get_text("\n")
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch(url: str, retries: int = 3) -> requests.Response:
    for i in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, timeout=60)
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            if i == retries - 1:
                raise
            print(f"   retry {i+1} after error: {e}")
            time.sleep(2)
    raise RuntimeError("unreachable")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/docs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    ok, failed = 0, []
    for fid, slug, title in CIRCULARS:
        url = BASE + slug
        dest = out / f"{fid}.md"
        print(f"- {fid}: {title}")
        try:
            html = fetch(url).text
            body = extract_text(html)
            dest.write_text(f"# {fid} — {title}\n\nSource: {url}\n\n{body}\n", encoding="utf-8")
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((url, str(e)))
            print(f"   FAILED: {e}")
        time.sleep(0.5)

    for name in ANNEXES:
        url = ASSETS + name
        dest = out / name
        print(f"- {name}")
        try:
            dest.write_bytes(fetch(url).content)
            ok += 1
        except Exception as e:  # noqa: BLE001
            failed.append((url, str(e)))
            print(f"   FAILED: {e}")

    print(f"\nSaved {ok} files to {out}/")
    if failed:
        print("Failed (download manually from sbp.org.pk and drop into data/docs):")
        for url, err in failed:
            print("  ", url, "->", err)
    print("\nNext: python scripts/ingest.py")


if __name__ == "__main__":
    main()
