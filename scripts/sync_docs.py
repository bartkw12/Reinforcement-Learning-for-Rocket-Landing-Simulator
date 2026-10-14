"""Copy the generated tables into the README and docs/ (between include markers).

Examples:
    python scripts/sync_docs.py           # update the documents
    python scripts/sync_docs.py --check   # exit 1 if any document is out of date (CI)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from lunarlander_rl.docs_sync import default_documents, stale, sync

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report, do not modify")
    args = parser.parse_args()
    documents = default_documents(ROOT)
    if args.check:
        out_of_date = stale(documents, ROOT)
        for document in out_of_date:
            print(f"out of date: {document.relative_to(ROOT)}")
        if out_of_date:
            print("run: python scripts/sync_docs.py")
        return 1 if out_of_date else 0
    for document in sync(documents, ROOT):
        print(f"updated {document.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
