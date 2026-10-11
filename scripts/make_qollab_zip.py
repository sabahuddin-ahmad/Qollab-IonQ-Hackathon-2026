"""Build the minimal upload for Qollab: the entry point, its data and the two modules it imports.

Qollab runs main.py at the top level of the project. The whole repository also fits Qollab's limits
(under 100 files, each under 2 MB), but these nine files are all a run needs.

Usage: .venv/bin/python scripts/make_qollab_zip.py [--out qollab_qr1.zip]
"""

from __future__ import annotations

import argparse
import pathlib
import zipfile

ROOT = pathlib.Path(__file__).resolve().parents[1]
FILES = ["main.py", "requirements.txt", "demo/qr1_demo.json", "src/__init__.py", "src/qrc/__init__.py",
         "src/qrc/circuit.py", "src/qrc/reference.py", "src/qrc/textplot.py", "src/qrc/plots.py"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="qollab_qr1.zip")
    args = ap.parse_args()
    out = pathlib.Path(args.out).expanduser()
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name in FILES:
            z.write(ROOT / name, name)
    print(f"wrote {out.resolve()} ({out.stat().st_size / 1024:.1f} KB, {len(FILES)} files)")


if __name__ == "__main__":
    main()
