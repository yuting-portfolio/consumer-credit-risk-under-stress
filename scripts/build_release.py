"""
Build a clean distributable copy of the project (code, data, docs, results, figures).
"""
from pathlib import Path
import shutil
import zipfile
from datetime import datetime


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "release_bundle"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
ZIP_NAME = ROOT / f"CREDIT_RISK_RESEARCH_PACKAGE_{STAMP}.zip"


def safe_copy(src: Path, dst: Path):
    if not src.exists():
        return False
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main():
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    include_paths = [
        "README.md",
        "requirements.txt",
        ".gitignore",
        "run_all.py",
        "scripts",
        "src",
        "data",
        "results",
        "figures",
        "paper",
        "docs",
    ]

    copied = []
    missing = []

    for rel in include_paths:
        src = ROOT / rel
        dst = OUT_DIR / rel
        ok = safe_copy(src, dst)
        (copied if ok else missing).append(rel)

    manifest = OUT_DIR / "PACKAGE_MANIFEST.txt"
    with manifest.open("w", encoding="utf-8") as f:
        f.write("Credit Risk Research Package Manifest\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n\n")
        f.write("Copied:\n")
        for p in copied:
            f.write(f"- {p}\n")
        if missing:
            f.write("\nMissing:\n")
            for p in missing:
                f.write(f"- {p}\n")

    with zipfile.ZipFile(ZIP_NAME, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in OUT_DIR.rglob("*"):
            zf.write(p, p.relative_to(ROOT))

    print(f"Release folder: {OUT_DIR}")
    print(f"Release zip: {ZIP_NAME}")
    print("Done.")


if __name__ == "__main__":
    main()
