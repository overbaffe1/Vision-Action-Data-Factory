#!/usr/bin/env python3
"""Copy LLaMA-Factory and scaling-lawyer into third_party/ (no symlinks).

Excludes weights and large artifacts. Training configs are excluded
(hackathon uses configs/). After copy, scaling-lawyer is pruned to the
hackathon subset (vla + libero_eval SmolVLA wrapper).

Usage:
  python3 scripts/vendor_deps.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LLAMA_FACTORY_SRC = Path(
    os.environ.get("LLAMA_FACTORY_SRC", "/home/jovyan/people/murtazin/LLaMA-Factory")
)
SCALING_LAWYER_SRC = Path(
    os.environ.get("SCALING_LAWYER_SRC", "/home/jovyan/people/murtazin/scaling-lawyer")
)
DST_LF = ROOT / "third_party" / "LLaMA-Factory"
DST_SL = ROOT / "third_party" / "scaling-lawyer"

SKIP_DIR_NAMES_TRAIN = {
    ".git",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    "node_modules",
    "saves",
    "output",
    "outputs",
    "checkpoints",
    "ckpts",
    "models",
    "runs",
    "cache",
    "wandb",
    "logs",
    "log",
    "datasets",
    "assets",
    "evaluation",
    "tests",
    "tests_v1",
    "docker",
    "docs",
    "notebooks",
    "environments",
    "active_learning",
    "optuna_report",
    "ocr_report_qwen35_assets",
    "reports",
    "inference",
    "cicd",
    "results",
    "clearml-export",
    "configs",
    "config",
    "examples",
    "tools",
}

SKIP_SUFFIXES = (
    ".safetensors",
    ".bin",
    ".pt",
    ".pth",
    ".ckpt",
    ".gguf",
    ".onnx",
    ".npz",
    ".h5",
    ".zip",
    ".tar",
    ".tar.gz",
    ".tgz",
    ".7z",
    ".mp4",
    ".avi",
    ".mov",
    ".parquet",
    ".arrow",
    ".feather",
    ".pyc",
    ".log",
    ".csv",
    ".tsv",
    ".pdf",
    ".html",
    ".htm",
)

SKIP_FILE_NAMES = {"uv.lock"}

LF_TOPLEVEL_ALLOW = {
    "src",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "requirements.txt",
    "MANIFEST.in",
    "LICENSE",
    "Makefile",
    "README.md",
    "README_zh.md",
    "CITATION.cff",
    "CLAUDE.md",
}

SL_TOPLEVEL_ALLOW = {
    "src",
    "pyproject.toml",
    "README.md",
    "LICENSE",
}

# Keep only these under src/scaling_lawyer/ after vendor.
SL_KEEP_FILES = {
    "__init__.py",
    "common.py",
    "subsets.py",
    "vla.py",
}
SL_KEEP_LIBERO_EVAL = {
    "__init__.py",
    "smolvla_model.py",
}

HACKATHON_SL_README = """# scaling-lawyer (hackathon subset)

Vendored subset for the AIJ robotics hackathon pipeline.

Kept modules:

- `scaling_lawyer.vla` — SmolVLA action-expert training (LeRobot)
- `scaling_lawyer.common` / `subsets` — helpers used by `vla`
- `scaling_lawyer.libero_eval.SmolVLALiberoModel` — LIBERO layout (image/image2) for submission validation

Full upstream lives outside this repo. Re-vendor with `./scripts/vendor_deps.sh` (applies prune automatically).
"""


def should_skip_dir(name: str, skip_dirs: set[str]) -> bool:
    return name in skip_dirs or name.startswith(".")


def should_skip_file(path: Path) -> bool:
    if path.name in SKIP_FILE_NAMES:
        return True
    lower = path.name.lower()
    if lower.endswith(SKIP_SUFFIXES):
        return True
    return False


def copy_tree(
    src: Path,
    dst: Path,
    *,
    toplevel_allow: set[str] | None = None,
    skip_dirs: set[str] | None = None,
) -> int:
    skip_dirs = skip_dirs or SKIP_DIR_NAMES_TRAIN
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    copied = 0

    entries = list(src.iterdir())
    if toplevel_allow is not None:
        entries = [p for p in entries if p.name in toplevel_allow]

    for entry in entries:
        if entry.is_symlink():
            continue
        if entry.is_dir():
            if should_skip_dir(entry.name, skip_dirs):
                continue
            for root, dirs, files in os.walk(entry):
                root_path = Path(root)
                rel = root_path.relative_to(src)
                dirs[:] = [d for d in dirs if not should_skip_dir(d, skip_dirs)]
                out_dir = dst / rel
                out_dir.mkdir(parents=True, exist_ok=True)
                for fname in files:
                    src_file = root_path / fname
                    if should_skip_file(src_file) or src_file.is_symlink():
                        continue
                    shutil.copy2(src_file, out_dir / fname)
                    copied += 1
        elif entry.is_file():
            if should_skip_file(entry):
                continue
            shutil.copy2(entry, dst / entry.name)
            copied += 1
    return copied


def prune_scaling_lawyer(dst: Path) -> None:
    """Drop upstream modules not used by the hackathon pipeline."""
    pkg = dst / "src" / "scaling_lawyer"
    if not pkg.is_dir():
        return
    for path in pkg.iterdir():
        if path.name == "libero_eval" and path.is_dir():
            for child in path.iterdir():
                if child.name not in SL_KEEP_LIBERO_EVAL:
                    if child.is_dir():
                        shutil.rmtree(child)
                    else:
                        child.unlink()
            continue
        if path.name in SL_KEEP_FILES:
            continue
        if path.name == "__pycache__":
            shutil.rmtree(path)
            continue
        if path.is_dir():
            shutil.rmtree(path)
        else:
            path.unlink()
    (dst / "README.md").write_text(HACKATHON_SL_README, encoding="utf-8")


def assert_no_symlinks(tree: Path) -> None:
    bad = [p for p in tree.rglob("*") if p.is_symlink()]
    if bad:
        raise RuntimeError("Symlinks found under third_party:\n" + "\n".join(map(str, bad[:20])))


def du(path: Path) -> str:
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return f"{total / (1024 * 1024):.1f} MiB"


def main() -> int:
    if not (LLAMA_FACTORY_SRC / "src" / "llamafactory").is_dir():
        print(f"LLaMA-Factory source not found: {LLAMA_FACTORY_SRC}", file=sys.stderr)
        return 1
    if not (SCALING_LAWYER_SRC / "src" / "scaling_lawyer").is_dir():
        print(f"scaling-lawyer source not found: {SCALING_LAWYER_SRC}", file=sys.stderr)
        return 1

    (ROOT / "third_party").mkdir(parents=True, exist_ok=True)

    print("[1/2] Vendoring LLaMA-Factory")
    n_lf = copy_tree(
        LLAMA_FACTORY_SRC, DST_LF, toplevel_allow=LF_TOPLEVEL_ALLOW, skip_dirs=SKIP_DIR_NAMES_TRAIN
    )
    (DST_LF / "data").mkdir(parents=True, exist_ok=True)
    (DST_LF / "data" / "dataset_info.json").write_text("{}\n", encoding="utf-8")

    print("[2/2] Vendoring scaling-lawyer (hackathon subset)")
    # libero_eval is hackathon-local (not upstream); preserve across wipe+prune.
    libero_pkg = DST_SL / "src" / "scaling_lawyer" / "libero_eval"
    libero_backup: Path | None = None
    if libero_pkg.is_dir():
        libero_backup = Path(tempfile.mkdtemp()) / "libero_eval"
        shutil.copytree(libero_pkg, libero_backup)

    n_sl = copy_tree(
        SCALING_LAWYER_SRC, DST_SL, toplevel_allow=SL_TOPLEVEL_ALLOW, skip_dirs=SKIP_DIR_NAMES_TRAIN
    )
    prune_scaling_lawyer(DST_SL)

    if libero_backup is not None and libero_backup.is_dir():
        dest = DST_SL / "src" / "scaling_lawyer" / "libero_eval"
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(libero_backup, dest)
        shutil.rmtree(libero_backup.parent, ignore_errors=True)
        # Re-apply keep-list filter on restored overlay.
        for child in list(dest.iterdir()):
            if child.name not in SL_KEEP_LIBERO_EVAL:
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
        (DST_SL / "README.md").write_text(HACKATHON_SL_README, encoding="utf-8")

    assert_no_symlinks(ROOT / "third_party")

    print(f"Done. LLaMA-Factory: {n_lf} files, {du(DST_LF)}")
    print(f"      scaling-lawyer: {n_sl} files (pre-prune), {du(DST_SL)}")
    print(f"  {DST_LF}")
    print(f"  {DST_SL}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
