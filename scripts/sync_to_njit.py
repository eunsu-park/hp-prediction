#!/usr/bin/env python3
"""Promote a validated hp-prediction version from eunsu-park to njit-research.

This is the recurring "dev -> prod" promotion tool for the two-tier operating
model documented in docs/realtime-regression-sw/njit-consolidation-plan.md:

    eunsu-park/hp-prediction (self-contained: engine + weights vendored in-tree)
        |   develop / validate here  (dev/staging tier)
        v   promotion (this script)
    njit-research/hp-prediction      (single self-contained repo)

It copies ONLY the payload paths (engine source, model weights, web templates,
the CI config, and the site post-processor) from a source checkout into a target
working copy of the njit repo, applying the same reference rewrites used at
bootstrap. It deliberately does NOT touch infra files (.github/, README, LICENSE,
docs/) or the bot-maintained site/data/, so autonomous forecast history in the
njit repo is preserved.

By default the script is a dry run; pass --apply to write, and --commit to also
create a commit in the target repo. Pushing is always left to the operator.

Example:
    # Inspect what would change (no writes):
    python scripts/sync_to_njit.py --target /path/to/njit/hp-prediction

    # Apply and commit (review, then push manually):
    python scripts/sync_to_njit.py --target /path/to/njit/hp-prediction \\
        --apply --commit --message "Promote v0.2.0 (engine + weights)"
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

# Payload paths, relative to the repo root, that this script keeps in sync.
# Engine source lives under the (now-inlined) vendor/realtime-regression-sw tree.
ENGINE_ROOT = "vendor/realtime-regression-sw"
PAYLOAD_TREES = [
    f"{ENGINE_ROOT}/src",
    f"{ENGINE_ROOT}/scripts",
]
PAYLOAD_BINARIES = [
    f"{ENGINE_ROOT}/checkpoint/model_best.pth",
    f"{ENGINE_ROOT}/checkpoint/table_stats.pkl",
]
# Text payload files get reference rewrites applied (see REWRITES).
PAYLOAD_TEXT = [
    "site/index.html",
    "site/main.js",
    "scripts/update_site_data.py",
    "configs/realtime.ci.yaml",
]

# Runtime-generated directories under the engine tree that must never be copied.
EXCLUDE_DIR_NAMES = {"__pycache__", "dataset", "results", ".pytest_cache"}

# eunsu-park -> njit-research reference rewrites, applied to PAYLOAD_TEXT files so
# a promotion never reintroduces dev-tier URLs into the prod repo.
REWRITES = [
    ("eunsu-park/hp-prediction", "njit-research/hp-prediction"),
    ("eunsu-park/realtime-regression-sw", "njit-research/hp-prediction"),
    ("www.eunsu.me/hp-prediction", "sites.njit.edu/hp-prediction"),
    ("eunsu-park.github.io/hp-prediction", "njit-research.github.io/hp-prediction"),
    ("/Users/eunsupark/realtime", "~/realtime"),
]

# The standalone production repo must not reveal the development repos. Files
# within a payload tree that describe the upstream vendoring process are skipped,
# and text payloads are sanitized of any upstream-repo identifiers on copy.
EXCLUDE_TREE_FILES = {"_vendor/README.md"}
TEXT_SUFFIXES = {
    ".py", ".md", ".yaml", ".yml", ".txt", ".html", ".js", ".css", ".json",
    ".cfg", ".ini",
}
_VENDOR_HEADER_RE = re.compile(r"^#\s*Vendored from .*DO NOT EDIT\..*$", re.MULTILINE)
_EXTRACTED_HEADER_RE = re.compile(r"^#\s*Extracted from .*\n", re.MULTILINE)
_RESYNC_LINE_RE = re.compile(r"^[ \t]*#\s*Re-sync: see .*\n", re.MULTILINE)
_UPSTREAM_PATH_RE = re.compile(r"`(?:setup-sw-db|(?<!realtime-)regression-sw)/[^`]*`")
_BARE_REGRESSION_RE = re.compile(r"(?<!realtime-)\bregression-sw\b")


def log(msg: str) -> None:
    """Print a progress line to stderr."""
    print(msg, file=sys.stderr)


def run(cmd: list[str], cwd: Path) -> str:
    """Run a command and return stdout, raising on non-zero exit."""
    result = subprocess.run(
        cmd, cwd=cwd, check=True, capture_output=True, text=True
    )
    return result.stdout


def ensure_source_ready(source: Path) -> None:
    """Validate that the source checkout has the engine and matched weights.

    Args:
        source: Root of the eunsu-park hp-prediction checkout.

    Raises:
        SystemExit: If the vendored engine source or the weights are missing.
    """
    engine_src = source / ENGINE_ROOT / "src"
    if not engine_src.is_dir():
        raise SystemExit(
            f"engine source not found at {engine_src} — the engine is vendored "
            "in-tree under vendor/realtime-regression-sw/; check out the source "
            "repo fully first."
        )
    for rel in PAYLOAD_BINARIES:
        if not (source / rel).is_file():
            raise SystemExit(
                f"weight file missing: {source / rel} — the checkpoint is "
                "committed in-tree under vendor/realtime-regression-sw/checkpoint/; "
                "check out the source repo fully first."
            )


def sanitize_text(text: str) -> str:
    """Strip upstream-repo identifiers and apply dev -> prod rewrites.

    The standalone production repo must not reveal the development repos, so this
    removes vendoring-provenance headers, drops re-sync pointers, neutralizes
    references to the upstream training/data repositories, and finally applies
    the URL/repo rewrites in REWRITES.
    """
    text = _RESYNC_LINE_RE.sub("", text)
    text = _VENDOR_HEADER_RE.sub(
        "# Bundled engine module - do not edit by hand.", text
    )
    text = _EXTRACTED_HEADER_RE.sub("", text)
    text = _UPSTREAM_PATH_RE.sub("`the training code`", text)
    text = text.replace("setup-sw-db", "the data pipeline")
    text = _BARE_REGRESSION_RE.sub("the training pipeline", text)
    for old, new in REWRITES:
        text = text.replace(old, new)
    return text


def copy_tree(src_dir: Path, dst_dir: Path, apply: bool) -> list[str]:
    """Mirror a source directory into the target, skipping generated dirs.

    Args:
        src_dir: Source directory (payload tree).
        dst_dir: Destination directory in the njit working copy.
        apply: When False, only report what would change.

    Returns:
        Relative paths (from dst_dir) that were copied/updated.
    """
    changed: list[str] = []
    for path in sorted(src_dir.rglob("*")):
        rel = path.relative_to(src_dir)
        if any(part in EXCLUDE_DIR_NAMES for part in rel.parts):
            continue
        if path.is_dir():
            continue
        if rel.as_posix() in EXCLUDE_TREE_FILES:
            continue
        dst = dst_dir / rel
        if path.suffix.lower() in TEXT_SUFFIXES:
            new_text = sanitize_text(path.read_text(encoding="utf-8"))
            if dst.is_file() and dst.read_text(encoding="utf-8") == new_text:
                continue
            changed.append(str(dst))
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(new_text, encoding="utf-8")
        else:
            if dst.is_file() and dst.read_bytes() == path.read_bytes():
                continue
            changed.append(str(dst))
            if apply:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dst)
    return changed


def copy_binary(src: Path, dst: Path, apply: bool) -> list[str]:
    """Copy a single binary file if it differs; return [dst] if changed."""
    if dst.is_file() and dst.read_bytes() == src.read_bytes():
        return []
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return [str(dst)]


def copy_text(src: Path, dst: Path, apply: bool) -> list[str]:
    """Copy a text file with sanitization + rewrites; return [dst] if changed."""
    new_text = sanitize_text(src.read_text(encoding="utf-8"))
    if dst.is_file() and dst.read_text(encoding="utf-8") == new_text:
        return []
    if apply:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(new_text, encoding="utf-8")
    return [str(dst)]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path(__file__).resolve().parent.parent,
        help="Root of the eunsu-park hp-prediction checkout (default: this repo).",
    )
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="Root of the njit-research/hp-prediction working copy.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Write changes (default is a dry run).",
    )
    parser.add_argument(
        "--commit",
        action="store_true",
        help="With --apply, git-commit the synced payload in the target repo.",
    )
    parser.add_argument(
        "--message",
        default="Promote engine + weights + web payload from eunsu-park",
        help="Commit message used with --commit.",
    )
    args = parser.parse_args()

    source: Path = args.source.resolve()
    target: Path = args.target.resolve()
    if not (target / ".git").exists():
        raise SystemExit(f"target is not a git repo: {target}")
    ensure_source_ready(source)

    changed: list[str] = []
    for rel in PAYLOAD_TREES:
        changed += copy_tree(source / rel, target / rel, args.apply)
    for rel in PAYLOAD_BINARIES:
        changed += copy_binary(source / rel, target / rel, args.apply)
    for rel in PAYLOAD_TEXT:
        changed += copy_text(source / rel, target / rel, args.apply)

    verb = "Updated" if args.apply else "Would update"
    log(f"{verb} {len(changed)} payload file(s).")
    for c in changed:
        log(f"  {c}")

    if not changed:
        log("Target already up to date with source payload.")
        return 0

    if args.apply and args.commit:
        # Stage only the payload paths so the bot-maintained site/data/ and infra
        # files are never swept into the promotion commit.
        stage = PAYLOAD_TREES + PAYLOAD_BINARIES + PAYLOAD_TEXT
        run(["git", "add", "--", *stage], cwd=target)
        run(["git", "commit", "-m", args.message], cwd=target)
        log(f"Committed in {target} — review and push manually.")
    elif not args.apply:
        log("Dry run — re-run with --apply (and optionally --commit) to write.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
