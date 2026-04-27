#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTAIN_DEFINITION = REPO_ROOT / "captain-definition"
BASE_IMAGE_ARG = "BROWSER_USE_BASE_IMAGE"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Bump CACHE_BUSTER and refresh the CapRover base image pin."
    )
    parser.add_argument("--repository", required=True, help="GitHub repository slug, e.g. owner/repo")
    parser.add_argument(
        "--source-ref",
        default="origin/main",
        help="Git ref to use when deriving the immutable image SHA tag",
    )
    return parser.parse_args()


def source_sha(ref: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}"],
            text=True,
            cwd=REPO_ROOT,
            stderr=subprocess.PIPE,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        details = exc.stderr.strip() if isinstance(exc, subprocess.CalledProcessError) and exc.stderr else str(exc)
        raise RuntimeError(f"Failed to resolve git ref '{ref}' to a commit: {details}") from exc


def image_ref(repository: str, sha: str) -> str:
    return f"ghcr.io/{repository.lower()}:sha-{sha[:12]}"


def update_definition(path: Path, base_image: str) -> None:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"Captain definition file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON in {path}: {exc}") from exc

    build_args = data.setdefault("buildArgs", {})
    current = str(build_args.get("CACHE_BUSTER", "0")).strip()
    try:
        value = int(current)
    except ValueError:
        value = 0

    build_args["CACHE_BUSTER"] = str(value + 1)
    build_args[BASE_IMAGE_ARG] = base_image

    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    update_definition(CAPTAIN_DEFINITION, image_ref(args.repository, source_sha(args.source_ref)))


if __name__ == "__main__":
    main()
