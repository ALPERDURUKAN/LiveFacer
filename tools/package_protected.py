#!/usr/bin/env python3
"""
External packaging pipeline for a protected single-EXE release.

This script is intentionally external to the runtime payload and must not
modify the workspace /system directory in-place.
"""

from __future__ import annotations

import argparse
import base64
import hmac
import json
import os
import secrets
import shutil
import subprocess
import sys
import stat
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Iterable, List, Tuple

DEFAULT_EXCLUDES = (
    ".git",
    ".idea",
    ".vscode",
    "__pycache__",
    "*.pyc",
    "system/tmp",
    "dist_protected",
)

CRITICAL_FILES = (
    "code/run.py",
    "code/modules/core.py",
    "code/modules/ui.py",
    "code/modules/integrity.py",
    "code/modules/license_manager.py",
    "code/modules/model_bootstrap.py",
)


@dataclass
class BuildContext:
    project_root: Path
    output_dir: Path
    stage_dir: Path
    work_dir: Path
    dist_dir: Path
    python_exe: Path
    skip_pyinstaller: bool
    product_id: str
    product_permalink: str
    grace_days: int
    minimal_stage: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--skip-pyinstaller", action="store_true", default=False)
    parser.add_argument("--product-id", default="")
    parser.add_argument("--product-permalink", default="")
    parser.add_argument("--grace-days", type=int, default=7)
    parser.add_argument("--minimal-stage", action="store_true", default=False)
    return parser.parse_args()


def should_ignore(path: Path, root: Path, excluded_prefixes: Iterable[str] = ()) -> bool:
    rel = path.relative_to(root).as_posix()
    for prefix in excluded_prefixes:
        if rel == prefix or rel.startswith(f"{prefix}/"):
            return True
    for pattern in DEFAULT_EXCLUDES:
        if pattern.startswith("*"):
            if path.match(pattern):
                return True
            continue
        if rel == pattern or rel.startswith(f"{pattern}/"):
            return True
    return False


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        raise


def copy_tree_filtered(
    source_root: Path,
    target_root: Path,
    minimal_stage: bool = False,
    excluded_prefixes: Iterable[str] = (),
) -> None:
    if target_root.exists():
        shutil.rmtree(target_root, onerror=_handle_remove_readonly)
    target_root.mkdir(parents=True, exist_ok=True)

    for item in source_root.rglob("*"):
        if minimal_stage:
            rel = item.relative_to(source_root).as_posix()
            if rel.startswith("system/") and not rel.startswith("system/security/"):
                continue
        if should_ignore(item, source_root, excluded_prefixes):
            continue
        rel = item.relative_to(source_root)
        dest = target_root / rel
        if item.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        elif item.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, dest)


def file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def collect_manifest_entries(stage_dir: Path, files: Iterable[str]) -> List[dict]:
    entries = []
    for rel in files:
        path = stage_dir / rel
        if not path.exists():
            continue
        entries.append(
            {
                "path": rel.replace("\\", "/"),
                "sha256": file_sha256(path),
                "required": True,
            }
        )
    return entries


def load_or_create_hmac_key(output_dir: Path) -> Tuple[bytes, Path]:
    env_b64 = os.getenv("LIVEFACER_MANIFEST_HMAC_KEY_B64", "").strip()
    key_path = output_dir / "secrets" / "manifest_hmac_key.b64"
    key_path.parent.mkdir(parents=True, exist_ok=True)

    if env_b64:
        key = base64.b64decode(env_b64)
        if len(key) < 32:
            raise RuntimeError("LIVEFACER_MANIFEST_HMAC_KEY_B64 must decode to at least 32 bytes.")
        return key, key_path

    if key_path.exists():
        key = base64.b64decode(key_path.read_text(encoding="utf-8").strip())
        if len(key) < 32:
            raise RuntimeError(f"Stored HMAC key is too short: {key_path}")
        return key, key_path

    key = secrets.token_bytes(32)
    key_path.write_text(base64.b64encode(key).decode("ascii"), encoding="utf-8")
    print(f"[WARN] Generated new HMAC signing key at: {key_path}")
    print("[WARN] Use LIVEFACER_MANIFEST_HMAC_KEY_B64 for deterministic production builds.")
    return key, key_path


def write_security_bundle(context: BuildContext) -> None:
    hmac_key, _ = load_or_create_hmac_key(context.output_dir)

    security_dir = context.stage_dir / "system" / "security"
    security_dir.mkdir(parents=True, exist_ok=True)

    manifest = {
        "version": 1,
        "algorithm": "hmac-sha256",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "files": collect_manifest_entries(context.stage_dir, CRITICAL_FILES),
    }
    manifest_bytes = json.dumps(manifest, separators=(",", ":"), sort_keys=True).encode("utf-8")
    signature = hmac.new(hmac_key, manifest_bytes, sha256).digest()

    (security_dir / "manifest.json").write_bytes(manifest_bytes)
    (security_dir / "manifest.sig").write_bytes(signature)
    (security_dir / "manifest_key.b64").write_text(
        base64.b64encode(hmac_key).decode("ascii"), encoding="utf-8"
    )

    license_config = {
        "license_enforced": True,
        "product_id": context.product_id,
        "product_permalink": context.product_permalink or "livefacer",
        "offline_grace_days": context.grace_days,
    }
    (security_dir / "license_config.json").write_text(
        json.dumps(license_config, indent=2), encoding="utf-8"
    )


def ensure_local_pyinstaller(context: BuildContext) -> Path:
    tools_site = context.work_dir / "pyi_site_packages"
    tools_site.mkdir(parents=True, exist_ok=True)

    pyinstaller_module = tools_site / "PyInstaller"
    if pyinstaller_module.exists():
        return tools_site

    subprocess.check_call(
        [
            str(context.python_exe),
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--target",
            str(tools_site),
            "pyinstaller==6.10.0",
        ]
    )
    return tools_site


def build_single_exe(context: BuildContext) -> None:
    tools_site = ensure_local_pyinstaller(context)

    stage_run = context.stage_dir / "code" / "run.py"
    if not stage_run.exists():
        raise RuntimeError(f"Stage entry script not found: {stage_run}")

    app_name = "LiveFacerProtected"
    pyinstaller_cmd = [
        str(context.python_exe),
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onefile",
        "--name",
        app_name,
        "--distpath",
        str(context.dist_dir),
        "--workpath",
        str(context.work_dir / "pyi-work"),
        "--specpath",
        str(context.work_dir / "pyi-spec"),
        "--add-data",
        f"{context.stage_dir / 'code'};code",
        "--add-data",
        f"{context.stage_dir / 'system'};system",
        str(stage_run),
    ]
    pyinstaller_env = os.environ.copy()
    existing_py_path = pyinstaller_env.get("PYTHONPATH", "")
    pyinstaller_env["PYTHONPATH"] = (
        str(tools_site)
        if not existing_py_path
        else str(tools_site) + os.pathsep + existing_py_path
    )
    subprocess.check_call(pyinstaller_cmd, env=pyinstaller_env)

    output_file = context.dist_dir / f"{app_name}.exe"
    if not output_file.exists():
        raise RuntimeError(f"Expected output executable was not generated: {output_file}")


def main() -> int:
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    stage_dir = output_dir / "stage_workspace"
    work_dir = output_dir / "build_work"
    dist_dir = output_dir / "single_exe"
    python_exe = project_root / "system" / "python" / "python.exe"

    context = BuildContext(
        project_root=project_root,
        output_dir=output_dir,
        stage_dir=stage_dir,
        work_dir=work_dir,
        dist_dir=dist_dir,
        python_exe=python_exe,
        skip_pyinstaller=args.skip_pyinstaller,
        product_id=args.product_id,
        product_permalink=args.product_permalink,
        grace_days=max(1, int(args.grace_days)),
        minimal_stage=bool(args.minimal_stage),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    context.work_dir.mkdir(parents=True, exist_ok=True)
    context.dist_dir.mkdir(parents=True, exist_ok=True)

    excluded_prefixes: List[str] = []
    try:
        rel_output = context.output_dir.relative_to(context.project_root).as_posix()
        excluded_prefixes.append(rel_output)
    except ValueError:
        pass

    print("[INFO] Creating stage workspace...", flush=True)
    copy_tree_filtered(
        context.project_root,
        context.stage_dir,
        context.minimal_stage,
        excluded_prefixes=excluded_prefixes,
    )
    print("[INFO] Stage workspace ready.", flush=True)

    print("[INFO] Generating signed security bundle...", flush=True)
    write_security_bundle(context)
    print("[INFO] Security bundle generated.", flush=True)

    if context.skip_pyinstaller:
        print("[INFO] Skip mode enabled. Stage and signed security artifacts created.", flush=True)
        return 0

    print("[INFO] Building single executable with PyInstaller...", flush=True)
    build_single_exe(context)
    print("[OK] Protected single executable build completed.", flush=True)
    print(f"[OK] Output: {context.dist_dir}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
