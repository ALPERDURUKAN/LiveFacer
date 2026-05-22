from __future__ import annotations

import base64
import hmac
import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Dict, List, Tuple


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def get_base_dir() -> Path:
    explicit_base = getattr(sys, "_base_dir", None)
    if explicit_base:
        return Path(explicit_base)

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent

    return Path(__file__).resolve().parents[2]


def get_security_dir() -> Path:
    return get_base_dir() / "system" / "security"


def _manifest_paths() -> Tuple[Path, Path, Path]:
    security_dir = get_security_dir()
    return (
        security_dir / "manifest.json",
        security_dir / "manifest.sig",
        security_dir / "manifest_key.b64",
    )


def is_integrity_enforced() -> bool:
    if "LIVEFACER_INTEGRITY_ENFORCE" in os.environ:
        return _env_bool("LIVEFACER_INTEGRITY_ENFORCE", default=False)

    manifest_path, signature_path, key_path = _manifest_paths()
    return manifest_path.exists() and signature_path.exists() and key_path.exists()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        while True:
            chunk = file.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_manifest(manifest_path: Path) -> Dict:
    content = manifest_path.read_bytes()
    return json.loads(content.decode("utf-8"))


def _verify_manifest_signature(
    manifest_path: Path, signature_path: Path, key_path: Path
) -> Tuple[bool, str]:
    try:
        manifest_bytes = manifest_path.read_bytes()
        signature = signature_path.read_bytes()
        key = base64.b64decode(key_path.read_text(encoding="utf-8").strip())
    except Exception as exception:
        return False, f"Failed to load integrity files: {exception}"

    if len(key) < 32:
        return False, "Integrity key is invalid."

    expected_signature = hmac.new(key, manifest_bytes, sha256).digest()
    if not hmac.compare_digest(expected_signature, signature):
        return False, "Manifest signature mismatch."
    return True, "Manifest signature is valid."


def verify_runtime_integrity() -> Tuple[bool, str]:
    if not is_integrity_enforced():
        return True, "Integrity enforcement is disabled."

    manifest_path, signature_path, key_path = _manifest_paths()
    if not manifest_path.exists() or not signature_path.exists() or not key_path.exists():
        return False, "Required integrity files are missing."

    signature_ok, signature_message = _verify_manifest_signature(
        manifest_path, signature_path, key_path
    )
    if not signature_ok:
        return False, signature_message

    try:
        manifest = _load_manifest(manifest_path)
    except Exception as exception:
        return False, f"Invalid manifest format: {exception}"

    algorithm = str(manifest.get("algorithm", "")).lower().strip()
    if algorithm != "hmac-sha256":
        return False, f"Unsupported integrity algorithm: {algorithm}"

    entries: List[Dict] = manifest.get("files", [])
    base_dir = get_base_dir()
    for entry in entries:
        relative_path = str(entry.get("path", "")).replace("\\", "/")
        expected_hash = str(entry.get("sha256", "")).lower()
        required = bool(entry.get("required", True))

        if not relative_path or not expected_hash:
            return False, f"Malformed manifest entry: {entry}"

        file_path = base_dir / Path(relative_path)
        if not file_path.exists():
            if required:
                return False, f"Required file missing: {relative_path}"
            continue

        actual_hash = _sha256_file(file_path).lower()
        if actual_hash != expected_hash:
            return False, f"Tamper detected in file: {relative_path}"

    return True, "Runtime integrity verified."
