from __future__ import annotations

import json
import os
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import request

from modules import license_manager


def _base_dir() -> Path:
    explicit_base = getattr(sys, "_base_dir", None)
    if explicit_base:
        return Path(explicit_base)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _models_dir() -> Path:
    return _base_dir() / "code" / "models"


def _encrypted_cache_dir() -> Path:
    return _base_dir() / "system" / "model-cache" / "encrypted"


def _manifest_candidates() -> List[Path]:
    base = _base_dir()
    return [
        base / "system" / "security" / "models_manifest.json",
        base / "code" / "security" / "models_manifest.json",
    ]


def _sha256_bytes(data: bytes) -> str:
    digest = sha256()
    digest.update(data)
    return digest.hexdigest()


def _derive_stream_key(license_key: str) -> bytes:
    machine_hash = license_manager.get_machine_fingerprint()
    seed = f"{license_key}|{machine_hash}|model-cache-v1".encode("utf-8")
    return sha256(seed).digest()


def _xor_stream_crypt(data: bytes, key: bytes) -> bytes:
    output = bytearray(len(data))
    counter = 0
    index = 0
    while index < len(data):
        stream_block = sha256(key + counter.to_bytes(8, "big")).digest()
        block_size = min(len(stream_block), len(data) - index)
        for i in range(block_size):
            output[index + i] = data[index + i] ^ stream_block[i]
        index += block_size
        counter += 1
    return bytes(output)


def _load_manifest() -> Dict[str, Any]:
    for candidate in _manifest_candidates():
        if candidate.exists():
            try:
                return json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                return {}
    return {}


def _download_bytes(url: str) -> bytes:
    req = request.Request(url, method="GET")
    with request.urlopen(req, timeout=60) as response:
        return response.read()


def _read_encrypted_cache(path: Path, key: bytes) -> Optional[bytes]:
    if not path.exists():
        return None
    try:
        encrypted = path.read_bytes()
        return _xor_stream_crypt(encrypted, key)
    except Exception:
        return None


def _write_encrypted_cache(path: Path, data: bytes, key: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encrypted = _xor_stream_crypt(data, key)
    path.write_bytes(encrypted)


def ensure_models_available(license_key: Optional[str]) -> Dict[str, Any]:
    key = (license_key or "").strip()
    if not key:
        return {"ok": False, "message": "Model bootstrap requires a valid license key."}

    manifest = _load_manifest()
    model_entries = manifest.get("models", [])
    if not model_entries:
        return {
            "ok": False,
            "message": "Missing models manifest for secure first-run bootstrap.",
        }

    models_dir = _models_dir()
    cache_dir = _encrypted_cache_dir()
    models_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    stream_key = _derive_stream_key(key)
    downloaded_count = 0
    restored_count = 0

    for entry in model_entries:
        filename = str(entry.get("filename", "")).strip()
        url = str(entry.get("url", "")).strip()
        expected_sha = str(entry.get("sha256", "")).strip().lower()
        if not filename or not url or not expected_sha:
            return {"ok": False, "message": f"Invalid model manifest entry: {entry}"}
        if expected_sha.startswith("replace_"):
            return {
                "ok": False,
                "message": f"Manifest checksum placeholder detected for {filename}. Set real SHA256 values first.",
            }

        target_file = models_dir / filename
        cache_file = cache_dir / f"{filename}.enc"

        # If plaintext target already exists and passes checksum, keep it.
        if target_file.exists():
            current_sha = _sha256_bytes(target_file.read_bytes()).lower()
            if current_sha == expected_sha:
                continue

        # Try encrypted cache first.
        cached_plain = _read_encrypted_cache(cache_file, stream_key)
        if cached_plain is not None and _sha256_bytes(cached_plain).lower() == expected_sha:
            target_file.write_bytes(cached_plain)
            restored_count += 1
            continue

        # Download if cache is missing/invalid.
        try:
            downloaded = _download_bytes(url)
        except Exception as exception:
            return {"ok": False, "message": f"Model download failed for {filename}: {exception}"}

        downloaded_sha = _sha256_bytes(downloaded).lower()
        if downloaded_sha != expected_sha:
            return {
                "ok": False,
                "message": f"Checksum mismatch for downloaded model: {filename}",
            }

        target_file.write_bytes(downloaded)
        _write_encrypted_cache(cache_file, downloaded, stream_key)
        downloaded_count += 1

    return {
        "ok": True,
        "message": f"Model bootstrap ready (downloaded={downloaded_count}, restored={restored_count}).",
    }
