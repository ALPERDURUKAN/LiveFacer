from __future__ import annotations

import hmac
import json
import os
import platform
import sys
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Dict, Optional
from urllib import parse, request


LOCAL_STATE_SIGNING_SALT = "livefacer-license-local-state-v1"
DEFAULT_GRACE_DAYS = 7
GUMROAD_VERIFY_URL = "https://api.gumroad.com/v2/licenses/verify"


@dataclass
class LicenseConfig:
    license_enforced: bool
    product_id: str
    product_permalink: str
    offline_grace_days: int


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except Exception:
        return None


def _base_dir() -> Path:
    explicit_base = getattr(sys, "_base_dir", None)
    if explicit_base:
        return Path(explicit_base)
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def _security_dir() -> Path:
    return _base_dir() / "system" / "security"


def _license_dir() -> Path:
    return _base_dir() / "system" / "license"


def _state_file() -> Path:
    return _license_dir() / "license_state.json"


def _machine_fingerprint() -> str:
    fingerprint_data = "|".join(
        [
            platform.system(),
            platform.release(),
            platform.machine(),
            platform.node(),
            os.getenv("PROCESSOR_IDENTIFIER", ""),
            str(uuid.getnode()),
        ]
    )
    return sha256(fingerprint_data.encode("utf-8")).hexdigest()


def get_machine_fingerprint() -> str:
    return _machine_fingerprint()


def _state_signing_key(machine_hash: str) -> bytes:
    data = f"{machine_hash}|{LOCAL_STATE_SIGNING_SALT}".encode("utf-8")
    return sha256(data).digest()


def _sign_payload(payload: Dict[str, Any], machine_hash: str) -> str:
    canonical_payload = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    digest = hmac.new(_state_signing_key(machine_hash), canonical_payload, sha256).hexdigest()
    return digest


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_license_config() -> LicenseConfig:
    config_file = _security_dir() / "license_config.json"
    file_config = _read_json(config_file)

    enforced = file_config.get("license_enforced", False)
    if "LIVEFACER_LICENSE_ENFORCE" in os.environ:
        enforced = _env_bool("LIVEFACER_LICENSE_ENFORCE", bool(enforced))

    product_id = str(os.getenv("LIVEFACER_GUMROAD_PRODUCT_ID", "")).strip() or str(
        file_config.get("product_id", "")
    ).strip()
    product_permalink = str(
        os.getenv("LIVEFACER_GUMROAD_PRODUCT_PERMALINK", "")
    ).strip() or str(file_config.get("product_permalink", "")).strip()

    grace_days = file_config.get("offline_grace_days", DEFAULT_GRACE_DAYS)
    env_grace = os.getenv("LIVEFACER_LICENSE_GRACE_DAYS")
    if env_grace:
        try:
            grace_days = int(env_grace)
        except ValueError:
            grace_days = DEFAULT_GRACE_DAYS
    grace_days = max(1, int(grace_days))

    return LicenseConfig(
        license_enforced=bool(enforced),
        product_id=product_id,
        product_permalink=product_permalink,
        offline_grace_days=grace_days,
    )


def load_state() -> Dict[str, Any]:
    raw_state = _read_json(_state_file())
    if not raw_state:
        return {}

    signature = str(raw_state.get("signature", "")).strip()
    payload = dict(raw_state)
    payload.pop("signature", None)
    machine_hash = str(payload.get("machine_hash", "")).strip()
    if not signature or not machine_hash:
        return {}

    expected = _sign_payload(payload, machine_hash)
    if not hmac.compare_digest(expected, signature):
        return {}
    return payload


def save_state(payload: Dict[str, Any]) -> None:
    state = dict(payload)
    machine_hash = str(state.get("machine_hash", "")).strip()
    if not machine_hash:
        machine_hash = _machine_fingerprint()
        state["machine_hash"] = machine_hash
    state["updated_at"] = _now_utc().isoformat()
    signature = _sign_payload(state, machine_hash)
    stored = dict(state)
    stored["signature"] = signature
    _write_json(_state_file(), stored)


def verify_with_gumroad(license_key: str, config: LicenseConfig) -> Dict[str, Any]:
    if not license_key:
        return {
            "ok": False,
            "network_error": False,
            "message": "License key is empty.",
        }

    if not config.product_id and not config.product_permalink:
        return {
            "ok": False,
            "network_error": False,
            "message": "Missing Gumroad product configuration.",
        }

    payload = {
        "license_key": license_key,
        "increment_uses_count": "false",
    }
    if config.product_id:
        payload["product_id"] = config.product_id
    else:
        payload["product_permalink"] = config.product_permalink

    encoded = parse.urlencode(payload).encode("utf-8")
    req = request.Request(
        GUMROAD_VERIFY_URL,
        data=encoded,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        method="POST",
    )

    try:
        with request.urlopen(req, timeout=15) as response:
            body = response.read().decode("utf-8", errors="replace")
            data = json.loads(body)
    except Exception as exception:
        return {
            "ok": False,
            "network_error": True,
            "message": f"Gumroad verification failed: {exception}",
        }

    success = bool(data.get("success"))
    purchase = data.get("purchase") or {}

    is_refunded = bool(purchase.get("refunded", False))
    is_chargebacked = bool(purchase.get("chargebacked", False))
    is_disputed = bool(purchase.get("disputed", False))
    if success and not is_refunded and not is_chargebacked and not is_disputed:
        return {
            "ok": True,
            "network_error": False,
            "message": "License verified online.",
            "purchase": purchase,
        }

    return {
        "ok": False,
        "network_error": False,
        "message": "License key is invalid or revoked.",
        "purchase": purchase,
    }


def activate_license(license_key: str) -> Dict[str, Any]:
    config = load_license_config()
    verify_result = verify_with_gumroad(license_key, config)
    if not verify_result.get("ok", False):
        return verify_result

    machine_hash = _machine_fingerprint()
    purchase = verify_result.get("purchase") or {}
    state = {
        "license_key": license_key,
        "machine_hash": machine_hash,
        "status": "valid",
        "last_verified_at": _now_utc().isoformat(),
        "purchase_email": str(purchase.get("email", "")),
        "purchase_id": str(purchase.get("id", "")),
        "purchase_permalink": str(purchase.get("product_permalink", "")),
    }
    save_state(state)
    return {
        "ok": True,
        "network_error": False,
        "message": "License activated successfully.",
        "status": "valid_online",
        "state": state,
    }


def _offline_grace_valid(state: Dict[str, Any], grace_days: int) -> Dict[str, Any]:
    last_verified = _parse_datetime(str(state.get("last_verified_at", "")))
    if not last_verified:
        return {
            "ok": False,
            "status": "invalid",
            "message": "No previous successful verification found.",
            "requires_activation": True,
        }

    deadline = last_verified + timedelta(days=grace_days)
    now = _now_utc()
    if now <= deadline and str(state.get("status", "")).lower() == "valid":
        remaining_seconds = max(0, int((deadline - now).total_seconds()))
        remaining_days = remaining_seconds // 86400
        return {
            "ok": True,
            "status": "valid_offline_grace",
            "message": f"Offline grace active ({remaining_days} day(s) left).",
            "requires_activation": False,
        }

    return {
        "ok": False,
        "status": "expired_grace",
        "message": "Offline grace period expired. Internet verification required.",
        "requires_activation": True,
    }


def validate_license(explicit_license_key: Optional[str] = None) -> Dict[str, Any]:
    config = load_license_config()
    if not config.license_enforced:
        return {
            "ok": True,
            "status": "not_enforced",
            "message": "License enforcement disabled.",
            "requires_activation": False,
        }

    state = load_state()
    machine_hash = _machine_fingerprint()
    state_machine_hash = str(state.get("machine_hash", "")).strip()
    if state_machine_hash and state_machine_hash != machine_hash:
        return {
            "ok": False,
            "status": "machine_mismatch",
            "message": "License state does not match this machine.",
            "requires_activation": True,
        }

    env_key = os.getenv("LIVEFACER_LICENSE_KEY", "").strip()
    key = (explicit_license_key or "").strip() or env_key or str(state.get("license_key", "")).strip()
    if not key:
        return {
            "ok": False,
            "status": "missing_key",
            "message": "No license key found. Activation is required.",
            "requires_activation": True,
        }

    online_result = verify_with_gumroad(key, config)
    if online_result.get("ok", False):
        purchase = online_result.get("purchase") or {}
        updated_state = dict(state)
        updated_state.update(
            {
                "license_key": key,
                "machine_hash": machine_hash,
                "status": "valid",
                "last_verified_at": _now_utc().isoformat(),
                "purchase_email": str(purchase.get("email", "")),
                "purchase_id": str(purchase.get("id", "")),
                "purchase_permalink": str(purchase.get("product_permalink", "")),
            }
        )
        save_state(updated_state)
        return {
            "ok": True,
            "status": "valid_online",
            "message": "License verified online.",
            "requires_activation": False,
            "state": updated_state,
        }

    if not online_result.get("network_error", False):
        return {
            "ok": False,
            "status": "invalid",
            "message": online_result.get("message", "License validation failed."),
            "requires_activation": True,
        }

    grace_result = _offline_grace_valid(state, config.offline_grace_days)
    if grace_result.get("ok", False):
        result = dict(grace_result)
        result["state"] = state
        return result
    return grace_result
