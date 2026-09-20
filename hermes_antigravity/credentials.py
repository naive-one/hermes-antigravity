from __future__ import annotations

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


def hermes_home() -> Path:
    return Path(os.getenv("HERMES_HOME") or Path.home() / ".hermes").expanduser()


def _accounts_path() -> Path:
    return hermes_home() / ".antigravity_accounts.json"


def _legacy_path() -> Path:
    return hermes_home() / ".antigravity_oauth.json"


def _account_key(credentials: dict[str, Any]) -> str:
    email = credentials.get("email")
    if isinstance(email, str) and email.strip():
        return email.strip().lower()
    refresh = str(credentials.get("refresh_token") or credentials.get("refresh") or "")
    access = str(credentials.get("access_token") or credentials.get("access") or credentials.get("token") or "")
    seed = refresh or access or "default"
    return "account-" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:12]


class CredentialStore:
    """Profile-local Antigravity account pool.

    File format:
      {"active": "user@example.com", "accounts": {"user@example.com": {...}}}

    The old single-account .antigravity_oauth.json is imported lazily on first
    read so existing users do not need to log in again.
    """

    def __init__(self, path: Path | None = None):
        self.path = Path(path or _accounts_path()).expanduser()

    @classmethod
    def default(cls) -> "CredentialStore":
        return cls()

    def _read_raw(self) -> dict[str, Any]:
        if not self.path.exists():
            self._migrate_legacy()
        if not self.path.exists():
            return {"active": None, "accounts": {}}
        try:
            with self.path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            return {"active": None, "accounts": {}}
        if not isinstance(data, dict):
            return {"active": None, "accounts": {}}
        accounts = data.get("accounts")
        if not isinstance(accounts, dict):
            # Accept a legacy dict accidentally written to the new path.
            key = _account_key(data)
            return {"active": key, "accounts": {key: data}}
        return {"active": data.get("active"), "accounts": accounts}

    def _write_raw(self, data: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        fd, tmp = tempfile.mkstemp(prefix=f".{self.path.name}.", dir=str(self.path.parent), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, sort_keys=True)
                f.write("\n")
            os.chmod(tmp, 0o600)
            os.replace(tmp, self.path)
            os.chmod(self.path, 0o600)
        finally:
            if os.path.exists(tmp):
                os.unlink(tmp)

    def _migrate_legacy(self) -> None:
        legacy = _legacy_path()
        if self.path.exists() or not legacy.exists():
            return
        try:
            with legacy.open("r", encoding="utf-8") as f:
                creds = json.load(f)
        except Exception:
            return
        if not isinstance(creds, dict) or not creds:
            return
        key = _account_key(creds)
        self._write_raw({"active": key, "accounts": {key: creds}})

    def list_accounts(self) -> list[tuple[str, dict[str, Any], bool]]:
        data = self._read_raw()
        active = data.get("active")
        out: list[tuple[str, dict[str, Any], bool]] = []
        for key, creds in (data.get("accounts") or {}).items():
            if isinstance(creds, dict):
                out.append((str(key), dict(creds), str(key) == active))
        out.sort(key=lambda item: (not item[2], item[0]))
        return out

    def load(self) -> dict[str, Any]:
        data = self._read_raw()
        accounts = data.get("accounts") or {}
        active = data.get("active")
        if isinstance(active, str) and isinstance(accounts.get(active), dict):
            return dict(accounts[active])
        for key, creds in accounts.items():
            if isinstance(creds, dict):
                data["active"] = key
                self._write_raw(data)
                return dict(creds)
        return {}

    def upsert(self, credentials: dict[str, Any], *, activate: bool = True) -> str:
        data = self._read_raw()
        accounts = dict(data.get("accounts") or {})
        key = _account_key(credentials)
        merged = dict(accounts.get(key) or {})
        merged.update(credentials)
        accounts[key] = merged
        data["accounts"] = accounts
        if activate or not data.get("active"):
            data["active"] = key
        self._write_raw(data)
        return key

    def save(self, credentials: dict[str, Any]) -> None:
        self.upsert(credentials, activate=True)

    def activate(self, key: str) -> bool:
        data = self._read_raw()
        accounts = data.get("accounts") or {}
        if key not in accounts:
            # Accept an email prefix or exact email value for convenience.
            matches = [name for name, creds in accounts.items() if name == key or (isinstance(creds, dict) and creds.get("email") == key)]
            if len(matches) != 1:
                return False
            key = str(matches[0])
        data["active"] = key
        self._write_raw(data)
        return True

    def remove(self, key: str) -> bool:
        data = self._read_raw()
        accounts = dict(data.get("accounts") or {})
        if key not in accounts:
            return False
        accounts.pop(key, None)
        data["accounts"] = accounts
        if data.get("active") == key:
            data["active"] = next(iter(accounts), None)
        self._write_raw(data)
        return True

    def ordered_credentials(self) -> list[tuple[str, dict[str, Any]]]:
        return [(key, creds) for key, creds, _ in self.list_accounts()]

    def delete(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
        try:
            _legacy_path().unlink()
        except FileNotFoundError:
            pass


def _expiry_to_epoch(value: object) -> float | object:
    if not isinstance(value, str):
        return value
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return value


def parse_agy_keychain_secret(raw: str) -> dict[str, Any]:
    raw = raw.strip()
    if raw.startswith("go-keyring-base64:"):
        raw = base64.b64decode(raw.split(":", 1)[1]).decode("utf-8")
    data = json.loads(raw)
    token = data.get("token") if isinstance(data, dict) else None
    if not isinstance(token, dict) or not token.get("access_token"):
        return {}
    return {
        "access_token": token.get("access_token"),
        "refresh_token": token.get("refresh_token"),
        "expires_at": _expiry_to_epoch(token.get("expiry")),
        "token_type": token.get("token_type", "Bearer"),
        "source": "agy-keychain",
    }


def load_agy_keychain_credentials(*, runner: Callable[[], str] | None = None) -> dict[str, Any]:
    if runner is None and sys.platform != "darwin":
        return {}

    def default_runner() -> str:
        return subprocess.check_output(
            ["security", "find-generic-password", "-a", "antigravity", "-s", "gemini", "-w"],
            stderr=subprocess.DEVNULL,
            timeout=5,
        ).decode("utf-8")

    try:
        return parse_agy_keychain_secret((runner or default_runner)())
    except Exception:
        return {}
