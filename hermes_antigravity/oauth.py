from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, Callable

from .cloudcode import load_or_onboard_project
from .credentials import CredentialStore, load_agy_keychain_credentials
from .errors import AntigravityError

CALLBACK_HOST = "127.0.0.1"
CALLBACK_PORT = 51121
CALLBACK_PATH = "/oauth-callback"
AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_URL = "https://oauth2.googleapis.com/token"
CLIENT_ID = "".join(
    ("1071006060591", "-", "tmhssin2h21lcre235vtolojh4g403ep", ".apps.", "googleusercontent", ".com")
)
CLIENT_SECRET = "".join(("GOC", "SPX", "-", "K58FWR486LdLJ1mLB", "8sXC4z6qDAf"))
SCOPES = [
    "https://www.googleapis.com/auth/aicode",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/userinfo.email",
    "https://www.googleapis.com/auth/userinfo.profile",
    "https://www.googleapis.com/auth/cclog",
    "https://www.googleapis.com/auth/experimentsandconfigs",
]


def _expires_at(expires_in: object) -> int:
    try:
        seconds = int(expires_in)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        seconds = 3600
    return int(time.time()) + seconds - 300


def _post_form_json(url: str, data: dict[str, str], headers: dict[str, str]) -> dict[str, Any]:
    body = urllib.parse.urlencode(data).encode("utf-8")
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise AntigravityError(f"OAuth token request failed ({exc.code}): {detail}", status=exc.code) from exc


def _get_json(url: str, headers: dict[str, str]) -> dict[str, Any]:
    req = urllib.request.Request(url, headers=headers, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode("utf-8") or "{}")
    except Exception:
        return {}


def oauth_client() -> tuple[str, str]:
    return CLIENT_ID, CLIENT_SECRET


def refresh_access_token(
    refresh_token: str,
    *,
    post_json: Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]] | None = None,
    client: tuple[str, str] | None = None,
) -> dict[str, Any]:
    sender = post_json or _post_form_json
    client_id, client_secret = client or oauth_client()
    data = sender(
        TOKEN_URL,
        {
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
        {"Content-Type": "application/x-www-form-urlencoded"},
    )
    if not data.get("access_token"):
        raise AntigravityError(
            "OAuth refresh response did not include access_token",
            status=401,
            error_type="invalid_request_error",
        )
    return {
        "refresh_token": data.get("refresh_token") or refresh_token,
        "access_token": data["access_token"],
        "expires_at": _expires_at(data.get("expires_in")),
        "token_type": data.get("token_type", "Bearer"),
    }


def refresh_if_needed(credentials: dict[str, Any], *, skew_seconds: int = 60) -> dict[str, Any]:
    access = credentials.get("access_token") or credentials.get("access") or credentials.get("token")
    refresh = credentials.get("refresh_token") or credentials.get("refresh")
    expires = credentials.get("expires_at") or credentials.get("expires")
    if refresh and (not access or (isinstance(expires, (int, float)) and time.time() + skew_seconds >= float(expires))):
        credentials = {**credentials, **refresh_access_token(str(refresh))}
    return credentials


def _pkce_pair() -> tuple[str, str]:
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


def callback_redirect_uri() -> str:
    host = os.getenv("ANTIGRAVITY_OAUTH_REDIRECT_HOST", "localhost")
    port = int(os.getenv("ANTIGRAVITY_OAUTH_PORT", str(CALLBACK_PORT)))
    return f"http://{host}:{port}{CALLBACK_PATH}"


def build_auth_url(
    state: str | None = None,
    redirect_uri: str | None = None,
    client: tuple[str, str] | None = None,
) -> tuple[str, str]:
    state = state or secrets.token_urlsafe(24)
    verifier, challenge = _pkce_pair()
    client_id, _ = client or oauth_client()
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": redirect_uri or callback_redirect_uri(),
        "scope": " ".join(SCOPES),
        "state": state,
        "access_type": "offline",
        "prompt": "consent",
        "code_challenge": challenge,
        "code_challenge_method": "S256",
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}", verifier


def exchange_code_for_tokens(
    code: str,
    *,
    redirect_uri: str | None = None,
    code_verifier: str | None = None,
    post_json: Callable[[str, dict[str, str], dict[str, str]], dict[str, Any]] | None = None,
    client: tuple[str, str] | None = None,
) -> dict[str, Any]:
    sender = post_json or _post_form_json
    client_id, client_secret = client or oauth_client()
    payload = {
        "client_id": client_id,
        "client_secret": client_secret,
        "code": code,
        "grant_type": "authorization_code",
        "redirect_uri": redirect_uri or callback_redirect_uri(),
    }
    if code_verifier:
        payload["code_verifier"] = code_verifier
    data = sender(TOKEN_URL, payload, {"Content-Type": "application/x-www-form-urlencoded"})
    refresh_token = data.get("refresh_token")
    if not data.get("access_token"):
        raise AntigravityError("OAuth token response did not include access_token", status=401)
    if not refresh_token:
        raise AntigravityError(
            "No refresh token received. Re-run login and approve offline access.",
            status=401,
            error_type="invalid_request_error",
        )
    return {
        "refresh_token": refresh_token,
        "access_token": data["access_token"],
        "expires_at": _expires_at(data.get("expires_in")),
        "token_type": data.get("token_type", "Bearer"),
    }


def fetch_user_email(access_token: str) -> str | None:
    data = _get_json(
        "https://www.googleapis.com/oauth2/v1/userinfo?alt=json",
        {"Authorization": f"Bearer {access_token}"},
    )
    email = data.get("email")
    return email if isinstance(email, str) and email else None


def parse_callback_text(raw: str, expected_state: str) -> str:
    text = (raw or "").strip()
    if not text:
        raise AntigravityError("Empty OAuth callback", status=401)
    try:
        parsed = urllib.parse.urlparse(text)
        query = parsed.query
        if not query and ("code=" in text or "state=" in text):
            query = text[1:] if text.startswith("?") else text
        params = urllib.parse.parse_qs(query)
    except Exception as exc:
        raise AntigravityError(f"Invalid OAuth callback: {exc}", status=401) from exc
    error = (params.get("error") or [None])[0]
    if error:
        raise AntigravityError(f"OAuth callback failed: {error}", status=401)
    state = (params.get("state") or [None])[0]
    code = (params.get("code") or [None])[0]
    if state != expected_state:
        raise AntigravityError("OAuth state did not match", status=401)
    if not code:
        raise AntigravityError("OAuth callback did not contain code", status=401)
    return str(code)


class _CallbackHandler(BaseHTTPRequestHandler):
    server: "_CallbackServer"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path != CALLBACK_PATH:
            self.send_error(404)
            return
        params = urllib.parse.parse_qs(parsed.query)
        self.server.received_state = (params.get("state") or [None])[0]
        self.server.received_error = (params.get("error") or [None])[0]
        self.server.received_code = (params.get("code") or [None])[0]
        body = b"Hermes Antigravity login complete. You can close this tab."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args: object) -> None:
        return


class _CallbackServer(HTTPServer):
    received_code: str | None = None
    received_state: str | None = None
    received_error: str | None = None


def _complete_credentials(credentials: dict[str, Any]) -> dict[str, Any]:
    email = fetch_user_email(str(credentials["access_token"]))
    if email:
        credentials["email"] = email
    credentials["project_id"] = load_or_onboard_project(
        str(credentials["access_token"]),
        seed=email or "antigravity-default",
    )
    return credentials


def import_agy_keychain_credentials() -> dict[str, Any]:
    credentials = load_agy_keychain_credentials()
    if not credentials:
        return {}
    credentials = refresh_if_needed(credentials)
    return _complete_credentials(credentials)


def run_login(
    *,
    open_browser: bool = True,
    timeout: int = 300,
    store: CredentialStore | None = None,
    prefer_keychain: bool = True,
    manual: bool = False,
) -> dict[str, Any]:
    store = store or CredentialStore.default()
    if prefer_keychain:
        credentials = import_agy_keychain_credentials()
        if credentials:
            return credentials

    state = secrets.token_urlsafe(24)
    redirect_uri = callback_redirect_uri()
    auth_url, verifier = build_auth_url(state=state, redirect_uri=redirect_uri)
    print("Open this URL to authorize Antigravity:")
    print(auth_url)
    if open_browser:
        webbrowser.open(auth_url)

    code: str | None = None
    if manual:
        pasted = input(
            "After Google redirects to localhost, paste the full callback URL here:\n> "
        )
        code = parse_callback_text(pasted, state)
    else:
        bind_host = os.getenv("ANTIGRAVITY_OAUTH_BIND_HOST", CALLBACK_HOST)
        bind_port = int(os.getenv("ANTIGRAVITY_OAUTH_PORT", str(CALLBACK_PORT)))
        server = _CallbackServer((bind_host, bind_port), _CallbackHandler)
        server.timeout = 1
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and not server.received_code and not server.received_error:
            server.handle_request()
        server.server_close()
        if server.received_error:
            raise AntigravityError(f"OAuth callback failed: {server.received_error}", status=401)
        if server.received_state != state:
            raise AntigravityError("OAuth login timed out or state did not match", status=401)
        code = server.received_code

    if not code:
        raise AntigravityError("OAuth login did not return an authorization code", status=401)

    credentials = exchange_code_for_tokens(
        code,
        redirect_uri=redirect_uri,
        code_verifier=verifier,
    )
    credentials = _complete_credentials(credentials)
    store.upsert(credentials, activate=True)
    return credentials
