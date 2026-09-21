import unittest
from unittest.mock import patch

from hermes_antigravity.errors import AntigravityError
from hermes_antigravity.openai_compat import ChatRequest
from hermes_antigravity.runtime import _runtime_candidates, generate_chat_completion


class FakeStore:
    def __init__(self):
        self.saved = []

    def ordered_credentials(self):
        return [
            ("a@example.com", {"email": "a@example.com", "access_token": "tok-a", "project_id": "p"},),
            ("b@example.com", {"email": "b@example.com", "access_token": "tok-b", "project_id": "p"},),
        ]

    def upsert(self, creds, activate=True):
        self.saved.append((dict(creds), activate))
        return creds.get("email", "account")


class FallbackClient:
    def __init__(self):
        self.calls = []

    def generate(self, *, access_token, body):
        self.calls.append((access_token, body["model"]))
        if body["model"].startswith("gemini-3.8"):
            raise AntigravityError("model not found", status=404)
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "ok"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        }


class QuotaClient:
    def __init__(self):
        self.calls = []

    def generate(self, *, access_token, body):
        self.calls.append((access_token, body["model"]))
        if access_token == "tok-a":
            raise AntigravityError("Quota reached.", status=429)
        return {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "second account"}]},
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {},
        }


class RuntimeTests(unittest.TestCase):
    def test_runtime_candidates_include_38_37_36(self):
        request = ChatRequest(
            model="google-antigravity/gemini-3.8-flash",
            messages=[{"role": "user", "content": "ping"}],
            reasoning_effort="high",
        )
        candidates = _runtime_candidates(request, {})
        self.assertEqual(
            candidates[:3],
            [
                "gemini-3.8-flash-high",
                "gemini-3.7-flash-high",
                "gemini-3.6-flash-high",
            ],
        )

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={})
    def test_404_falls_back_model(self, _catalog, _keychain):
        client = FallbackClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            },
            client=client,
            store=FakeStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        self.assertEqual(client.calls[0][1], "gemini-3.8-flash-high")
        self.assertEqual(client.calls[1][1], "gemini-3.7-flash-high")

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={})
    def test_429_rotates_account(self, _catalog, _keychain):
        client = QuotaClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
            },
            client=client,
            store=FakeStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "second account")
        self.assertEqual(client.calls[0][0], "tok-a")
        self.assertEqual(client.calls[1][0], "tok-b")


if __name__ == "__main__":
    unittest.main()
