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
    def test_explicit_disabled_id_is_rejected_without_credentials(self):
        with patch("hermes_antigravity.runtime._credential_candidates") as candidates:
            with self.assertRaises(AntigravityError) as raised:
                generate_chat_completion({"model": "gemini-4-argon-disabled", "messages": []})
        self.assertEqual(raised.exception.status, 403)
        candidates.assert_not_called()

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={
        "gemini-4-argon-disabled": {"disabled": True, "tagDescription": "Paid plan required"},
    })
    def test_disabled_family_fails_before_inference_or_account_rotation(self, catalog, _keychain):
        client = FallbackClient()
        with self.assertRaises(AntigravityError) as raised:
            generate_chat_completion({
                "model": "google-antigravity/gemini-4-argon",
                "messages": [{"role": "user", "content": "ping"}],
            }, client=client, store=FakeStore())
        self.assertEqual(raised.exception.status, 403)
        self.assertIn("Paid plan required", str(raised.exception))
        self.assertEqual(client.calls, [])
        catalog.assert_called_once()

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={
        "gemini-3.8-flash-high": {"disabled": True, "tagDescription": "Model access denied"},
    })
    def test_disabled_known_model_fails_before_inference_or_account_rotation(self, catalog, _keychain):
        client = QuotaClient()
        store = FakeStore()
        with self.assertRaises(AntigravityError) as raised:
            generate_chat_completion({
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            }, client=client, store=store)
        self.assertEqual(raised.exception.status, 403)
        self.assertIn("Model access denied", str(raised.exception))
        self.assertEqual(client.calls, [])
        catalog.assert_called_once_with("tok-a", "p")
        self.assertEqual([(creds["email"], active) for creds, active in store.saved], [
            ("a@example.com", False),
        ])

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={
        "gemini-3.8-flash-low": {"disabled": False},
        "gemini-3.8-flash-high": {"disabled": True, "tagDescription": "High reasoning access denied"},
    })
    def test_disabled_requested_variant_fails_before_inference_or_account_rotation(self, catalog, _keychain):
        client = QuotaClient()
        store = FakeStore()
        with self.assertRaises(AntigravityError) as raised:
            generate_chat_completion({
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            }, client=client, store=store)
        self.assertEqual(raised.exception.status, 403)
        self.assertEqual(str(raised.exception), "High reasoning access denied")
        self.assertEqual(client.calls, [])
        catalog.assert_called_once_with("tok-a", "p")
        self.assertEqual([(creds["email"], active) for creds, active in store.saved], [
            ("a@example.com", False),
        ])

    def test_disabled_runtime_candidates_are_excluded(self):
        request = ChatRequest(model="google-antigravity/gemini-4-argon", messages=[], reasoning_effort="high")
        catalog = {"gemini-4-argon-high": {"disabled": True}, "gemini-4-argon-low": {"disabled": False}}
        self.assertNotIn("gemini-4-argon-high", _runtime_candidates(request, catalog))

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
        _catalog.assert_called_once_with("tok-a", "p")

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch(
        "hermes_antigravity.runtime.fetch_available_models",
        return_value={
            "gemini-3.8-flash-tiered": {
                "model": "MODEL_DYNAMIC_38",
                "displayName": "Gemini 3.8 Flash (High)",
            }
        },
    )
    def test_preflight_discovery_preserves_static_404_chain(self, catalog, _keychain):
        class TieredClient:
            def __init__(self):
                self.calls = []

            def generate(self, *, access_token, body):
                catalog.assert_called_once_with("tok-a", "p")
                self.calls.append(body["model"])
                if body["model"] != "gemini-3.8-flash-tiered":
                    raise AntigravityError("model not found", status=404)
                return {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "dynamic ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {},
                }

        client = TieredClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            },
            client=client,
            store=FakeStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "dynamic ok")
        self.assertEqual(
            client.calls[:3],
            [
                "gemini-3.8-flash-high",
                "gemini-3.7-flash-high",
                "gemini-3.6-flash-high",
            ],
        )
        self.assertEqual(client.calls[3], "gemini-3.8-flash-tiered")
        catalog.assert_called_once()

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch(
        "hermes_antigravity.runtime.fetch_available_models",
        return_value={
            "gemini-3.9-flash-high": {
                "model": "MODEL_DYNAMIC_39",
                "displayName": "Gemini 3.9 Flash (High)",
            }
        },
    )
    def test_future_model_uses_discovered_runtime_first(self, catalog, _keychain):
        class FutureClient:
            def __init__(self):
                self.calls = []

            def generate(self, *, access_token, body):
                self.calls.append(body["model"])
                return {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "future ok"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {},
                }

        client = FutureClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.9-flash",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            },
            client=client,
            store=FakeStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "future ok")
        self.assertEqual(client.calls[0], "gemini-3.9-flash-high")
        catalog.assert_called_once()

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={
        "gemini-4-argon-low": {"disabled": False},
        "gemini-4-argon-medium": {"disabled": False},
        "gemini-4-argon-high": {"disabled": False},
    })
    def test_enabled_argon_defaults_to_medium_during_live_resolution(self, catalog, _keychain):
        client = FallbackClient()
        with patch.object(client, "generate", wraps=client.generate) as generate:
            result = generate_chat_completion({
                "model": "google-antigravity/gemini-4-argon",
                "messages": [{"role": "user", "content": "ping"}],
            }, client=client, store=FakeStore())
        self.assertEqual(result["choices"][0]["message"]["content"], "ok")
        generate.assert_called_once()
        body = generate.call_args.kwargs["body"]
        self.assertEqual(body["model"], "gemini-4-argon-medium")
        thinking = body["request"]["generationConfig"]["thinkingConfig"]
        self.assertEqual(thinking["thinkingLevel"], "MEDIUM")
        self.assertNotIn("thinkingBudget", thinking)
        self.assertEqual(client.calls, [("tok-a", "gemini-4-argon-medium")])
        catalog.assert_called_once_with("tok-a", "p")

    @patch("hermes_antigravity.runtime.time.sleep", return_value=None)
    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch("hermes_antigravity.runtime.fetch_available_models", return_value={})
    def test_transient_429_retries_same_account(self, _catalog, _keychain, _sleep):
        class TransientClient:
            def __init__(self):
                self.calls = []

            def generate(self, *, access_token, body):
                self.calls.append(access_token)
                if len(self.calls) < 3:
                    raise AntigravityError(
                        "Resource has been exhausted (e.g. check quota).",
                        status=429,
                    )
                return {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "recovered"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {},
                }

        client = TransientClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.8-flash",
                "messages": [{"role": "user", "content": "ping"}],
            },
            client=client,
            store=FakeStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        self.assertEqual(client.calls, ["tok-a", "tok-a", "tok-a"])

    @patch("hermes_antigravity.runtime.load_agy_keychain_credentials", return_value={})
    @patch(
        "hermes_antigravity.runtime.fetch_available_models",
        return_value={
            "gemini-3.1-pro-high": {
                "model": "MODEL_PLACEHOLDER_M37",
                "displayName": "Gemini 3.1 Pro (High)",
            }
        },
    )
    @patch(
        "hermes_antigravity.runtime.refresh_access_token",
        return_value={
            "access_token": "tok-new",
            "refresh_token": "refresh-a",
            "expires_at": 99999999999,
        },
    )
    def test_post_refresh_404_uses_preflight_discovered_runtime(self, refresh, catalog, _keychain):
        class RefreshStore:
            def ordered_credentials(self):
                return [
                    (
                        "a@example.com",
                        {
                            "email": "a@example.com",
                            "access_token": "tok-old",
                            "refresh_token": "refresh-a",
                            "project_id": "p",
                        },
                    )
                ]

            def upsert(self, creds, activate=True):
                return "a@example.com"

        class RefreshClient:
            def __init__(self):
                self.calls = []

            def generate(self, *, access_token, body):
                self.calls.append((access_token, body["model"]))
                if access_token == "tok-old":
                    from hermes_antigravity.errors import TokenExpired
                    raise TokenExpired()
                if body["model"] == "gemini-pro-agent":
                    raise AntigravityError("model not found", status=404)
                return {
                    "candidates": [
                        {
                            "content": {"role": "model", "parts": [{"text": "dynamic after refresh"}]},
                            "finishReason": "STOP",
                        }
                    ],
                    "usageMetadata": {},
                }

        client = RefreshClient()
        result = generate_chat_completion(
            {
                "model": "google-antigravity/gemini-3.1-pro",
                "messages": [{"role": "user", "content": "ping"}],
                "reasoning_effort": "high",
            },
            client=client,
            store=RefreshStore(),
        )
        self.assertEqual(result["choices"][0]["message"]["content"], "dynamic after refresh")
        self.assertEqual(
            client.calls,
            [
                ("tok-old", "gemini-pro-agent"),
                ("tok-new", "gemini-pro-agent"),
                ("tok-new", "gemini-3.1-pro-high"),
            ],
        )
        refresh.assert_called_once()
        catalog.assert_called_once()

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
