import importlib
import sys
import types
import unittest
from unittest.mock import patch


def _completion(content):
    return {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "model": "google-antigravity/gemini-3.8-flash",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 1,
            "completion_tokens": 2,
            "total_tokens": 3,
        },
    }


class HermesIntegrationTests(unittest.TestCase):
    def test_provider_profile_registers_with_installed_hermes_contract(self):
        import providers
        import hermes_antigravity.hermes_provider as module

        captured = {}

        def register_provider(profile):
            captured["profile"] = profile

        with patch.object(providers, "register_provider", register_provider):
            self.assertTrue(module.register_provider_profile())

        self.assertEqual(captured["profile"].name, "antigravity")

    def test_provider_profile_supplies_native_auxiliary_client(self):
        import providers
        import hermes_antigravity.hermes_provider as module

        captured = {}

        def register_provider(profile):
            captured["profile"] = profile

        with patch.object(providers, "register_provider", register_provider):
            self.assertTrue(module.register_provider_profile())

        client = captured["profile"].create_client(
            api_key="hermes-antigravity",
            base_url="http://127.0.0.1:8765/v1",
        )
        self.assertIsNotNone(client)
        with patch(
            "hermes_antigravity.hermes_client.generate_chat_completion",
            return_value=_completion("aux ok"),
        ) as generate:
            response = client.chat.completions.create(
                model="google-antigravity/gemini-3.8-flash",
                messages=[{"role": "user", "content": "ping"}],
            )

        self.assertEqual(response.choices[0].message.content, "aux ok")
        generate.assert_called_once()

    def test_native_auxiliary_client_streams_an_openai_chunk(self):
        from hermes_antigravity.hermes_client import AntigravityHermesClient

        client = AntigravityHermesClient()
        with patch(
            "hermes_antigravity.hermes_client.generate_chat_completion",
            return_value=_completion("stream aux ok"),
        ):
            chunks = list(
                client.chat.completions.create(
                    model="google-antigravity/gemini-3.8-flash",
                    messages=[{"role": "user", "content": "ping"}],
                    stream=True,
                )
            )

        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].choices[0].delta.content, "stream aux ok")
        self.assertEqual(chunks[0].choices[0].finish_reason, "stop")

    def test_native_auxiliary_client_supports_async_calls(self):
        import asyncio

        from hermes_antigravity.hermes_client import AntigravityHermesClient

        client = AntigravityHermesClient()
        with patch(
            "hermes_antigravity.hermes_client.generate_chat_completion",
            return_value=_completion("async aux ok"),
        ):
            async def run_call():
                return await client.chat.completions.create(
                    model="google-antigravity/gemini-3.8-flash",
                    messages=[{"role": "user", "content": "ping"}],
                )

            response = asyncio.run(run_call())

        self.assertEqual(response.choices[0].message.content, "async aux ok")

    def test_native_auxiliary_client_supports_async_streams(self):
        import asyncio

        from hermes_antigravity.hermes_client import AntigravityHermesClient

        client = AntigravityHermesClient()
        with patch(
            "hermes_antigravity.hermes_client.generate_chat_completion",
            return_value=_completion("async stream ok"),
        ):
            async def run_call():
                stream = await client.chat.completions.create(
                    model="google-antigravity/gemini-3.8-flash",
                    messages=[{"role": "user", "content": "ping"}],
                    stream=True,
                )
                return [chunk async for chunk in stream]

            chunks = asyncio.run(run_call())

        self.assertEqual(chunks[0].choices[0].delta.content, "async stream ok")

    def test_provider_profile_registers_current_hermes_contract(self):
        captured = {}

        providers = types.ModuleType("providers")
        base = types.ModuleType("providers.base")

        class ProviderProfile:
            def __init__(self, **kwargs):
                for key, value in kwargs.items():
                    setattr(self, key, value)

        def register_provider(profile):
            captured["profile"] = profile

        providers.register_provider = register_provider
        base.ProviderProfile = ProviderProfile
        base.OMIT_TEMPERATURE = object()

        with patch.dict(
            sys.modules,
            {"providers": providers, "providers.base": base},
        ):
            import hermes_antigravity.hermes_provider as module

            module = importlib.reload(module)
            profile = captured["profile"]
            self.assertEqual(profile.name, "antigravity")
            self.assertEqual(profile.auth_type, "api_key")
            self.assertTrue(profile.supports_model_listing)
            self.assertEqual(
                profile.native_reasoning_details_type,
                "antigravity.native_assistant",
            )
            self.assertEqual(
                module._efforts_for("google-antigravity/gemini-3.9-flash"),
                ("low", "medium", "high"),
            )

    def test_general_plugin_registers_cli_and_execution_middleware(self):
        import hermes_antigravity.hermes_plugin as plugin

        class FakeContext:
            def __init__(self):
                self.cli = []
                self.middleware = []

            def register_cli_command(self, **kwargs):
                self.cli.append(kwargs)

            def register_middleware(self, kind, callback):
                self.middleware.append((kind, callback))

        ctx = FakeContext()
        with (
            patch.object(plugin, "register_provider_profile"),
            patch.object(plugin, "ensure_provider_profile_files"),
            patch.object(plugin, "_save_placeholder_api_key"),
        ):
            plugin.register(ctx)

        self.assertEqual(ctx.cli[0]["name"], "agy")
        self.assertEqual(ctx.middleware[0][0], "llm_execution")
        self.assertIs(ctx.middleware[0][1], plugin.antigravity_llm_execution)


if __name__ == "__main__":
    unittest.main()
