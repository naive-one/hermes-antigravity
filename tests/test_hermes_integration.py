import importlib
import sys
import types
import unittest
from unittest.mock import patch


class HermesIntegrationTests(unittest.TestCase):
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
