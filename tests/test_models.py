import unittest

from hermes_antigravity.models import (
    DEFAULT_MODEL,
    get_fallback_runtime_model,
    get_max_output_tokens,
    public_models_from_catalog,
    resolve_wire_model_id,
)


class ModelTests(unittest.TestCase):
    def test_default_model(self):
        self.assertEqual(DEFAULT_MODEL, "google-antigravity/gemini-3.8-flash")

    def test_flash_routing(self):
        self.assertEqual(resolve_wire_model_id("gemini-3.8-flash", "low"), "gemini-3.8-flash-low")
        self.assertEqual(resolve_wire_model_id("gemini-3.8-flash", "medium"), "gemini-3.8-flash-medium")
        self.assertEqual(resolve_wire_model_id("gemini-3.8-flash", "high"), "gemini-3.8-flash-high")

    def test_fallback_chain(self):
        self.assertEqual(
            get_fallback_runtime_model("gemini-3.8-flash-high", "high"),
            "gemini-3.7-flash-high",
        )
        self.assertEqual(
            get_fallback_runtime_model("gemini-3.7-flash-high", "high"),
            "gemini-3.6-flash-high",
        )

    def test_future_gemini_infers_full_output_limit(self):
        self.assertEqual(
            get_max_output_tokens(
                "google-antigravity/gemini-3.9-flash",
                "gemini-3.9-flash-high",
            ),
            65536,
        )

    def test_catalog_filters_internal_and_image_models(self):
        models = public_models_from_catalog(
            {
                "gemini-3.9-flash-low": {"displayName": "Gemini 3.9 Flash (Low)"},
                "gemini-secret-low": {"isInternal": True},
                "gemini-3.9-image": {"displayName": "Gemini image"},
            }
        )
        self.assertIn("google-antigravity/gemini-3.9-flash", models)
        self.assertNotIn("google-antigravity/gemini-secret", models)
        self.assertNotIn("google-antigravity/gemini-3.9-image", models)

    def test_dynamic_future_family_is_exposed(self):
        models = public_models_from_catalog(
            {
                "gemini-3.9-flash-low": {"displayName": "Gemini 3.9 Flash (Low)"},
                "gemini-3.9-flash-high": {"displayName": "Gemini 3.9 Flash (High)"},
            }
        )
        self.assertIn("google-antigravity/gemini-3.9-flash", models)


if __name__ == "__main__":
    unittest.main()
