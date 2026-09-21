import unittest

from hermes_antigravity.transform import build_generate_content_request, thinking_config


class TransformTests(unittest.TestCase):
    def test_gemini_38_thinking_budgets(self):
        self.assertEqual(thinking_config("gemini-3.8-flash-low", "low")["thinkingBudget"], 1000)
        self.assertEqual(thinking_config("gemini-3.8-flash-medium", "medium")["thinkingBudget"], 4000)
        self.assertEqual(thinking_config("gemini-3.8-flash-high", "high")["thinkingBudget"], -1)

    def test_off_disables_thoughts(self):
        config = thinking_config("gemini-3.8-flash-low", "off")
        self.assertEqual(config["thinkingBudget"], 0)
        self.assertFalse(config["includeThoughts"])

    def test_request_preserves_tool_signature(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[
                {"role": "user", "content": "test"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "thought_signature": "signed-value",
                            "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_1", "content": "ok"},
            ],
            reasoning_effort="high",
        )
        model_parts = body["request"]["contents"][1]["parts"]
        call = next(p for p in model_parts if "functionCall" in p)
        self.assertEqual(call["thoughtSignature"], "signed-value")

    def test_request_uses_model_enum(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[{"role": "user", "content": "ping"}],
            reasoning_effort="medium",
        )
        self.assertEqual(body["model"], "gemini-3.8-flash-medium")
        self.assertEqual(body["request"]["labels"]["model_enum"], "MODEL_PLACEHOLDER_M319")


if __name__ == "__main__":
    unittest.main()
