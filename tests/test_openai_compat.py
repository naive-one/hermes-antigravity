import unittest

from hermes_antigravity.openai_compat import openai_completion_object, to_openai_completion


class OpenAICompatTests(unittest.TestCase):
    def test_tool_thought_signature_survives_for_hermes_transport(self):
        completion = to_openai_completion(
            "google-antigravity/gemini-3.8-flash",
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "read_file",
                                        "args": {"path": "a"},
                                    },
                                    "thoughtSignature": "abcdabcd",
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            },
        )
        obj = openai_completion_object(completion)
        tool_call = obj.choices[0].message.tool_calls[0]
        self.assertIsInstance(tool_call.extra_content, dict)
        self.assertEqual(
            tool_call.extra_content["google"]["thought_signature"],
            "abcdabcd",
        )


if __name__ == "__main__":
    unittest.main()
