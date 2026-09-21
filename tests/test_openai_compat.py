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


    def test_signed_text_and_thinking_use_native_reasoning_carrier(self):
        completion = to_openai_completion(
            "google-antigravity/gemini-3.8-flash",
            {
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "thought": True,
                                    "text": "thinking",
                                    "thoughtSignature": "abcdabcd",
                                },
                                {
                                    "text": "answer",
                                    "thoughtSignature": "efghefgh",
                                },
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {},
            },
        )
        message = completion["choices"][0]["message"]
        details = message["reasoning_details"]
        self.assertEqual(details[0]["type"], "antigravity.native_assistant")
        self.assertEqual(details[0]["parts"][0]["thoughtSignature"], "abcdabcd")
        self.assertEqual(details[0]["parts"][1]["thoughtSignature"], "efghefgh")
        obj = openai_completion_object(completion)
        self.assertIsInstance(obj.choices[0].message.reasoning_details, list)
        self.assertIsInstance(obj.choices[0].message.reasoning_details[0], dict)
        self.assertIsInstance(obj.choices[0].message.reasoning_details[0]["parts"][0], dict)


if __name__ == "__main__":
    unittest.main()
