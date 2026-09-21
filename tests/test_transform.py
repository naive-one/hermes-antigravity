import unittest

from hermes_antigravity.transform import build_generate_content_request, thinking_config


class TransformTests(unittest.TestCase):
    def test_gemini_38_thinking_budgets(self):
        self.assertEqual(thinking_config("gemini-3.8-flash-low", "low")["thinkingBudget"], 1000)
        self.assertEqual(thinking_config("gemini-3.8-flash-medium", "medium")["thinkingBudget"], 4000)
        self.assertEqual(thinking_config("gemini-3.8-flash-high", "high")["thinkingBudget"], -1)

    def test_unspecified_reasoning_keeps_thoughts_off(self):
        config = thinking_config("gemini-3.8-flash-low", None)
        self.assertEqual(config["thinkingBudget"], 0)
        self.assertFalse(config["includeThoughts"])

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
                            "thought_signature": "abcdabcd",
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
        self.assertEqual(call["thoughtSignature"], "abcdabcd")

    def test_unsigned_gemini_tool_history_becomes_observation(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[
                {"role": "user", "content": "run it"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_2",
                            "type": "function",
                            "function": {"name": "read_file", "arguments": "{\"path\":\"a\"}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_2", "content": "file text"},
            ],
            reasoning_effort="high",
        )
        contents = body["request"]["contents"]
        self.assertFalse(
            any(
                "functionCall" in part
                for turn in contents
                for part in turn.get("parts", [])
            )
        )
        joined = "\n".join(
            part.get("text", "")
            for turn in contents
            for part in turn.get("parts", [])
        )
        self.assertIn("Observation from", joined)
        self.assertIn("file text", joined)

    def test_wire_envelope_matches_antigravity_shape(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[
                {"role": "system", "content": "system rule"},
                {"role": "user", "content": "ping"},
            ],
        )
        request = body["request"]
        self.assertEqual(request["systemInstruction"]["role"], "user")
        self.assertEqual(request["labels"]["last_step_index"], "0")
        self.assertIn("request_id", request["labels"])
        self.assertIn("trajectory_id", request["labels"])
        self.assertEqual(request["labels"]["used_non_gemini_model"], "false")
        self.assertTrue(body["requestId"].startswith("agent/"))

    def test_gemini_tools_use_parameters_json_schema_and_dereference_local_refs(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[{"role": "user", "content": "ping"}],
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "lookup",
                        "description": "lookup",
                        "parameters": {
                            "type": "object",
                            "$defs": {
                                "Query": {
                                    "type": "object",
                                    "properties": {"q": {"type": "string"}},
                                    "required": ["q"],
                                }
                            },
                            "properties": {"query": {"$ref": "#/$defs/Query"}},
                        },
                    },
                }
            ],
        )
        declaration = body["request"]["tools"][0]["functionDeclarations"][0]
        self.assertIn("parametersJsonSchema", declaration)
        self.assertNotIn("parameters", declaration)
        query_schema = declaration["parametersJsonSchema"]["properties"]["query"]
        self.assertEqual(query_schema["type"], "object")
        self.assertEqual(query_schema["properties"]["q"]["type"], "string")

    def test_native_reasoning_carrier_replays_text_and_thinking_signatures(self):
        body = build_generate_content_request(
            model="google-antigravity/gemini-3.8-flash",
            project_id="p",
            messages=[
                {"role": "user", "content": "first"},
                {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_content": "thinking",
                    "reasoning_details": [
                        {
                            "type": "antigravity.native_assistant",
                            "parts": [
                                {
                                    "kind": "thinking",
                                    "text": "thinking",
                                    "thoughtSignature": "abcdabcd",
                                },
                                {
                                    "kind": "text",
                                    "text": "answer",
                                    "thoughtSignature": "efghefgh",
                                },
                            ],
                        }
                    ],
                },
                {"role": "user", "content": "continue"},
            ],
            reasoning_effort="high",
        )
        model_turn = next(
            turn for turn in body["request"]["contents"] if turn["role"] == "model"
        )
        thinking = next(part for part in model_turn["parts"] if part.get("thought"))
        answer = next(
            part
            for part in model_turn["parts"]
            if part.get("text") == "answer" and not part.get("thought")
        )
        self.assertEqual(thinking["thoughtSignature"], "abcdabcd")
        self.assertEqual(answer["thoughtSignature"], "efghefgh")

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
