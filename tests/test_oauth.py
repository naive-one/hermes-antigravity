import unittest
from unittest.mock import patch

from hermes_antigravity.oauth import SCOPES, oauth_client, parse_callback_text


class OAuthTests(unittest.TestCase):
    def test_aicode_scope_present(self):
        self.assertIn("https://www.googleapis.com/auth/aicode", SCOPES)

    @patch.dict(
        "os.environ",
        {
            "ANTIGRAVITY_CLIENT_ID": "custom-client",
            "ANTIGRAVITY_CLIENT_SECRET": "custom-secret",
        },
    )
    def test_oauth_client_env_override(self):
        self.assertEqual(oauth_client(), ("custom-client", "custom-secret"))

    def test_manual_callback_parser(self):
        code = parse_callback_text(
            "http://localhost:51121/oauth-callback?code=abc123&state=s1",
            "s1",
        )
        self.assertEqual(code, "abc123")


if __name__ == "__main__":
    unittest.main()
