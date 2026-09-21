import unittest
from unittest.mock import patch

from hermes_antigravity.cloudcode import (
    default_project_id,
    load_or_onboard_project,
    stable_project_id,
)


class CloudCodeTests(unittest.TestCase):
    def test_stable_project_id_is_deterministic_uuid(self):
        first = stable_project_id("user@example.com")
        second = stable_project_id("user@example.com")
        other = stable_project_id("other@example.com")
        self.assertEqual(first, second)
        self.assertNotEqual(first, other)
        self.assertEqual(len(first), 36)
        self.assertEqual(first[14], "5")

    @patch.dict("os.environ", {}, clear=False)
    def test_project_discovery_failure_uses_stable_account_fallback(self):
        def fail_post(url, body, headers):
            raise RuntimeError("offline")

        project = load_or_onboard_project(
            "token",
            seed="user@example.com",
            post_json=fail_post,
        )
        self.assertEqual(project, stable_project_id("user@example.com"))

    @patch.dict(
        "os.environ",
        {"ANTIGRAVITY_PROJECT_ID": "explicit-project"},
        clear=False,
    )
    def test_project_env_override_wins(self):
        self.assertEqual(default_project_id("anything"), "explicit-project")
        self.assertEqual(
            load_or_onboard_project("token", seed="anything"),
            "explicit-project",
        )


if __name__ == "__main__":
    unittest.main()
