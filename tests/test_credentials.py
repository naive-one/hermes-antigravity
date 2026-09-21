import json
import tempfile
import unittest
from pathlib import Path

from hermes_antigravity.credentials import CredentialStore


class CredentialStoreTests(unittest.TestCase):
    def test_account_pool_add_activate_remove(self):
        with tempfile.TemporaryDirectory() as tmp:
            store = CredentialStore(Path(tmp) / "accounts.json")
            a = store.upsert(
                {"email": "a@example.com", "access_token": "a", "refresh_token": "ra"},
                activate=True,
            )
            b = store.upsert(
                {"email": "b@example.com", "access_token": "b", "refresh_token": "rb"},
                activate=False,
            )
            self.assertEqual(store.load()["email"], "a@example.com")
            self.assertTrue(store.activate(b))
            self.assertEqual(store.load()["email"], "b@example.com")
            self.assertTrue(store.remove(b))
            self.assertEqual(store.load()["email"], "a@example.com")

    def test_file_contains_no_plaintext_permissions_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "accounts.json"
            store = CredentialStore(path)
            store.upsert({"email": "a@example.com", "access_token": "a"})
            data = json.loads(path.read_text())
            self.assertIn("accounts", data)
            self.assertEqual(data["active"], "a@example.com")


if __name__ == "__main__":
    unittest.main()
