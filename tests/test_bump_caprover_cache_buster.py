from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.bump_caprover_cache_buster import image_ref, update_definition


class BumpCaproverCacheBusterTest(unittest.TestCase):
    def test_updates_existing_definition(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "captain-definition"
            path.write_text(
                json.dumps(
                    {
                        "schemaVersion": 2,
                        "dockerfilePath": "Dockerfile.app",
                        "buildArgs": {
                            "CACHE_BUSTER": "4",
                            "BROWSER_USE_BASE_IMAGE": "browseruse/browseruse:latest",
                        },
                    }
                ),
                encoding="utf-8",
            )

            update_definition(path, "ghcr.io/owner/browser-use:sha-123456789abc")

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["buildArgs"]["CACHE_BUSTER"], "5")
            self.assertEqual(data["buildArgs"]["BROWSER_USE_BASE_IMAGE"], "ghcr.io/owner/browser-use:sha-123456789abc")

    def test_invalid_cache_buster_resets_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = Path(tmp_dir) / "captain-definition"
            path.write_text(
                json.dumps({"schemaVersion": 2, "dockerfilePath": "Dockerfile.app", "buildArgs": {"CACHE_BUSTER": "x"}}),
                encoding="utf-8",
            )

            update_definition(path, "ghcr.io/owner/browser-use:sha-123456789abc")

            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(data["buildArgs"]["CACHE_BUSTER"], "1")

    def test_image_ref_uses_lowercase_repository_and_12_char_sha(self) -> None:
        self.assertEqual(
            image_ref("Owner/Browser-Use", "123456789abcdef"),
            "ghcr.io/owner/browser-use:sha-123456789abc",
        )


if __name__ == "__main__":
    unittest.main()
