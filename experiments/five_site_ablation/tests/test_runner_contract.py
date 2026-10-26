from __future__ import annotations

import re
import unittest
from pathlib import Path


EXPERIMENT_DIR = Path(__file__).resolve().parents[1]


class RunnerContractTests(unittest.TestCase):
    def test_default_smoke_index_is_silver_bow(self):
        runner = (EXPERIMENT_DIR / "run_ablation_arm.slurm").read_text(encoding="utf-8")
        sites = (EXPERIMENT_DIR / "five_site_benchmark.tsv").read_text(encoding="utf-8").splitlines()
        match = re.search(r'SMOKE_SITE_INDEX:-([0-9]+)', runner)
        self.assertIsNotNone(match)
        row_index = int(match.group(1))
        site_id, folder, name = sites[row_index].split("\t")
        self.assertEqual(site_id, "MTD980502777")
        self.assertEqual(folder, "SILVER_BOW_CREEK_BUTTE_AREA")
        self.assertEqual(name, "Silver Bow Creek/Butte Area")

    def test_site_manifest_is_hashed_into_run_identity(self):
        runner = (EXPERIMENT_DIR / "run_ablation_arm.slurm").read_text(encoding="utf-8")
        self.assertIn('SITE_FILE_SHA=$(sha256sum "$SITE_FILE"', runner)
        self.assertIn('printf \'site_file_sha256\\t%s\\n\'', runner)
        self.assertIn('"$SITE_FILE_SHA" | sha256sum', runner)


if __name__ == "__main__":
    unittest.main()
