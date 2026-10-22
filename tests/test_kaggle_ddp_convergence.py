"""YU-344: the Kaggle 2xT4 DDP path converged onto the canonical training primitives.

Weight-free checks. This fork moved the RLCD loop out of the notebook into
`laya/finetune.py`, so the same code runs on the Kaggle T4 pair, MPS and CPU; the notebook
drives it instead of owning a copy. The loop keeps the process-group / shard orchestration
but delegates the model, the reward and the temperature fit to `laya.common` /
`laya.calibrate`. Upstream still writes the loop into a `%%writefile train_ddp.py` cell and
asserts its imports there; asserting the delegating shape here is what keeps this fork's
notebook from growing a private copy back. The last four classes cover the shared
primitives the loop is built from, and pass either way.

Run: python tests/test_kaggle_ddp_convergence.py
"""
import json
import os
import re
import sys
import unittest
from pathlib import Path

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402

from laya.calibrate import fit_temperature_map  # noqa: E402
from laya.common import TEMP_MAX, TEMP_MIN, collate_items  # noqa: E402
from laya.train import sigma_at, split_calibration  # noqa: E402

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "laya_finetune_typed_decisions_2xT4_kaggle.ipynb"
MODULE = Path(__file__).resolve().parents[1] / "laya" / "finetune.py"


class NotebookDelegationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
        cls.cells = ["".join(cell["source"]) for cell in cls.nb["cells"]]
        cls.loop = MODULE.read_text(encoding="utf-8")

    def test_notebook_writes_no_training_script(self):
        written = [cell.splitlines()[0] for cell in self.cells if cell.startswith("%%writefile")]
        self.assertEqual(written, [], "the notebook owns a private training script again")

    def test_notebook_delegates_the_loop(self):
        self.assertTrue(
            any(re.search(r"from laya\.finetune import .*train_rlcd", cell) for cell in self.cells),
            "the notebook no longer drives laya.finetune.train_rlcd")

    def test_module_owns_the_orchestration(self):
        self.assertIn("def train_rlcd", self.loop)
        # Sharding is module-private: it is the part that stays out of the notebook.
        self.assertIn("train_items[rank::world_size]", self.loop)

    def test_module_delegates_model_and_reward(self):
        # ...while the semantics come from the canonical primitives, not a second copy of them.
        self.assertIn("from .common import build_model, proper_reward", self.loop)
        for gone in ("def proper_reward", "class DecisionModel"):
            self.assertNotIn(gone, self.loop)

    def test_calibration_held_out_before_sharding(self):
        holdout_at = self.loop.index("calib_idx")
        shard_at = self.loop.index("train_items[rank::world_size]")
        self.assertLess(holdout_at, shard_at)


class SplitCalibrationTest(unittest.TestCase):
    def test_disjoint_deterministic_and_seeded(self):
        items = list(range(1000))
        tr1, ca1 = split_calibration(items, calib_max=400, calib_frac=0.1, seed=20260922)
        tr2, ca2 = split_calibration(items, calib_max=400, calib_frac=0.1, seed=20260922)
        self.assertEqual(ca1, ca2)
        self.assertEqual(tr1, tr2)
        self.assertEqual(len(ca1), 100)
        self.assertFalse(set(tr1) & set(ca1))
        self.assertEqual(len(tr1) + len(ca1), 1000)

    def test_calib_max_caps_count(self):
        items = list(range(500))
        _, ca = split_calibration(items, calib_max=10, calib_frac=0.5, seed=0)
        self.assertEqual(len(ca), 10)


class CollateTest(unittest.TestCase):
    def test_collate_items_matches_notebook_shape(self):
        encoded = [
            {"ids": [1, 2, 3], "markers": [0, 2], "qtype": 0, "target": [0.7, 0.3]},
            {"ids": [4, 5], "markers": [1], "qtype": 2, "target": [1.0]},
        ]
        batch = collate_items([encoded], pad_id=0)
        self.assertEqual(batch["input_ids"].shape, (2, 3))
        self.assertEqual(batch["attention_mask"].tolist(), [[1, 1, 1], [1, 1, 0]])
        self.assertEqual(batch["marker_pos"].tolist(), [[0, 2], [1, 0]])
        self.assertEqual(batch["marker_mask"].tolist(), [[True, True], [True, False]])
        got = batch["target"].tolist()
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(got[0][0], 0.7, places=6)
        self.assertAlmostEqual(got[0][1], 0.3, places=6)
        self.assertEqual(got[1], [1.0, 0.0])
        self.assertEqual(batch["qtype"].tolist(), [0, 2])


class SigmaScheduleTest(unittest.TestCase):
    def test_linear_endpoints(self):
        self.assertAlmostEqual(sigma_at(0, 4, 0.4, 0.1), 0.4)
        self.assertAlmostEqual(sigma_at(3, 4, 0.4, 0.1), 0.1)
        self.assertAlmostEqual(sigma_at(1, 4, 0.4, 0.1), 0.3)


class TemperatureFitTest(unittest.TestCase):
    def test_fit_temperature_map_respects_runtime_bounds(self):
        rng = np.random.default_rng(0)
        records = []
        for i in range(40):
            k = int(rng.integers(2, 5))
            logits = rng.normal(0, 1, k)
            target = rng.dirichlet(np.ones(k))
            records.append((i % 3, logits, target, k))
        fitted = fit_temperature_map(records)
        for t in fitted["temperature"]:
            self.assertGreaterEqual(t, TEMP_MIN)
            self.assertLessEqual(t, TEMP_MAX)
        for v in fitted["temperature_by_options"].values():
            self.assertGreaterEqual(v, TEMP_MIN)
            self.assertLessEqual(v, TEMP_MAX)


if __name__ == "__main__":
    unittest.main()
