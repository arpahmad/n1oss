"""Which quant n1os recommends for a GPU configuration. No GPU, network or downloads."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import n1os


def cards(n, vram=24.0, name="NVIDIA GeForce RTX 3090"):
    return [{"index": i, "name": name, "vram_gb": vram, "arch": "86"} for i in range(n)]


class GpuPlanTests(unittest.TestCase):
    def test_3090_count_picks_the_quant_that_stays_on_the_gpus(self):
        # 1-3x 24 GB: experts mostly in RAM, so the 4-bit attention (Maya-S24) leaves more room.
        # 4x: Maya-S keeps ~90% of its experts in VRAM. 5x: Maya-M. 6x and up: the 3.5-bit quant.
        expect = {
            1: "Maya-S24", 2: "Maya-S24", 3: "Maya-S24",
            4: "Maya-S-v2-IQ2_XXS", 5: "Maya-M",
            6: "GSQ-RCO-3.5bit", 7: "GSQ-RCO-3.5bit", 8: "GSQ-RCO-3.5bit",
        }
        for n, model in expect.items():
            plan = n1os.gpu_plan(cards(n), ram_gb=128)
            self.assertEqual(plan["model"], model, plan["summary"])
            self.assertIn("NVIDIA GeForce RTX 3090", plan["label"])
            if n > 1:
                self.assertIn(f"{n}x NVIDIA GeForce RTX 3090", plan["label"])
            self.assertGreaterEqual(plan["coverage"][model], 0.15)

    def test_four_3090s_keep_maya_s_and_a_64k_context(self):
        plan = n1os.gpu_plan(cards(4), ram_gb=128)
        self.assertEqual(plan["model"], "Maya-S-v2-IQ2_XXS")
        self.assertGreaterEqual(plan["coverage"]["Maya-S-v2-IQ2_XXS"], n1os.FIT_GOOD)
        self.assertLess(plan["coverage"]["Maya-M"], n1os.FIT_GOOD)
        self.assertEqual(plan["context"], 65536)
        self.assertIn("Maya-S", plan["summary"])

    def test_five_3090s_keep_maya_m(self):
        plan = n1os.gpu_plan(cards(5), ram_gb=128)
        self.assertEqual(plan["model"], "Maya-M")
        self.assertGreaterEqual(plan["coverage"]["Maya-M"], n1os.FIT_GOOD)
        self.assertLess(plan["coverage"]["GSQ-RCO-3.5bit"], n1os.FIT_GOOD)
        self.assertEqual(plan["context"], 65536)

    def test_six_3090s_keep_the_largest_quant(self):
        plan = n1os.gpu_plan(cards(6), ram_gb=128)
        self.assertEqual(plan["model"], "GSQ-RCO-3.5bit")
        self.assertGreaterEqual(plan["coverage"]["GSQ-RCO-3.5bit"], n1os.FIT_GOOD)
        self.assertEqual(plan["context"], 65536)

    def test_eight_3090s_fit_the_largest_quant_at_128k(self):
        plan = n1os.gpu_plan(cards(8), ram_gb=256)
        self.assertEqual(plan["model"], "GSQ-RCO-3.5bit")
        self.assertGreaterEqual(plan["coverage"]["GSQ-RCO-3.5bit"], 1.0)
        self.assertEqual(plan["context"], 131072)

    def test_a_pair_keeps_a_drafting_quant(self):
        # Two 80 GB cards fit GSQ-RCO too, but a pair drafts and GSQ-RCO has no draft block.
        plan = n1os.gpu_plan(cards(2, vram=80, name="NVIDIA A100"), ram_gb=256)
        self.assertEqual(plan["model"], "Maya-M")
        self.assertGreaterEqual(plan["coverage"]["GSQ-RCO-3.5bit"], 1.0)

    def test_one_32gb_card_keeps_maya_s(self):
        plan = n1os.gpu_plan(cards(1, vram=32, name="Tesla V100"), ram_gb=64)
        self.assertEqual(plan["model"], "Maya-S-v2-IQ2_XXS")
        self.assertEqual(plan["context"], 32768)

    def test_two_32gb_cards_keep_maya_s(self):
        plan = n1os.gpu_plan(cards(2, vram=32, name="Tesla V100"), ram_gb=64)
        self.assertEqual(plan["model"], "Maya-S-v2-IQ2_XXS")

    def test_mixed_nine_gpus_fit_the_largest_quant(self):
        gpus = cards(8, vram=16, name="NVIDIA GeForce RTX 5060 Ti")
        gpus.append({"index": 8, "name": "NVIDIA GeForce RTX 3090", "vram_gb": 24, "arch": "86"})
        plan = n1os.gpu_plan(gpus, ram_gb=128)
        self.assertEqual(plan["model"], "GSQ-RCO-3.5bit")
        self.assertIn("152 GB", plan["label"])

    def test_ram_pin_follows_the_spill(self):
        four = cards(4)
        self.assertEqual(n1os.optimize_env(four, "Maya-S-v2-IQ2_XXS", 128),
                         {"STRATA_GLM_RAM_RESIDENT": "1"})
        self.assertEqual(n1os.optimize_env(four, "Maya-S-v2-IQ2_XXS", 24), {})
        self.assertEqual(n1os.optimize_env(cards(1), "Maya-S24", 64), {})
        self.assertEqual(n1os.optimize_env(cards(1), "Maya-S24", 128),
                         {"STRATA_GLM_RAM_RESIDENT": "1"})
        self.assertEqual(n1os.optimize_env(cards(8), "GSQ-RCO-3.5bit", 256), {})

    def test_unified_memory_is_sized_from_ram(self):
        apu = [{"index": 0, "name": "Radeon 8060S", "vram_gb": 0.5, "arch": "gfx1151"}]
        plan = n1os.gpu_plan(apu, ram_gb=192)
        self.assertEqual(plan["model"], "GSQ-RCO-3.5bit")
        self.assertIn("192 GB shared memory", plan["label"])
        self.assertEqual(n1os.optimize_env(apu, plan["model"], 192), {})

    def test_config_records_the_profile_and_pins_ram(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        gpus = cards(4)
        pc = {"backend": "cuda", "gpus": gpus, "archs": [86]}
        a = SimpleNamespace(port=8080, env=[], host=None, api_key=None, gguf_dir=None)
        with patch.object(n1os, "ROOT", root), patch.object(n1os, "EXE", root / "strata"), \
                patch.object(n1os, "mem_gb", return_value=(128, 100)), \
                patch.object(n1os, "saved_calibration", return_value=None):
            path = n1os.write_config(a, pc, {"lib_dirs": []}, root / "pack",
                                     "Maya-S-v2-IQ2_XXS", 32768, root / "models", None)
        cfg = json.loads(path.read_text())
        self.assertEqual(cfg["gpu"], [0, 1, 2, 3])
        self.assertEqual(cfg["gpu_profile"]["recommended"], "Maya-S-v2-IQ2_XXS")
        self.assertIn("4x NVIDIA GeForce RTX 3090", cfg["gpu_profile"]["label"])
        self.assertGreaterEqual(cfg["gpu_profile"]["expert_coverage"], 0.85)
        self.assertEqual(cfg["env"]["STRATA_GLM_RAM_RESIDENT"], "1")

    def test_an_explicit_env_wins_over_the_pin(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        root = Path(tmp.name)
        a = SimpleNamespace(port=8080, env=["STRATA_GLM_RAM_RESIDENT=0"], host=None, api_key=None, gguf_dir=None)
        with patch.object(n1os, "ROOT", root), patch.object(n1os, "EXE", root / "strata"), \
                patch.object(n1os, "mem_gb", return_value=(128, 100)), \
                patch.object(n1os, "saved_calibration", return_value=None):
            path = n1os.write_config(a, {"backend": "cuda", "gpus": cards(4)}, {"lib_dirs": []},
                                     root / "pack", "Maya-S-v2-IQ2_XXS", 32768, root / "models", None)
        self.assertEqual(json.loads(path.read_text())["env"]["STRATA_GLM_RAM_RESIDENT"], "0")


if __name__ == "__main__":
    unittest.main()
