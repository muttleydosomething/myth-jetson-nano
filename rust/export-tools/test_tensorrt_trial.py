"""Focused ownership/cancellation proof; no CUDA or speech service required."""
import os
from pathlib import Path
import sys
import tempfile
import unittest
import tensorrt_trial as trial


class CleanupTest(unittest.TestCase):
    def test_tensorrt_trial_cleanup(self):
        # A pressure stop must terminate the exact child process group and retain
        # its failure receipt. The helper never manipulates other service states.
        with tempfile.TemporaryDirectory() as directory:
            result = trial.guarded([sys.executable, "-c", "import time; time.sleep(30)"],
                                   Path(directory), reserve=1, available=lambda: 0)
            self.assertEqual(result["reason"], "memory-reserve")
            self.assertIsNotNone(result["returncode"])
            self.assertTrue((Path(directory) / "command.json").exists())
            with self.assertRaises(ProcessLookupError):
                os.kill(result["pid"], 0)


class DistributionTest(unittest.TestCase):
    def test_identical_distributions(self):
        result = trial.distribution_metrics([0.25, 0.75, 0], [0.25, 0.75, 0])
        self.assertEqual(result, {"kl": 0.0, "kl_infinite": False, "js": 0.0,
                                  "total_variation": 0.0})

    def test_filtered_support_change_is_not_hidden(self):
        import math
        result = trial.distribution_metrics([1, 0], [0, 1])
        self.assertTrue(result["kl_infinite"])
        self.assertIsNone(result["kl"])
        self.assertAlmostEqual(result["js"], math.log(2))
        self.assertEqual(result["total_variation"], 1)

    def test_invalid_distribution_cannot_pass(self):
        for values in ([0, 0], [float("nan"), 1], [-1, 2]):
            with self.assertRaises(AssertionError):
                trial.distribution_metrics(values, [0.5, 0.5])


class EmergencyQuantizationTest(unittest.TestCase):
    def test_gpt2_transposed_weights_preserve_channel_orientation(self):
        import numpy as np
        weights = np.array([[1, -2, 3], [4, 1, -2]], dtype=np.float32)
        maxima = np.array([5, 2], dtype=np.float32)
        plain = trial.emergency_projection(weights, maxima)
        transposed = trial.emergency_projection(weights.T, maxima, transposed=True)
        np.testing.assert_array_equal(plain[0].T, transposed[0])
        np.testing.assert_array_equal(plain[1], transposed[1])
        values = np.array([[3, -1]], dtype=np.float32)
        rebuilt = plain[0].astype(np.float32) * plain[1][None, :]
        np.testing.assert_allclose((values * plain[2]) @ rebuilt,
                                   values @ weights, atol=0.06, rtol=0.01)

    def test_zero_channels_keep_finite_positive_scales(self):
        import numpy as np
        quantized, scales, inverse, activation = trial.emergency_projection(
            np.zeros((2, 3), dtype=np.float32), np.zeros(2))
        self.assertEqual(quantized.dtype, np.int8)
        self.assertTrue(np.all(quantized == 0))
        self.assertTrue(np.all(scales > 0) and np.isfinite(scales).all())
        self.assertTrue(np.isfinite(inverse).all() and activation > 0)

    def test_invalid_calibration_cannot_create_an_engine(self):
        import numpy as np
        for maxima in ([1], [float("nan"), 1], [-1, 1]):
            with self.assertRaises(ValueError):
                trial.emergency_projection(np.ones((2, 3)), np.array(maxima))


class EmergencyLifecycleTest(unittest.TestCase):
    def test_failed_restoration_keeps_explicit_failure_receipt(self):
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "candidate"
            receipt = Path(directory) / "backup.json"
            receipt.write_text(json.dumps({"archive_readback": True,
                "image_layers_verified": True, "original_image": "original",
                "snapshot_image": "snapshot"}))
            deployed = {"Image": "original", "State": {"Running": True}, "Config": {"Env": []}}

            def run(argv, **kwargs):
                if argv[-2:] == ["start", "chatterbox-tts"]:
                    raise OSError("start unavailable")
                return SimpleNamespace(returncode=1)

            with patch.object(trial.subprocess, "check_output", return_value=json.dumps([deployed])), \
                 patch.object(trial.subprocess, "run", side_effect=run), \
                 patch.object(trial, "available", return_value=0):
                with self.assertRaisesRegex(OSError, "start unavailable"):
                    trial.emergency_trial(root, receipt)
            result = json.loads((root / "emergency-result.json").read_text())
            self.assertFalse(result["original_restored"])
            self.assertIn("start unavailable", result["restoration_error"])
            self.assertTrue(result["finished_at"])

    def test_failed_or_cancelled_trial_restores_original_with_matching_paths(self):
        import io
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        for cancelled, absent in ((False, False), (True, False), (False, True)):
            with self.subTest(cancelled=cancelled, absent=absent), tempfile.TemporaryDirectory() as directory:
                root = Path(directory) / "candidate"
                receipt = Path(directory) / "backup.json"
                receipt.write_text(json.dumps({"archive_readback": True,
                    "image_layers_verified": True, "original_image": "original",
                    "snapshot_image": "snapshot"}))
                deployed = {"Image": "original", "State": {"Running": True},
                            "Config": {"Env": []}}

                def inspect(argv, **kwargs):
                    return json.dumps([deployed if argv[-1] == "chatterbox-tts"
                                       else {"State": {"Running": False}}])

                def command(argv, **kwargs):
                    if argv[-2] == "inspect":
                        return SimpleNamespace(returncode=1 if absent else 0,
                            stderr="No such object: " + argv[-1] if absent else "",
                            stdout=json.dumps([{"State": {"Running": False}}]))
                    return SimpleNamespace(returncode=1)

                def interrupted(argv, output, **kwargs):
                    # The executing path and writable mount must agree, so
                    # sibling corpus authorities resolve inside the container.
                    self.assertEqual(argv[-1], str(root))
                    self.assertIn(f"{root}:{root}:rw", argv)
                    self.assertIn(f"{root.parent}:{root.parent}:ro", argv)
                    if cancelled:
                        raise InterruptedError("operator cancellation")
                    return {"reason": "test rejection", "returncode": 1}

                with patch.object(trial.subprocess, "check_output", side_effect=inspect), \
                     patch.object(trial.subprocess, "run", side_effect=command) as run, \
                     patch.object(trial, "available", return_value=5 * 1024**3), \
                     patch.object(trial, "guarded", side_effect=interrupted), \
                     patch("urllib.request.urlopen", return_value=io.StringIO(
                         '{"model_info":{"loaded":true}}')):
                    with self.assertRaises(InterruptedError if cancelled else AssertionError):
                        trial.emergency_trial(root, receipt)
                    self.assertTrue(any(call.args[0][-2:] == ["start", "chatterbox-tts"]
                                        for call in run.call_args_list))
                    result = json.loads((root / "emergency-result.json").read_text())
                    self.assertTrue(result["original_restored"])
                    self.assertFalse(result["production_activation"])


class AudioAssessmentTest(unittest.TestCase):
    def test_word_accounting(self):
        self.assertEqual(trial.word_errors("Ready, Holmes.", "ready holmes"), (0, 2))
        self.assertEqual(trial.word_errors("The ship is ready", "ship was ready now"), (3, 4))

    def test_bootstrap_keeps_paired_word_weights(self):
        rows = len(trial.REAL_TEXTS)
        result = trial.paired_bound([1] * rows, [10] * rows)
        self.assertAlmostEqual(result["difference"], 0.1)
        self.assertAlmostEqual(result["upper_95"], 0.1)
        self.assertEqual(trial.paired_bound([0] * rows), {"difference": 0.0, "upper_95": 0.0})


if __name__ == "__main__":
    unittest.main()
