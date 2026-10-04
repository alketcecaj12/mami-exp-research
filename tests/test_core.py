import unittest
from mami.vlm import parse_label
from mami.metrics import evaluate

class CoreTests(unittest.TestCase):
    def test_parser(self):
        self.assertEqual(parse_label("NOT_MISOGYNOUS"), 0)
        self.assertEqual(parse_label(" MISOGYNOUS\n"), 1)
        self.assertIsNone(parse_label("NOT_MISOGYNOUS because..."))

    def test_invalid_is_failure(self):
        report = evaluate([0, 1], [None, 1])
        self.assertEqual(report["accuracy_all"], 0.5)
        self.assertEqual(report["invalid_responses"], 1)
        self.assertAlmostEqual(report["macro_f1_all"], 0.5)

    def test_all_invalid(self):
        report = evaluate([0, 1], [None, None])
        self.assertEqual(report["macro_f1_all"], 0)
        self.assertEqual(report["misogyny_recall"], 0)
        self.assertIsNone(report["cohen_kappa_valid_only"])
