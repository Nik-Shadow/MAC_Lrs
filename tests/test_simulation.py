import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from mas_sim import allocation_trial, resource_trial, reliability_trial, jain_fairness


class MASTests(unittest.TestCase):
    def test_lab2_all_tasks_completed(self):
        for method in ("central", "cnp"):
            result, log = allocation_trial(0, method)
            self.assertEqual(result["tasks"], 48)
            self.assertGreater(result["makespan"], 0)
            self.assertTrue(any(message.kind == "INFORM_DONE" for message in log.items))

    def test_contract_network_has_explicit_protocol(self):
        central, _ = allocation_trial(5, "central")
        cnp, log = allocation_trial(5, "cnp")
        self.assertGreater(cnp["messages"], central["messages"])
        kinds = {message.kind for message in log.items}
        self.assertTrue({"CFP", "PROPOSE", "ACCEPT_PROPOSAL", "REJECT_PROPOSAL", "INFORM_DONE"} <= kinds)

    def test_lab3_exclusive_access(self):
        for seed in range(5):
            result, log = resource_trial(seed, "negotiation")
            self.assertEqual(result["conflict_slots"], 0)
            self.assertLessEqual(result["completed"], 30)
            self.assertTrue(any(message.kind == "COUNTER" for message in log.items))
            self.assertTrue(any(message.kind == "GRANT" for message in log.items))

    def test_jain(self):
        self.assertEqual(jain_fairness([1, 1, 1, 1]), 1.0)
        self.assertLess(jain_fairness([1, 0, 0, 0]), 1.0)

    def test_lab4_validation(self):
        bad, _ = reliability_trial(0, "forged_completion", False)
        good, _ = reliability_trial(0, "forged_completion", True)
        self.assertEqual(bad["false_accepted"], 6)
        self.assertEqual(good["false_accepted"], 0)
        self.assertEqual(good["rejected_invalid"], 6)
        self.assertGreater(good["true_completed"], bad["true_completed"])

    def test_lab4_bid_filter(self):
        bad, _ = reliability_trial(0, "tampered_bid", False)
        good, _ = reliability_trial(0, "tampered_bid", True)
        self.assertGreater(good["true_completed"], bad["true_completed"])
        self.assertGreater(good["rejected_invalid"], 0)

    def test_lab4_reproducible(self):
        x, _ = reliability_trial(17, "message_loss", True)
        y, _ = reliability_trial(17, "message_loss", True)
        self.assertEqual(x, y)


if __name__ == "__main__":
    unittest.main()
