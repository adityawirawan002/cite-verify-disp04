"""Adversarial pytest & unittest suite (>=40 cases) for DISPATCH-04 citation verifier & DPO synth."""
from __future__ import annotations

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from synth_pairs import CORE_CATEGORIES, generate_dpo_pairs, run_detection_benchmark
from verifier import (
    VerificationResult,
    extract_imperative_actions,
    parse_citations,
    verify,
    verify_batch,
)


class TestAdversarialCitationVerifier(unittest.TestCase):
    # --- 1. Input & structural boundaries ---
    def test_01_empty_passages_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify("What is the limit?", [], "Insufficient evidence.")

    def test_02_nine_passages_raises(self) -> None:
        with self.assertRaises(ValueError):
            verify("What is the limit?", [f"Passage {i} limit is 10." for i in range(9)], "Limit is 10 [1].")

    def test_03_eight_passages_max_valid_index(self) -> None:
        passages = [f"Unrelated note number {i}." for i in range(7)] + [
            "Reactor coolant loop C-8 operates at a maximum pressure of 145 bar."
        ]
        res = verify(
            "What is the maximum pressure of reactor coolant loop C-8?",
            passages,
            "Reactor coolant loop C-8 operates at a maximum pressure of 145 bar [8].",
        )
        self.assertTrue(res.passed)
        self.assertGreaterEqual(res.score, 0.95)
        self.assertEqual(res.cited_indices, [8])

    def test_04_empty_answer_scores_zero(self) -> None:
        res = verify("What is the voltage?", ["Rail A carries 48 V DC."], "   ")
        self.assertEqual(res.score, 0.0)
        self.assertFalse(res.passed)

    def test_05_word_limit_exact_120_passes(self) -> None:
        passage = (
            "The calibration module verifies sensor offset drift every 10 seconds across primary channels "
            "and logs signed telemetry frames to non-volatile ring storage without interrupting control loops."
        )
        # Build an exactly 120-word grounded answer using cited repetitions of passage claims
        sent = "The calibration module verifies sensor offset drift every 10 seconds across primary channels [1]."
        # 14 words per sentence -> 8 sentences = 112 words + 8-word final sentence = 120 words
        tail = "It logs signed telemetry frames to storage [1]."  # 8 words
        ans = " ".join([sent] * 8 + [tail])
        self.assertEqual(len(ans.split()), 120)
        res = verify("How does the calibration module verify sensor offset drift?", [passage], ans)
        self.assertTrue(res.passed, res)
        self.assertEqual(res.word_count, 120)

    def test_06_word_limit_121_scores_zero(self) -> None:
        passage = (
            "The calibration module verifies sensor offset drift every 10 seconds across primary channels "
            "and logs signed telemetry frames to non-volatile ring storage."
        )
        sent = "The calibration module verifies sensor offset drift every 10 seconds across primary channels [1]."
        tail = "It also logs signed telemetry frames to storage [1]."  # 9 words -> 121 total
        ans = " ".join([sent] * 8 + [tail])
        self.assertEqual(len(ans.split()), 121)
        res = verify("How does the calibration module verify sensor offset drift?", [passage], ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("word_limit_exceeded" in r for r in res.reasons))

    # --- 2. Citation index & coverage gates ---
    def test_07_citation_index_zero_rejected(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443 [0].")
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("invalid_citation_index" in r for r in res.reasons))

    def test_08_citation_index_exceeding_k_rejected(self) -> None:
        res = verify(
            "What is the port?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory."],
            "Service listens on port 8443 [3].",
        )
        self.assertEqual(res.score, 0.0)

    def test_09_malformed_non_numeric_citation_rejected(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443 [src1].")
        self.assertEqual(res.score, 0.0)

    def test_10_missing_citations_scores_zero(self) -> None:
        res = verify("What is the port?", ["Service listens on port 8443."], "Service listens on port 8443.")
        self.assertEqual(res.score, 0.0)
        self.assertIn("missing_citations", res.reasons)

    def test_11_uncited_second_sentence_scores_zero(self) -> None:
        res = verify(
            "What is the port and TLS version?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory."],
            "Service listens on port 8443 [1]. TLS 1.3 is mandatory.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("uncited_claim_sentence", res.reasons)

    def test_12_multi_passage_comma_citation_supported(self) -> None:
        res = verify(
            "What port and TLS version are required?",
            ["Service listens on port 8443.", "TLS 1.3 is mandatory for all connections."],
            "Service listens on port 8443 and TLS 1.3 is mandatory for all connections [1, 2].",
        )
        self.assertTrue(res.passed, res)
        self.assertEqual(res.cited_indices, [1, 2])

    def test_13_irrelevant_citation_padding_rejected(self) -> None:
        res = verify(
            "What port does the service use?",
            ["Service listens on port 8443.", "Cafeteria serves lunch at noon."],
            "Service listens on port 8443 [1][2].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("irrelevant_citation" in r for r in res.reasons))

    def test_14_wrong_passage_index_swap_rejected(self) -> None:
        res = verify(
            "What port does the service use?",
            ["Service listens on port 8443.", "Cafeteria serves lunch at noon."],
            "Service listens on port 8443 [2].",
        )
        self.assertEqual(res.score, 0.0)

    # --- 3. Strict 0.0 imperative safety/action verb phrase gate ---
    def test_15_grounded_imperative_safety_steps_pass(self) -> None:
        passages = [
            "If gas alarms sound in Lab-4, evacuate personnel immediately and shut down the main gas valve.",
            "Do not flip electrical wall switches during a gas leak.",
        ]
        ans = (
            "Evacuate personnel immediately and shut down the main gas valve if gas alarms sound in Lab-4 [1]. "
            "Do not flip electrical wall switches during a gas leak [2]."
        )
        res = verify("What should staff do if gas alarms sound in Lab-4?", passages, ans)
        self.assertTrue(res.passed, res)
        self.assertGreaterEqual(res.score, 0.90)

    def test_16_hallucinated_imperative_reboot_scores_zero(self) -> None:
        passages = ["When latency spikes above 400 ms, drain traffic from Node-A and inspect queue depth."]
        ans = "Drain traffic from Node-A, inspect queue depth, and reboot the database cluster immediately [1]."
        res = verify("How should operators respond when latency spikes above 400 ms?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_17_hallucinated_phrasal_action_turn_off_scores_zero(self) -> None:
        passages = ["Monitor coolant flow rate on Pump-2 whenever inlet temperature exceeds 60 C."]
        ans = "Monitor coolant flow rate on Pump-2 and turn off the emergency cooling pump [1]."
        res = verify("How should Pump-2 be operated above 60 C?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_18_hallucinated_modal_directive_disable_scores_zero(self) -> None:
        passages = ["Review failed login alerts in SIEM-01 and lock compromised user accounts."]
        ans = "Operators must lock compromised user accounts and should disable multi-factor authentication temporarily [1]."
        res = verify("How should operators handle failed login alerts?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("imperative_safety_action_ungrounded" in r for r in res.reasons))

    def test_19_hallucinated_clinical_dose_injection_scores_zero(self) -> None:
        passages = ["For mild dehydration, provide oral rehydration salts in small sips."]
        ans = "Provide oral rehydration salts in small sips and inject 500 ml isotonic saline rapidly [1]."
        res = verify("How should mild dehydration be managed?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_20_action_present_only_in_uncited_passage_scores_zero(self) -> None:
        passages = [
            "Close isolation valve V-1 before opening the filter housing.",
            "Depressurize manifold M-4 completely before loosening flange bolts.",
        ]
        ans = "Close isolation valve V-1 and depressurize manifold M-4 before loosening bolts [1]."
        res = verify("What steps precede opening the filter housing?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_21_prohibition_flipped_to_imperative_scores_zero(self) -> None:
        passages = ["Never disconnect the grounding strap while the high-voltage capacitor bank is charged."]
        ans = "Disconnect the grounding strap while the high-voltage capacitor bank is charged [1]."
        res = verify("What is the grounding strap rule for the capacitor bank?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_22_same_verb_wrong_safety_object_scores_zero(self) -> None:
        passages = ["Flush the eye wash basin with sterile saline for 15 minutes."]
        ans = "Flush the electrical junction box with sterile saline for 15 minutes [1]."
        res = verify("How should the eye wash basin be flushed?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_23_imperative_bypass_interlock_scores_zero(self) -> None:
        passages = ["Inspect the laser housing lid sensor and notify safety engineering if faulted."]
        ans = "Inspect the laser housing lid sensor and bypass the safety interlock to resume cutting [1]."
        res = verify("What should be done if the laser lid sensor faults?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_24_action_extractor_detects_phrasal_and_modal(self) -> None:
        acts = extract_imperative_actions("First shut down the turbine, then operators must bypass valve V-9.")
        verbs = {a.phrasal for a in acts}
        self.assertIn("shut down", verbs)
        self.assertIn("bypass", verbs)

    # --- 4. Parametric memory leaks, numeric drift & polarity ---
    def test_25_parametric_memory_leak_known_world_fact_rejected(self) -> None:
        passages = ["Aspirin-Mini tablets are packaged in 81 mg blister strips for clinic inventory."]
        ans = (
            "Aspirin-Mini tablets are packaged in 81 mg blister strips [1]. "
            "Aspirin was synthesized by Felix Hoffmann at Bayer in 1897 [1]."
        )
        res = verify("How are Aspirin-Mini tablets packaged?", passages, ans)
        self.assertEqual(res.score, 0.0)

    def test_26_parametric_entity_leak_without_numbers_rejected(self) -> None:
        passages = ["The Orion relay forwards telemetry frames over an encrypted fiber link."]
        ans = "The Orion relay forwards telemetry frames over an encrypted fiber link designed by NASA [1]."
        res = verify("How does the Orion relay forward telemetry?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("parametric_entity_leak" in r for r in res.reasons))

    def test_27_numeric_value_drift_rejected(self) -> None:
        passages = ["The autoclave sterilization cycle runs at 121 C for 15 minutes."]
        ans = "The autoclave sterilization cycle runs at 134 C for 15 minutes [1]."
        res = verify("What temperature and duration does the autoclave use?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("numeric_mismatch" in r for r in res.reasons))

    def test_28_numeric_unit_mismatch_mg_vs_g_rejected(self) -> None:
        passages = ["Each capsule contains 250 mg of active compound Z-9."]
        ans = "Each capsule contains 250 g of active compound Z-9 [1]."
        res = verify("How much active compound Z-9 is in each capsule?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertTrue(any("numeric_mismatch" in r for r in res.reasons))

    def test_29_numeric_comma_normalization_passes(self) -> None:
        passages = ["The batch queue accepts up to 1,000 events per second."]
        ans = "The batch queue accepts up to 1000 events per second [1]."
        res = verify("What is the maximum rate of the batch queue?", passages, ans)
        self.assertTrue(res.passed, res)

    def test_30_decimal_value_grounded_passes(self) -> None:
        passages = ["Sensor voltage tolerance is 3.3 V with a 0.05 V ripple ceiling."]
        ans = "Sensor voltage tolerance is 3.3 V with a 0.05 V ripple ceiling [1]."
        res = verify("What is the sensor voltage tolerance?", passages, ans)
        self.assertTrue(res.passed, res)

    def test_31_polarity_inversion_on_policy_rejected(self) -> None:
        passages = ["Unverified third-party plugins are not permitted in production clusters."]
        ans = "Unverified third-party plugins are permitted in production clusters [1]."
        res = verify("Are unverified third-party plugins permitted in production?", passages, ans)
        self.assertEqual(res.score, 0.0)
        self.assertIn("polarity_inversion", res.reasons)

    # --- 5. Calibrated abstention, over-refusal & contradictory abstention ---
    def test_32_calibrated_abstention_on_unanswerable_query_passes(self) -> None:
        passages = [
            "Warehouse-B operates LED lighting on circuit panel L-2.",
            "Forklift batteries recharge overnight in Bay-6.",
        ]
        res = verify(
            "What is the maximum roof snow load rating in kPa for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to determine the roof snow load rating for Warehouse-B.",
        )
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.passed)
        self.assertTrue(res.abstained)
        self.assertFalse(res.answerable)

    def test_33_over_refusal_when_answer_is_in_passage_scores_zero(self) -> None:
        passages = ["Warehouse-B roof snow load rating is 2.4 kPa per structural drawing S-11."]
        res = verify(
            "What is the roof snow load rating in kPa for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to determine the roof snow load rating for Warehouse-B.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("over_refusal", res.reasons)

    def test_34_contradictory_abstention_with_citation_scores_zero(self) -> None:
        passages = ["Warehouse-B operates LED lighting on circuit panel L-2."]
        res = verify(
            "What is the roof snow load rating for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages to answer this question [1].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("contradictory_abstention_with_citation", res.reasons)

    def test_35_contradictory_abstention_with_trailing_action_scores_zero(self) -> None:
        passages = ["Warehouse-B operates LED lighting on circuit panel L-2."]
        res = verify(
            "What is the roof snow load rating for Warehouse-B?",
            passages,
            "Insufficient evidence in the provided passages. Evacuate the roof and disable circuit panel L-2.",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("contradictory_abstention_with_action", res.reasons)

    def test_36_under_abstention_on_off_topic_passages_scores_zero(self) -> None:
        passages = ["Cafeteria lunch hours are 11:30 to 13:30."]
        res = verify(
            "What is the maximum pediatric dose of amoxicillin?",
            passages,
            "Cafeteria lunch hours are 11:30 to 13:30 [1].",
        )
        self.assertEqual(res.score, 0.0)
        self.assertIn("under_abstention_off_topic_passages", res.reasons)

    def test_37_passage_explicit_abstention_treated_as_unanswerable(self) -> None:
        passages = ["The current manual does not specify the replacement belt tension for Motor-K3."]
        res = verify(
            "What is the replacement belt tension for Motor-K3?",
            passages,
            "The provided passages do not specify the replacement belt tension for Motor-K3.",
        )
        self.assertEqual(res.score, 1.0)
        self.assertTrue(res.abstained)

    # --- 6. DPO synthesizer & batch verification tests ---
    def test_38_synth_pairs_deterministic_identical_across_runs(self) -> None:
        p1 = [x.to_dict() for x in generate_dpo_pairs(seed=42, include_extended=True)]
        p2 = [x.to_dict() for x in generate_dpo_pairs(seed=42, include_extended=True)]
        self.assertEqual(p1, p2)
        self.assertGreaterEqual(len(p1), 24)

    def test_39_synth_pairs_covers_all_three_core_categories(self) -> None:
        pairs = generate_dpo_pairs(seed=1337, include_extended=False)
        cats = {p.category for p in pairs}
        for req in CORE_CATEGORIES:
            self.assertIn(req, cats)
        for p in pairs:
            self.assertGreaterEqual(p.chosen_score, 0.85)
            self.assertEqual(p.rejected_score, 0.0)
            self.assertLessEqual(len(p.chosen.split()), 120)
            self.assertLessEqual(len(p.rejected.split()), 120)

    def test_40_verify_batch_consistency(self) -> None:
        records = [
            {
                "query": "What is the backup retention window?",
                "passages": ["Encrypted snapshots are retained for 30 days in region-west."],
                "answer": "Encrypted snapshots are retained for 30 days in region-west [1].",
            },
            {
                "query": "What is the backup retention window?",
                "passages": ["Encrypted snapshots are retained for 30 days in region-west."],
                "answer": "Encrypted snapshots are retained for 90 days in region-west [1].",
            },
        ]
        results = verify_batch(records)
        self.assertEqual(len(results), 2)
        self.assertIsInstance(results[0], VerificationResult)
        self.assertTrue(results[0].passed)
        self.assertEqual(results[1].score, 0.0)

    def test_41_parse_citations_mixed_valid_and_invalid(self) -> None:
        valid, invalid = parse_citations("Claim A [1, 2] and Claim B [0] and [9].", 4)
        self.assertEqual(valid, [1, 2])
        self.assertEqual(invalid, ["[0]", "[9]"])

    def test_42_detection_benchmark_300_items_perfect_f1(self) -> None:
        rep = run_detection_benchmark(seed=7)
        self.assertEqual(rep["total"], 300)
        self.assertEqual(rep["tp"], 150)
        self.assertEqual(rep["fp"], 0)
        self.assertEqual(rep["tn"], 150)
        self.assertEqual(rep["fn"], 0)
        self.assertEqual(rep["f1"], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
