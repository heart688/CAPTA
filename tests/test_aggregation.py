"""Synthetic regression tests for negative-parallel-only source filtering."""
import unittest
from unittest.mock import patch

import torch

from capta.aggregation import energy, match, project, remove_conflicting_parallel, server_aggregate
from capta.engine import server_aggregate as engine_aggregate
from capta.utils import group


class DirectionalAggregationTests(unittest.TestCase):
    def test_positive_negative_and_orthogonal_components(self):
        for dtype in (torch.float32, torch.float64):
            target = {"head.weight": torch.tensor([1., 0.], dtype=dtype)}
            for source, expected in (([2., 3.], [2., 3.]),
                                     ([-2., 3.], [0., 3.]),
                                     ([0., 3.], [0., 3.]),
                                     ([2., 0.], [2., 0.]),
                                     ([-2., 0.], [0., 0.])):
                with self.subTest(dtype=dtype, source=source):
                    update = {"head.weight": torch.tensor(source, dtype=dtype)}
                    before = update["head.weight"].clone()
                    got = remove_conflicting_parallel(update, target, [list(update)])
                    torch.testing.assert_close(got["head.weight"], torch.tensor(expected, dtype=dtype))
                    torch.testing.assert_close(update["head.weight"], before, rtol=0, atol=0)
                    torch.testing.assert_close(target["head.weight"], torch.tensor([1., 0.], dtype=dtype))

    def test_filter_is_blockwise_not_per_tensor_or_global(self):
        target = {"head.weight": torch.tensor([1.]), "head.bias": torch.tensor([1.]),
                  "history_embed.0.weight": torch.tensor([1., 0.])}
        source = {"head.weight": torch.tensor([2.]), "head.bias": torch.tensor([-1.]),
                  "history_embed.0.weight": torch.tensor([-3., 4.])}
        blocks = [["head.weight", "head.bias"], ["history_embed.0.weight"]]
        got = remove_conflicting_parallel(source, target, blocks)
        torch.testing.assert_close(got["head.weight"], source["head.weight"])
        torch.testing.assert_close(got["head.bias"], source["head.bias"])
        torch.testing.assert_close(got["history_embed.0.weight"], torch.tensor([0., 4.]))

    def test_zero_target_preserves_source(self):
        source = {"head.weight": torch.tensor([-3., 4.])}
        target = {"head.weight": torch.zeros(2)}
        got = remove_conflicting_parallel(source, target, [list(source)])
        torch.testing.assert_close(got["head.weight"], source["head.weight"], rtol=0, atol=0)

    def test_energy_match_does_not_invent_a_missing_direction(self):
        zero = {"head.weight": torch.zeros(2)}
        nonzero = {"head.weight": torch.tensor([1., 2.])}
        for candidate, reference, matched in ((zero, zero, True), (zero, nonzero, False),
                                               (nonzero, zero, True)):
            got, audit = match(candidate, reference, [list(zero)])
            torch.testing.assert_close(got["head.weight"], zero["head.weight"])
            self.assertEqual(audit["all_blocks_matched"], matched)
            if not matched:
                self.assertEqual(audit["block_statuses"], ["zero_candidate_unmatched_reference"])
                self.assertEqual(audit["max_relative_energy_error"], 1.)
        self.assertEqual(audit["max_relative_energy_error"], 0.)

    def test_random_geometry_preserves_perpendicular_and_nonnegative_parallel(self):
        rng = torch.Generator().manual_seed(716)
        block = ["head.weight", "head.bias"]
        for dtype in (torch.float32, torch.float64):
            for _ in range(30):
                target = {k: torch.randn(7, generator=rng, dtype=dtype) for k in block}
                source = {k: torch.randn(7, generator=rng, dtype=dtype) for k in block}
                kept = remove_conflicting_parallel(source, target, [block])
                parallel, perpendicular = project(source, target, [block])
                kept_parallel, kept_perpendicular = project(kept, target, [block])
                coefficient = sum(float((source[k] * target[k]).sum()) for k in block)
                again = remove_conflicting_parallel(kept, target, [block])
                for k in block:
                    torch.testing.assert_close(kept_perpendicular[k], perpendicular[k], rtol=1e-5, atol=1e-6)
                    expected = parallel[k] if coefficient >= 0 else torch.zeros_like(parallel[k])
                    torch.testing.assert_close(kept_parallel[k], expected, rtol=1e-5, atol=1e-6)
                    torch.testing.assert_close(again[k], kept[k], rtol=1e-5, atol=1e-6)

    def test_filter_does_not_discard_a_small_nonzero_target_direction(self):
        source = {"head.weight": torch.tensor([-2., 3.])}
        for scale in (1e-12, 1., 1e12):
            target = {"head.weight": torch.tensor([scale, 0.])}
            got = remove_conflicting_parallel(source, target, [list(source)])
            torch.testing.assert_close(got["head.weight"], torch.tensor([0., 3.]))

    def test_cancelled_candidate_stops_main_aggregation(self):
        shared = {"head.weight": torch.zeros(2)}
        target = {"head.weight": torch.tensor([1., 0.])}
        sources = [{"head.weight": torch.tensor([value, 0.])} for value in (4., -1., -1.)]
        with self.assertRaisesRegex(FloatingPointError, "projected source group prediction"):
            server_aggregate(shared, sources, target, {"target": "shenzhen", "beta": .6})

    def test_server_matches_filtered_reference_and_keeps_positive_alignment(self):
        target = {"head.weight": torch.tensor([1., 0.])}
        shared = {"head.weight": torch.tensor([5., 6.])}
        for source, safe in (([2., 3.], [2., 3.]), ([-2., 3.], [0., 3.]),
                             ([2., 0.], [2., 0.]), ([0., 3.], [0., 3.])):
            with self.subTest(source=source):
                sources = [{"head.weight": torch.tensor(source)} for _ in range(3)]
                result, audit = server_aggregate(shared, sources, target, {"target": "shenzhen", "beta": .6})
                expected = shared["head.weight"] + target["head.weight"] + .6 * torch.tensor(safe)
                torch.testing.assert_close(result["head.weight"], expected)
                self.assertLess(audit["nonconflict_relative_error"], 1e-5)
                self.assertTrue(audit["energy"]["all_blocks_matched"])
                if source[0] > 0:
                    self.assertGreater(audit["correction_target_cosines"]["prediction"], 0.)

    def test_server_rejects_zero_or_tiny_target_per_group(self):
        shared = {"head.weight": torch.zeros(2)}
        cfg = {"target": "shenzhen", "beta": .6}
        sources = [{"head.weight": torch.tensor([2., 3.])} for _ in range(3)]
        for magnitude in (0., 1e-12):
            with self.assertRaisesRegex(FloatingPointError, "target update group prediction"):
                server_aggregate(shared, sources, {"head.weight": torch.tensor([magnitude, 0.])}, cfg)
        shared["history_embed.0.weight"] = torch.zeros(2)
        target = {"head.weight": torch.zeros(2), "history_embed.0.weight": torch.tensor([100., 0.])}
        sources = [{key: torch.tensor([2., 3.]) for key in shared} for _ in range(3)]
        with self.assertRaisesRegex(FloatingPointError, "target update group prediction"):
            server_aggregate(shared, sources, target, cfg)

    def test_server_rejects_zero_source_and_tiny_reference(self):
        shared = {"head.weight": torch.zeros(2)}
        target = {"head.weight": torch.tensor([1., 0.])}
        for source, stage in (([0., 0.], "projected source"), ([-2., 0.], "projected source"),
                              ([1e-12, 1e-12], "energy reference")):
            sources = [{"head.weight": torch.tensor(source)} for _ in range(3)]
            with self.assertRaisesRegex(FloatingPointError, stage):
                server_aggregate(shared, sources, target, {"target": "shenzhen", "beta": .6})

    def test_norm_guard_uses_strict_greater_than(self):
        shared = {"head.weight": torch.zeros(2, dtype=torch.float64)}
        sources = [{"head.weight": torch.tensor([1., 1.], dtype=torch.float64)} for _ in range(3)]
        with patch("capta.aggregation.MIN_GROUP_SQUARED_NORM", .25):
            target = {"head.weight": torch.tensor([.5, 0.], dtype=torch.float64)}
            with self.assertRaisesRegex(FloatingPointError, "target update"):
                server_aggregate(shared, sources, target, {"target": "shenzhen", "beta": .6})

    def test_engine_geometry_matches_executed_update(self):
        keys = ("head.weight", "history_embed.0.weight", "blocks.0.out.weight")
        shared = {k: torch.zeros(2) for k in keys}
        target = {k: torch.tensor([1., 0.]) for k in keys}
        source = {k: torch.tensor([2., 3.]) for k in keys}
        updates = {c: source for c in ("metr-la", "pems-bay", "chengdu")}
        updates["shenzhen"] = target
        new, audit = engine_aggregate(shared, updates, {"target": "shenzhen", "beta": .6, "aggregation": "tra"})
        for k in keys:
            torch.testing.assert_close(new[k], target[k] + .6 * source[k])
            geometry = audit["block_geometry"][group(k)]
            self.assertAlmostEqual(geometry["nonconflicting_reference_squared_norm"], energy(source, [k]))
            self.assertAlmostEqual(geometry["actual_matched_correction_energy"], energy(source, [k]), places=5)
            self.assertGreater(geometry["matched_correction_target_dot"], 0.)
        self.assertTrue(audit["geometry_audit_deterministically_recomputed_from_uploaded_deltas"])

if __name__ == "__main__":
    unittest.main()
