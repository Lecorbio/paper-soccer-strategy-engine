from __future__ import annotations

import copy
import unittest
import xml.etree.ElementTree as ET

from benchmarks.flagship_study import charts


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG = {"svg": SVG_NAMESPACE}

LABELS = {
    "mcts-2000": "Tactical MctsBot",
    "alpha-beta-50k": "Hand-evaluated AlphaBetaBot",
    "jacek-50k": "Neural alpha-beta (JacekInspiredBot)",
    "rank5-fixed-50k": "Rank5DerivedBot — fixed 50k demo profile",
}


def _reliability_bins(offset: float, count: int) -> list[dict[str, float | int]]:
    bins: list[dict[str, float | int]] = []
    for index in range(10):
        prediction = (index + 0.5) / 10.0
        observed = min(0.98, max(0.02, prediction + offset))
        bins.append({
            "count": count,
            "mean_prediction": prediction,
            "observed_frequency": observed,
            "pair_clusters": max(1, count // 2),
            "bootstrap_successful_resamples": 10_000,
            "observed_frequency_pair_bootstrap_95": {
                "method": "pair_cluster_percentile_stratified",
                "confidence": 0.95,
                "resamples": 10_000,
                "successful_resamples": 10_000,
                "lower": max(0.0, observed - 0.08),
                "upper": min(1.0, observed + 0.08),
            },
        })
    return bins


def _calibration_metrics(brier: float, log_loss: float, samples: int,
                         bin_count: int, offset: float) -> dict[str, object]:
    return {
        "brier_score": brier,
        "log_loss": log_loss,
        "samples": samples,
        "pair_clusters": samples // 2,
        "pair_cluster_bootstrap_95": {
            "method": "pair_cluster_percentile_stratified",
            "seed": "17",
            "resamples": 10_000,
            "successful_resamples": 10_000,
            "confidence": 0.95,
            "stratify_by": "matchup_and_opening_depth",
            "brier_score": {
                "lower": max(0.0, brier - 0.02),
                "upper": min(1.0, brier + 0.02),
            },
            "log_loss": {
                "lower": max(0.0, log_loss - 0.03),
                "upper": log_loss + 0.03,
            },
        },
        "reliability_bins": _reliability_bins(offset, bin_count),
    }


def _analysis_fixture() -> dict[str, object]:
    return {
        "bradley_terry": {
            "intervals": {
                "rank5-fixed-50k": {
                    "estimate": -0.45,
                    "lower": -0.70,
                    "upper": -0.20,
                },
                "jacek-50k": {
                    "estimate": -0.20,
                    "lower": -0.40,
                    "upper": 0.05,
                },
                "mcts-2000": {
                    "estimate": 0.55,
                    "lower": 0.30,
                    "upper": 0.80,
                },
                "alpha-beta-50k": {
                    "estimate": 0.10,
                    "lower": -0.10,
                    "upper": 0.30,
                },
            },
        },
        "sample_sizes": {"games": 4_800, "pairs": 2_400},
        "calibration": {
            "mcts-2000": _calibration_metrics(0.1814, 0.5412, 120, 12, 0.01),
            "rank5-fixed-50k": _calibration_metrics(
                0.2326, 0.6557, 150, 15, -0.04
            ),
            "alpha-beta-50k": _calibration_metrics(
                0.1946, 0.5763, 130, 13, 0.03
            ),
            "jacek-50k": _calibration_metrics(0.1682, 0.5094, 140, 14, -0.02),
        },
    }


def _selection_fixture() -> dict[str, object]:
    return {
        "validation_pareto": [
            {
                "id": "mcts-2000",
                "validation_p95_ms": 31.0,
                "validation_strength": 0.63,
                "validation_strength_pairs": 200,
                "validation_strength_pair_bootstrap_95": {"lower": 0.57, "upper": 0.69},
                "validation_latency_decisions": 4_200,
                "pareto_optimal": True,
                "constrained_pareto_optimal": True,
                "gate_eligible": True,
                "selected": True,
                "fixed": False,
            },
            {
                "id": "alpha-beta-50k",
                "validation_p95_ms": 43.0,
                "validation_strength": 0.69,
                "validation_strength_pairs": 200,
                "validation_strength_pair_bootstrap_95": {"lower": 0.63, "upper": 0.75},
                "validation_latency_decisions": 3_900,
                "pareto_optimal": True,
                "constrained_pareto_optimal": True,
                "gate_eligible": True,
                "selected": False,
                "fixed": False,
            },
            {
                "id": "jacek-50k",
                "validation_p95_ms": 56.0,
                "validation_strength": 0.66,
                "validation_strength_pairs": 200,
                "validation_strength_pair_bootstrap_95": {"lower": 0.60, "upper": 0.72},
                "validation_latency_decisions": 3_700,
                "pareto_optimal": False,
                "constrained_pareto_optimal": False,
                "gate_eligible": False,
                "selected": False,
                "fixed": False,
            },
            {
                "id": "rank5-fixed-50k",
                "validation_p95_ms": 52.0,
                "validation_strength": 0.50,
                "validation_strength_pairs": None,
                "validation_strength_pair_bootstrap_95": None,
                "validation_latency_decisions": 2_100,
                "pareto_optimal": False,
                "constrained_pareto_optimal": False,
                "gate_eligible": False,
                "selected": True,
                "fixed": True,
            },
        ],
    }


def _render_all() -> dict[str, str]:
    analysis = _analysis_fixture()
    return {
        "bradley_terry": charts.bradley_terry_svg(analysis, LABELS),
        "pareto": charts.pareto_svg(_selection_fixture(), LABELS),
        "calibration": charts.calibration_svg(analysis, LABELS),
    }


def _parse(svg: str) -> ET.Element:
    root = ET.fromstring(svg)
    if root.tag != f"{{{SVG_NAMESPACE}}}svg":
        raise AssertionError(f"unexpected SVG root: {root.tag}")
    return root


def _visible_labels(root: ET.Element) -> str:
    return " ".join(
        "".join(element.itertext()) for element in root.findall(".//svg:text", SVG)
    )


class DeterministicSvgTests(unittest.TestCase):
    def test_every_chart_is_byte_identical_and_xml_valid(self) -> None:
        first = _render_all()
        second = _render_all()

        self.assertEqual(first, second)
        self.assertEqual(
            set(first), {"bradley_terry", "pareto", "calibration"}
        )
        for name, svg in first.items():
            with self.subTest(chart=name):
                root = _parse(svg)
                self.assertEqual(root.attrib["role"], "img")
                self.assertEqual(root.attrib["aria-labelledby"], "title desc")
                self.assertIsNotNone(root.find("svg:title", SVG))
                self.assertIsNotNone(root.find("svg:desc", SVG))
                self.assertNotIn("nan", svg.casefold())
                self.assertNotIn("infinity", svg.casefold())

    def test_colorblind_safe_palette_is_used_consistently(self) -> None:
        self.assertEqual(
            charts.PALETTE,
            ("#0072B2", "#D55E00", "#009E73", "#CC79A7"),
        )
        for name, svg in _render_all().items():
            with self.subTest(chart=name):
                for color in charts.PALETTE:
                    self.assertIn(color, svg)

    def test_no_chart_bytes_or_visible_labels_name_a_random_benchmark(self) -> None:
        for name, svg in _render_all().items():
            with self.subTest(chart=name):
                self.assertNotIn("random", svg.casefold())
                self.assertNotIn("uniform_legal_move_generator", svg.casefold())
                self.assertNotIn("opening generator", svg.casefold())
                self.assertNotIn("random", _visible_labels(_parse(svg)).casefold())


class BradleyTerryChartTests(unittest.TestCase):
    def test_strength_axis_intervals_and_sample_sizes_are_explicit(self) -> None:
        svg = charts.bradley_terry_svg(_analysis_fixture(), LABELS)
        root = _parse(svg)

        self.assertIn(
            "Bradley–Terry estimates; pair-clustered bootstrap 95% intervals",
            svg,
        )
        self.assertIn("Relative log ability (sum constrained to zero)", svg)
        self.assertIn(
            "Zero is an identifiability convention, not absolute skill.", svg
        )
        self.assertIn(
            "n=4800 decisive games in 2400 color-swapped pairs.", svg
        )
        for formatted_interval in (
            "0.55 [0.30, 0.80]",
            "0.10 [-0.10, 0.30]",
            "-0.20 [-0.40, 0.05]",
            "-0.45 [-0.70, -0.20]",
        ):
            self.assertIn(formatted_interval, svg)
        self.assertEqual(len(root.findall(".//svg:circle", SVG)), 4)
        self.assertEqual(len(root.findall(".//svg:text[@class='label']", SVG)), 4)


class ParetoChartTests(unittest.TestCase):
    def test_axes_and_fifty_millisecond_gate_are_unambiguous(self) -> None:
        svg = charts.pareto_svg(_selection_fixture(), LABELS)
        root = _parse(svg)

        self.assertIn(
            "Validation p95 decision latency (ms; Rank5DerivedBot — fixed 50k "
            "demo profile uses fresh-root searches)", svg
        )
        self.assertIn(
            "Mean color-swapped pair score vs Rank5DerivedBot — fixed 50k demo profile",
            svg,
        )
        self.assertNotIn("win rate", svg.casefold())
        self.assertIn("50 ms gate", svg)
        gate_lines = [
            element for element in root.findall(".//svg:line", SVG)
            if element.attrib.get("stroke") == "#E69F00"
        ]
        self.assertEqual(len(gate_lines), 1)
        self.assertEqual(gate_lines[0].attrib["x1"], gate_lines[0].attrib["x2"])
        self.assertEqual(gate_lines[0].attrib["stroke-dasharray"], "8 6")

    def test_selected_fixed_ineligible_and_frontier_encodings_are_distinct(self) -> None:
        svg = charts.pareto_svg(_selection_fixture(), LABELS)
        root = _parse(svg)
        circles = root.findall(".//svg:circle", SVG)
        by_fill = {circle.attrib["fill"]: circle.attrib for circle in circles}

        self.assertEqual(len(circles), 4)
        self.assertEqual(by_fill[charts.PALETTE[0]]["r"], "11")
        self.assertEqual(by_fill[charts.PALETTE[0]]["stroke"], "#FFFFFF")
        self.assertEqual(by_fill[charts.PALETTE[1]]["r"], "7")
        self.assertEqual(by_fill[charts.PALETTE[1]]["stroke"], "#FFFFFF")
        self.assertEqual(by_fill[charts.PALETTE[2]]["fill-opacity"], "0.42")
        self.assertEqual(by_fill[charts.PALETTE[2]]["stroke"], "#555555")
        self.assertEqual(by_fill[charts.PALETTE[3]]["r"], "11")
        self.assertEqual(by_fill[charts.PALETTE[3]]["fill-opacity"], "0.42")
        self.assertEqual(by_fill[charts.PALETTE[3]]["stroke"], "#111111")

        frontier = [
            element for element in root.findall(".//svg:path", SVG)
            if element.attrib.get("stroke") == "#666666"
        ]
        self.assertEqual(len(frontier), 1)
        self.assertTrue(frontier[0].attrib["d"].startswith("M "))
        self.assertIn(" L ", frontier[0].attrib["d"])
        self.assertIn("large points are selected", svg)
        self.assertIn("black outlines mark fixed", svg)
        self.assertIn("faded points miss the gate", svg)
        self.assertIn("line is the constrained frontier", svg)
        self.assertIn("Vertical bars are pair-bootstrap 95% intervals", svg)
        self.assertIn("strength n=200 pairs; latency n=4200 decisions", svg)
        self.assertIn(
            "defined reference; strength n=N/A; fresh-root latency n=2100 decisions",
            svg,
        )
        self.assertEqual(
            len(root.findall(".//svg:line[@class='strength-ci']", SVG)), 3
        )


class CalibrationChartTests(unittest.TestCase):
    def test_ten_bin_curves_metrics_and_prediction_sample_sizes_are_visible(self) -> None:
        analysis = _analysis_fixture()
        svg = charts.calibration_svg(analysis, LABELS)
        root = _parse(svg)

        self.assertIn("Validation-fitted mappings applied once to test predictions", svg)
        self.assertIn("Mean predicted win probability", svg)
        self.assertIn("Observed win frequency", svg)
        self.assertIn(
            "n counts prediction opportunities, not independent games.", svg
        )
        self.assertIn("whole-pair bootstraps", svg)
        self.assertIn("point area scales with bin prediction n", svg)
        for metrics in analysis["calibration"].values():
            self.assertEqual(len(metrics["reliability_bins"]), 10)
            self.assertIn(f"prediction n={metrics['samples']}", svg)
            self.assertIn(f"pair n={metrics['pair_clusters']}", svg)
            self.assertIn(f"Brier {metrics['brier_score']:.3f}", svg)
            self.assertIn(f"Log loss {metrics['log_loss']:.3f}", svg)

        circles = root.findall(".//svg:circle", SVG)
        self.assertEqual(len(circles), 40)
        for color in charts.PALETTE:
            self.assertEqual(
                sum(circle.attrib.get("fill") == color for circle in circles), 10
            )
        reliability_paths = [
            element for element in root.findall(".//svg:path", SVG)
            if element.attrib.get("stroke") in charts.PALETTE
        ]
        self.assertEqual(len(reliability_paths), 4)
        self.assertEqual(
            len(root.findall(".//svg:line[@class='reliability-ci']", SVG)), 40
        )
        self.assertEqual(svg.count("bin prediction n="), 40)

    def test_calibration_rejects_anything_other_than_ten_bins(self) -> None:
        analysis = copy.deepcopy(_analysis_fixture())
        analysis["calibration"]["mcts-2000"]["reliability_bins"].pop()

        with self.assertRaisesRegex(charts.ChartError, "ten reliability bins"):
            charts.calibration_svg(analysis, LABELS)


if __name__ == "__main__":
    unittest.main()
