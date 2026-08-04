from __future__ import annotations

import copy
import json
import pathlib
import unittest
import xml.etree.ElementTree as ET

from benchmarks.flagship_study import charts


SVG_NAMESPACE = "http://www.w3.org/2000/svg"
SVG = {"svg": SVG_NAMESPACE}
REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parents[2]

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
                "unconstrained_pareto_optimal": True,
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
                "unconstrained_pareto_optimal": True,
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
                "unconstrained_pareto_optimal": False,
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
                "unconstrained_pareto_optimal": False,
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


def _production_pareto_fixture() -> tuple[dict[str, object], dict[str, str]]:
    selection = json.loads(
        (REPOSITORY_ROOT / "benchmarks/flagship_study/selection_lock.json")
        .read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (REPOSITORY_ROOT / "benchmarks/flagship_study/manifest.json")
        .read_text(encoding="utf-8")
    )
    labels: dict[str, str] = {}
    for configuration in manifest["configurations"]:
        settings = configuration["settings"]
        if configuration["kind"] == "mcts":
            suffix = f" ({settings['iterations']} iter)"
        elif configuration["kind"] in ("alpha-beta", "jacek-inspired"):
            suffix = f" ({settings['max_nodes'] // 1000}k nodes)"
        else:
            suffix = ""
        labels[configuration["id"]] = configuration["public_label"] + suffix
    return selection, labels


def _rectangle(element: ET.Element) -> tuple[float, float, float, float]:
    return tuple(
        float(element.attrib[name]) for name in ("x", "y", "width", "height")
    )


def _point_rectangle(element: ET.Element) -> tuple[float, float, float, float]:
    center_x = float(element.attrib["cx"])
    center_y = float(element.attrib["cy"])
    radius = float(element.attrib["r"])
    return center_x - radius, center_y - radius, 2 * radius, 2 * radius


def _overlap(
    first: tuple[float, float, float, float],
    second: tuple[float, float, float, float],
) -> bool:
    first_x, first_y, first_width, first_height = first
    second_x, second_y, second_width, second_height = second
    return not (
        first_x + first_width <= second_x
        or second_x + second_width <= first_x
        or first_y + first_height <= second_y
        or second_y + second_height <= first_y
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

    def test_pareto_only_styles_do_not_change_the_other_chart_documents(self) -> None:
        rendered = _render_all()

        self.assertIn(".point-key{", rendered["pareto"])
        self.assertIn(".status{", rendered["pareto"])
        for name in ("bradley_terry", "calibration"):
            with self.subTest(chart=name):
                self.assertNotIn(".point-key{", rendered[name])
                self.assertNotIn(".status{", rendered[name])

    def test_checked_in_production_charts_match_the_generator(self) -> None:
        selection, labels = _production_pareto_fixture()
        manifest = json.loads(
            (REPOSITORY_ROOT / "benchmarks/flagship_study/manifest.json")
            .read_text(encoding="utf-8")
        )
        test = json.loads(
            (REPOSITORY_ROOT / "benchmarks/flagship_study/data/test.json")
            .read_text(encoding="utf-8")
        )
        rendered = {
            "bradley_terry": charts.bradley_terry_svg(test, labels),
            "pareto": charts.pareto_svg(selection, labels),
            "calibration": charts.calibration_svg(test, labels),
        }

        for name, svg in rendered.items():
            with self.subTest(chart=name):
                path = REPOSITORY_ROOT / manifest["outputs"]["charts"][name]
                self.assertEqual(svg, path.read_text(encoding="utf-8"))

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
        circles = root.findall(".//svg:circle[@class='pareto-point']", SVG)
        by_identifier = {
            circle.attrib["data-config-id"]: circle.attrib for circle in circles
        }

        self.assertEqual(len(circles), 4)
        self.assertEqual(by_identifier["mcts-2000"]["fill"], charts.PALETTE[2])
        self.assertEqual(by_identifier["mcts-2000"]["r"], "11")
        self.assertEqual(by_identifier["mcts-2000"]["stroke"], "#FFFFFF")
        self.assertEqual(by_identifier["alpha-beta-50k"]["fill"], charts.PALETTE[0])
        self.assertEqual(by_identifier["alpha-beta-50k"]["r"], "7")
        self.assertEqual(by_identifier["alpha-beta-50k"]["stroke"], "#FFFFFF")
        self.assertEqual(by_identifier["jacek-50k"]["fill"], charts.PALETTE[1])
        self.assertEqual(by_identifier["jacek-50k"]["fill-opacity"], "0.42")
        self.assertEqual(by_identifier["jacek-50k"]["stroke"], "#555555")
        self.assertEqual(by_identifier["rank5-fixed-50k"]["fill"], charts.PALETTE[3])
        self.assertEqual(by_identifier["rank5-fixed-50k"]["r"], "11")
        self.assertEqual(by_identifier["rank5-fixed-50k"]["fill-opacity"], "0.42")
        self.assertEqual(by_identifier["rank5-fixed-50k"]["stroke"], "#111111")

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
        self.assertIn(
            "score 63.0% [57.0%, 69.0%] · p95 31.0 ms · "
            "n=200 pairs / 4,200 decisions",
            svg,
        )
        self.assertIn(
            "defined score 50.0% (strength n=N/A) · fresh-root p95 52.0 ms · "
            "n=2,100 decisions",
            svg,
        )
        for status in (
            "constrained Pareto",
            "unconstrained Pareto",
            "gate rejected",
            "unconstrained dominated",
            "selected",
            "fixed",
        ):
            self.assertIn(status, svg)
        self.assertEqual(
            len(root.findall(".//svg:line[@class='strength-ci']", SVG)), 3
        )

    def test_rendering_is_invariant_to_pareto_input_order(self) -> None:
        selection = _selection_fixture()
        reversed_selection = copy.deepcopy(selection)
        reversed_selection["validation_pareto"].reverse()

        self.assertEqual(
            charts.pareto_svg(selection, LABELS),
            charts.pareto_svg(reversed_selection, LABELS),
        )

    def test_production_callouts_and_detail_rows_are_collision_free(self) -> None:
        selection, labels = _production_pareto_fixture()
        svg = charts.pareto_svg(selection, labels)
        root = _parse(svg)

        self.assertEqual(root.attrib["width"], "1600")
        self.assertEqual(root.attrib["height"], "860")
        self.assertEqual(root.attrib["viewBox"], "0 0 1600 860")
        callouts = root.findall(".//svg:rect[@class='pareto-callout']", SVG)
        details = root.findall(".//svg:rect[@class='pareto-detail-row']", SVG)
        detail_groups = root.findall(".//svg:g[@class='pareto-detail']", SVG)
        points = root.findall(".//svg:circle[@class='pareto-point']", SVG)
        self.assertEqual(len(callouts), 10)
        self.assertEqual(len(details), 10)
        self.assertEqual(len(detail_groups), 10)
        self.assertEqual(len(points), 10)

        callout_rectangles = [_rectangle(element) for element in callouts]
        detail_rectangles = [_rectangle(element) for element in details]
        point_rectangles = [_point_rectangle(element) for element in points]
        for rectangle in callout_rectangles:
            x, y, width, height = rectangle
            self.assertGreaterEqual(x, 120.0)
            self.assertGreaterEqual(y, 155.0)
            self.assertLessEqual(x + width, 920.0)
            self.assertLessEqual(y + height, 690.0)
        for rectangle in detail_rectangles:
            x, y, width, height = rectangle
            self.assertGreaterEqual(x, 980.0)
            self.assertLessEqual(x + width, 1570.0)
            self.assertGreaterEqual(y, 0.0)
            self.assertLessEqual(y + height, 860.0)
        for rectangles in (callout_rectangles, detail_rectangles):
            for index, first in enumerate(rectangles):
                for second in rectangles[index + 1:]:
                    self.assertFalse(_overlap(first, second))
        for callout in callout_rectangles:
            for point in point_rectangles:
                self.assertFalse(_overlap(callout, point))

        point_identifiers = {point.attrib["data-config-id"] for point in points}
        expected_identifiers = {
            point["id"] for point in selection["validation_pareto"]
        }
        self.assertEqual(point_identifiers, expected_identifiers)
        details_by_identifier = {
            group.attrib["data-config-id"]: "".join(group.itertext())
            for group in detail_groups
        }
        for point in selection["validation_pareto"]:
            row = details_by_identifier[point["id"]]
            self.assertIn(labels[point["id"]], row)
            self.assertIn(
                f"{point['validation_latency_decisions']:,} decisions", row
            )
            if point["fixed"]:
                self.assertIn("strength n=N/A", row)
                self.assertIn("fresh-root p95", row)
            else:
                interval = point["validation_strength_pair_bootstrap_95"]
                self.assertIn(
                    f"score {100 * point['validation_strength']:.1f}% "
                    f"[{100 * interval['lower']:.1f}%, "
                    f"{100 * interval['upper']:.1f}%]",
                    row,
                )
                self.assertIn(
                    f"n={point['validation_strength_pairs']:,} pairs", row
                )
            expected_status = (
                "constrained Pareto" if point["constrained_pareto_optimal"]
                else "constrained dominated" if point["gate_eligible"]
                else "gate rejected"
            )
            expected_status += " · " + (
                "unconstrained Pareto" if point["unconstrained_pareto_optimal"]
                else "unconstrained dominated"
            )
            if point["selected"]:
                expected_status += " · selected"
            if point["fixed"]:
                expected_status += " · fixed"
            self.assertTrue(row.endswith(expected_status), row)

    def test_dense_points_still_receive_distinct_in_bounds_callouts(self) -> None:
        selection = _selection_fixture()
        for index, point in enumerate(selection["validation_pareto"]):
            point["validation_p95_ms"] = 49.5 + index * 0.05
            point["validation_strength"] = 0.60 + index * 0.001
            if not point["fixed"]:
                point["validation_strength_pair_bootstrap_95"] = {
                    "lower": 0.55,
                    "upper": 0.65,
                }
        root = _parse(charts.pareto_svg(selection, LABELS))
        callouts = root.findall(".//svg:rect[@class='pareto-callout']", SVG)
        points = root.findall(".//svg:circle[@class='pareto-point']", SVG)
        rectangles = [_rectangle(element) for element in callouts]
        point_rectangles = [_point_rectangle(element) for element in points]

        self.assertEqual(len(rectangles), 4)
        for index, first in enumerate(rectangles):
            x, y, width, height = first
            self.assertGreaterEqual(x, 120.0)
            self.assertGreaterEqual(y, 155.0)
            self.assertLessEqual(x + width, 920.0)
            self.assertLessEqual(y + height, 690.0)
            for second in rectangles[index + 1:]:
                self.assertFalse(_overlap(first, second))
            for point in point_rectangles:
                self.assertFalse(_overlap(first, point))


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
