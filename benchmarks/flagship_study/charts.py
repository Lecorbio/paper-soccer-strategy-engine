"""Small deterministic SVG renderer for the three flagship study charts."""

from __future__ import annotations

import html
import math
from typing import Any, Mapping, Sequence


PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7")
GRID = "#D9D9D9"
TEXT = "#202124"
MUTED = "#5F6368"
BACKGROUND = "#FFFFFF"


class ChartError(ValueError):
    pass


def _escape(value: Any) -> str:
    return html.escape(str(value), quote=True)


def _finite(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ChartError(f"{name} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ChartError(f"{name} must be finite")
    return result


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ChartError(f"{name} must be a positive integer")
    return value


def _document(title: str, description: str, content: str,
              *, width: int = 1200, height: int = 760) -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">\n'
        f'<title id="title">{_escape(title)}</title>\n'
        f'<desc id="desc">{_escape(description)}</desc>\n'
        f'<rect width="{width}" height="{height}" fill="{BACKGROUND}"/>\n'
        '<style>text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;'
        f'fill:{TEXT}}}.title{{font-size:28px;font-weight:700}}.subtitle{{font-size:15px;'
        f'fill:{MUTED}}}.axis{{font-size:14px;fill:{MUTED}}}.label{{font-size:16px}}'
        '.small{font-size:13px}.legend{font-size:14px}</style>\n'
        f'{content}</svg>\n'
    )


def bradley_terry_svg(analysis: Mapping[str, Any],
                      labels: Mapping[str, str]) -> str:
    bt = analysis.get("bradley_terry")
    if not isinstance(bt, Mapping):
        raise ChartError("test analysis lacks Bradley-Terry results")
    intervals = bt.get("intervals")
    if not isinstance(intervals, Mapping) or len(intervals) != 4:
        raise ChartError("Bradley-Terry chart requires four intervals")
    rows = []
    for identifier, interval in intervals.items():
        if not isinstance(interval, Mapping):
            raise ChartError("invalid Bradley-Terry interval")
        rows.append((identifier, _finite(interval.get("estimate"), "estimate"),
                     _finite(interval.get("lower"), "lower"),
                     _finite(interval.get("upper"), "upper")))
    rows.sort(key=lambda row: (-row[1], row[0]))
    minimum = min(row[2] for row in rows)
    maximum = max(row[3] for row in rows)
    padding = max(0.2, (maximum - minimum) * 0.12)
    x_min = minimum - padding
    x_max = maximum + padding
    plot_left, plot_right = 390.0, 1110.0
    plot_top, plot_bottom = 190.0, 620.0

    def x(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * (plot_right - plot_left)

    parts = [
        '<text class="title" x="60" y="62">Test relative strength</text>',
        '<text class="subtitle" x="60" y="92">Bradley–Terry estimates; pair-clustered bootstrap 95% intervals</text>',
    ]
    ticks = 6
    for index in range(ticks + 1):
        value = x_min + index * (x_max - x_min) / ticks
        coordinate = x(value)
        parts.append(f'<line x1="{coordinate:.2f}" y1="{plot_top:.2f}" x2="{coordinate:.2f}" '
                     f'y2="{plot_bottom:.2f}" stroke="{GRID}"/>')
        parts.append(f'<text class="axis" x="{coordinate:.2f}" y="{plot_bottom + 30:.2f}" '
                     f'text-anchor="middle">{value:.2f}</text>')
    if x_min <= 0.0 <= x_max:
        coordinate = x(0.0)
        parts.append(f'<line x1="{coordinate:.2f}" y1="{plot_top - 15:.2f}" '
                     f'x2="{coordinate:.2f}" y2="{plot_bottom:.2f}" stroke="{MUTED}" '
                     'stroke-width="2" stroke-dasharray="6 6"/>')
    spacing = (plot_bottom - plot_top) / len(rows)
    for index, (identifier, estimate, lower, upper) in enumerate(rows):
        y = plot_top + spacing * (index + 0.5)
        color = PALETTE[index]
        parts.append(f'<text class="label" x="60" y="{y + 5:.2f}">{_escape(labels[identifier])}</text>')
        parts.append(f'<line x1="{x(lower):.2f}" y1="{y:.2f}" x2="{x(upper):.2f}" '
                     f'y2="{y:.2f}" stroke="{color}" stroke-width="6" stroke-linecap="round"/>')
        parts.append(f'<circle cx="{x(estimate):.2f}" cy="{y:.2f}" r="9" fill="{color}" '
                     'stroke="#FFFFFF" stroke-width="2"/>')
        parts.append(f'<text class="small" x="{plot_right:.2f}" y="{y - 14:.2f}" '
                     f'text-anchor="end">{estimate:.2f} [{lower:.2f}, {upper:.2f}]</text>')
    games = analysis.get("sample_sizes", {}).get("games", "?")
    pairs = analysis.get("sample_sizes", {}).get("pairs", "?")
    parts += [
        f'<text class="axis" x="{(plot_left + plot_right) / 2:.2f}" y="700" '
        'text-anchor="middle">Relative log ability (sum constrained to zero)</text>',
        f'<text class="subtitle" x="60" y="730">n={_escape(games)} decisive games in '
        f'{_escape(pairs)} color-swapped pairs. Zero is an identifiability convention, not absolute skill.</text>',
    ]
    return _document(
        "Test Bradley-Terry relative strength",
        "Four competitive bots with pair-clustered bootstrap intervals and no external baseline.",
        "\n".join(parts),
    )


def pareto_svg(selection: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    points = selection.get("validation_pareto")
    if not isinstance(points, Sequence) or not points:
        raise ChartError("selection lock lacks validation Pareto points")
    normalized: list[dict[str, Any]] = []
    for point in points:
        if not isinstance(point, Mapping):
            raise ChartError("invalid Pareto point")
        fixed = bool(point["fixed"])
        strength = 100.0 * _finite(
            point["validation_strength"], "validation_strength"
        )
        pairs_value = point.get("validation_strength_pairs")
        pairs = (
            None if fixed else _positive_integer(
                pairs_value, "validation_strength_pairs"
            )
        )
        if fixed and pairs_value is not None:
            raise ChartError("defined fixed reference cannot have a strength sample size")
        latency_decisions = _positive_integer(
            point.get("validation_latency_decisions"),
            "validation_latency_decisions",
        )
        interval_value = point.get("validation_strength_pair_bootstrap_95")
        interval: tuple[float, float] | None
        if fixed:
            if interval_value is not None:
                raise ChartError("fixed reference strength cannot have a sampled interval")
            interval = None
        else:
            if not isinstance(interval_value, Mapping):
                raise ChartError("candidate Pareto point lacks a pair-bootstrap interval")
            lower = 100.0 * _finite(interval_value.get("lower"), "strength lower")
            upper = 100.0 * _finite(interval_value.get("upper"), "strength upper")
            if not 0.0 <= lower <= upper <= 100.0:
                raise ChartError("Pareto strength interval is invalid")
            interval = (lower, upper)
        normalized.append({
            "id": str(point["id"]),
            "latency": _finite(point["validation_p95_ms"], "validation_p95_ms"),
            "strength": strength,
            "interval": interval,
            "pairs": pairs,
            "latency_decisions": latency_decisions,
            "optimal": bool(point["constrained_pareto_optimal"]),
            "eligible": bool(point["gate_eligible"]),
            "selected": bool(point["selected"]),
            "fixed": fixed,
        })
    x_max = max(60.0, max(point["latency"] for point in normalized) * 1.12)
    strengths = [
        bound
        for point in normalized
        for bound in (
            point["interval"] if point["interval"] is not None
            else (point["strength"], point["strength"])
        )
    ]
    y_min = max(0.0, min(strengths) - 8.0)
    y_max = min(100.0, max(strengths) + 8.0)
    if y_max - y_min < 20.0:
        center = (y_min + y_max) / 2.0
        y_min, y_max = max(0.0, center - 10.0), min(100.0, center + 10.0)
    left, right, top, bottom = 120.0, 1120.0, 155.0, 640.0

    def x(value: float) -> float:
        return left + value / x_max * (right - left)

    def y(value: float) -> float:
        return bottom - (value - y_min) / (y_max - y_min) * (bottom - top)

    parts = [
        '<text class="title" x="60" y="58">Validation strength vs latency</text>',
        '<text class="subtitle" x="60" y="88">Common-opponent pair score; native single-thread p95</text>',
    ]
    for index in range(7):
        value = index * x_max / 6
        parts.append(f'<line x1="{x(value):.2f}" y1="{top}" x2="{x(value):.2f}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(f'<text class="axis" x="{x(value):.2f}" y="{bottom + 28}" text-anchor="middle">{value:.0f}</text>')
    for index in range(6):
        value = y_min + index * (y_max - y_min) / 5
        parts.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{right}" y2="{y(value):.2f}" stroke="{GRID}"/>')
        parts.append(f'<text class="axis" x="{left - 12}" y="{y(value) + 5:.2f}" text-anchor="end">{value:.0f}%</text>')
    parts.append(f'<line x1="{x(50):.2f}" y1="{top}" x2="{x(50):.2f}" y2="{bottom}" '
                 'stroke="#E69F00" stroke-width="3" stroke-dasharray="8 6"/>')
    parts.append(f'<text class="small" x="{x(50) + 8:.2f}" y="{top + 18}">50 ms gate</text>')
    frontier = sorted(
        (point for point in normalized if point["optimal"]),
        key=lambda point: point["latency"],
    )
    if len(frontier) > 1:
        path = " ".join(
            ("M" if index == 0 else "L")
            + f" {x(point['latency']):.2f} {y(point['strength']):.2f}"
            for index, point in enumerate(frontier)
        )
        parts.append(f'<path d="{path}" fill="none" stroke="#666666" stroke-width="2"/>')
    family_colors: dict[str, str] = {}
    for point in normalized:
        identifier = point["id"]
        latency = point["latency"]
        strength = point["strength"]
        optimal = point["optimal"]
        eligible = point["eligible"]
        selected = point["selected"]
        fixed = point["fixed"]
        family = identifier.split("-")[0]
        if identifier.startswith("alpha-beta"):
            family = "alpha-beta"
        if identifier.startswith("rank5"):
            family = "rank5"
        if family not in family_colors:
            family_colors[family] = PALETTE[len(family_colors) % len(PALETTE)]
        color = family_colors[family]
        opacity = "1" if eligible else "0.42"
        radius = 11 if selected else 7
        stroke = "#111111" if fixed else ("#FFFFFF" if optimal else "#555555")
        if point["interval"] is not None:
            lower, upper = point["interval"]
            interval_x = x(latency)
            parts.append(
                f'<line class="strength-ci" x1="{interval_x:.2f}" y1="{y(lower):.2f}" '
                f'x2="{interval_x:.2f}" y2="{y(upper):.2f}" stroke="{color}" '
                f'stroke-opacity="{opacity}" stroke-width="2"/>'
            )
            for bound in (lower, upper):
                parts.append(
                    f'<line class="strength-ci-cap" x1="{interval_x - 5:.2f}" '
                    f'y1="{y(bound):.2f}" x2="{interval_x + 5:.2f}" '
                    f'y2="{y(bound):.2f}" stroke="{color}" '
                    f'stroke-opacity="{opacity}" stroke-width="2"/>'
                )
        parts.append(f'<circle cx="{x(latency):.2f}" cy="{y(strength):.2f}" r="{radius}" '
                     f'fill="{color}" fill-opacity="{opacity}" stroke="{stroke}" stroke-width="2"/>')
        label = labels.get(identifier, identifier)
        parts.append(f'<text class="small" x="{x(latency) + 12:.2f}" y="{y(strength) - 9:.2f}">{_escape(label)}</text>')
        sample_label = (
            f"defined reference; strength n=N/A; fresh-root latency "
            f"n={point['latency_decisions']} decisions"
            if fixed else (
                f"strength n={point['pairs']} pairs; latency "
                f"n={point['latency_decisions']} decisions"
            )
        )
        parts.append(
            f'<text class="small" x="{x(latency) + 12:.2f}" '
            f'y="{y(strength) + 10:.2f}">{_escape(sample_label)}</text>'
        )
    parts += [
        f'<text class="axis" x="{(left + right) / 2:.2f}" y="704" text-anchor="middle">Validation p95 decision latency (ms; Rank5DerivedBot — fixed 50k demo profile uses fresh-root searches)</text>',
        f'<text class="axis" x="28" y="{(top + bottom) / 2:.2f}" transform="rotate(-90 28 {(top + bottom) / 2:.2f})" text-anchor="middle">Mean color-swapped pair score vs Rank5DerivedBot — fixed 50k demo profile</text>',
        '<text class="subtitle" x="60" y="738">Vertical bars are pair-bootstrap 95% intervals; large points are selected; black outlines mark fixed; faded points miss the gate; the line is the constrained frontier.</text>',
    ]
    return _document(
        "Validation strength versus p95 latency",
        "Validation-only constrained Pareto frontier with candidate pair-bootstrap intervals, sample sizes, a fifty millisecond gate, and the Rank5DerivedBot — fixed 50k demo profile reference point.",
        "\n".join(parts),
    )


def calibration_svg(analysis: Mapping[str, Any], labels: Mapping[str, str]) -> str:
    calibration = analysis.get("calibration")
    if not isinstance(calibration, Mapping) or len(calibration) != 4:
        raise ChartError("calibration chart requires four bot metrics")
    left, right, top, bottom = 130.0, 760.0, 150.0, 650.0

    def x(value: float) -> float:
        return left + value * (right - left)

    def y(value: float) -> float:
        return bottom - value * (bottom - top)

    parts = [
        '<text class="title" x="60" y="58">Test calibration</text>',
        '<text class="subtitle" x="60" y="88">Validation-fitted mappings applied once to test predictions</text>',
    ]
    for index in range(6):
        value = index / 5
        parts.append(f'<line x1="{x(value):.2f}" y1="{top}" x2="{x(value):.2f}" y2="{bottom}" stroke="{GRID}"/>')
        parts.append(f'<line x1="{left}" y1="{y(value):.2f}" x2="{right}" y2="{y(value):.2f}" stroke="{GRID}"/>')
        parts.append(f'<text class="axis" x="{x(value):.2f}" y="{bottom + 28}" text-anchor="middle">{value:.1f}</text>')
        parts.append(f'<text class="axis" x="{left - 12}" y="{y(value) + 5:.2f}" text-anchor="end">{value:.1f}</text>')
    parts.append(f'<line x1="{x(0):.2f}" y1="{y(0):.2f}" x2="{x(1):.2f}" y2="{y(1):.2f}" '
                 'stroke="#777777" stroke-width="2" stroke-dasharray="7 6"/>')
    for index, identifier in enumerate(sorted(calibration)):
        metrics = calibration[identifier]
        bins = metrics.get("reliability_bins")
        if not isinstance(bins, Sequence) or len(bins) != 10:
            raise ChartError("calibration metrics require ten reliability bins")
        points = []
        for bin_value in bins:
            count = bin_value.get("count", 0)
            if count <= 0:
                continue
            interval = bin_value.get("observed_frequency_pair_bootstrap_95")
            prediction = _finite(bin_value.get("mean_prediction"), "mean prediction")
            observed = _finite(bin_value.get("observed_frequency"), "observed frequency")
            if interval is None:
                lower = upper = None
            else:
                if (not isinstance(interval, Mapping)
                        or interval.get("method") != "pair_cluster_percentile_stratified"):
                    raise ChartError(
                        "calibration bin has an invalid pair-cluster interval"
                    )
                lower = _finite(interval.get("lower"), "reliability lower")
                upper = _finite(interval.get("upper"), "reliability upper")
                if not 0.0 <= lower <= upper <= 1.0:
                    raise ChartError("pair-cluster reliability interval is invalid")
            points.append((prediction, observed, lower, upper,
                           _positive_integer(count, "reliability bin count")))
        color = PALETTE[index]
        if points:
            path = " ".join(
                ("M" if point_index == 0 else "L") + f" {x(px):.2f} {y(py):.2f}"
                for point_index, (px, py, _, _, _) in enumerate(points)
            )
            parts.append(f'<path d="{path}" fill="none" stroke="{color}" stroke-width="3"/>')
            for px, py, lower, upper, count in points:
                if lower is not None and upper is not None:
                    parts.append(
                        f'<line class="reliability-ci" x1="{x(px):.2f}" '
                        f'y1="{y(lower):.2f}" x2="{x(px):.2f}" '
                        f'y2="{y(upper):.2f}" stroke="{color}" stroke-width="2"/>'
                    )
                radius = 3.5 + min(5.0, math.sqrt(count) * 0.4)
                interval_title = (
                    f"pair-cluster 95% [{lower:.3f}, {upper:.3f}]"
                    if lower is not None and upper is not None
                    else "pair-cluster interval unavailable"
                )
                parts.append(
                    f'<circle cx="{x(px):.2f}" cy="{y(py):.2f}" r="{radius:.2f}" '
                    f'fill="{color}"><title>bin prediction n={count}; '
                    f'{interval_title}</title></circle>'
                )
        legend_y = 180 + index * 88
        bootstrap = metrics.get("pair_cluster_bootstrap_95")
        if (not isinstance(bootstrap, Mapping)
                or bootstrap.get("method") != "pair_cluster_percentile_stratified"):
            raise ChartError("calibration metrics lack pair-cluster bootstrap intervals")
        brier_interval = bootstrap.get("brier_score")
        log_interval = bootstrap.get("log_loss")
        if not isinstance(brier_interval, Mapping) or not isinstance(log_interval, Mapping):
            raise ChartError("calibration score intervals are invalid")
        brier_lower = _finite(brier_interval.get("lower"), "Brier lower")
        brier_upper = _finite(brier_interval.get("upper"), "Brier upper")
        log_lower = _finite(log_interval.get("lower"), "log lower")
        log_upper = _finite(log_interval.get("upper"), "log upper")
        if not 0.0 <= brier_lower <= brier_upper <= 1.0 or \
           not 0.0 <= log_lower <= log_upper:
            raise ChartError("calibration score intervals are invalid")
        parts.append(f'<rect x="820" y="{legend_y - 13}" width="22" height="6" fill="{color}"/>')
        parts.append(f'<text class="legend" x="854" y="{legend_y}">{_escape(labels[identifier])}</text>')
        parts.append(
            f'<text class="small" x="854" y="{legend_y + 24}">Brier '
            f'{_finite(metrics["brier_score"], "Brier"):.3f} '
            f'[{brier_lower:.3f}, {brier_upper:.3f}]</text>'
        )
        parts.append(
            f'<text class="small" x="854" y="{legend_y + 43}">Log loss '
            f'{_finite(metrics["log_loss"], "log loss"):.3f} '
            f'[{log_lower:.3f}, {log_upper:.3f}]</text>'
        )
        parts.append(
            f'<text class="small" x="854" y="{legend_y + 62}">prediction n='
            f'{_escape(metrics["samples"])}; pair n='
            f'{_escape(_positive_integer(metrics.get("pair_clusters"), "pair clusters"))}</text>'
        )
    parts += [
        f'<text class="axis" x="{(left + right) / 2:.2f}" y="708" text-anchor="middle">Mean predicted win probability</text>',
        f'<text class="axis" x="30" y="{(top + bottom) / 2:.2f}" transform="rotate(-90 30 {(top + bottom) / 2:.2f})" text-anchor="middle">Observed win frequency</text>',
        '<text class="subtitle" x="60" y="728">Vertical bars and score intervals use 10,000 whole-pair bootstraps within matchup × depth; point area scales with bin prediction n.</text>',
        '<text class="subtitle" x="60" y="750">Within-game predictions are dependent; n counts prediction opportunities, not independent games.</text>',
    ]
    return _document(
        "Test reliability and calibration",
        "Ten-bin reliability comparison with prediction and pair sample sizes, whole-pair clustered bootstrap intervals, Brier score, and log loss for four competitive bots.",
        "\n".join(parts),
    )


__all__ = ["ChartError", "bradley_terry_svg", "calibration_svg", "pareto_svg"]
