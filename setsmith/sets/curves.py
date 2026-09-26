"""Energy curves: target energy (1-10) for each position in a set."""

from __future__ import annotations

from bisect import bisect_right

from setsmith.scoring.weights import DEFAULT_CONFIG, CurveTemplate, ScoringConfig


class EnergyCurve:
    def __init__(self, name: str, template: CurveTemplate, cfg: ScoringConfig = DEFAULT_CONFIG):
        self.name = name
        self.template = template
        self._lo, self._hi = cfg.energy.scale_min, cfg.energy.scale_max

    def at(self, position: float) -> float:
        """Base curve value at a normalized position in 0..1 (no dips)."""
        points = self.template.points
        position = min(1.0, max(0.0, position))
        i = bisect_right([p for p, _ in points], position)
        if i >= len(points):
            value = points[-1][1]
        else:
            (x0, y0), (x1, y1) = points[i - 1], points[i]
            value = y0 if x1 == x0 else y0 + (y1 - y0) * (position - x0) / (x1 - x0)
        if self.template.cap is not None:
            value = min(value, self.template.cap)
        return min(self._hi, max(self._lo, value))

    def targets(self, n: int) -> list[float]:
        """Target energy for each of n tracks, including dips."""
        if n <= 0:
            return []
        values = [self.at(i / (n - 1) if n > 1 else 0.0) for i in range(n)]
        every = self.template.dip_every_tracks
        if every:
            for i in range(every, n - 1, every):
                values[i] = max(self._lo, values[i] - self.template.dip_depth)
        return values


def parse_curve(spec: str, cfg: ScoringConfig = DEFAULT_CONFIG) -> EnergyCurve:
    """A template name ("journey"), or custom points.

    Custom points are either evenly spaced energies ("3,5,8,6") or position:energy pairs
    ("0:3,0.6:9,1:5").
    """
    spec = spec.strip()
    if spec in cfg.sets.curves:
        return EnergyCurve(spec, cfg.sets.curves[spec], cfg)
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if len(parts) < 2:  # noqa: PLR2004 - a curve needs a start and an end
        names = ", ".join(cfg.sets.curves)
        raise ValueError(f"unknown curve {spec!r}: use one of {names}, or custom points")
    try:
        if all(":" in p for p in parts):
            points = tuple(
                (float(pos), float(val)) for pos, val in (p.split(":", 1) for p in parts)
            )
        else:
            values = [float(p) for p in parts]
            points = tuple((i / (len(values) - 1), v) for i, v in enumerate(values))
    except ValueError as exc:
        raise ValueError(f"could not read curve points {spec!r}: {exc}") from exc
    for _, value in points:
        if not cfg.energy.scale_min <= value <= cfg.energy.scale_max:
            raise ValueError(f"curve energy {value:g} is outside 1-10")
    return EnergyCurve("custom", CurveTemplate(points=points), cfg)
