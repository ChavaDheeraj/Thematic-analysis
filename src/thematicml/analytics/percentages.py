"""Percentages that always carry their numerator, denominator and definition."""
from __future__ import annotations

import math


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n <= 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(100 * max(0.0, centre - half), 2), round(100 * min(1.0, centre + half), 2))


def percentage(numerator: int, denominator: int, definition: str, level: str | None = None, ci: bool = False) -> dict:
    value = round(100.0 * numerator / denominator, 2) if denominator else None
    out = {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": value,
        "definition": definition,
    }
    if level:
        out["level"] = level
    if ci and denominator:
        out["ci95"] = wilson_ci(numerator, denominator)
    return out
