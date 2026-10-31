"""Trend statistics for short ordered series."""
from __future__ import annotations

import math

import numpy as np
from scipy import stats


def mann_kendall(y: list[float], alpha: float = 0.05) -> dict:
    """Mann-Kendall monotonic trend test with tie correction, plus Sen's slope.

    Assumes observations are serially independent; autocorrelation inflates
    false positives. Requires at least 4 observations.
    """
    x = np.asarray(y, dtype=float)
    n = len(x)
    if n < 4:
        return {"available": False, "reason": "Mann-Kendall needs at least 4 periods."}
    s = 0
    for i in range(n - 1):
        s += np.sign(x[i + 1:] - x[i]).sum()
    _, counts = np.unique(x, return_counts=True)
    var = (n * (n - 1) * (2 * n + 5) - sum(t * (t - 1) * (2 * t + 5) for t in counts)) / 18.0
    if var <= 0:
        return {"available": True, "trend": "no trend", "s": int(s), "z": 0.0, "p_value": 1.0, "sens_slope": 0.0, "n": n}
    if s > 0:
        z = (s - 1) / math.sqrt(var)
    elif s < 0:
        z = (s + 1) / math.sqrt(var)
    else:
        z = 0.0
    p = float(2 * stats.norm.sf(abs(z)))
    slopes = [(x[j] - x[i]) / (j - i) for i in range(n - 1) for j in range(i + 1, n)]
    trend = "no significant trend"
    if p < alpha:
        trend = "increasing" if z > 0 else "decreasing"
    tau = s / (0.5 * n * (n - 1))
    return {
        "available": True,
        "trend": trend,
        "s": int(s),
        "z": round(float(z), 4),
        "p_value": round(p, 6),
        "kendall_tau": round(float(tau), 4),
        "sens_slope": round(float(np.median(slopes)), 6),
        "n": n,
        "alpha": alpha,
        "assumption": "serially independent observations",
    }


def rolling_mean(y: list[float], window: int = 3) -> list[float | None]:
    out: list[float | None] = []
    for i in range(len(y)):
        if i + 1 < window:
            out.append(None)
        else:
            out.append(round(float(np.mean(y[i + 1 - window:i + 1])), 4))
    return out


def exponential_smoothing(y: list[float], alpha: float = 0.3) -> list[float]:
    if not y:
        return []
    out = [float(y[0])]
    for v in y[1:]:
        out.append(alpha * float(v) + (1 - alpha) * out[-1])
    return [round(v, 4) for v in out]


def change_points(y: list[float], min_size: int = 2, max_points: int = 5) -> list[int]:
    """Binary segmentation for shifts in mean (squared-error cost, BIC-style
    penalty 2*log(n)*variance). Returns indices where a new regime starts."""
    x = np.asarray(y, dtype=float)
    n = len(x)
    if n < 2 * min_size + 1:
        return []
    var = float(np.var(x)) or 1e-12
    penalty = 2 * math.log(n) * var

    def cost(a, b):
        seg = x[a:b]
        return float(((seg - seg.mean()) ** 2).sum())

    segments = [(0, n)]
    found: list[int] = []
    while len(found) < max_points:
        best = None
        for a, b in segments:
            if b - a < 2 * min_size:
                continue
            base = cost(a, b)
            for t in range(a + min_size, b - min_size + 1):
                gain = base - cost(a, t) - cost(t, b)
                if best is None or gain > best[0]:
                    best = (gain, t, (a, b))
        if best is None or best[0] <= penalty:
            break
        _, t, (a, b) = best
        found.append(t)
        segments.remove((a, b))
        segments += [(a, t), (t, b)]
    return sorted(found)


SEASONAL_PERIOD = {"D": 7, "W": 52, "M": 12, "Q": 4}


def seasonal_decomposition(y: list[float], freq: str) -> dict:
    period = SEASONAL_PERIOD.get(freq)
    if period is None or len(y) < 2 * period:
        return {"available": False, "reason": f"Seasonal decomposition needs at least two full cycles ({2 * (period or 0)} periods); found {len(y)}."}
    try:
        from statsmodels.tsa.seasonal import seasonal_decompose
    except ImportError:
        return {"available": False, "reason": "statsmodels is not installed (pip install thematicml[temporal])."}
    res = seasonal_decompose(np.asarray(y, dtype=float), period=period, model="additive", extrapolate_trend="freq")
    return {
        "available": True,
        "period": period,
        "trend": [round(float(v), 4) for v in res.trend],
        "seasonal": [round(float(v), 4) for v in res.seasonal],
        "residual": [round(float(v), 4) for v in res.resid],
    }


def arima_forecast(y: list[float], steps: int = 3, min_obs: int = 24) -> dict:
    if len(y) < min_obs:
        return {"available": False, "reason": f"ARIMA is only fitted with >= {min_obs} periods; found {len(y)}."}
    try:
        import warnings

        from statsmodels.tsa.arima.model import ARIMA
    except ImportError:
        return {"available": False, "reason": "statsmodels is not installed (pip install thematicml[temporal])."}
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for order in [(0, 1, 1), (1, 1, 0), (1, 1, 1), (1, 0, 0), (2, 1, 1)]:
            try:
                fit = ARIMA(np.asarray(y, dtype=float), order=order).fit()
            except Exception:
                continue
            if best is None or fit.aic < best[1].aic:
                best = (order, fit)
    if best is None:
        return {"available": False, "reason": "No ARIMA model converged."}
    order, fit = best
    fc = fit.get_forecast(steps)
    ci = fc.conf_int(alpha=0.05)
    return {
        "available": True,
        "order": list(order),
        "aic": round(float(fit.aic), 3),
        "forecast": [round(float(v), 4) for v in fc.predicted_mean],
        "ci95": [[round(float(a), 4), round(float(b), 4)] for a, b in ci],
        "selection": "lowest AIC among a small set of orders",
    }
