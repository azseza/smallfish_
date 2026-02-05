import time
import math
from typing import Sequence

# Simulated time for backtesting.  When set, time_now_ms() returns this
# value instead of the wall clock, making all time-windowed analytics
# (tape, book, state) work correctly with historical data.
_sim_time_ms: int | None = None


def set_sim_time(ts_ms: int) -> None:
    """Set simulated time (call per-candle in backtest)."""
    global _sim_time_ms
    _sim_time_ms = ts_ms


def clear_sim_time() -> None:
    """Restore wall-clock time (call after backtest)."""
    global _sim_time_ms
    _sim_time_ms = None


def time_now_ms() -> int:
    if _sim_time_ms is not None:
        return _sim_time_ms
    return int(time.time() * 1000)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def clamp(value: float, lo: float, hi: float) -> float:
    if value < lo:
        return lo
    if value > hi:
        return hi
    return value


def ema_update(prev: float, new_val: float, alpha: float) -> float:
    """Single-step EMA update: out = alpha * new_val + (1 - alpha) * prev."""
    return alpha * new_val + (1.0 - alpha) * prev


def ema_alpha(span: float) -> float:
    """Compute EMA decay factor for a given span (number of observations)."""
    return 2.0 / (span + 1.0)


def ema_alpha_seconds(halflife_s: float, dt_s: float) -> float:
    """Compute EMA alpha for time-based decay.
    alpha = 1 - exp(-dt / halflife)
    """
    if halflife_s <= 0 or dt_s <= 0:
        return 1.0
    return 1.0 - math.exp(-dt_s / halflife_s)


def ewma_series(data: Sequence[float], span: int) -> float:
    """Compute EWMA of a series using the standard span formula."""
    if not data:
        return 0.0
    alpha = ema_alpha(span)
    result = data[0]
    for v in data[1:]:
        result = alpha * v + (1.0 - alpha) * result
    return result


def realized_vol(prices: Sequence[float]) -> float:
    """Annualized realized volatility from a series of prices.
    Uses log returns, annualized assuming 1-second sampling.
    """
    if len(prices) < 2:
        return 0.0
    sum_sq = 0.0
    n = 0
    for i in range(1, len(prices)):
        if prices[i - 1] <= 0:
            continue
        log_ret = math.log(prices[i] / prices[i - 1])
        sum_sq += log_ret * log_ret
        n += 1
    if n == 0:
        return 0.0
    # variance per sample, then annualize (86400 seconds/day * 365 days)
    var_per_sample = sum_sq / n
    return math.sqrt(var_per_sample * n)  # realized vol over the window


def z_score(value: float, mean: float, std: float) -> float:
    if std <= 1e-12:
        return 0.0
    return (value - mean) / std


def tick_round(price: float, tick_size: float) -> float:
    """Round price to nearest tick."""
    if tick_size <= 0:
        return price
    return round(round(price / tick_size) * tick_size, 10)


def qty_round(qty: float, step: float) -> float:
    """Round quantity to step size, floored."""
    if step <= 0:
        return qty
    return math.floor(qty / step) * step


def safe_div(a: float, b: float, default: float = 0.0) -> float:
    if abs(b) < 1e-15:
        return default
    return a / b
