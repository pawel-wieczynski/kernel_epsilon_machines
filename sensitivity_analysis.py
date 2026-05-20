from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
from typing import Any, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from rkhs_epsilon import RKHSEpsilonMachine

GridLike = int | Iterable[int]
ExecutorKind = Literal["process", "thread"]

_WORKER_SERIES: np.ndarray | None = None
_WORKER_MODEL_KWARGS: dict[str, Any] = {}
_WORKER_RANDOM_STATE: int | None = None


# Stores one evaluated grid point and any fit failure.
@dataclass(slots=True)
class SensitivityRecord:
    L_past: int
    L_future: int
    discovered_states: int | None
    statistical_complexity: float | None
    runtime_seconds: float | None
    error: str | None


# Normalizes one grid value to a positive integer.
def _coerce_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} values must be integers")

    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} values must be integers") from exc

    if coerced != value:
        raise ValueError(f"{name} values must be integers")
    if coerced < 1:
        raise ValueError(f"{name} values must be positive")
    return coerced


# Converts one int or iterable into a sorted unique grid.
def _normalize_grid(values: GridLike, name: str) -> tuple[int, ...]:
    if isinstance(values, (int, np.integer)):
        grid = (_coerce_positive_int(values, name),)
    else:
        grid = tuple(_coerce_positive_int(value, name) for value in values)

    if not grid:
        raise ValueError(f"{name} must contain at least one value")
    return tuple(sorted(dict.fromkeys(grid)))


# Coerces the input series into a valid contiguous 1D array.
def _as_1d_series(series: ArrayLike) -> np.ndarray:
    values = np.asarray(series, dtype=float)
    if values.ndim != 1:
        raise ValueError("series must be one-dimensional")
    if values.size == 0:
        raise ValueError("series must not be empty")
    return np.ascontiguousarray(values)


# Seeds worker-local state so each job only receives grid coordinates.
def _init_worker(
    series: np.ndarray,
    model_kwargs: dict[str, Any],
    random_state: int | None,
) -> None:
    global _WORKER_SERIES, _WORKER_MODEL_KWARGS, _WORKER_RANDOM_STATE
    _WORKER_SERIES = series
    _WORKER_MODEL_KWARGS = model_kwargs
    _WORKER_RANDOM_STATE = random_state


# Fits one L_past/L_future pair and captures the core outputs.
def _evaluate_pair(job: tuple[int, int]) -> SensitivityRecord:
    if _WORKER_SERIES is None:
        raise RuntimeError("worker state was not initialized")

    L_past, L_future = job
    started = perf_counter()

    try:
        model = RKHSEpsilonMachine(
            L_past=L_past,
            L_future=L_future,
            clustering_method="kmeans",
            random_state=_WORKER_RANDOM_STATE,
            **_WORKER_MODEL_KWARGS,
        )
        model.fit(_WORKER_SERIES)
        return SensitivityRecord(
            L_past=L_past,
            L_future=L_future,
            discovered_states=int(model.n_states_found_),
            statistical_complexity=float(model.statistical_complexity_),
            runtime_seconds=perf_counter() - started,
            error=None,
        )
    except Exception as exc:
        return SensitivityRecord(
            L_past=L_past,
            L_future=L_future,
            discovered_states=None,
            statistical_complexity=None,
            runtime_seconds=perf_counter() - started,
            error=str(exc),
        )


# Maps the executor label to the matching pool implementation.
def _resolve_executor(executor: ExecutorKind) -> type[ProcessPoolExecutor] | type[ThreadPoolExecutor]:
    if executor == "process":
        return ProcessPoolExecutor
    if executor == "thread":
        return ThreadPoolExecutor
    raise ValueError("executor must be 'process' or 'thread'")


# Runs the full sensitivity grid in parallel and returns tidy results.
def run_sensitivity_analysis(
    series: ArrayLike,
    L_past: GridLike,
    L_future: GridLike,
    *,
    executor: ExecutorKind = "process",
    max_workers: int | None = None,
    random_state: int | None = 42,
    model_kwargs: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run a parallel L_past x L_future grid."""
    series_values = _as_1d_series(series)
    L_past_values = _normalize_grid(L_past, "L_past")
    L_future_values = _normalize_grid(L_future, "L_future")

    cleaned_model_kwargs = dict(model_kwargs or {})
    cleaned_model_kwargs.pop("L_past", None)
    cleaned_model_kwargs.pop("L_future", None)
    cleaned_model_kwargs.pop("clustering_method", None)
    cleaned_model_kwargs.pop("random_state", None)

    jobs = list(product(L_past_values, L_future_values))
    executor_cls = _resolve_executor(executor)
    with executor_cls(
        max_workers=max_workers,
        initializer=_init_worker,
        initargs=(series_values, cleaned_model_kwargs, random_state),
    ) as pool:
        records = list(pool.map(_evaluate_pair, jobs))

    results = pd.DataFrame(asdict(record) for record in records)
    return results.sort_values(["L_past", "L_future"], ignore_index=True)


# Reshapes one metric into a heatmap-friendly grid.
def pivot_sensitivity_metric(
    results: pd.DataFrame,
    metric: Literal["statistical_complexity", "discovered_states"] = "statistical_complexity",
) -> pd.DataFrame:
    """Pivot one sensitivity metric into a grid."""
    if metric not in {"statistical_complexity", "discovered_states"}:
        raise ValueError("metric must be 'statistical_complexity' or 'discovered_states'")

    return results.pivot(index="L_past", columns="L_future", values=metric)


__all__ = ["pivot_sensitivity_metric", "run_sensitivity_analysis", "SensitivityRecord"]