from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import asdict, dataclass
from itertools import product
from time import perf_counter
from typing import TYPE_CHECKING, Any, Iterable, Literal

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from rkhs_epsilon import RKHSEpsilonMachine

if TYPE_CHECKING:
    from matplotlib.axes import Axes

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


# Thins tick labels so dense grids stay readable.
def _tick_step(size: int, target_ticks: int = 12) -> int:
    return max(1, int(np.ceil(size / target_ticks)))


# Plots one metric with defaults adapted to discrete or continuous values.
def plot_sensitivity_heatmap(
    results: pd.DataFrame,
    metric: Literal["statistical_complexity", "discovered_states"] = "statistical_complexity",
    *,
    ax: Axes | None = None,
    cmap: str | None = None,
    annotate: bool | None = None,
) -> Axes:
    """Plot a readable sensitivity heatmap."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import MaxNLocator

    grid = pivot_sensitivity_metric(results, metric)
    rows, cols = grid.shape

    if ax is None:
        width = min(18.0, max(6.0, cols * 0.45 + 2.0))
        height = min(14.0, max(5.0, rows * 0.45 + 1.5))
        _, ax = plt.subplots(figsize=(width, height))

    show_annotations = annotate if annotate is not None else rows * cols <= 144
    is_discrete = metric == "discovered_states"

    heatmap_kwargs: dict[str, Any] = {
        "ax": ax,
        "annot": show_annotations,
        "fmt": "d" if is_discrete else ".2f",
        "linewidths": 0.25,
        "linecolor": "white",
        "cbar_kws": {"shrink": 0.85, "pad": 0.02},
    }

    if is_discrete:
        values = grid.to_numpy(dtype=float)
        finite_values = values[np.isfinite(values)]
        if finite_values.size:
            state_values = np.unique(finite_values.astype(int))
            boundaries = np.arange(state_values.min() - 0.5, state_values.max() + 1.5, 1.0)
            heatmap_kwargs["cmap"] = cmap or "viridis"
            heatmap_kwargs["norm"] = BoundaryNorm(boundaries, plt.get_cmap(heatmap_kwargs["cmap"]).N)
            heatmap_kwargs["vmin"] = boundaries[0]
            heatmap_kwargs["vmax"] = boundaries[-1]
            plot_data = grid.astype(float)
            if show_annotations:
                heatmap_kwargs["annot"] = grid.apply(
                    lambda column: column.map(lambda value: "" if pd.isna(value) else f"{int(value)}")
                )
                heatmap_kwargs["fmt"] = ""
        else:
            heatmap_kwargs["cmap"] = cmap or "viridis"
            plot_data = grid
    else:
        heatmap_kwargs["cmap"] = cmap or "mako"
        plot_data = grid

    sns.heatmap(plot_data, **heatmap_kwargs)

    row_step = _tick_step(rows)
    col_step = _tick_step(cols)
    ax.set_xticks(np.arange(cols)[::col_step] + 0.5)
    ax.set_xticklabels(grid.columns[::col_step], rotation=45, ha="right")
    ax.set_yticks(np.arange(rows)[::row_step] + 0.5)
    ax.set_yticklabels(grid.index[::row_step], rotation=0)

    ax.set_xlabel("L_future")
    ax.set_ylabel("L_past")
    ax.set_title(metric.replace("_", " ").title())

    colorbar = ax.collections[0].colorbar
    if colorbar is not None and is_discrete:
        colorbar.locator = MaxNLocator(integer=True)
        colorbar.update_ticks()
        colorbar.set_label("Discovered states")
    elif colorbar is not None:
        colorbar.set_label("Statistical complexity")

    return ax


__all__ = [
    "pivot_sensitivity_metric",
    "plot_sensitivity_heatmap",
    "run_sensitivity_analysis",
    "SensitivityRecord",
]