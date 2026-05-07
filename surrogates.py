import numpy as np

def shuffle_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = series.copy()
    rng.shuffle(out)
    return out

def phase_randomize_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(series)
    fft_vals = np.fft.rfft(series)
    amplitudes = np.abs(fft_vals)
    n_freqs = len(fft_vals)
    phases = rng.uniform(0, 2 * np.pi, n_freqs)
    phases[0] = 0.0
    if n % 2 == 0:
        phases[-1] = 0.0
    new_fft = amplitudes * np.exp(1j * phases)
    return np.fft.irfft(new_fft, n=n).astype(series.dtype)

def block_shuffle_surrogate(
    series: np.ndarray,
    rng: np.random.Generator,
    block_size: int | None = None,
) -> np.ndarray:
    n = len(series)
    if block_size is None:
        block_size = max(5, int(np.sqrt(n)))
    n_blocks = n // block_size
    if n_blocks < 2:
        return shuffle_surrogate(series, rng)
    blocks = [series[i * block_size:(i + 1) * block_size] for i in range(n_blocks)]
    rng.shuffle(blocks)
    tail = series[n_blocks * block_size:]
    return np.concatenate(blocks + ([tail] if len(tail) else []))

def garch_surrogate(series: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(series)
    r = series - series.mean()
    r2 = r ** 2
    unconditional_var = r2.mean()
    acf1 = float(np.corrcoef(r2[:-1], r2[1:])[0, 1])
    ab = np.clip(acf1, 0.05, 0.97)
    alpha = min(0.15, ab * 0.2)
    beta = ab - alpha
    omega = max(unconditional_var * (1.0 - alpha - beta), 1e-10)
    sigma2 = np.empty(n)
    out = np.empty(n)
    sigma2[0] = unconditional_var
    z = rng.standard_normal(n)
    out[0] = np.sqrt(sigma2[0]) * z[0]
    for t in range(1, n):
        sigma2[t] = omega + alpha * out[t - 1] ** 2 + beta * sigma2[t - 1]
        out[t] = np.sqrt(sigma2[t]) * z[t]
    return out.astype(series.dtype)

_FACTORIES = {
    "shuffle":       shuffle_surrogate,
    "phase_random":  phase_randomize_surrogate,
    "block_shuffle": block_shuffle_surrogate,
    "garch":         garch_surrogate,
}

def _run_job(job: tuple) -> dict:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent))

    from rkhs_epsilon import RKHSEpsilonMachine

    surrogate_name, series, seed, model_kwargs = job
    rng = np.random.default_rng(int(seed))
    surr = _FACTORIES[surrogate_name](series, rng)
    m = RKHSEpsilonMachine(**model_kwargs)
    m.fit(surr)
    return {
        "surrogate":              surrogate_name,
        "entropy_rate":           m.entropy_rate_,
        "statistical_complexity": m.statistical_complexity_,
        "n_states_found":         m.n_states_found_,
        "dbscan_noise_count":     m.dbscan_noise_count_,
    }
