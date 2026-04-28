import numpy as np

def two_regime_gauss(
    N=2000,
    mu_A=0.5,
    sigma_A=0.3,
    mu_B=-0.5,
    sigma_B=1.0,
    p_stay_A=0.98,
    p_stay_B=0.95,
    init_dist=None,
    seed=2137,
):
    np.random.seed(seed)
    states_true = np.zeros(N, dtype=int)
    series = np.zeros(N)

    if init_dist is None:
        state = 0
    else:
        init_dist = np.asarray(init_dist, dtype=float)
        init_dist = init_dist / init_dist.sum()
        state = int(np.random.choice(len(init_dist), p=init_dist))
    for t in range(N):
        states_true[t] = state
        if state == 0:
            series[t] = np.random.normal(mu_A, sigma_A)
            if np.random.rand() > p_stay_A:
                state = 1
        else:
            series[t] = np.random.normal(mu_B, sigma_B)
            if np.random.rand() > p_stay_B:
                state = 0

    T_true = np.array([
        [p_stay_A, 1 - p_stay_A],
        [1 - p_stay_B, p_stay_B],
    ])
    pi_true = np.array([
        (1 - p_stay_B) / ((1 - p_stay_A) + (1 - p_stay_B)),
        (1 - p_stay_A) / ((1 - p_stay_A) + (1 - p_stay_B)),
    ])

    return series, states_true, T_true, pi_true