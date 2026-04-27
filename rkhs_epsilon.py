import numpy as np
from scipy.spatial.distance import cdist, pdist
from scipy.linalg import eigh
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

def _rbf_kernel(X, Y=None, bandwidth=1.0):
    """Compute RBF (Gaussian) kernel matrix.
    K(x, y) = exp(-||x - y||^2 / (2 * bandwidth^2))
    """
    if Y is None:
        Y = X
    dists = cdist(X, Y, 'sqeuclidean')
    return np.exp(-dists / (2.0 * bandwidth ** 2))

def _median_heuristic(X):
    """Compute bandwidth via median heuristic."""
    dists = pdist(X, 'euclidean')
    return np.median(dists) if len(dists) > 0 else 1.0

def compute_mmd(X, Y, bandwidth=1.0):
    """
    Compute the Maximum Mean Discrepancy between two sample sets.

    MMD^2 = E[k(x,x')] - 2E[k(x,y)] + E[k(y,y')]

    Parameters
    ----------
    X : array (n, d)
    Y : array (m, d)
    bandwidth : float

    Returns
    -------
    mmd2 : float
        Squared MMD estimate.
    """
    Kxx = _rbf_kernel(X, X, bandwidth)
    Kyy = _rbf_kernel(Y, Y, bandwidth)
    Kxy = _rbf_kernel(X, Y, bandwidth)

    n = X.shape[0]
    m = Y.shape[0]

    # Unbiased estimator
    np.fill_diagonal(Kxx, 0)
    np.fill_diagonal(Kyy, 0)

    if n > 1 and m > 1:
        mmd2 = (Kxx.sum() / (n * (n - 1))
                - 2 * Kxy.sum() / (n * m)
                + Kyy.sum() / (m * (m - 1)))
    else:
        mmd2 = 0.0

    return max(mmd2, 0.0)

def build_delay_embedding(series, L, step=1, start=0, n_points=None):
    """
    Build a forward delay-window embedding from a time series.

    Parameters
    ----------
    series : array (N,)
        Scalar time series.
    L : int
        Window length.
    step : int
        Delay step.
    start : int
        Starting offset in the series.
    n_points : int or None
        Optional fixed number of windows to extract.

    Returns
    -------
    embedded : array (n_points, L)
        Each row is [x_t, x_{t+step}, ..., x_{t+(L-1)step}] for a valid
        starting index t >= start.
    """
    N = len(series)
    if n_points is None:
        n_points = N - start - (L - 1) * step
    if n_points <= 0:
        raise ValueError("Series too short for given L, step, and start")
    if start + (L - 1) * step + n_points > N:
        raise ValueError("Requested embedding exceeds series length")

    embedded = np.zeros((n_points, L))
    for i in range(L):
        offset = start + i * step
        embedded[:, i] = series[offset: offset + n_points]

    return embedded

class RKHSEpsilonMachine:
    """
    RKHS epsilon-machine for continuous time series.

    Implements the Brodu & Crutchfield (2022) approach:
    1. Construct past/future delay embeddings
    2. Compute kernel mean embeddings of conditional future distributions
    3. Use diffusion maps to find the causal state manifold
    4. Cluster the reduced coordinates to identify discrete causal states
    5. Estimate transitions and information-theoretic measures

    Parameters
    ----------
    L_past : int
        Past embedding dimension.
    L_future : int
        Future embedding dimension.
    bandwidth : float or 'median'
        Shared kernel bandwidth shortcut. If past_bandwidth and future_bandwidth
        are not given, this value is used for both.
    past_bandwidth : float or 'median' or None
        Past kernel bandwidth. If None, falls back to bandwidth.
    future_bandwidth : float or 'median' or None
        Future kernel bandwidth. If None, falls back to bandwidth.
    diffusion_bandwidth : float or 'median'
        Bandwidth for diffusion map kernel.
    diffusion_alpha : float
        Coifman-Lafon density-normalization exponent.
    diffusion_time : float
        Diffusion time used to scale the nontrivial eigenvectors.
    n_components : int
        Number of diffusion map components.
    clustering_method : {'dbscan', 'kmeans'}
        Discretization method applied to the diffusion coordinates.
    cluster_dims : int or None
        Number of leading diffusion coordinates used for clustering. If None,
        uses all retained coordinates.
    cluster_standardize : bool
        Whether to standardize clustering coordinates before clustering.
    n_states : int or None
        Number of causal states. Required for kmeans. Ignored by dbscan.
    dbscan_eps : float
        DBSCAN radius parameter.
    dbscan_min_samples : int
        DBSCAN minimum sample count.
    dbscan_noise_strategy : {'nearest_centroid', 'separate_state'}
        How to handle DBSCAN noise points.
    kmeans_n_init : int
        Number of KMeans initializations.
    random_state : int or None
        Random seed used by KMeans and any subsampling-based utilities.
    regularization : float
        Regularization parameter for kernel operations.
    max_mmd_samples : int or None
        Optional maximum number of future samples per state when computing MMD.
        If None, uses all samples.
    """

    def __init__(self, L_past=5, L_future=5, bandwidth='median',
                 past_bandwidth=None, future_bandwidth=None,
                 diffusion_bandwidth='median', diffusion_alpha=1.0,
                 diffusion_time=1.0, n_components=5,
                 clustering_method='dbscan', cluster_dims=2,
                 cluster_standardize=True, n_states=None,
                 dbscan_eps=0.0702, dbscan_min_samples=22,
                 dbscan_noise_strategy='nearest_centroid',
                 kmeans_n_init=10, random_state=42,
                 regularization=1e-4, max_mmd_samples=None):
        self.L_past = L_past
        self.L_future = L_future
        self.bandwidth = bandwidth
        self.past_bandwidth = bandwidth if past_bandwidth is None else past_bandwidth
        self.future_bandwidth = bandwidth if future_bandwidth is None else future_bandwidth
        self.diffusion_bandwidth = diffusion_bandwidth
        self.diffusion_alpha = diffusion_alpha
        self.diffusion_time = diffusion_time
        self.n_components = n_components
        self.clustering_method = clustering_method
        self.cluster_dims = cluster_dims
        self.cluster_standardize = cluster_standardize
        self.n_states = n_states
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_noise_strategy = dbscan_noise_strategy
        self.kmeans_n_init = kmeans_n_init
        self.random_state = random_state
        self.regularization = regularization
        self.max_mmd_samples = max_mmd_samples

    def fit(self, series):
        """
        Fit the RKHS epsilon machine to a time series.

        Parameters
        ----------
        series : array (N,)
            Scalar time series.

        Returns
        -------
        self
        """
        self._validate_parameters()
        series = np.asarray(series, dtype=float)
        self.series_ = series
        self.rng_ = np.random.default_rng(self.random_state)

        # Step 1: Build delay embeddings
        self._build_embeddings(series)

        # Step 2: Compute kernel mean embeddings
        self._compute_kernel_embeddings()

        # Step 3: Diffusion map on causal state space
        self._diffusion_map()

        # Step 4: Cluster into discrete causal states
        self._cluster_states()

        # Step 5: Compute transition probabilities and metrics
        self._compute_transitions()
        self._compute_metrics()

        return self

    def _validate_parameters(self):
        """Validate estimator parameters before fitting."""
        if self.L_past < 1 or self.L_future < 1:
            raise ValueError("L_past and L_future must be positive integers")
        if self.n_components < 1:
            raise ValueError("n_components must be at least 1")
        if self.cluster_dims is not None and self.cluster_dims < 1:
            raise ValueError("cluster_dims must be at least 1 when provided")
        if self.regularization < 0:
            raise ValueError("regularization must be nonnegative")

        method = self.clustering_method.lower()
        if method not in {'dbscan', 'kmeans'}:
            raise ValueError("clustering_method must be 'dbscan' or 'kmeans'")
        if method == 'kmeans' and self.n_states is None:
            raise ValueError("n_states must be provided when clustering_method='kmeans'")
        if self.dbscan_noise_strategy not in {'nearest_centroid', 'separate_state'}:
            raise ValueError(
                "dbscan_noise_strategy must be 'nearest_centroid' or 'separate_state'")

    def _resolve_bandwidth(self, bandwidth, X):
        """Resolve a bandwidth specifier against a dataset."""
        if bandwidth == 'median':
            return _median_heuristic(X)
        return float(bandwidth)

    def _build_embeddings(self, series):
        """Build past and future delay embeddings."""
        N = len(series)
        total_L = self.L_past + self.L_future
        n_points = N - total_L + 1

        if n_points <= 0:
            raise ValueError(
                f"Series length {N} too short for L_past={self.L_past}, "
                f"L_future={self.L_future}")

        if n_points < 2:
            raise ValueError("Need at least two embedded samples to fit the model")

        # Past: [x_{t-L_past+1}, ..., x_t]
        self.X_past = build_delay_embedding(series, self.L_past, start=0, n_points=n_points)

        # Future: [x_{t+1}, ..., x_{t+L_future}]
        self.X_future = build_delay_embedding(
            series,
            self.L_future,
            start=self.L_past,
            n_points=n_points,
        )

        self.n_points_ = n_points

        # Standardize for numerical stability
        self.past_scaler_ = StandardScaler()
        self.future_scaler_ = StandardScaler()
        self.X_past_scaled = self.past_scaler_.fit_transform(self.X_past)
        self.X_future_scaled = self.future_scaler_.fit_transform(self.X_future)

    def _compute_kernel_embeddings(self):
        """Compute kernel mean embeddings of P(future | past = x)."""
        X_past = self.X_past_scaled
        X_future = self.X_future_scaled
        n = self.n_points_

        # Determine bandwidths
        bw_past = self._resolve_bandwidth(self.past_bandwidth, X_past)
        bw_future = self._resolve_bandwidth(self.future_bandwidth, X_future)

        self.bw_past_ = bw_past
        self.bw_future_ = bw_future

        # Kernel matrices
        K_past = _rbf_kernel(X_past, bandwidth=bw_past)
        K_future = _rbf_kernel(X_future, bandwidth=bw_future)
        self.K_past_ = K_past
        self.K_future_ = K_future

        # Regularized conditional embedding weights (Brodu & Crutchfield Sec. III A 2):
        #   omega(x) = (G^X + epsilon * I)^{-1} K(x)
        # For all training samples the weight matrix is:
        #   Omega = (G^X + n*epsilon * I)^{-1} G^X
        reg = self.regularization * n * np.eye(n)
        Omega = np.linalg.solve(K_past + reg, K_past)
        self.Omega_ = Omega

        # Gram matrix on conditional embeddings (Eq. 7 in the paper):
        #   G^S_{ij} = sum_a sum_b omega_a(x_i) omega_b(x_j) k^Y(y_a, y_b)
        #            = Omega^T @ G^Y @ Omega
        self.embedding_gram_ = Omega.T @ K_future @ Omega

    def _diffusion_map(self):
        """Apply diffusion maps to the kernel embedding space."""
        G = self.embedding_gram_
        n = G.shape[0]

        # Convert gram matrix to distance-like quantity
        diag = np.diag(G).copy()
        # D^2(i,j) = G(i,i) - 2*G(i,j) + G(j,j)
        D2 = diag[:, None] - 2 * G + diag[None, :]
        D2 = np.maximum(D2, 0)

        # Diffusion kernel
        if self.diffusion_bandwidth == 'median':
            dists = D2[np.triu_indices(n, k=1)]
            epsilon = np.median(np.sqrt(dists + 1e-12))
        else:
            epsilon = float(self.diffusion_bandwidth)

        epsilon = max(epsilon, 1e-8)
        self.diffusion_eps_ = epsilon
        self.rkhs_distance_sq_ = D2

        W = np.exp(-D2 / (2.0 * epsilon ** 2))
        self.diffusion_kernel_ = W

        # Normalized graph Laplacian (Coifman-Lafon)
        q = W.sum(axis=1)
        q_alpha = np.power(q + 1e-12, self.diffusion_alpha)
        W_alpha = W / (q_alpha[:, None] * q_alpha[None, :])
        d = W_alpha.sum(axis=1)
        d_inv_sqrt = 1.0 / np.sqrt(d + 1e-12)
        L = W_alpha * d_inv_sqrt[:, None] * d_inv_sqrt[None, :]
        self.diffusion_operator_ = L

        # Eigendecomposition
        n_comp = min(self.n_components + 1, n - 1)
        eigenvalues, eigenvectors = eigh(L, subset_by_index=[n - n_comp, n - 1])

        # Sort by descending eigenvalue, skip the trivial one
        idx = np.argsort(-eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Scale by eigenvalues (diffusion coordinates)
        self.diffusion_eigenvalues_ = eigenvalues[1:self.n_components + 1]
        self.diffusion_eigenvectors_ = eigenvectors[:, 1:self.n_components + 1]
        weights = np.power(np.clip(self.diffusion_eigenvalues_, 0, None), self.diffusion_time)
        self.diffusion_weights_ = weights
        self.diffusion_coords_ = (
            self.diffusion_eigenvectors_
            * weights[None, :]
        )

    def _cluster_states(self):
        """Cluster diffusion coordinates into causal states."""
        coords = self.diffusion_coords_
        if self.cluster_dims is None:
            cluster_dims = coords.shape[1]
        else:
            cluster_dims = min(self.cluster_dims, coords.shape[1])

        cluster_data = coords[:, :cluster_dims]
        self.cluster_dims_ = cluster_dims
        self.cluster_data_ = cluster_data

        if self.cluster_standardize:
            scaler = StandardScaler()
            cluster_input = scaler.fit_transform(cluster_data)
            self.cluster_scaler_ = scaler
        else:
            cluster_input = cluster_data
            self.cluster_scaler_ = None

        self.cluster_input_ = cluster_input
        method = self.clustering_method.lower()

        if method == 'kmeans':
            kmeans = KMeans(
                n_clusters=self.n_states,
                n_init=self.kmeans_n_init,
                random_state=self.random_state,
            )
            raw_labels = kmeans.fit_predict(cluster_input)
            self.raw_labels_ = raw_labels.copy()
            self.labels_ = raw_labels
            self.cluster_centers_ = kmeans.cluster_centers_
            self.dbscan_noise_count_ = 0
        else:
            db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
            raw_labels = db.fit_predict(cluster_input)
            labels = raw_labels.copy()
            self.raw_labels_ = raw_labels.copy()
            self.dbscan_noise_count_ = int(np.sum(raw_labels == -1))

            if self.dbscan_noise_count_:
                if self.dbscan_noise_strategy == 'nearest_centroid':
                    cluster_ids = np.array([label for label in np.unique(labels) if label != -1])
                    if len(cluster_ids) == 0:
                        labels = np.zeros_like(labels)
                    else:
                        centroids = np.vstack([
                            cluster_input[labels == label].mean(axis=0)
                            for label in cluster_ids
                        ])
                        noise_mask = labels == -1
                        labels[noise_mask] = cluster_ids[
                            cdist(cluster_input[noise_mask], centroids).argmin(axis=1)
                        ]
                else:
                    labels[labels == -1] = labels.max() + 1

            _, labels = np.unique(labels, return_inverse=True)
            self.labels_ = labels
            self.cluster_centers_ = np.vstack([
                cluster_input[self.labels_ == label].mean(axis=0)
                for label in range(len(np.unique(self.labels_)))
            ])

        self.n_states_found_ = len(np.unique(self.labels_))
        self.cluster_sizes_ = np.bincount(self.labels_, minlength=self.n_states_found_)

    def _compute_transitions(self):
        """Estimate transition probabilities between causal states."""
        n_states = self.n_states_found_
        self.transition_matrix_ = np.zeros((n_states, n_states))

        labels = self.labels_
        for t in range(len(labels) - 1):
            i = labels[t]
            j = labels[t + 1]
            self.transition_matrix_[i, j] += 1

        # Normalize rows
        row_sums = self.transition_matrix_.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        self.transition_matrix_ /= row_sums

    def _compute_metrics(self):
        """Compute information-theoretic measures."""
        # Stationary distribution
        T = self.transition_matrix_
        n = T.shape[0]

        try:
            eigenvalues, eigenvectors = np.linalg.eig(T.T)
            idx = np.argmin(np.abs(eigenvalues - 1.0))
            pi = np.real(eigenvectors[:, idx])
            pi = np.abs(pi)
            pi_sum = pi.sum()
            pi = pi / pi_sum if pi_sum > 0 else np.ones(n) / n
        except np.linalg.LinAlgError:
            pi = np.ones(n) / n

        self.stationary_dist_ = pi

        # Statistical complexity: Cmu = -sum pi_i log2 pi_i
        pi_pos = pi[pi > 0]
        self.statistical_complexity_ = -np.sum(pi_pos * np.log2(pi_pos))

        # Entropy rate: h = -sum_i pi_i sum_j T_ij log2 T_ij
        h = 0.0
        for i in range(n):
            for j in range(n):
                if T[i, j] > 0:
                    h -= pi[i] * T[i, j] * np.log2(T[i, j])
        self.entropy_rate_ = h

    def _align_true_states(self, true_states):
        """Align a hidden-state sequence with the fitted embedding windows."""
        true_states = np.asarray(true_states)
        start = self.L_past - 1
        stop = start + self.n_points_
        if len(true_states) < stop:
            raise ValueError("true_states is too short to align with fitted windows")
        return true_states[start:stop]

    def compare_to_reference(self, true_states=None, true_transition_matrix=None,
                             true_stationary_dist=None):
        """
        Compare the fitted machine to a known toy-model reference.

        Parameters
        ----------
        true_states : array-like or None
            Hidden-state sequence for the original series.
        true_transition_matrix : array-like or None
            Reference transition matrix.
        true_stationary_dist : array-like or None
            Reference stationary distribution.

        Returns
        -------
        summary : dict
            Comparison summary with cluster counts and optional error metrics.
        """
        summary = {
            'n_states': self.n_states_found_,
            'cluster_sizes': self.cluster_sizes_.copy(),
        }

        if true_states is not None:
            aligned_true_states = self._align_true_states(true_states)
            n_true_states = int(np.max(aligned_true_states)) + 1
            contingency = np.zeros((self.n_states_found_, n_true_states), dtype=int)
            for state_id in range(self.n_states_found_):
                for regime_id in range(n_true_states):
                    contingency[state_id, regime_id] = np.sum(
                        (self.labels_ == state_id) & (aligned_true_states == regime_id)
                    )

            summary['aligned_true_states'] = aligned_true_states
            summary['contingency'] = contingency
            summary['accuracy'] = contingency.max(axis=1).sum() / len(self.labels_)

        if true_transition_matrix is not None:
            true_transition_matrix = np.asarray(true_transition_matrix, dtype=float)
            if true_transition_matrix.shape == self.transition_matrix_.shape:
                summary['transition_matrix_error'] = np.linalg.norm(
                    self.transition_matrix_ - true_transition_matrix
                )
            else:
                summary['transition_matrix_error'] = np.nan

        if true_stationary_dist is not None:
            true_stationary_dist = np.asarray(true_stationary_dist, dtype=float)
            if true_stationary_dist.shape == self.stationary_dist_.shape:
                summary['stationary_distribution_error'] = np.linalg.norm(
                    self.stationary_dist_ - true_stationary_dist
                )
            else:
                summary['stationary_distribution_error'] = np.nan

        return summary

    def predict_state(self, past_window):
        """
        Predict the causal state for a given past window.

        Parameters
        ----------
        past_window : array (L_past,)
            Recent past values.

        Returns
        -------
        state : int
            Predicted causal state label.
        """
        past_scaled = self.past_scaler_.transform(past_window.reshape(1, -1))

        # Find nearest neighbor in past embeddings
        dists = np.sum((self.X_past_scaled - past_scaled) ** 2, axis=1)
        nearest = np.argmin(dists)
        return self.labels_[nearest]

    def mmd_distance_matrix(self):
        """
        Compute pairwise MMD between all discovered causal states.

        Returns
        -------
        mmd_matrix : array (n_states, n_states)
        """
        n_states = self.n_states_found_
        mmd_matrix = np.zeros((n_states, n_states))

        for i in range(n_states):
            mask_i = self.labels_ == i
            X_i = self.X_future_scaled[mask_i]
            for j in range(i + 1, n_states):
                mask_j = self.labels_ == j
                X_j = self.X_future_scaled[mask_j]

                # Subsample for efficiency
                max_samples = self.max_mmd_samples
                if max_samples is not None and len(X_i) > max_samples:
                    idx_i = self.rng_.choice(len(X_i), max_samples, replace=False)
                    X_i_sub = X_i[idx_i]
                else:
                    X_i_sub = X_i
                if max_samples is not None and len(X_j) > max_samples:
                    idx_j = self.rng_.choice(len(X_j), max_samples, replace=False)
                    X_j_sub = X_j[idx_j]
                else:
                    X_j_sub = X_j

                mmd2 = compute_mmd(X_i_sub, X_j_sub, self.bw_future_)
                mmd_matrix[i, j] = np.sqrt(mmd2)
                mmd_matrix[j, i] = mmd_matrix[i, j]

        return mmd_matrix

def rolling_rkhs(series, window_size, step_size, L_past=5, L_future=5,
                 n_states=None, bandwidth='median', n_components=5,
                 **model_kwargs):
    """
    Apply RKHS epsilon machine on rolling windows.

    Parameters
    ----------
    series : array
        Original continuous time series.
    window_size : int
        Window size.
    step_size : int
        Step between windows.
    L_past, L_future : int
        Embedding dimensions.
    n_states : int or None
        Number of causal states per window. If None, uses the model's automatic
        clustering path.
    bandwidth : float or 'median'
        Kernel bandwidth.
    n_components : int
        Diffusion map components.
    **model_kwargs
        Additional keyword arguments passed to RKHSEpsilonMachine.

    Returns
    -------
    results : list of dict
    """
    results = []
    N = len(series)

    for start in range(0, N - window_size + 1, step_size):
        end = start + window_size
        window = series[start:end]

        try:
            model = RKHSEpsilonMachine(
                L_past=L_past,
                L_future=L_future,
                bandwidth=bandwidth,
                n_components=n_components,
                n_states=n_states,
                **model_kwargs,
            )
            model.fit(window)

            results.append({
                'window_start': start,
                'window_end': end,
                'num_states': model.n_states_found_,
                'statistical_complexity': model.statistical_complexity_,
                'entropy_rate': model.entropy_rate_,
                'model': model,
            })
        except Exception as e:
            results.append({
                'window_start': start,
                'window_end': end,
                'num_states': np.nan,
                'statistical_complexity': np.nan,
                'entropy_rate': np.nan,
                'model': None,
                'error': str(e),
            })

    return results
