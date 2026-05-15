"""
Bayesian state-space model with Sequential Monte Carlo (particle filter).

Models the latent company health state z_t evolving over time, from which
observed financial and clinical signals y_t are generated.

State vector z_t ∈ R^d (d=4 components):
  z[0]: log(cash_months_runway) — financial health
  z[1]: burn_trend               — acceleration signal (-1 to +1)
  z[2]: clinical_progress        — enrollment completion signal (0 to 1)
  z[3]: market_condition         — latent financing market state (-1 to +1)

Transition model: z_{t+1} = A z_t + noise_transition
  - A is a diagonal transition matrix (autoregressive per component)
  - Noise ~ N(0, Q) captures state evolution uncertainty

Observation model: y_t = H z_t + noise_obs
  - y_t ∈ R^m: observed financial / clinical metrics
  - H maps latent state to observations
  - Noise ~ N(0, R)

Particle filter (Bootstrap filter):
  1. Propagate particles: x_i_{t+1} ~ p(z|z_i_t)
  2. Weight: w_i ∝ p(y_t | z_i_t)
  3. Resample: systematic resampling to avoid weight collapse
  4. Estimate: posterior mean / std of state components

This provides:
  - Filtered state estimate (best guess of current hidden health)
  - State uncertainty (credible interval)
  - Predictive distribution of future states
  - Anomaly detection (particle weight collapse indicates surprise)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StateSpaceParams:
    """Parameters for the latent state-space model."""
    # State dimension
    d: int = 4

    # Transition matrix diagonals (AR coefficients per state component)
    # Values < 1 → mean-reverting; values close to 1 → persistent
    transition_diag: tuple[float, ...] = (0.85, 0.70, 0.90, 0.75)

    # Transition noise std per component
    transition_noise: tuple[float, ...] = (0.10, 0.15, 0.05, 0.20)

    # Observation noise std (scalar, applied to all observations)
    observation_noise: float = 0.20

    # Particle count for filter
    n_particles: int = 1000


@dataclass
class ParticleFilterState:
    particles: np.ndarray          # shape (n_particles, d)
    weights: np.ndarray            # shape (n_particles,), sums to 1
    log_marginal_likelihood: float # running log-likelihood (anomaly detection)


@dataclass
class StateEstimate:
    posterior_mean: np.ndarray     # shape (d,)
    posterior_std: np.ndarray      # shape (d,)
    effective_sample_size: float
    log_marginal_likelihood: float
    state_labels: list[str]


@dataclass
class StateSpaceResult:
    current_state_estimate: StateEstimate
    predicted_state_estimate: StateEstimate
    cash_health_score: float        # z[0] normalised to [0, 1]
    burn_acceleration_signal: float # z[1] normalised
    clinical_progress_signal: float # z[2]
    market_condition_signal: float  # z[3] normalised to [0, 1]
    anomaly_score: float            # 1 - ESS/n_particles; high → surprising observations
    interpretation: str
    methodology_note: str = (
        "Bootstrap particle filter with 1000 particles. "
        "Latent state z_t: [log_runway, burn_trend, clinical_progress, market_condition]. "
        "Observation likelihood via Gaussian with sigma=0.20. "
        "Anomaly score = 1 - ESS/N; high values indicate inputs inconsistent with prior."
    )


_STATE_LABELS = ["log_runway", "burn_trend", "clinical_progress", "market_condition"]


def _systematic_resample(weights: np.ndarray, n: int, rng: np.random.Generator) -> np.ndarray:
    """Systematic resampling — O(n) with lower variance than multinomial."""
    cumsum = np.cumsum(weights)
    positions = (rng.uniform(0.0, 1.0) + np.arange(n)) / n
    indices = np.searchsorted(cumsum, positions)
    return np.clip(indices, 0, len(weights) - 1)


def initialise_particles(
    observation: np.ndarray,
    params: StateSpaceParams,
    rng: np.random.Generator,
) -> ParticleFilterState:
    """
    Initialise particle cloud from prior, then update weights with first observation.

    Prior: z ~ N(observation_mean_map(y), I * prior_std)
    """
    n = params.n_particles
    d = params.d

    # Prior mean: set from observation signal (approximate identifiability)
    prior_mean = np.zeros(d)
    if len(observation) >= 1:
        prior_mean[0] = float(observation[0])  # log_runway
    if len(observation) >= 2:
        prior_mean[1] = float(observation[1])  # burn_trend
    if len(observation) >= 3:
        prior_mean[2] = float(observation[2])  # clinical_progress
    if len(observation) >= 4:
        prior_mean[3] = float(observation[3])  # market_condition

    particles = rng.normal(prior_mean, 0.5, size=(n, d))
    weights = np.ones(n) / n

    return ParticleFilterState(
        particles=particles,
        weights=weights,
        log_marginal_likelihood=0.0,
    )


def propagate(
    state: ParticleFilterState,
    params: StateSpaceParams,
    rng: np.random.Generator,
) -> ParticleFilterState:
    """Propagate particles one step forward via transition model."""
    n = params.n_particles
    A = np.array(params.transition_diag)
    Q_std = np.array(params.transition_noise)

    noise = rng.normal(0.0, 1.0, size=(n, params.d)) * Q_std
    new_particles = state.particles * A + noise

    return ParticleFilterState(
        particles=new_particles,
        weights=state.weights.copy(),
        log_marginal_likelihood=state.log_marginal_likelihood,
    )


def update(
    state: ParticleFilterState,
    observation: np.ndarray,
    params: StateSpaceParams,
    rng: np.random.Generator,
) -> ParticleFilterState:
    """
    Weight update step: w_i ∝ p(y | z_i).

    Observation likelihood: y ~ N(H z, sigma * I)
    where H = I[:m, :d] (first m components of state map to observation).
    """
    obs_dim = len(observation)
    sigma = params.observation_noise

    # Predicted observations: first obs_dim components of state
    predicted = state.particles[:, :obs_dim]
    residuals = observation - predicted  # shape (n, obs_dim)
    log_liks = -0.5 * np.sum((residuals / sigma) ** 2, axis=1)

    # Normalise
    log_w = np.log(state.weights + 1e-300) + log_liks
    log_w -= np.max(log_w)
    new_weights = np.exp(log_w)
    log_marginal = float(np.log(np.mean(new_weights) + 1e-300))
    new_weights /= new_weights.sum()

    # Effective sample size
    ess = 1.0 / float(np.sum(new_weights ** 2))

    # Resample if ESS < N/2
    if ess < params.n_particles / 2:
        indices = _systematic_resample(new_weights, params.n_particles, rng)
        new_particles = state.particles[indices]
        new_weights = np.ones(params.n_particles) / params.n_particles
    else:
        new_particles = state.particles

    return ParticleFilterState(
        particles=new_particles,
        weights=new_weights,
        log_marginal_likelihood=state.log_marginal_likelihood + log_marginal,
    )


def extract_estimate(
    state: ParticleFilterState,
    params: StateSpaceParams,
) -> StateEstimate:
    """Compute posterior mean and std from weighted particles."""
    mean = np.average(state.particles, weights=state.weights, axis=0)
    var = np.average((state.particles - mean) ** 2, weights=state.weights, axis=0)
    std = np.sqrt(np.maximum(var, 0.0))
    ess = 1.0 / float(np.sum(state.weights ** 2))

    return StateEstimate(
        posterior_mean=mean,
        posterior_std=std,
        effective_sample_size=round(ess, 1),
        log_marginal_likelihood=state.log_marginal_likelihood,
        state_labels=_STATE_LABELS[:params.d],
    )


def run_particle_filter(
    observations: np.ndarray,
    params: StateSpaceParams,
    rng: np.random.Generator,
) -> tuple[list[StateEstimate], ParticleFilterState]:
    """
    Run particle filter over a sequence of observations.

    observations: shape (T, m) — T time steps, m observed components
    Returns: (list of T StateEstimates, final ParticleFilterState)
    """
    T, m = observations.shape
    state = initialise_particles(observations[0], params, rng)

    estimates: list[StateEstimate] = []
    for t in range(T):
        state = update(state, observations[t], params, rng)
        estimates.append(extract_estimate(state, params))
        if t < T - 1:
            state = propagate(state, params, rng)

    return estimates, state


def run_state_space_analysis(
    cash_months_runway: float,
    burn_acceleration: float,
    enrollment_fraction: float,
    biotech_market_score: float,
    rng: np.random.Generator,
    params: StateSpaceParams | None = None,
) -> StateSpaceResult:
    """
    Single-step state-space analysis from current company observables.

    Constructs a synthetic observation vector, runs one filtering step,
    and extracts the latent state estimate and predictive distribution.
    """
    if params is None:
        params = StateSpaceParams()

    # Normalise observables to state-space units
    log_runway = float(np.log(max(cash_months_runway, 0.5)))
    burn_signal = float(np.clip(burn_acceleration - 1.0, -1.0, 1.0))
    clin_signal = float(np.clip(enrollment_fraction, 0.0, 1.0))
    market_signal = float(np.clip((biotech_market_score - 5.0) / 5.0, -1.0, 1.0))

    observation = np.array([log_runway, burn_signal, clin_signal, market_signal])

    state = initialise_particles(observation, params, rng)
    state = update(state, observation, params, rng)
    current_est = extract_estimate(state, params)

    # One-step prediction
    state_pred = propagate(state, params, rng)
    predicted_est = extract_estimate(state_pred, params)

    # Anomaly score (high = observations surprising given prior)
    ess = current_est.effective_sample_size
    anomaly = 1.0 - ess / params.n_particles

    # Normalised state scores for output
    log_r = float(current_est.posterior_mean[0])
    cash_score = float(np.clip(1.0 - np.exp(-max(log_r, 0.0) / 3.0), 0.0, 1.0))

    burn_acc = float(current_est.posterior_mean[1])
    burn_norm = float(np.clip((burn_acc + 1.0) / 2.0, 0.0, 1.0))

    clin = float(np.clip(current_est.posterior_mean[2], 0.0, 1.0))
    mkt = float(current_est.posterior_mean[3])
    mkt_norm = float(np.clip((mkt + 1.0) / 2.0, 0.0, 1.0))

    health = (cash_score + (1.0 - burn_norm) + clin + mkt_norm) / 4.0

    if health > 0.65:
        interp = "Latent company state: healthy — cash, clinical, and market signals aligned."
    elif health > 0.40:
        interp = "Latent company state: mixed — at least one risk dimension under stress."
    else:
        interp = "Latent company state: stressed — multiple risk dimensions indicate elevated distress."

    if anomaly > 0.50:
        interp += " High anomaly score: observed inputs are surprising given prior expectations."

    return StateSpaceResult(
        current_state_estimate=current_est,
        predicted_state_estimate=predicted_est,
        cash_health_score=round(cash_score, 3),
        burn_acceleration_signal=round(burn_norm, 3),
        clinical_progress_signal=round(clin, 3),
        market_condition_signal=round(mkt_norm, 3),
        anomaly_score=round(anomaly, 3),
        interpretation=interp,
    )
