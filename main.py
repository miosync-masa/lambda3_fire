!pip install jax jaxlib plotly tensorboardX wandb --quiet

import jax
import wandb
import numpy as np
from jax import lax
import jax.numpy as jnp
from jax import jit, grad
from jax import random
from functools import partial
import plotly.graph_objects as go
from typing import Union, List, Any, Tuple, Optional
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

@dataclass
class Lambda3FireConfig:
    # Core Parameters
    embedding_dim: int = 16
    structure_radius_base: float = 3.0
    rho_t0: float = 1.0
    cutoff_rho_exponent: float = 0.5
    cutoff_sigma_exponent: float = 0.3
    entropic_spread_coeff: float = 1.0
    target_acceptance: float = 0.5

    entropic_spread_coeff: float = 1.0
    entropy_weight: float = 0.1
    energy_density_base: float = 1.0
    projection_angle: float = 0.0
    structure_length_ref: float = 1.0
    pairwise_a: float = 1.0
    pairwise_b: float = 0.5
    pairwise_c_decay: float = 0.1

    # Random seed
    key_global: jax.random.KeyArray = field(default_factory=lambda: jax.random.PRNGKey(42))

    # Coherence penalty factors
    alpha_distance: float = 0.1  # exp(-α d^2)
    gamma_charge: float = 4.0
    gamma_color: float = 4.0

    # Structural coherence weights
    w_spin: float = 1.0
    w_color: float = 1.0
    w_charge: float = 1.0
    w_dist: float = 1.0

    # Transaction direction vectors
    lambda_f_bind: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0]))
    lambda_f_move: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 1.0, 0.0]))
    lambda_f_split: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 1.0]))

    # EMA Smoothing Parameters
    ema_energy_window: int = 5
    ema_energy_current_weight: float = 0.8
    ema_energy_history_weight: float = 0.2

    ema_score_window: int = 5
    ema_score_current_weight: float = 0.7
    ema_score_history_weight: float = 0.3

    # Cooldown Control
    warmup_step: int = 30
    warmup_buffer: int = 10
    cooldown_ewma_alpha: float = 0.1
    cooldown_target_on: float = 3.0
    cooldown_target_off: float = 0.0
    cooling_intensity_scaling: float = 200.0
    
    # Spin Flip Control
    spin_flip_interval: int = 100
    spin_flip_base_prob: float = 0.15
    spin_flip_split_decay: float = 3.0

    # Setup Parameters
    n_steps: int = 300
    project_name: str = "lambda3-fire-chloroferrocene"
    experiment_types: Tuple[str, ...] = ("photo_irradiation", "heating", "pressure")
    intensities: List[float] = field(default_factory=lambda: [1.0e6, 1000.0, 5.0e7])

# Utility function for safe division
def safe_divide(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

   """
    Compute dynamic interaction cutoff radius based on structural tension density (ρT)
    and synchronization rate (σs), following the Λ³ transaction principle.

    The cutoff adapts according to the local structural configuration:
    - ρT (tension density) reflects the level of transactional crowding.
    - σs (synchronization rate) quantifies local structural coherence.
    - Scaling exponents (α for ρT, β for σs) modulate the influence of each factor.

    Args:
        r (jnp.ndarray): Atomic or particle positions, shape (N, 3).
        config (Lambda3FireConfig): Simulation configuration containing cutoff parameters.

    Returns:
        float: Adaptive cutoff radius for the current structural configuration.
    """
@jit
def compute_dynamic_cutoff(
    r: jnp.ndarray,
    config: Any  # Should be Lambda3FireConfig or equivalent
):
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1)) + jnp.eye(n_el) * 1e10  # Avoid self-interaction

    # Compute tension density ρT
    rho_T = jnp.sum(1.0 / dists, where=dists < 10.0) / n_el

    # Compute synchronization rate σs
    mask = dists < config.structure_radius_base
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * mask
    sigma_s = jnp.sum(sigma_s_ij) / jnp.sum(mask)

    # Apply scaling for dynamic cutoff
    cutoff = (
        config.structure_radius_base *
        (rho_T / config.rho_t0) ** config.cutoff_rho_exponent *
        sigma_s ** config.cutoff_sigma_exponent
    )
    return cutoff

    """
    Compose external transaction parameters for the Λ³ tensor model
    based on experimental operations (e.g., heating, electric field, pressure).

    Each experimental type contributes to:
    - ρT (semantic tension density)
    - σs (synchronization rate)
    - ΛF (transaction direction vector)

    Correlation terms boost ρT nonlinearly when certain experiment types coexist,
    capturing cooperative or interfering effects between operations.

    Args:
        experiment_types (Tuple[str, ...]): List of experimental conditions 
            ('heating', 'electric_field', 'pressure', 'photo_irradiation', 'cooling').
        intensities (List[float]): Corresponding intensity values for each operation.

    Returns:
        Tuple[jnp.ndarray, float, float]: 
            - ΛF_ext (normalized external transaction direction vector),
            - ρT_ext (boosted tension density),
            - σs_ext (combined synchronization rate).
    """
    
@partial(jit, static_argnames=["experiment_types"])
def experiment_to_transaction_params(
    experiment_types: Tuple[str, ...],
    intensities: List[float]
) -> Tuple[jnp.ndarray, float, float]:
    LambdaF_total = jnp.zeros(3)
    rhoT_total = 0.0
    sigmaS_list = []

    for exp_type, intensity in zip(experiment_types, intensities):
        if exp_type == "heating":
            T = intensity
            rhoT = 0.02 * T
            sigmaS = jnp.exp(-0.01 * T)
            weights = jnp.array([
                jnp.exp(-0.01 * T),
                0.01 * T,
                0.02 * T
            ])
        elif exp_type == "electric_field":
            E = intensity
            rhoT = 1e-6 * E
            sigmaS = jnp.exp(-1e-6 * E)
            weights = jnp.array([
                1e-6 * E,
                3e-6 * E,
                jnp.exp(1e-6 * E)
            ])
        elif exp_type == "pressure":
            P = intensity
            rhoT = 1e-6 * P
            sigmaS = 1.0 / (1.0 + 1e-7 * jnp.abs(P))
            weights_pos = jnp.array([
                1.0 + 1e-7 * P,
                5e-8 * P,
                1e-7 / (1.0 + P)
            ])
            weights_neg = jnp.array([
                jnp.exp(1e-7 * P),
                5e-8 * jnp.abs(P),
                1e-7 * jnp.abs(P)
            ])
            weights = lax.cond(P > 0, lambda _: weights_pos, lambda _: weights_neg, operand=None)
        elif exp_type == "photo_irradiation":
            I = intensity
            rhoT = 1e-4 * I**0.75
            sigmaS = jnp.exp(-1e-4 * I)
            weights = jnp.array([
                jnp.exp(-1e-4 * I),
                2e-4 * I,
                1e-4 * I
            ])
        elif exp_type == "cooling":
            T = intensity
            rhoT = -0.01 * T
            sigmaS = jnp.clip(0.9 + 0.01 * T, 0.0, 1.0)
            weights = jnp.array([
                1.0 + 0.02 * T,
                0.1 * T,
                jnp.exp(-0.05 * T)
            ])
        else:
            raise ValueError(f"Unsupported experiment type: {exp_type}")

        LambdaF = weights / jnp.sum(weights)
        LambdaF_total += LambdaF
        rhoT_total += rhoT
        sigmaS_list.append(sigmaS)

    # Nonlinear interaction boosts (correlation effects between experimental types)
    correlation_rhoT_boost = 0.0
    if "heating" in experiment_types and "photo_irradiation" in experiment_types:
        T = intensities[experiment_types.index("heating")]
        I = intensities[experiment_types.index("photo_irradiation")]
        correlation_rhoT_boost += 0.001 * jnp.sqrt(T * I)

    if "heating" in experiment_types and "pressure" in experiment_types:
        T = intensities[experiment_types.index("heating")]
        P = intensities[experiment_types.index("pressure")]
        correlation_rhoT_boost += 1e-5 * jnp.log1p(P * T)

    if "photo_irradiation" in experiment_types and "pressure" in experiment_types:
        I = intensities[experiment_types.index("photo_irradiation")]
        P = intensities[experiment_types.index("pressure")]
        correlation_rhoT_boost += 5e-6 * jnp.sqrt(I * jnp.log1p(P))

    if "electric_field" in experiment_types and "photo_irradiation" in experiment_types:
        E = intensities[experiment_types.index("electric_field")]
        I = intensities[experiment_types.index("photo_irradiation")]
        correlation_rhoT_boost += 1e-6 * jnp.sqrt(E * I)

    # Final combined output
    LambdaF_ext = LambdaF_total / jnp.sum(LambdaF_total)
    sigmaS_ext = jnp.prod(jnp.array(sigmaS_list))
    rhoT_ext = rhoT_total + correlation_rhoT_boost

    return LambdaF_ext, rhoT_ext, sigmaS_ext

 """
    Compute the semantic tension density (ρT) using a hybrid approach that combines:
    - Energy-based interaction terms (pairwise potential)
    - Synchronization tensor (spin, color, distance coherence)
    - Disorder-driven entropy contribution (σs misalignment)
    - Optional external transaction input (rhoT_ext)

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        spins (jnp.ndarray): Spin values of elements (N,).
        colors (jnp.ndarray): Color charges or identifiers (N, 3).
        cutoff_radius (float): Interaction cutoff for synchronization evaluation.
        entropy_weight (float): Weight of the entropy contribution.
        disorder_amplitude (float): Amplitude of the disorder-based entropy term.
        energy_density_base (float): Base scale for energy contribution.
        projection_angle (float): Optional projection angle factor.
        structure_length_ref (float): Reference length for structural scaling.
        a, b, c_decay (float): Pairwise interaction coefficients.
        rhoT_ext (float): External tension contribution.

    Returns:
        Tuple[float, float]:
            - rho_T_out: Computed semantic tension density.
            - S_gen: Generated entropy (degree of misalignment).
    """
@jit
def compute_rho_T_with_hybrid(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    colors: jnp.ndarray,
    cutoff_radius: float,
    config: Any,  # Lambda3FireConfig expected
    rhoT_ext: float = 0.0
) -> Tuple[float, float]:
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    dists = jnp.where(dists < 1e-2, 1e-2, dists)  # Avoid singularity
    dists = dists + jnp.eye(n_el) * 1e10          # Avoid self-interaction

    entropy_weight = config.entropy_weight
    disorder_amplitude = config.entropic_spread_coeff
    energy_density_base = config.energy_density_base
    projection_angle = config.projection_angle
    structure_length_ref = config.structure_length_ref
    a = config.pairwise_a
    b = config.pairwise_b
    c_decay = config.pairwise_c_decay

    rho_T_base = energy_density_base * (structure_length_ref / dists) ** 2 * jnp.cos(projection_angle)
    rho_T_base = safe_divide(jnp.sum(rho_T_base, where=dists < cutoff_radius), n_el)

    pairwise_term = (a / dists) + b * jnp.exp(-c_decay * dists)
    rho_T_phys = jnp.sum(pairwise_term, where=dists > cutoff_radius) / n_el

    spin_match = (spins[:, None] == spins[None, :]).astype(float)
    dcolor = colors[:, None, :] - colors[None, :, :]
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_match = jnp.exp(-GAMMA_COLOR * color_norm2)
    local_overlap = jnp.exp(-0.1 * dists ** 2)
    mask = (dists < cutoff_radius) & (dists > 0)

    sigma_s_ij = spin_match * color_match * local_overlap * mask

    S_gen = disorder_amplitude * safe_divide(
        jnp.sum(1.0 - sigma_s_ij, where=mask),
        jnp.sum(mask)
    )

    rho_T_out = (
        rho_T_base + 
        entropy_weight * S_gen + 
        0.1 * rho_T_phys + 
        rhoT_ext
    )

    return rho_T_out, S_gen 

    """
    Determine the transaction direction (Bind, Move, Split) based on structural coherence tensors
    and semantic tension density (ρT), following the Λ³ transactional principle.

    Scoring is derived from:
    - Local spin, color, and charge coherence tensors.
    - Geometric distance overlap.
    - External transaction influence (if provided).
    - Distortion feedback from dipole moment and Laplacian curvature (structure deformation).

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        spins (jnp.ndarray): Spin values of elements (N,).
        charges (jnp.ndarray): Charge values of elements (N,).
        colors (jnp.ndarray): Color charges (N, 3).
        k_vectors (jnp.ndarray): k-vectors for phase interaction (N, 3).
        rho_T (float): Semantic tension density.
        c (float): Cutoff distance for coherence evaluation.
        LambdaF_ext (Optional[jnp.ndarray]): External transaction direction vector (3,).
        rhoT_ext (float): External tension density contribution.
        sigmaS_ext (float): External synchronization rate contribution.

    Returns:
        Tuple[int, jnp.ndarray, float, float, float]:
            - direction (int): 0 = Bind, 1 = Move, 2 = Split.
            - phase (jnp.ndarray): Phase factor array (N,).
            - score_bind (float): Normalized score for Bind.
            - score_move (float): Normalized score for Move.
            - score_split (float): Normalized score for Split.
    """
def _compute_transaction_direction_full(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    rho_T: float,
    c: float,
    config: Any,  # Lambda3FireConfig expected
    LambdaF_ext: Optional[jnp.ndarray] = None,
    rhoT_ext: float = 0.0,
    sigmaS_ext: float = 0.0
) -> Tuple[int, jnp.ndarray, float, float, float]:
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    mask = (dists < c) & (dists > 0)

    spin_tensor = config.w_spin * (spins[:, None] == spins[None, :]).astype(float)
    dcolor = colors[:, None, :] - colors[None, :, :]
    color_tensor = config.w_color * jnp.exp(-config.gamma_color * jnp.sum(dcolor ** 2, axis=-1))
    dq = charges[:, None] - charges[None, :]
    charge_tensor = config.w_charge * jnp.exp(-config.gamma_charge * dq ** 2)
    distance_tensor = config.w_dist * jnp.exp(-config.alpha_distance * dists ** 2)
    base_sigma_s = spin_tensor * color_tensor * charge_tensor * distance_tensor * mask

    rho_T = jnp.clip(rho_T, 1e-5, 10.0)

    charge_align = charges[:, None] * charges[None, :] / (dists + 1e-10)
    sigma_s_bind = safe_divide(jnp.sum(base_sigma_s * (charge_align > 0.0), where=mask), jnp.sum(mask))
    sigma_s_move = safe_divide(jnp.sum(base_sigma_s * (1.0 - spin_tensor), where=mask), jnp.sum(mask))
    sigma_s_split = safe_divide(jnp.sum(base_sigma_s * (charge_align < 0.0), where=mask), jnp.sum(mask))

    score_bind = sigma_s_bind * rho_T
    score_move = sigma_s_move * rho_T
    score_split = sigma_s_split * rho_T

    if LambdaF_ext is not None:
        ext_scores = LambdaF_ext * rhoT_ext * sigmaS_ext
        score_bind += ext_scores[0]
        score_move += ext_scores[1]
        score_split += ext_scores[2]

    dipole_vector = compute_dipole_tensor(r, charges)
    lap_term = laplacian_term(r)
    dipole_strength = jnp.linalg.norm(dipole_vector)
    lap_strength_scaled = jnp.log1p(jnp.sum(lap_term ** 2))
    total_distortion = dipole_strength + lap_strength_scaled + 1e-8

    distortion_move_weight = jnp.sum(jnp.abs(lap_term)) / total_distortion * 0.8
    distortion_split_weight = dipole_strength / total_distortion * 0.8

    score_move += score_bind * distortion_move_weight
    score_split += score_bind * distortion_split_weight
    score_bind *= (1.0 - distortion_move_weight - distortion_split_weight)

    score_bind = jnp.clip(score_bind, 1e-5, 10.0)
    score_move = jnp.clip(score_move, 1e-5, 10.0)
    score_split = jnp.clip(score_split, 1e-5, 10.0)

    total_score = score_bind + score_move + score_split + 1e-8
    score_bind /= total_score
    score_move /= total_score
    score_split /= total_score

    scores = jnp.array([score_bind, score_move, score_split])
    direction = jnp.argmax(scores)

    alpha_R = config.alpha_r
    E_field = config.e_field

    def generate_phase_valid(kv):
        k_cross_E = jnp.cross(kv, E_field)
        phi = alpha_R * jnp.sum(k_cross_E * spins[:, None], axis=-1)
        return jnp.exp(1j * phi)

    def generate_phase_dummy(_):
        return jnp.ones(r.shape[0], dtype=jnp.complex64)

    enable_phase = jnp.logical_not(jnp.allclose(k_vectors, 0.0))
    apply_kvec_shape = (k_vectors.shape[-1] == 3 and k_vectors.shape[0] == r.shape[0])
    apply_kvec = jnp.logical_and(apply_kvec_shape, enable_phase)

    phase = lax.cond(apply_kvec, generate_phase_valid, generate_phase_dummy, k_vectors)

    return direction, phase, score_bind, score_move, score_split

# ✅ JIT compilation (external transaction params treated as static)
compute_transaction_direction_full = jit(_compute_transaction_direction_full)

   """
    Compute the embedding vector h_i(r) for each element based on the Λ³ transaction structure,
    modulating interaction strengths by structural coherence, distance overlap, and entropy-driven attention.

    Embedding reflects:
    - Local transactional coherence (spin alignment × overlap)
    - Radial influence weighted by transaction direction (ΛF)
    - Entropy-coupled diffusion scaling (tau_dynamic)

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        spins (jnp.ndarray): Spin states of elements (N,).
        charges (jnp.ndarray): Charge values of elements (N,).
        k_vectors (jnp.ndarray): k-vectors (momentum-space interactions) (N, 3).
        c (float): Cutoff radius for overlap evaluation.
        direction (Optional[int]): Transaction direction index (0 = Bind, 1 = Move, 2 = Split).
        phase (jnp.ndarray): Phase factor from transaction evaluation.
        lambda_f_override (Optional[jnp.ndarray]): Override vector for ΛF (if provided).
        s_gen (float): Generated entropy (disorder level).
        tau_base (float): Base attention scaling factor.
        alpha_entropy (float): Scaling factor for entropy-linked diffusion broadening.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - h (jnp.ndarray): Embedding matrix (N, EMBEDDING_DIM).
            - lambda_f (jnp.ndarray): Transaction direction vector ΛF (3,).
    """

@jit
def compute_embedding(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    k_vectors: jnp.ndarray,
    c: float,
    config: Any,  # Lambda3FireConfig expected
    direction: Optional[int] = None,
    phase: jnp.ndarray = 1.0,
    lambda_f_override: Optional[jnp.ndarray] = None,
    s_gen: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_el = r.shape[0]
    h = jnp.zeros((n_el, config.embedding_dim), dtype=jnp.complex64)

    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists2 = jnp.sum((r_i - r_j) ** 2, axis=-1)
    dists = jnp.sqrt(dists2) + jnp.eye(n_el) * 1e10
    mask = (dists < c) & (dists > 0)

    spin_align = (spins[:, None] == spins[None, :]).astype(float)
    sigma_s_ij = jnp.exp(-0.1 * dists2) * spin_align * mask
    rhoT_ij = (1.0 / dists) * mask

    tau_dynamic = config.tau_base * (1.0 + config.alpha_entropy * s_gen)
    A_ij = jnp.exp(-dists2 / tau_dynamic) * mask

    if lambda_f_override is not None:
        lambda_f = lambda_f_override
    else:
        lambda_f = jnp.where(
            direction == 0, config.lambda_f_bind,
            jnp.where(direction == 1, config.lambda_f_move, config.lambda_f_split)
        )

    lambda_f_extended = jnp.tile(lambda_f, config.embedding_dim // 3 + 1)[:config.embedding_dim]
    lambda_f_complex = lambda_f_extended.astype(jnp.complex64) * jnp.mean(phase)

    for i in range(n_el):
        contrib_scalar = jnp.sum(sigma_s_ij[i] * rhoT_ij[i] * A_ij[i])
        h = h.at[i].set(contrib_scalar * lambda_f_complex)

    return h, lambda_f

 """
    Compute the structural energy of the system based on transactional coherence tensors,
    distance interactions, and directional alignment (ΛF vector), with thermal fluctuation.

    The energy includes:
    - Distance-dependent interaction (ρT)
    - Synchronization tensor (spin, charge, color matching, and overlap)
    - ΛF directionality coupling (dot product between displacement vectors and ΛF)
    - Periodic potential correction using k-vectors (optional)
    - Thermal noise injection for direction (Monte Carlo fluctuation)

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        spins (jnp.ndarray): Spin states of elements (N,).
        charges (jnp.ndarray): Charge values of elements (N,).
        colors (jnp.ndarray): Color charges of elements (N, 3).
        k_vectors (jnp.ndarray): k-vectors for periodic potential (N, 3).
        c (float): Cutoff radius for overlap evaluation.
        direction (int): Transaction direction (0 = Bind, 1 = Move, 2 = Split).
        temp_beta (float): Inverse temperature for fluctuation strength.
        key (jax.random.PRNGKey): Random key for thermal fluctuation.

    Returns:
        Tuple[float, jnp.ndarray]:
            - total_energy (float): Computed energy (normalized).
            - key (jax.random.PRNGKey): Updated random key after noise injection.
    """
@jit
def compute_energy_structural(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    c: float,
    direction: int,
    config: Any,  # Lambda3FireConfig expected
    key: jnp.ndarray = None
) -> Tuple[float, jnp.ndarray]:
    n_el = r.shape[0]

    dR = r[:, None, :] - r[None, :, :]
    dists = jnp.linalg.norm(dR, axis=-1)
    dists = jnp.maximum(dists, 1e-2)

    rho_T_ij = 1.0 / dists

    def add_periodic_potential(kv):
        dot_k_dr = jnp.tensordot(kv, dR, axes=[[1], [2]])
        rho_T_periodic = jnp.mean(jnp.cos(dot_k_dr), axis=0)
        return rho_T_periodic / (dists ** 2 + 1.0)

    def zero_potential(_):
        return jnp.zeros_like(rho_T_ij)

    apply_kvec_shape = (k_vectors.shape[-1] == 3) & (k_vectors.shape[0] == r.shape[0])
    enable_phase = jnp.logical_not(jnp.all(jnp.isclose(k_vectors, 0.0)))
    apply_kvec = apply_kvec_shape & enable_phase

    rho_T_ij += lax.cond(apply_kvec, add_periodic_potential, zero_potential, k_vectors)

    spin_match = (spins[:, None] == spins[None, :]).astype(float)
    dq = charges[:, None] - charges[None, :]
    charge_match = jnp.exp(-config.gamma_charge * dq ** 2)
    dcolor = colors[:, None, :] - colors[None, :, :]
    color_match = jnp.exp(-config.gamma_color * jnp.sum(dcolor ** 2, axis=-1))
    local_overlap = jnp.exp(-config.alpha_distance * dists ** 2)
    sigma_s_ij = spin_match * charge_match * color_match * local_overlap

    λ_nominal = jnp.select(
        [direction == 0, direction == 1, direction == 2],
        [config.lambda_f_bind, config.lambda_f_move, config.lambda_f_split],
        default=jnp.array([1.0, 0.0, 0.0])
    )

    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, shape=(3,)) / jnp.sqrt(config.temp_beta + 1e-9)
    lambda_f = λ_nominal + noise
    lambda_f /= (jnp.linalg.norm(lambda_f) + 1e-9)

    dot_products = jnp.einsum('ijk,k->ij', dR, lambda_f) / dists

    energy_matrix = rho_T_ij * sigma_s_ij * dot_products
    total_energy = jnp.sum(energy_matrix) / (n_el ** 2)

    return jnp.nan_to_num(total_energy, nan=1e-6), key
  
"""
    Estimate the local semantic tension density (ρT) and entropy disorder (S_gen)
    for a given element, considering only its surrounding neighborhood.

    Local evaluation uses:
    - Distance-based interactions within the cutoff radius.
    - Synchronization coherence between spin, color, and local overlap.
    - Entropy term derived from misalignment in the local coherence tensor.

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        spins (jnp.ndarray): Spin states of elements (N,).
        colors (jnp.ndarray): Color charges of elements (N, 3).
        idx (int): Index of the target element for localized evaluation.
        cutoff_radius (float): Interaction cutoff radius.

    Returns:
        Tuple[float, float]:
            - rho_T_est (float): Estimated local semantic tension density.
            - S_gen (float): Generated entropy (local misalignment disorder).
    """
@jit
def compute_rho_T_localized(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    colors: jnp.ndarray,
    idx: int,
    cutoff_radius: float,
    config: Any  # Lambda3FireConfig expected
) -> Tuple[float, float]:
    ri = r[idx]
    dists = jnp.sqrt(jnp.sum((r - ri) ** 2, axis=1))
    mask = (dists < cutoff_radius) & (dists > 0)

    spin_match = (spins[idx] == spins).astype(float)
    dcolor = colors[idx] - colors
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_match = jnp.exp(-config.gamma_color * color_norm2)
    local_overlap = jnp.exp(-0.1 * dists ** 2)

    sigma_s = spin_match * color_match * local_overlap * mask

    rho_T_base = (1.0 / (dists ** 2 + 1e-5)) * mask
    rho_T_est = jnp.sum(rho_T_base)

    S_gen = jnp.sum(1.0 - sigma_s) / (jnp.sum(mask) + 1e-8)

    return rho_T_est, S_gen

"""
    Perform adaptive Metropolis sampling with structural transaction evaluation.

    Features:
    - Energy evaluation via transactional structural model.
    - Local ρT estimation using the region of maximum displacement (low-rank inspired).
    - Adaptive proposal scaling (σ adjustment) based on acceptance rate and transaction mode (Split-sensitive).
    - Transaction direction re-evaluation at each step based on localized structural feedback.

    Args:
        r0 (jnp.ndarray): Initial positions of elements (N, 3).
        spins (jnp.ndarray): Spin states of elements (N,).
        charges (jnp.ndarray): Charge values of elements (N,).
        colors (jnp.ndarray): Color charges of elements (N, 3).
        k_vectors (jnp.ndarray): k-vectors (N, 3).
        c (float): Cutoff radius for coherence evaluation.
        direction (int): Transaction direction (0 = Bind, 1 = Move, 2 = Split).
        n_steps (int): Number of sampling steps.
        sigma_init (float): Initial proposal standard deviation.
        temp_beta (float): Inverse temperature (for thermal fluctuation strength).
        target_acceptance (float): Target acceptance rate for proposal tuning.
        rng_key (jax.random.PRNGKey): Random seed key.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray]:
            - r_samples (jnp.ndarray): Sampled positions (n_steps + 1, N, 3).
            - sigmas (jnp.ndarray): Adaptive σ values per step.
    """
@jit
def adaptive_metropolis_sampling_structural(
    r0: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    c: float,
    direction: int,
    config: Any,  # Lambda3FireConfig expected
    rng_key: jnp.ndarray = jax.random.PRNGKey(0)
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_steps = config.n_steps

    @jit
    def safe_divide(a, b, eps=1e-12):
        return a / (b + eps)

    @jit
    def single_step(state, step):
        r, sigma, E_curr, key, acc_cnt, r_samples, sigmas, score_split = state
        key, k_prop, k_energy = jax.random.split(key, 3)
        noise = jax.random.normal(k_prop, shape=r.shape)
        r_proposed = jnp.clip(r + sigma * noise, -5.0, 5.0)

        E_prop, key = compute_energy_structural(
            r_proposed, spins, charges, colors,
            k_vectors, c, direction,
            config=config, key=k_energy
        )

        delta_r = jnp.linalg.norm(r_proposed - r, axis=1)
        idx_changed = jnp.argmax(delta_r)
        rho_T, _ = compute_rho_T_localized(
            r_proposed, spins, colors, idx_changed, cutoff_radius=c, config=config
        )

        _, _, _, score_split_new, _ = compute_transaction_direction_full(
            r_proposed, spins, charges, colors, k_vectors, rho_T, c, config=config
        )

        acc_ratio = safe_divide(E_curr, E_prop)
        accept = jax.random.uniform(k_prop) < acc_ratio
        r_new = lax.select(accept, r_proposed, r)
        E_new = lax.select(accept, E_prop, E_curr)
        acc_cnt = acc_cnt + accept

        r_samples = r_samples.at[step + 1].set(r_new)
        sigmas = sigmas.at[step + 1].set(sigma)

        target_acc = lax.cond(
            score_split_new < 0.5,
            lambda _: config.target_acceptance,
            lambda _: config.target_acceptance_split,
            operand=None
        )

        def _update_sigma(args):
            acc_rate, sigma = args
            factor = lax.cond(acc_rate > target_acc, lambda _: 1.1, lambda _: 0.9, operand=None)
            scale = 1.0 / (1.0 + step / n_steps)
            return jnp.clip(sigma * (1 + (factor - 1) * scale), 0.05, 2.0)

        sigma = lax.cond(
            (step + 1) % 100 == 0,
            _update_sigma,
            lambda _: sigma,
            (acc_cnt / 100.0, sigma)
        )

        return (r_new, sigma, E_new, key, acc_cnt, r_samples, sigmas, score_split_new), None

    r_samples = jnp.zeros((n_steps + 1, *r0.shape))
    r_samples = r_samples.at[0].set(r0)

    sigmas = jnp.zeros(n_steps + 1)
    sigmas = sigmas.at[0].set(config.sigma_init)

    rng_key, k0_energy = jax.random.split(rng_key)
    E0, rng_key = compute_energy_structural(
        r0, spins, charges, colors,
        k_vectors, c, direction,
        config=config, key=k0_energy
    )

    rho_T, _ = compute_rho_T_with_hybrid(r0, spins, colors, c, config=config)
    _, _, _, score_split_init, _ = compute_transaction_direction_full(
        r0, spins, charges, colors, k_vectors, rho_T, c, config=config
    )

    init_state = (r0, config.sigma_init, E0, rng_key, 0, r_samples, sigmas, score_split_init)
    final_state, _ = lax.scan(single_step, init_state, jnp.arange(n_steps))
    _, _, _, _, _, r_samples, sigmas, _ = final_state

    return r_samples, sigmas

 """
    Visualize the transaction direction vector ΛF in 3D space,
    along with optional dipole vector and Cp ring normal axis.

    This provides insight into the orientation of transactional dynamics
    (Bind, Move, Split components) and their alignment with structural features.

    Args:
        lambda_f_history (list): History of ΛF vectors across simulation steps.
        step (int): Target step for visualization.
        r (jnp.ndarray): Current atomic positions (unused here, but reserved).
        dipole_vector (np.ndarray, optional): Dipole moment vector to visualize.
        filename (str): Output HTML filename for the plot.
    """
def visualize_lambda_f_with_crystal_axis(
    lambda_f_history: list,
    step: int,
    r: jnp.ndarray,
    config: Any,  # Lambda3FireConfig expected
    dipole_vector: np.ndarray = None,
    filename: str = "lambda_f_visualization.html"
):
    lambda_f = lambda_f_history[step]
    fig = go.Figure()

    fig.add_trace(go.Scatter3d(
        x=[0, lambda_f[0]], y=[0, lambda_f[1]], z=[0, lambda_f[2]],
        mode='lines+markers', line=dict(color='blue', width=6),
        marker=dict(size=6), name=f'ΛF vector (Step {step})'
    ))

    cp_normal = jnp.array([0.0, 0.0, 1.0])
    fig.add_trace(go.Scatter3d(
        x=[0, cp_normal[0]], y=[0, cp_normal[1]], z=[0, cp_normal[2]],
        mode='lines', line=dict(color='red', width=4, dash='dash'),
        name='Cp ring normal'
    ))

    if dipole_vector is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, float(dipole_vector[0])],
            y=[0, float(dipole_vector[1])],
            z=[0, float(dipole_vector[2])],
            mode='lines+markers', line=dict(color='orange', width=4),
            marker=dict(size=4), name='Dipole Vector'
        ))

    fig.add_trace(go.Scatter3d(
        x=[lambda_f[0]], y=[lambda_f[1]], z=[lambda_f[2]],
        mode='text', text=[f"B:{lambda_f[0]:.2f}<br>M:{lambda_f[1]:.2f}<br>S:{lambda_f[2]:.2f}"],
        textposition='top center', showlegend=False
    ))

    fig.update_layout(
        scene=dict(
            xaxis_title='Bind',
            yaxis_title='Move',
            zaxis_title='Split',
            xaxis=dict(range=[-1, 1]),
            yaxis=dict(range=[-1, 1]),
            zaxis=dict(range=[-1, 1])
        ),
        title=f"ΛF Transaction Direction at Step {step} (Chloroferrocene)"
    )

    fig.write_html(filename)

 """
    Perform PCA (Principal Component Analysis) on the ΛF transaction history
    to identify dominant transition modes.

    Useful for:
    - Characterizing transaction mode dynamics.
    - Understanding principal directions of state changes over time.

    Args:
        lambda_f_matrix (list): One-hot encoded history of transaction directions.
    """
def analyze_lambda_f(lambda_f_matrix: list):
   
    data = np.stack(lambda_f_matrix)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data)

    print("Principal components of transaction modes (PCA):", pca.components_)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # --- 2D PCA visualization ---
    plt.figure(figsize=(6, 5))
    plt.plot(transformed[:, 0], transformed[:, 1], '-o', label="ΛF trajectory")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("ΛF Transaction Mode Trajectory (PCA)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    """
    Apply quantum-like spin flipping (±1) with a given flip probability.

    This simulates spin fluctuation or quantum noise, introducing
    probabilistic flipping for each element independently.

    Args:
        spins (jnp.ndarray): Spin states of elements (N,).
        flip_prob (float): Probability of spin flip at each site.
        key (jax.random.PRNGKey, optional): Random seed key.

    Returns:
        jnp.ndarray: Flipped spin array.
    """
@jit
def randomize_spins(
    spins: jnp.ndarray,
    config: Any,  # Lambda3FireConfig expected
    score_split: float = None,
    key: jnp.ndarray = None
) -> jnp.ndarray:
    if key is None:
        key = jax.random.PRNGKey(0)

    if score_split is not None:
        flip_prob = config.spin_flip_base_prob * jnp.exp(-config.spin_flip_split_decay * score_split)
    else:
        flip_prob = config.spin_flip_base_prob

    flip_mask = jax.random.bernoulli(key, p=flip_prob, shape=spins.shape)
    flipped_spins = spins * jnp.where(flip_mask, -1, 1)
    return flipped_spins

 """
    Compute the dipole moment tensor D = Σ q_i * r_i.

    Represents the first moment of charge distribution,
    typically used to assess charge polarization or structural asymmetry.

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).
        charges (jnp.ndarray): Charge values of elements (N,).

    Returns:
        jnp.ndarray: Dipole moment vector (3,).
    """

@jit
def compute_dipole_tensor(
    r: jnp.ndarray,
    charges: jnp.ndarray
) -> jnp.ndarray:
   
    return jnp.sum(r * charges[:, None], axis=0)

   """
    Compute the geometric Laplacian term for each element,
    measuring the local curvature or distortion of the positional structure.

    The calculation excludes self-interaction (i ≠ j) and applies
    an inverse cubic distance scaling.

    Args:
        r (jnp.ndarray): Positions of elements (N, 3).

    Returns:
        jnp.ndarray: Laplacian vector for each element (N, 3).
    """
@jit
def laplacian_term(
    r: jnp.ndarray
) -> jnp.ndarray:
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    rij = r_i - r_j
    dists = jnp.linalg.norm(rij, axis=-1) + 1e-5
    dists = jnp.maximum(dists, 0.1)

    mask = 1.0 - jnp.eye(r.shape[0])
    scale = -1.0 / (dists ** 3 + 1e-5) * mask
    contrib = rij * scale[..., None]
    lap = jnp.sum(contrib, axis=1)
    return lap

def lambda3_fire_vmc(
    r, spins, charges, colors, k_vectors,
    lambdaF_ext=None, rhoT_ext=0.0, sigmaS_ext=0.0,
    experiment_type=None, intensity=None,
    adaptive_stop=False,
    config=Lambda3FireConfig()
):
    wandb.init(project=config.project_name, config={"n_steps": config.n_steps})
    r_current = r
    lambda_f_history = []
    lambda_f_matrix = []
    dipole_history = []
    score_bind_history = []
    score_move_history = []
    score_split_history = []
    energy_history = []
    cooldown_level = 0.0

    global key_global

    for step in range(config.n_steps):
        c = compute_dynamic_cutoff(r_current, config=config)

        rho_T, s_gen = compute_rho_T_with_hybrid(r_current, spins, colors, cutoff_radius=c, config=config)
        rho_T = jnp.clip(rho_T, a_min=1e-5, a_max=10.0)

        direction, phase, score_bind, score_move, score_split = compute_transaction_direction_full(
            r_current, spins, charges, colors, k_vectors, rho_T, c, config=config,
            LambdaF_ext=lambdaF_ext, rhoT_ext=rhoT_ext, sigmaS_ext=sigmaS_ext
        )

        if step % config.spin_flip_interval == 0:
            spins = randomize_spins(spins, config=config, score_split=score_split, key=jax.random.PRNGKey(step))

        score_bind = jnp.clip(score_bind, a_min=1e-5, a_max=10.0)
        score_move = jnp.clip(score_move, a_min=1e-5, a_max=10.0)
        score_split = jnp.clip(score_split, a_min=1e-5, a_max=10.0)

        if step > config.ema_score_window:
            score_bind = config.ema_score_current_weight * score_bind + config.ema_score_history_weight * jnp.mean(jnp.array(score_bind_history[-config.ema_score_window:]))
            score_move = config.ema_score_current_weight * score_move + config.ema_score_history_weight * jnp.mean(jnp.array(score_move_history[-config.ema_score_window:]))
            score_split = config.ema_score_current_weight * score_split + config.ema_score_history_weight * jnp.mean(jnp.array(score_split_history[-config.ema_score_window:]))

        score_bind_history.append(float(score_bind))
        score_move_history.append(float(score_move))
        score_split_history.append(float(score_split))

        lambda_f_weighted = jnp.array([score_bind, score_move, score_split])
        lambda_f_weighted /= jnp.sum(lambda_f_weighted)
        h, lambda_f = compute_embedding(r_current, spins, charges, k_vectors, c, config=config, direction=None, phase=phase, lambda_f_override=lambda_f_weighted)

        if step > (config.warmup_step + config.warmup_buffer):
            cooldown_target = config.cooldown_target_off if lambda_f[0] > 0.95 else config.cooldown_target_on
            cooldown_level = (1 - config.cooldown_ewma_alpha) * cooldown_level + config.cooldown_ewma_alpha * cooldown_target
            cooling_intensity = config.cooling_intensity_scaling * (cooldown_level / config.cooldown_target_on)
            lambdaF_ext, rhoT_ext, sigmaS_ext = experiment_to_transaction_params(("cooling",), [cooling_intensity])

        lap_term = laplacian_term(r_current)
        score_bind += 0.001 * jnp.sum(lap_term ** 2)

        if direction == 2:
            split_phase = (step % 12) / 12.0 * 2 * np.pi
            phase_vector = jnp.exp(1j * split_phase)
        else:
            phase_vector = 1.0 + 0j

        h, lambda_f = compute_embedding(r_current, spins, charges, k_vectors, c, config=config, direction=direction, phase=phase * phase_vector, s_gen=s_gen)

        energy, key_global = compute_energy_structural(r_current, spins, charges, colors, k_vectors, c, direction, config=config, key=key_global)

        if len(energy_history) >= config.ema_energy_window:
            energy = config.ema_energy_current_weight * energy + config.ema_energy_history_weight * jnp.mean(jnp.array(energy_history[-config.ema_energy_window:]))
        energy_history.append(float(energy))

        dipole_vector = compute_dipole_tensor(r_current, charges)
        dipole_magnitude = float(jnp.linalg.norm(dipole_vector))
        dipole_history.append(dipole_magnitude)

        dipole_delta = jnp.abs(dipole_magnitude - jnp.mean(jnp.array(dipole_history[-config.ema_energy_window:]))) if len(dipole_history) >= config.ema_energy_window else 0.0
        reward = -jnp.tanh(jnp.abs(rho_T - 1.0)) - 0.1 * jnp.tanh(dipole_delta)

        r_samples, sigmas = adaptive_metropolis_sampling_structural(
            r_current, spins, charges, colors, k_vectors, c, direction, config=config, rng_key=jax.random.PRNGKey(step)
        )
        r_current = r_samples[-1] + 0.002 * lap_term

        lambda_f_history.append(lambda_f)
        lambda_f_onehot = jnp.array([int(direction == 0), int(direction == 1), int(direction == 2)])
        lambda_f_matrix.append(np.array(lambda_f_onehot))

        wandb.log({
            "step": step,
            "rho_T": float(rho_T),
            "energy": float(energy),
            "lambda_f_bind": float(lambda_f[0]),
            "lambda_f_move": float(lambda_f[1]),
            "lambda_f_split": float(lambda_f[2]),
            "score_bind": float(score_bind),
            "score_move": float(score_move),
            "score_split": float(score_split),
            "dipole_x": float(dipole_vector[0]),
            "dipole_y": float(dipole_vector[1]),
            "dipole_z": float(dipole_vector[2]),
            "dipole_magnitude": dipole_magnitude,
            "sigma": float(sigmas[-1]),
            "reward": float(reward),
            "electron_positions": wandb.Object3D({
                "type": "lidar/beta",
                "points": np.array(r_current, dtype=np.float32)
            })
        })

        if step % 100 == 0:
            visualize_lambda_f_with_crystal_axis(lambda_f_history, step, r_current, config=config, dipole_vector=dipole_vector)

    wandb.finish()
    return r_current, lambda_f_matrix

# Define initial atomic positions for chloroferrocene:
# Fe center, five-membered Cp rings (C1–C4, C6–C9), and Cl ligand.
def setup_chloroferrocene():
    n_el = 10
    r = jnp.array([
        [0.0, 0.0, 0.0],        # Fe
        [1.4, 0.0, 2.05],       # C1
        [0.7, 1.21, 2.05],      # C2
        [-0.7, 1.21, 2.05],     # C3
        [-1.4, 0.0, 2.05],      # C4
        [0.0, 0.0, 2.05],       # Cl
        [1.4, 0.0, -2.05],      # C6
        [0.7, 1.21, -2.05],     # C7
        [-0.7, 1.21, -2.05],    # C8
        [-1.4, 0.0, -2.05],     # C9
    ], dtype=jnp.float32)

    spins = jnp.array([0.0] * n_el, dtype=jnp.float32)
    charges = jnp.array([1.0] * n_el, dtype=jnp.float32)

    # ✅ Element-specific color charge (RGB-like visualization), 
    # compatible with QCD color charges at the quark level.
    colors = jnp.array([
        [0.8, 0.0, 0.0],   # Fe（red）
        [0.2, 0.2, 0.2],   # C1
        [0.2, 0.2, 0.2],   # C2
        [0.2, 0.2, 0.2],   # C3
        [0.2, 0.2, 0.2],   # C4
        [0.1, 0.1, 0.5],   # Cl（blue）
        [0.2, 0.2, 0.2],   # C6
        [0.2, 0.2, 0.2],   # C7
        [0.2, 0.2, 0.2],   # C8
        [0.2, 0.2, 0.2],   # C9
    ], dtype=jnp.float32)

    k_vectors = jnp.array([[1.0, 0.0, 0.0]] * n_el, dtype=jnp.float32)

    return r, spins, charges, colors, k_vectors

if __name__ == "__main__":
    config = Lambda3FireConfig()

    r, spins, charges, colors, k_vectors = setup_chloroferrocene()

    lambdaF_ext, rhoT_ext, sigmaS_ext = experiment_to_transaction_params(
        config.experiment_types, config.intensities
    )

    r_final, lambda_f_matrix = lambda3_fire_vmc(
        r, spins, charges, colors, k_vectors,
        lambdaF_ext=lambdaF_ext,
        rhoT_ext=rhoT_ext,
        sigmaS_ext=sigmaS_ext,
        experiment_type=config.experiment_types,
        intensity=config.intensities,
        adaptive_stop=True,
        config=config
    )

    print(f"Final electron positions for chloroferrocene:\n{r_final}")
