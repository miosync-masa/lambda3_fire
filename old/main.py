import jax
from jax import lax
import jax.numpy as jnp
from jax import jit, grad
import numpy as np
from jax import random
from functools import partial
import plotly.graph_objects as go
import wandb
from typing import Union, List
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt

# Core Parameters for Λ³ Transaction Engine
EMBEDDING_DIM = 16
STRUCTURE_RADIUS_BASE = 3.0
RHO_T0 = 1.0
CUTOFF_RHO_EXPONENT = 0.5
CUTOFF_SIGMA_EXPONENT = 0.3
ENTROPIC_SPREAD_COEFF = 1.0
TARGET_ACCEPTANCE = 0.5

key_global = jax.random.PRNGKey(42)
ALPHA_DISTANCE   = 0.1     # 同期率の exp(-α d^2) の α
GAMMA_CHARGE     = 4.0     # 電荷不一致の抑制係数
GAMMA_COLOR      = 4.0     # 色荷不一致の抑制係数

# --- Factor weights for structural coherence tensor ---
W_SPIN   = 1.0   # スピン整合性の寄与重み
W_COLOR  = 1.0   # 色荷整合性の寄与重み
W_CHARGE = 1.0   # 電荷整合性の寄与重み
W_DIST   = 1.0   # 距離整合性の寄与重み（オーバーラップ的性質）

# Transaction direction vectors
LAMBDA_F_BIND = jnp.array([1.0, 0.0, 0.0])
LAMBDA_F_MOVE = jnp.array([0.0, 1.0, 0.0])
LAMBDA_F_SPLIT = jnp.array([0.0, 0.0, 1.0])

# Utility function for safe division
def safe_divide(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

# 1. Dynamic cutoff calculation
@jit
def compute_dynamic_cutoff(
    r,
    structure_radius_base=STRUCTURE_RADIUS_BASE,        # Λ₀：構造基準半径
    rho_T_base=RHO_T0,                                  # ρT₀：基準テンション密度
    rho_scaling_exp=CUTOFF_RHO_EXPONENT,               # α：テンション密度の指数スケーリング
    sigma_scaling_exp=CUTOFF_SIGMA_EXPONENT            # β：同期率の指数スケーリング
):
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1)) + jnp.eye(n_el) * 1e10

    # ρT 計算
    rho_T = jnp.sum(1.0 / dists, where=dists < 10.0) / n_el

    # σs 計算
    mask = dists < structure_radius_base
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * mask
    sigma_s = jnp.sum(sigma_s_ij) / jnp.sum(mask)

    return structure_radius_base * (rho_T / rho_T_base) ** rho_scaling_exp * sigma_s ** sigma_scaling_exp

# 2. 実験操作からΛ³テンソルへの外部トランザクションパラメータ
@partial(jit, static_argnames=["experiment_types"])
def experiment_to_transaction_params(experiment_types: tuple, intensities: list):
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

    # 🔁 非線形ブースト項（相互作用モデル拡張）
    correlation_rhoT_boost = 0.0

    if "heating" in experiment_types and "photo_irradiation" in experiment_types:
        i_T = experiment_types.index("heating")
        i_I = experiment_types.index("photo_irradiation")
        T = intensities[i_T]
        I = intensities[i_I]
        correlation_rhoT_boost += 0.001 * jnp.sqrt(T * I)

    if "heating" in experiment_types and "pressure" in experiment_types:
        i_T = experiment_types.index("heating")
        i_P = experiment_types.index("pressure")
        T = intensities[i_T]
        P = intensities[i_P]
        correlation_rhoT_boost += 1e-5 * jnp.log1p(P * T)

    if "photo_irradiation" in experiment_types and "pressure" in experiment_types:
        i_I = experiment_types.index("photo_irradiation")
        i_P = experiment_types.index("pressure")
        I = intensities[i_I]
        P = intensities[i_P]
        correlation_rhoT_boost += 5e-6 * jnp.sqrt(I * jnp.log1p(P))

    if "electric_field" in experiment_types and "photo_irradiation" in experiment_types:
        i_E = experiment_types.index("electric_field")
        i_I = experiment_types.index("photo_irradiation")
        E = intensities[i_E]
        I = intensities[i_I]
        correlation_rhoT_boost += 1e-6 * jnp.sqrt(E * I)

    # 合成トランザクションの出力
    LambdaF_ext = LambdaF_total / jnp.sum(LambdaF_total)
    sigmaS_ext = jnp.prod(jnp.array(sigmaS_list))
    rhoT_ext = rhoT_total + correlation_rhoT_boost

    return LambdaF_ext, rhoT_ext, sigmaS_ext

# 3. rho_T の計算
@jit
def compute_rho_T_with_hybrid(
    r, spins, colors, cutoff_radius,
    entropy_weight=0.1,
    disorder_amplitude=ENTROPIC_SPREAD_COEFF,
    energy_density_base=1.0,
    projection_angle=0.0,
    structure_length_ref=1.0,
    a=1.0, b=0.5, c_decay=0.1,
    rhoT_ext=0.0
):
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    dists = jnp.where(dists < 1e-2, 1e-2, dists)
    dists = dists + jnp.eye(n_el) * 1e10

    # --- ρT base ---
    rho_T_base = energy_density_base * (structure_length_ref / dists) ** 2 * jnp.cos(projection_angle)
    rho_T_base = safe_divide(jnp.sum(rho_T_base, where=dists < cutoff_radius), n_el)

    # --- physical pair term ---
    pairwise_term = (a / dists) + b * jnp.exp(-c_decay * dists)
    rho_T_phys = jnp.sum(pairwise_term, where=dists > cutoff_radius) / n_el

    # --- σs 整合テンソル（spin × color × distance）---
    spin_match = (spins[:, None] == spins[None, :]).astype(float)

    dcolor = colors[:, None, :] - colors[None, :, :]
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_match = jnp.exp(-GAMMA_COLOR * color_norm2)

    local_overlap = jnp.exp(-0.1 * dists ** 2)
    mask = (dists < cutoff_radius) & (dists > 0)

    sigma_s_ij = spin_match * color_match * local_overlap * mask

    # --- エントロピー項（非整合度）---
    S_gen = disorder_amplitude * safe_divide(jnp.sum(1.0 - sigma_s_ij, where=mask), jnp.sum(mask))

    # --- 統合テンション出力 ---
    scale_factor = 1.0
    rho_T_out = (rho_T_base + entropy_weight * S_gen + 0.1 * rho_T_phys + rhoT_ext) * scale_factor

    return rho_T_out, S_gen

def _compute_transaction_direction_full(
    r, spins, charges, colors, k_vectors, rho_T, c,
    LambdaF_ext=None, rhoT_ext=0.0, sigmaS_ext=0.0
):
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    mask = (dists < c) & (dists > 0)

    spin_tensor = W_SPIN * (spins[:, None] == spins[None, :]).astype(float)
    dcolor = colors[:, None, :] - colors[None, :, :]
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_tensor = W_COLOR * jnp.exp(-GAMMA_COLOR * color_norm2)
    dq = charges[:, None] - charges[None, :]
    charge_tensor = W_CHARGE * jnp.exp(-GAMMA_CHARGE * dq ** 2)
    distance_tensor = W_DIST * jnp.exp(-ALPHA_DISTANCE * dists ** 2)

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

    alpha_R = 0.1
    E_field = jnp.array([0.0, 0.0, 1.0])

    def generate_phase_valid(kv):
        k_cross_E = jnp.cross(kv, E_field)
        phi = alpha_R * jnp.sum(k_cross_E * spins[:, None], axis=-1)
        return jnp.exp(1j * phi)

    def generate_phase_dummy(_):
        return jnp.ones(r.shape[0], dtype=jnp.complex64)

    enable_phase = jnp.logical_not(jnp.allclose(k_vectors, 0.0))
    apply_kvec_shape = (k_vectors.shape[-1] == 3 and k_vectors.shape[0] == r.shape[0])
    apply_kvec = jnp.logical_and(apply_kvec_shape, enable_phase)

    phase = lax.cond(
        apply_kvec,
        generate_phase_valid,
        generate_phase_dummy,
        k_vectors
    )

    return direction, phase, score_bind, score_move, score_split

# ✅ jit適用（LambdaF_extをstaticに明示）
compute_transaction_direction_full = jax.jit(_compute_transaction_direction_full)

# 埋め込み h_i(r) の計算
@jit
def compute_embedding(
    r, spins, charges, k_vectors, c,
    direction=None,
    phase=1.0,
    lambda_f_override=None,
    s_gen=0.0,                  # 🔥構造エントロピー（ゆらぎ）
    tau_base=4.0,
    alpha_entropy=2.5           # 🔧 ゆらぎによる拡散ブースト係数
):
    n_el = r.shape[0]
    h = jnp.zeros((n_el, EMBEDDING_DIM), dtype=jnp.complex64)

    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists2 = jnp.sum((r_i - r_j) ** 2, axis=-1)
    dists = jnp.sqrt(dists2) + jnp.eye(n_el) * 1e10
    mask = (dists < c) & (dists > 0)

    spin_align = (spins[:, None] == spins[None, :]).astype(float)
    sigma_s_ij = jnp.exp(-0.1 * dists2) * spin_align * mask
    rhoT_ij = (1.0 / dists) * mask

    # 🔥 エントロピー連動で attention 幅を拡張！
    tau_dynamic = tau_base * (1.0 + alpha_entropy * s_gen)
    A_ij = jnp.exp(-dists2 / tau_dynamic) * mask

    if lambda_f_override is not None:
        lambda_f = lambda_f_override
    else:
        lambda_f = jnp.where(
            direction == 0, LAMBDA_F_BIND,
            jnp.where(direction == 1, LAMBDA_F_MOVE, LAMBDA_F_SPLIT)
        )

    lambda_f_extended = jnp.tile(lambda_f, EMBEDDING_DIM // 3 + 1)[:EMBEDDING_DIM]
    lambda_f_complex = lambda_f_extended.astype(jnp.complex64) * jnp.mean(phase)

    for i in range(n_el):
        contrib_scalar = jnp.sum(sigma_s_ij[i] * rhoT_ij[i] * A_ij[i])
        h = h.at[i].set(contrib_scalar * lambda_f_complex)

    return h, lambda_f

# 9. エネルギー計算
@jit
def compute_energy_structural(r, spins, charges, colors,
                              k_vectors, c, direction,
                              temp_beta=10.0,
                              key=None):
    n_el = r.shape[0]

    # --- 幾何テンソル定義 ---
    dR = r[:, None, :] - r[None, :, :]
    dists = jnp.linalg.norm(dR, axis=-1)
    dists = jnp.maximum(dists, 1e-2)

    # --- ρT: base + 周期補正 ---
    rho_T_ij = 1.0 / dists

    def add_periodic_potential(kv):
        dot_k_dr = jnp.tensordot(kv, dR, axes=[[1], [2]])  # (M,N,N)
        rho_T_periodic = jnp.mean(jnp.cos(dot_k_dr), axis=0)
        return rho_T_periodic / (dists ** 2 + 1.0)

    def zero_potential(_):
        return jnp.zeros_like(rho_T_ij)

    apply_kvec_shape = jnp.logical_and(k_vectors.shape[-1] == 3, k_vectors.shape[0] == r.shape[0])
    enable_phase = jnp.logical_not(jnp.all(jnp.isclose(k_vectors, 0.0)))
    apply_kvec = jnp.logical_and(apply_kvec_shape, enable_phase)

    rho_T_ij += lax.cond(apply_kvec, add_periodic_potential, zero_potential, k_vectors)

    # --- σs: resonance tensor ---
    spin_match     = (spins[:, None] == spins[None, :]).astype(float)
    dq             = charges[:, None] - charges[None, :]
    charge_match   = jnp.exp(-GAMMA_CHARGE * dq ** 2)
    dcolor         = colors[:, None, :] - colors[None, :, :]
    color_match    = jnp.exp(-GAMMA_COLOR * jnp.sum(dcolor ** 2, axis=-1))
    local_overlap  = jnp.exp(-ALPHA_DISTANCE * dists ** 2)

    sigma_s_ij = spin_match * charge_match * color_match * local_overlap

    # --- ΛF方向定義 + 熱ゆらぎ ---
    λ_nominal = jnp.select(
        [direction == 0, direction == 1, direction == 2],
        [LAMBDA_F_BIND, LAMBDA_F_MOVE, LAMBDA_F_SPLIT],
        default=jnp.array([1.0, 0.0, 0.0])
    )

    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, shape=(3,)) / jnp.sqrt(temp_beta + 1e-9)
    lambda_f = λ_nominal + noise
    lambda_f /= (jnp.linalg.norm(lambda_f) + 1e-9)

    # --- cosθ = (Δr·λ) / |Δr| ---
    dot_products = jnp.einsum('ijk,k->ij', dR, lambda_f) / dists

    # --- エネルギー定義（NaNガード付き）---
    energy_matrix = rho_T_ij * sigma_s_ij * dot_products
    total_energy = jnp.sum(energy_matrix) / (n_el ** 2)

    return jnp.nan_to_num(total_energy, nan=1e-6), key

@jit
def compute_rho_T_localized(r, spins, colors, idx, cutoff_radius):
    ri = r[idx]
    dists = jnp.sqrt(jnp.sum((r - ri) ** 2, axis=1))
    mask = (dists < cutoff_radius) & (dists > 0)

    spin_match = (spins[idx] == spins).astype(float)
    dcolor = colors[idx] - colors
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_match = jnp.exp(-GAMMA_COLOR * color_norm2)
    local_overlap = jnp.exp(-0.1 * dists ** 2)

    sigma_s = spin_match * color_match * local_overlap * mask

    rho_T_base = (1.0 / (dists**2 + 1e-5)) * mask
    rho_T_est = jnp.sum(rho_T_base)
    S_gen = jnp.sum(1.0 - sigma_s) / (jnp.sum(mask) + 1e-8)

    return rho_T_est, S_gen

# 🔧 core_sampling
def adaptive_metropolis_sampling_structural(
    r0, spins, charges, colors,
    k_vectors, c, direction,
    *,
    n_steps=1000,
    sigma_init=0.5,
    temp_beta=10.0,
    target_acceptance=0.25,
    rng_key=jax.random.PRNGKey(0)
):
    n_steps = int(n_steps)

    @jit
    def safe_divide(a, b, eps=1e-12):
        return a / (b + eps)

    @jit
    def single_step(state, step):
        r, sigma, E_curr, key, acc_cnt, r_samples, sigmas, score_split = state
        key, k_prop, k_energy = jax.random.split(key, 3)
        noise = jax.random.normal(k_prop, shape=r.shape)
        r_proposed = jnp.clip(r + sigma * noise, -5.0, 5.0)

        # ✅ エネルギーは通常評価
        E_prop, key = compute_energy_structural(
            r_proposed, spins, charges, colors,
            k_vectors, c, direction,
            temp_beta=temp_beta, key=k_energy
        )

        # 🔧 Low-rank風評価：最大変位の電子周辺のみで ρT 推定
        delta_r = jnp.linalg.norm(r_proposed - r, axis=1)
        idx_changed = jnp.argmax(delta_r)

        rho_T, _ = compute_rho_T_localized(
            r_proposed, spins, colors, idx_changed, cutoff_radius=c
        )

        # ✅ トランザクション方向評価（局所ρTベース）
        _, _, _, score_split_new, _ = compute_transaction_direction_full(
            r_proposed, spins, charges, colors, k_vectors, rho_T, c
        )

        acc_ratio = safe_divide(E_curr, E_prop)
        accept = jax.random.uniform(k_prop) < acc_ratio
        r_new = lax.select(accept, r_proposed, r)
        E_new = lax.select(accept, E_prop, E_curr)
        acc_cnt = acc_cnt + accept

        r_samples = r_samples.at[step + 1].set(r_new)
        sigmas = sigmas.at[step + 1].set(sigma)

        # ✅ acceptance調整（スコアによる分岐）
        target_acc = lax.cond(
            score_split_new < 0.5,
            lambda _: 0.25,
            lambda _: 0.1,
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

    # 初期状態の準備
    r_samples = jnp.zeros((n_steps + 1, *r0.shape))
    r_samples = r_samples.at[0].set(r0)

    sigmas = jnp.zeros(n_steps + 1)
    sigmas = sigmas.at[0].set(sigma_init)

    rng_key, k0_energy = jax.random.split(rng_key)
    E0, rng_key = compute_energy_structural(
        r0, spins, charges, colors,
        k_vectors, c, direction,
        temp_beta=temp_beta, key=k0_energy
    )

    rho_T, _ = compute_rho_T_with_hybrid(r0, spins, colors, c)
    _, _, _, score_split_init, _ = compute_transaction_direction_full(
        r0, spins, charges, colors, k_vectors, rho_T, c
    )

    init_state = (r0, sigma_init, E0, rng_key, 0, r_samples, sigmas, score_split_init)
    final_state, _ = lax.scan(single_step, init_state, jnp.arange(n_steps))
    _, _, _, _, _, r_samples, sigmas, _ = final_state

    return r_samples, sigmas

# 11. Lambda ベクトルの3D可視化（dipole表示つき）
def visualize_lambda_f_with_crystal_axis(lambda_f_history, step, r, dipole_vector=None, filename="lambda_f_visualization.html"):
    lambda_f = lambda_f_history[step]
    fig = go.Figure()

    # ΛF ベクトル（青）
    fig.add_trace(go.Scatter3d(
        x=[0, lambda_f[0]], y=[0, lambda_f[1]], z=[0, lambda_f[2]],
        mode='lines+markers', line=dict(color='blue', width=6),
        marker=dict(size=6), name=f'ΛF vector (Step {step})'
    ))

    # Cp環法線（ダミーでz軸）
    cp_normal = jnp.array([0.0, 0.0, 1.0])
    fig.add_trace(go.Scatter3d(
        x=[0, cp_normal[0]], y=[0, cp_normal[1]], z=[0, cp_normal[2]],
        mode='lines', line=dict(color='red', width=4, dash='dash'),
        name='Cp ring normal'
    ))

    # 💡 Dipoleベクトル（オレンジ）
    if dipole_vector is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, float(dipole_vector[0])],
            y=[0, float(dipole_vector[1])],
            z=[0, float(dipole_vector[2])],
            mode='lines+markers', line=dict(color='orange', width=4),
            marker=dict(size=4),
            name='Dipole Vector'
        ))

    # ΛFラベル
    fig.add_trace(go.Scatter3d(
        x=[lambda_f[0]], y=[lambda_f[1]], z=[lambda_f[2]],
        mode='text',
        text=[f"B:{lambda_f[0]:.2f}<br>M:{lambda_f[1]:.2f}<br>S:{lambda_f[2]:.2f}"],
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

def analyze_lambda_f(lambda_f_matrix):
    data = np.stack(lambda_f_matrix)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data)

    print("遷移モード主成分（PCA）:", pca.components_)
    print("寄与率:", pca.explained_variance_ratio_)

    # 可視化
    plt.figure(figsize=(6, 5))
    plt.plot(transformed[:, 0], transformed[:, 1], '-o', label="λF")
    plt.xlabel("main1")
    plt.ylabel("main2")
    plt.title("ΛFtransactionlog(PCA)")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

@jit
def randomize_spins(spins, flip_prob=0.1, key=None):
    """スピン ±1 に量子揺らぎ的にフリップ"""
    if key is None:
        key = jax.random.PRNGKey(0)
    flip_mask = jax.random.bernoulli(key, p=flip_prob, shape=spins.shape)
    flipped_spins = spins * jnp.where(flip_mask, -1, 1)
    return flipped_spins

@jit
def compute_dipole_tensor(r, charges):
    """双極子モーメントテンソル D = Σ q_i * r_i"""
    return jnp.sum(r * charges[:, None], axis=0)

@jit
def laplacian_term(r):
    r_i = r[:, None, :]        # (N,1,3)
    r_j = r[None, :, :]        # (1,N,3)
    rij = r_i - r_j            # (N,N,3)
    dists = jnp.linalg.norm(rij, axis=-1) + 1e-5
    dists = jnp.maximum(dists, 0.1)

    # maskで i ≠ j のみ取り出す（対角成分を0に）
    mask = 1.0 - jnp.eye(r.shape[0])
    scale = -1.0 / (dists ** 3 + 1e-5) * mask  # (N,N)
    contrib = rij * scale[..., None]          # (N,N,3)
    lap = jnp.sum(contrib, axis=1)            # (N,3)
    return lap

def lambda3_fire_vmc(
    r, spins, charges, colors, k_vectors, n_steps=1000,
    project_name="lambda3-fire",
    lambdaF_ext=None, rhoT_ext=0.0, sigmaS_ext=0.0,
    experiment_type=None, intensity=None,
    adaptive_stop=False
):
    wandb.init(project=project_name, config={"n_steps": n_steps})
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

    for step in range(n_steps):
        # --- 動的 cutoff 計算 ---
        c = compute_dynamic_cutoff(
            r_current,
            structure_radius_base=STRUCTURE_RADIUS_BASE,
            rho_T_base=RHO_T0,
            rho_scaling_exp=CUTOFF_RHO_EXPONENT,
            sigma_scaling_exp=CUTOFF_SIGMA_EXPONENT
        )

        # --- テンション密度 ρT 計算 ---
        rho_T, s_gen = compute_rho_T_with_hybrid(
            r_current, spins, colors, cutoff_radius=c,
            entropy_weight=0.1,
            disorder_amplitude=ENTROPIC_SPREAD_COEFF
        )
        rho_T = jnp.clip(rho_T, a_min=1e-5, a_max=10.0)

        # --- トランザクション方向判定 ---
        direction, phase, score_bind, score_move, score_split = compute_transaction_direction_full(
            r_current, spins, charges, colors, k_vectors, rho_T, c,
            LambdaF_ext=lambdaF_ext, rhoT_ext=rhoT_ext, sigmaS_ext=sigmaS_ext
        )

        # --- スピンフリップ処理 ---
        if step % 100 == 0:
            flip_prob = 0.15 * jnp.exp(-3.0 * score_split)
            spins = randomize_spins(spins, flip_prob=flip_prob, key=jax.random.PRNGKey(step))

        # --- スコア履歴の記録（平滑化含む） ---
        score_bind = jnp.clip(score_bind, a_min=1e-5, a_max=10.0)
        score_move = jnp.clip(score_move, a_min=1e-5, a_max=10.0)
        score_split = jnp.clip(score_split, a_min=1e-5, a_max=10.0)

        if step > 5:
            score_bind = 0.7 * score_bind + 0.3 * jnp.mean(jnp.array(score_bind_history[-5:]))
            score_move = 0.7 * score_move + 0.3 * jnp.mean(jnp.array(score_move_history[-5:]))
            score_split = 0.7 * score_split + 0.3 * jnp.mean(jnp.array(score_split_history[-5:]))

        score_bind_history.append(float(score_bind))
        score_move_history.append(float(score_move))
        score_split_history.append(float(score_split))

        # --- ΛFの暫定ベクトル（スコア比率） ---
        lambda_f_weighted = jnp.array([score_bind, score_move, score_split])
        lambda_f_weighted /= jnp.sum(lambda_f_weighted)
        h, lambda_f = compute_embedding(
            r_current, spins, charges, k_vectors, c,
            direction=None, phase=phase, lambda_f_override=lambda_f_weighted
        )

        # --- EMA冷却制御：Bindを条件に連続緩和していく ---
        warmup_step = 30
        warmup_buffer = 10
        if step > (warmup_step + warmup_buffer):
            # ΛFがBindを指しているなら緩和（目標は0）
            cooldown_target = 0.0 if lambda_f[0] > 0.95 else 3.0
            # EMA更新（緩やかに変化）
            cooldown_level = 0.9 * cooldown_level + 0.1 * cooldown_target
            # 冷却強度を連続スケールで注入
            cooling_intensity = 200.0 * (cooldown_level / 3.0)
            lambdaF_ext, rhoT_ext, sigmaS_ext = experiment_to_transaction_params(("cooling",), [cooling_intensity])

        # --- ΛF再計算（phase更新付き） ---
        lap_term = laplacian_term(r_current)
        score_bind += 0.001 * jnp.sum(lap_term ** 2)

        if direction == 2:
            split_phase = (step % 12) / 12.0 * 2 * np.pi
            phase_vector = jnp.exp(1j * split_phase)
        else:
            phase_vector = 1.0 + 0j

        h, lambda_f = compute_embedding(
            r_current, spins, charges, k_vectors, c,
            direction=direction,
            phase=phase * phase_vector,
            s_gen=s_gen  # 🔥 disorderからの影響を反映
        )

        # --- エネルギー計算 ---
        energy, key_global = compute_energy_structural(
            r_current, spins, charges, colors,
            k_vectors, c, direction,
            temp_beta=10.0,
            key=key_global
        )

        if len(energy_history) >= 5:
            energy = 0.8 * energy + 0.2 * jnp.mean(jnp.array(energy_history[-5:]))
        energy_history.append(float(energy))

        # --- 双極子と報酬処理 ---
        dipole_vector = compute_dipole_tensor(r_current, charges)
        dipole_magnitude = float(jnp.linalg.norm(dipole_vector))
        dipole_history.append(dipole_magnitude)

        dipole_delta = jnp.abs(dipole_magnitude - jnp.mean(jnp.array(dipole_history[-5:]))) if len(dipole_history) >= 5 else 0.0
        reward = -jnp.tanh(jnp.abs(rho_T - 1.0)) - 0.1 * jnp.tanh(dipole_delta)

        # --- VMCステップ実行 ---
        r_samples, sigmas = adaptive_metropolis_sampling_structural(
            r_current, spins, charges, colors,
            k_vectors, c, direction,
            n_steps=1, temp_beta=10.0,
            rng_key=jax.random.PRNGKey(step)
        )
        r_current = r_samples[-1] + 0.002 * lap_term

        lambda_f_history.append(lambda_f)
        lambda_f_onehot = jnp.array([
            int(direction == 0),
            int(direction == 1),
            int(direction == 2)
        ])
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
            visualize_lambda_f_with_crystal_axis(lambda_f_history, step, r_current, dipole_vector=dipole_vector)

    wandb.finish()
    return r_current, lambda_f_matrix

# クロロフェロセンの設定
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

    # ✅ 元素ごとの color charge（RGB風）
    colors = jnp.array([
        [0.8, 0.0, 0.0],   # Fe（赤系）
        [0.2, 0.2, 0.2],   # C1
        [0.2, 0.2, 0.2],   # C2
        [0.2, 0.2, 0.2],   # C3
        [0.2, 0.2, 0.2],   # C4
        [0.1, 0.1, 0.5],   # Cl（青っぽく）
        [0.2, 0.2, 0.2],   # C6
        [0.2, 0.2, 0.2],   # C7
        [0.2, 0.2, 0.2],   # C8
        [0.2, 0.2, 0.2],   # C9
    ], dtype=jnp.float32)

    k_vectors = jnp.array([[1.0, 0.0, 0.0]] * n_el, dtype=jnp.float32)

    return r, spins, charges, colors, k_vectors

if __name__ == "__main__":
    # 初期構造：クロロフェロセン
    r, spins, charges, colors, k_vectors = setup_chloroferrocene()

    # 実験操作（イオン化誘導の3成分）
    experiment_types = ("photo_irradiation", "heating", "pressure")
    intensities = [1.0e6, 1000.0, 5.0e7]

    # トランザクション構造を合成
    lambdaF_ext, rhoT_ext, sigmaS_ext = experiment_to_transaction_params(
        experiment_types, intensities
    )

    # ✅ シミュレーション実行（adaptive_stopあり、内部で冷却・停止処理）
    r_final, lambda_f_matrix = lambda3_fire_vmc(
        r, spins, charges, colors, k_vectors,
        n_steps=300,
        lambdaF_ext=lambdaF_ext,
        rhoT_ext=rhoT_ext,
        sigmaS_ext=sigmaS_ext,
        experiment_type=experiment_types,
        intensity=intensities,
        adaptive_stop=True
    )

    print(f"Final electron positions for chloroferrocene:\n{r_final}")
