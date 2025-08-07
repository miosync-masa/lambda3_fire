# === Standard Library ===
import copy
import hashlib
import io
import logging
import traceback
from collections import namedtuple
from typing import Any, List, NamedTuple, Optional, Tuple, Union
from functools import partial

import jax
import jax.numpy as jnp
from jax import grad, jit, lax, random, vmap
from functools import partial

import numpy as np
import wandb

import plotly.graph_objects as go
from matplotlib import pyplot as plt

from dataclasses import dataclass, field, asdict
from sklearn.decomposition import PCA
from scipy.sparse import csr_matrix


logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
#jax.debug.print = lambda *args, **kwargs: None

def log_error(step, func_name, error, context=None):
    error_msg = f"Step {step} in {func_name}: {str(error)}\n{traceback.format_exc()}"
    if context:
        error_msg += f"\nContext: {context}"
    logger.error(error_msg)
    try:
        wandb.log({"error": error_msg, "step": step})
    except:
        logger.warning("Failed to log error to wandb")

@dataclass
class UpdateKeyStruct:
    projector: object  # jax.Array
    eigenvector: object

@dataclass
class Lambda3Fire_tamaki_Config:
    # 基本
    embedding_dim: int = 16
    sigma: float = 0.35
    structure_radius_base: float = 2.5
    rho_t0: float = 1.0
    cutoff_rho_exponent: float = 0.4
    cutoff_sigma_exponent: float = 0.25
    entropic_spread_coeff: float = 1.0
    entropy_weight: float = 1.8   # レーザーによるエントロピー増大対応
    energy_density_base: float = 1.2
    projection_angle: float = 0.0
    structure_length_ref: float = 1.2
    tau_base: float = 3.0
    alpha_entropy: float = 1.8    # エントロピー感度をレーザー照射に最適化
    pairwise_a: float = 1.0
    pairwise_b: float = 0.6
    pairwise_c_decay: float = 0.15
    k_vector_update_method: str = "dipole+laplacian"
    k_vector_norm_eps: float = 1e-8
    split_threshold: float = 0.07   # レーザー誘起分離を促進
    q_lambda_jump_threshold: float = 0.15

    # 🎲 Noise/Temp
    key_global: Any = field(default_factory=lambda: jax.random.PRNGKey(314))
    temp_beta: float = 15.0
    base_scale: float = 0.2
    strength: float = 0.15
    observe_prob: float = 0.6
    noise_scale: float = 0.02
    global_noise_strength: float = 0.03
    ham_noise_strength: float = 0.015
    alpha_mixing: float = 0.85
    disorder_amplitude: float = 6.5  # レーザー加熱による揺らぎを反映
    phase_noise_strength: float = 0.03

    # 🔄 Spin Flip
    spin_flip_interval: int = 1
    spin_flip_base_prob: float = 0.35    # レーザーエネルギー吸収でスピン反転活性化
    spin_flip_split_decay: float = 6.5
    beta_spin_flip: float = 0.02

    # 🔄 color＆Chargeアップデート
    threshold_ionization: float = 0.08
    threshold_redox: float = 1.5
    threshold_excitation: float = 0.5
    threshold_proton_hop: float = 0.5
    proton_move_delta: float = 0.10 #　一般的なC-H, N-H, O-H結合距離：1.0~1.1 Å程度、Hoppingで「0.1~0.3Å」くらいだと、分子内・分子間の跳躍も現実的
    color_noise_scale: float = 0.0015
    charge_noise_scale: float = 0.0005
    redox_delta: float = 1.0

    # 🌀 分岐・同期
    alpha_distance: float = 0.25
    gamma_charge: float = 2.5
    gamma_color: float = 2.5
    gamma_lambdaF: float = 0.5
    sigma_init: float = 0.45
    target_acceptance_split: float = 0.3
    target_acceptance_move: float = 0.5
    target_acceptance_bind: float = 0.4
    target_acceptance: float = 0.55
    distance_overlap_alpha: float = 0.2

    # ⚡ 同期重み
    w_spin: float = 0.55
    w_color: float = 0.2
    w_charge: float = 0.2
    w_dist: float = 0.15
    w_lambdaF: float = 0.1

    # 🏹 ΛF基底
    lambda_f_bind: jnp.ndarray = field(default_factory=lambda: jnp.array([1.0, 0.0, 0.0]))
    lambda_f_move: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 1.0, 0.0]))
    lambda_f_split: jnp.ndarray = field(default_factory=lambda: jnp.array([0.0, 0.0, 1.0]))

    # 📉 EMA
    ema_energy_window: int = 15
    ema_energy_current_weight: float = 0.75
    ema_energy_history_weight: float = 0.25
    ema_score_window: int = 10
    ema_score_current_weight: float = 0.7
    ema_score_history_weight: float = 0.3
    ema_alpha: float = 0.9

    # quantum Parameter
    quantum_progression_step: float = 0.10
    HAMILTONIAN_MODE: str = "qed_field"
    # "heisenberg","hubbard","custom","huckel","dirac_field","dirac","qed_field"
    measurement_prob: float = 0.40   # ← 測定頻度UP
    qed_alpha: float = 1.0 / 8.0
    qed_beta: float = 1.0
    qed_gamma: float = 3.5
    cutoff_qed_field: float = 3.0
    a_mu_field_mode: str = "zero"
    boundary_mode: str = "periodic"
    grid_size: int = 30
    grid_extent: float = 5.0
    heisenberg_J: float = 2.0
    hubbard_t: float = 2.0
    hubbard_U: float = 2.2
    huckel_alpha: float = -1.2
    huckel_beta: float = -0.6
    dirac_m: float = 1.0
    dirac_c: float = 1.0
    custom_Delta: float = 0.1
    cutoff_huckel: float = 1.5
    cutoff_heisenberg: float = 2.0
    cutoff_hubbard: float = 1.2
    cutoff_dirac: float = 0.01
    cutoff_custom: float = 1.4
    cutoff_dirac_field: float = 0.02

    # ❄️ Cooldown
    warmup_step: int = 20
    warmup_buffer: int = 15
    cooldown_ewma_alpha: float = 0.12
    cooldown_target_on: float = 1.0
    cooldown_target_off: float = 0.0
    cooling_intensity_scaling: float = 25.0
    spin_quench_factor: float = 0.2

    # 📡 位相
    alpha_r: float = 1.0
    e_field: jnp.ndarray = field(default_factory=lambda: jnp.array([8.0, 2.0, 5.0]))

    # 🧪 Experiment
    n_steps: int = 60
    project_name: str = "lambda3-fire-chloroferrocene-laser-ionization"
    experiment_types: List[str] = field(default_factory=lambda: ["photo_irradiation", "heating", "pressure"])
    intensities: List[float] = field(default_factory=lambda: [3.0e7, 5.0e4, 0])

# 🎉 Config初期化
CONFIG = Lambda3Fire_tamaki_Config()

# ========================
# 🟢 ✅ グローバル展開（関数間で共有する）
# ========================
embedding_dim = CONFIG.embedding_dim
sigma=CONFIG.sigma
structure_radius_base = CONFIG.structure_radius_base
rho_t0 = CONFIG.rho_t0
cutoff_rho_exponent = CONFIG.cutoff_rho_exponent
cutoff_sigma_exponent = CONFIG.cutoff_sigma_exponent
entropic_spread_coeff = CONFIG.entropic_spread_coeff
entropy_weight = CONFIG.entropy_weight
energy_density_base = CONFIG.energy_density_base
projection_angle = CONFIG.projection_angle
structure_length_ref = CONFIG.structure_length_ref
tau_base = CONFIG.tau_base
alpha_entropy = CONFIG.alpha_entropy
noise_scale=CONFIG.noise_scale
delta_rhoT = CONFIG.quantum_progression_step
HAMILTONIAN_MODE = CONFIG.HAMILTONIAN_MODE
measurement_prob = CONFIG.measurement_prob
heisenberg_J = CONFIG.heisenberg_J
hubbard_t = CONFIG.hubbard_t
hubbard_U = CONFIG.hubbard_U
huckel_alpha = CONFIG.huckel_alpha
huckel_beta = CONFIG.huckel_beta
huckel_r_cut = CONFIG.cutoff_huckel
heisenberg_r_cut = CONFIG.cutoff_heisenberg
hubbard_r_cut = CONFIG.cutoff_hubbard
dirac_r_cut = CONFIG.cutoff_dirac
custom_r_cut = CONFIG.cutoff_custom
dirac_field_r_cut = CONFIG.cutoff_dirac_field
qed_field_r_cut = CONFIG.cutoff_qed_field
dirac_m = CONFIG.dirac_m
dirac_c = CONFIG.dirac_c
custom_Delta = CONFIG.custom_Delta
grid_size = CONFIG.grid_size
grid_extent = CONFIG.grid_extent
qed_alpha = CONFIG.qed_alpha
qed_beta = CONFIG.qed_beta
qed_gamma = CONFIG.qed_gamma
a_mu_field_mode = CONFIG.a_mu_field_mode
boundary_mode = CONFIG.boundary_mode

base_scale = CONFIG.base_scale
strength = CONFIG.strength
observe_prob = CONFIG.observe_prob
noise_scale = CONFIG.noise_scale
global_noise_strength = CONFIG.global_noise_strength
ham_noise_strength = CONFIG.ham_noise_strength
alpha_mixing = CONFIG.alpha_mixing
phase_noise_strength = CONFIG.phase_noise_strength
color_noise_scale = CONFIG.color_noise_scale
charge_noise_scale = CONFIG.charge_noise_scale
split_threshold = CONFIG.split_threshold

warmup_step = CONFIG.warmup_step
warmup_buffer = CONFIG.warmup_buffer
pairwise_a = CONFIG.pairwise_a
pairwise_b = CONFIG.pairwise_b
pairwise_c_decay = CONFIG.pairwise_c_decay
disorder_amplitude = CONFIG.disorder_amplitude

key_global = CONFIG.key_global
temp_beta = CONFIG.temp_beta
alpha_distance = CONFIG.alpha_distance
gamma_charge = CONFIG.gamma_charge
gamma_color = CONFIG.gamma_color
gamma_lambdaF = CONFIG.gamma_lambdaF
sigma_init = CONFIG.sigma_init
target_acceptance_split = CONFIG.target_acceptance_split
target_acceptance_move = CONFIG.target_acceptance_move
target_acceptance_bind = CONFIG.target_acceptance_bind
target_acceptance = CONFIG.target_acceptance
distance_overlap_alpha = CONFIG.distance_overlap_alpha

w_spin = CONFIG.w_spin
w_color = CONFIG.w_color
w_charge = CONFIG.w_charge
w_dist = CONFIG.w_dist
w_lambdaF = CONFIG.w_lambdaF

threshold_ionization = CONFIG.threshold_ionization
threshold_redox = CONFIG.threshold_redox
redox_delta = CONFIG.redox_delta
threshold_excitation = CONFIG.threshold_excitation
threshold_proton_hop = CONFIG.threshold_proton_hop
proton_move_delta = CONFIG.proton_move_delta

lambda_f_bind = CONFIG.lambda_f_bind
lambda_f_move = CONFIG.lambda_f_move
lambda_f_split = CONFIG.lambda_f_split

ema_alpha = CONFIG.ema_alpha
ema_energy_window = CONFIG.ema_energy_window
ema_energy_current_weight = CONFIG.ema_energy_current_weight
ema_energy_history_weight = CONFIG.ema_energy_history_weight
ema_score_window = CONFIG.ema_score_window
ema_score_current_weight = CONFIG.ema_score_current_weight
ema_score_history_weight = CONFIG.ema_score_history_weight

cooldown_ewma_alpha = CONFIG.cooldown_ewma_alpha
cooldown_target_on = CONFIG.cooldown_target_on
cooldown_target_off = CONFIG.cooldown_target_off
cooling_intensity_scaling = CONFIG.cooling_intensity_scaling
spin_quench_factor = CONFIG.spin_quench_factor

spin_flip_interval = CONFIG.spin_flip_interval
spin_flip_base_prob = CONFIG.spin_flip_base_prob
spin_flip_split_decay = CONFIG.spin_flip_split_decay

alpha_r = CONFIG.alpha_r
e_field = CONFIG.e_field

n_steps = CONFIG.n_steps
project_name = CONFIG.project_name
experiment_types = CONFIG.experiment_types
intensities = CONFIG.intensities
T = CONFIG.intensities[CONFIG.experiment_types.index("heating")]
I = CONFIG.intensities[CONFIG.experiment_types.index("photo_irradiation")]
beta_spin_flip = CONFIG.beta_spin_flip
q_lambda_jump_threshold = CONFIG.q_lambda_jump_threshold
k_vector_update_method = CONFIG.k_vector_update_method
k_vector_norm_eps = CONFIG.k_vector_norm_eps

# Utility function for safe division
def safe_divide(numerator, denominator, eps=1e-8):
    return numerator / (denominator + eps)

# ========================
# 🟢 ✅ 基本処理関数
# ========================
@jax.jit
def compute_dynamic_structure_radius(rho_T, sigma_s, T, I):
    base_radius = structure_radius_base

    # --- 物理的スケーリングの推奨例 ---
    temp_extension  = 0.002 * jnp.clip(T, 0, 5000)         # [0, 10] くらい
    laser_extension = 0.005 * jnp.sqrt(jnp.clip(I, 0, 1e8)) # [0, 50] くらい
    sigma_extension = jnp.where(sigma_s < 0.5, (0.5 - sigma_s) * 2.5, 0.0)  # ～1.25
    rho_extension   = jnp.clip(rho_T / 20.0, 0.0, 3.0)       # ～3.0くらい

    radius = base_radius + temp_extension + laser_extension + sigma_extension + rho_extension
    radius = jnp.clip(radius, 1.2, 8.0)

    # 必要なら、最小/最大制限も
    radius = jnp.clip(radius, base_radius * 0.8, base_radius * 30)
    return radius

@jax.jit
def compute_dynamic_cutoff(r: jnp.ndarray, step: int, T: float, I: float):
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1)) + jnp.eye(n_el) * 1e10

    # (1) まず仮のmaskでsigma_s計算
    temp_mask = dists < CONFIG.structure_radius_base
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * temp_mask
    sigma_s = jnp.sum(sigma_s_ij) / (jnp.sum(temp_mask) + 1e-8)

    # (2) dynamic_radiusを計算
    rho_T = jnp.sum(1.0 / dists, where=dists < 10.0) / n_el
    dynamic_radius = compute_dynamic_structure_radius(rho_T, sigma_s, T, I)

    # (3) dynamic_radiusでmask再計算
    mask = dists < dynamic_radius
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * mask
    sigma_s = jnp.sum(sigma_s_ij) / (jnp.sum(mask) + 1e-8)

    mask_sum = jnp.sum(mask)

    # (4) cutoffもdynamic_radiusベースで
    cutoff = dynamic_radius * (rho_T / rho_t0) ** cutoff_rho_exponent * sigma_s ** cutoff_sigma_exponent
    return cutoff

def compute_rhoT_from_temperature(T: float) -> float:
    """
    意味エネルギー密度を温度Tから計算。室温(300 K)を基準にスケール。
    Λ³流スケール設計で、時間次元を持たず意味駆動量として整合。
    """
    k_B = 8.617e-5  # Boltzmann 定数 (eV/K)
    T_ref = 300.0   # 室温 (K)
    scale_factor = 1e4  # eVスケールへの調整
    rhoT = scale_factor * k_B * T / T_ref * (T / T_ref) ** 0.5  # 非線形増幅
    return jnp.clip(rhoT, 0.0, 100.0)  # 上限を緩和

def compute_spin_flip_probability(T):
    """
    温度Tからスピンフリップ確率を計算（意味駆動設計）。
    CONFIG.beta_spin_flip を使って、熱自由度スケールを意味論的に管理。
    """
    return jnp.clip(beta_spin_flip * T, 0.0, 1.0)

# ========================
# 🟢 ✅ 量子選択関数
# ========================
def get_hamiltonian(
    *,
    i=None, r=None, Lambda=None, psi=None, identity_ids=None,
    ix=None, iy=None, iz=None,
    Lambda_field=None, psi_field=None,
    A_mu_field=None, F_mu_nu_field=None,
    gammas=None,
    r_field=None
):
    """
    汎用ハミルトニアン選択関数。
    グローバル定数/Config参照。
    HAMILTONIAN_MODEの値により「粒子系」「場系」分岐。
    必要な引数だけ個別に渡す設計に統一！（他はグローバル参照）
    """

    mode = HAMILTONIAN_MODE  # グローバルに展開された現在のモード

    # === 1. 粒子モード ===
    PARTICLE_MODES = {
        "heisenberg": select_hamiltonian_heisenberg,
        "hubbard":    select_hamiltonian_hubbard,
        "huckel":     select_hamiltonian_huckel,
        "custom":     select_hamiltonian_custom,
    }

    # === 2. 粒子Diracは粒子系で分岐 ===
    if mode in PARTICLE_MODES:
        if None in (i, r, Lambda, psi, identity_ids):
            raise ValueError("get_hamiltonian: Missing arguments for particle mode")
        return PARTICLE_MODES[mode](i, r, Lambda, psi, identity_ids)

    elif mode == "dirac":
        # 粒子Dirac用（通常の粒子系扱い、ただし2x2や4x4対応）
        if None in (i, r, Lambda, psi, identity_ids):
            raise ValueError("get_hamiltonian: Missing arguments for dirac (particle) mode")
        return select_hamiltonian_dirac(i, r, Lambda, psi, identity_ids)

    # === 3. 場モード ===
    elif mode == "dirac_field":
        # Dirac場（格子場）モード
        if None in (ix, iy, iz, psi_field, Lambda_field, r_field):
            raise ValueError("get_hamiltonian: Missing arguments for dirac_field mode")
        return select_hamiltonian_dirac_field(
            ix, iy, iz, psi_field, Lambda_field, r_field
        )

    elif mode == "qed_field":
        # QED場（格子場＋電磁場）モード
        if None in (ix, iy, iz, Lambda_field, psi_field, A_mu_field, F_mu_nu_field, r):
            raise ValueError("get_hamiltonian: Missing arguments for qed_field mode")
        return select_hamiltonian_qed_field(
            ix, iy, iz,
            Lambda_field, psi_field,
            A_mu_field, F_mu_nu_field,
            r, gammas  # gammasはグローバルまたは明示でOK
        )

    else:
        raise ValueError(f"get_hamiltonian: Unknown HAMILTONIAN_MODE: {mode}")

# ========================
# 🟢 ✅ 次元選択関数
# ========================

def get_quantum_state_dim(mode: str) -> int:
    # モードごとに適切な状態空間次元を返す
    if mode == "heisenberg":
        return 2
    elif mode == "hubbard":
        return 4
    elif mode == "custom":
        return 2
    elif mode == "huckel":
        return 2
    elif mode == "dirac_field":
        return 2
    elif mode == "dirac":
        return 2  # 2や4どちらでも設計OK
    elif mode == "qed_field":
        return 2  # 必要に応じて4や8にも拡張可能
    else:
        raise ValueError(f"Unknown HAMILTONIAN_MODE: {mode}")

def pad_hamiltonian(H, size):
    if H.shape[0] == size and H.shape[1] == size:
        return H
    print(f"⚠️ [pad_hamiltonian] 異常サイズ発見: H.shape={H.shape}, size={size}")
    H_padded = jnp.zeros((size, size), dtype=H.dtype)
    H_padded = H_padded.at[:H.shape[0], :H.shape[1]].set(H)
    return H_padded

# ========================
# 🟢 ✅ heisenberg関数
# ========================

@jax.jit
def select_hamiltonian_heisenberg(i, r, Lambda, psi, identity_ids):
    try:
        n_el = r.shape[0]
        S_i = Lambda[i, :2, :2]
        quantum_state_dim = Lambda.shape[-1]  # ← ここで動的取得OK！
        H = jnp.zeros((2, 2), dtype=jnp.complex64)
        dists = jnp.linalg.norm(r - r[i], axis=1)
        dists = jnp.where(jnp.arange(n_el) == i, jnp.inf, dists)  # 自己相互作用除外
        max_neighbors = 3
        neighbor_indices = jnp.argsort(dists)[:max_neighbors]
        neighbor_mask = dists[neighbor_indices] < heisenberg_r_cut
        valid_neighbors = jnp.where(neighbor_mask, neighbor_indices, n_el)  # n_elは絶対に無効インデックス！

        def add_neighbor_term(carry_H, j):
            # n_elだったら何もしない
            do_add = (j < n_el)
            S_j = jnp.where(do_add, Lambda[j, :2, :2], jnp.zeros((2, 2), dtype=jnp.complex64))
            term = jnp.where(
                do_add,
                heisenberg_J * (S_i[0, 1] * S_j[0, 1] + S_i[1, 0] * S_j[1, 0]),
                0.0
            )
            return carry_H + term, None

        H, _ = jax.lax.scan(add_neighbor_term, H, valid_neighbors)
        return pad_hamiltonian(H, size=2)
    except Exception as e:
        context = {
            "i": i,
            "r_shape": str(r.shape),
            "dists_shape": str(dists.shape) if 'dists' in locals() else "N/A",
            "valid_neighbors": str(valid_neighbors) if 'valid_neighbors' in locals() else "N/A"
        }
        log_error(0, "select_hamiltonian_heisenberg", e, context)
        return pad_hamiltonian(jnp.zeros((2, 2), dtype=jnp.complex64))

# ========================
# 🟢 ✅ hubbard関数
# ========================
@jax.jit
def select_hamiltonian_hubbard(i, r, Lambda, psi, identity_ids):
    try:
        n_el = r.shape[0]
        H = jnp.zeros((4, 4), dtype=jnp.complex64)
        quantum_state_dim = Lambda.shape[-1]  # ← ここで動的取得OK！

        # --- 距離計算 ---
        dists = jnp.linalg.norm(r - r[i], axis=1)    # (n_el,)
        dists = jnp.where(jnp.arange(n_el) == i, jnp.inf, dists)  # 自分自身は除外

        max_neighbors = 4  # Hubbard (4x4) 用
        neighbor_indices = jnp.argsort(dists)[:max_neighbors]  # 近い順インデックス(4,)
        mask = dists[neighbor_indices] < hubbard_r_cut          # True/False(4,)
        # 使わないindexは-1に（JAXでバッチできる！）
        valid_neighbors = jnp.where(mask, neighbor_indices, -1)  # (4,)

        def add_neighbor_term(carry_H, idx):
            # idxが有効ならtermを追加（idx >= 0のときだけ）
            return jax.lax.cond(
                idx >= 0,
                lambda _: carry_H - hubbard_t * (
                    jnp.kron(jnp.array([[0, 1], [0, 0]]), jnp.eye(2)) +
                    jnp.kron(jnp.eye(2), jnp.array([[0, 1], [0, 0]]))
                ),
                lambda _: carry_H,
                operand=None
            ), None

        # --- 近傍ごとに項を追加（maskでindex有効のみ演算！）---
        H, _ = jax.lax.scan(add_neighbor_term, H, valid_neighbors)

        # --- オンサイト相互作用（diag）---
        H += hubbard_U * jnp.diag(jnp.array([0, 1, 1, 2]))
        return pad_hamiltonian(H, size=4)

    except Exception as e:
        context = {
            "i": i,
            "r_shape": str(r.shape),
            "dists_shape": str(dists.shape) if 'dists' in locals() else "N/A",
            "valid_neighbors": str(valid_neighbors) if 'valid_neighbors' in locals() else "N/A"
        }
        log_error(0, "select_hamiltonian_hubbard", e, context)
        return pad_hamiltonian(jnp.zeros((4, 4), dtype=jnp.complex64), size=4)

# ========================
# 🟢 ✅ huckel関数
# ========================
@jax.jit
def select_hamiltonian_huckel(i, r, Lambda, psi, identity_ids):
    try:
        n_el = r.shape[0]
        quantum_state_dim = Lambda.shape[-1]   # ← 最初に取得！

        H = huckel_alpha * jnp.eye(quantum_state_dim, dtype=jnp.complex64)
        dists = jnp.linalg.norm(r - r[i], axis=1)
        dists = jnp.where(jnp.arange(n_el) == i, jnp.inf, dists)
        max_neighbors = quantum_state_dim - 1
        neighbor_indices = jnp.argsort(dists)[:max_neighbors]
        mask = dists[neighbor_indices] < huckel_r_cut
        valid_neighbors = jnp.where(mask, neighbor_indices, n_el)
        valid_mask = mask.astype(jnp.int32)

        def update_H(carry_H, idx_mask):
            idx, m = idx_mask
            H_new = jnp.where(
                m & (idx < n_el),
                carry_H.at[0, idx + 1].set(huckel_beta).at[idx + 1, 0].set(huckel_beta),
                carry_H
            )
            return H_new, None

        H, _ = jax.lax.scan(
            update_H,
            H,
            (jnp.arange(max_neighbors), valid_mask),
            length=max_neighbors
        )

        H = jnp.where(jnp.isnan(H) | jnp.isinf(H), huckel_alpha * jnp.eye(quantum_state_dim), H)
        return H
    except Exception as e:
        context = {
            "i": str(i),
            "r_shape": str(r.shape),
            "dists_shape": str(dists.shape) if 'dists' in locals() else "N/A",
            "valid_neighbors": str(valid_neighbors) if 'valid_neighbors' in locals() else "N/A"
        }
        log_error(0, "select_hamiltonian_huckel", e, context)
        return huckel_alpha * jnp.eye(quantum_state_dim, dtype=jnp.complex64)

# ========================
# 🟢 ✅ カスタム関数
# ========================
@jax.jit
def select_hamiltonian_custom(i, r, Lambda, psi, identity_ids):
    H = jnp.array([[0, custom_Delta], [jnp.conj(custom_Delta), 0]], dtype=jnp.complex64)
    # Hのshapeが違ってもpad_hamiltonianで吸収
    return pad_hamiltonian(H, size=2)

# ========================
# 🟢 ✅dirac関数
# ========================
@jax.jit
def select_hamiltonian_dirac(i, r, Lambda, psi, identity_ids):
    n_el = r.shape[0]
    alpha = get_dirac_alpha()   # (3,2,2)
    beta  = get_dirac_beta()    # (2,2)
    quantum_state_dim = Lambda.shape[-1]  # ← ここで動的取得OK！
    # --- 距離探索 ---
    dists = jnp.linalg.norm(r - r[i], axis=1)
    dists = jnp.where(jnp.arange(n_el) == i, jnp.inf, dists)
    max_neighbors = min(quantum_state_dim-1, n_el-1)
    neighbor_indices = jnp.argsort(dists)[:max_neighbors]
    mask = dists[neighbor_indices] < dirac_r_cut
    valid_neighbors = jnp.where(mask, neighbor_indices, n_el)

    # --- 近傍Λ,ψ情報から動的p_vec計算 ---
    # 例：平均Λ[0,1]ベクトルから“運動量”推定
    p_vec = jnp.mean(jnp.where(mask[:, None], Lambda[neighbor_indices, 0, 1], 0.0), axis=0)
    p_vec = jnp.pad(p_vec, (0, 3 - p_vec.shape[0]))  # 安全のため3次元にpad

    H = dirac_c * jnp.tensordot(alpha, p_vec, axes=([0],[0])) + dirac_m * dirac_c**2 * beta

    # 近傍ごとに結合項追加例（Λ情報使って追加設計可）
    def add_neighbor_term(carry_H, j):
        do_add = (j < n_el)
        # 例えばΛのtraceでinteraction項を追加
        neighbor_term = jnp.where(do_add, 0.01 * jnp.trace(Lambda[j]) * jnp.eye(quantum_state_dim), 0.0)
        return carry_H + neighbor_term, None

    H, _ = jax.lax.scan(add_neighbor_term, H, valid_neighbors)
    H = jnp.where(jnp.isnan(H) | jnp.isinf(H), 0.0, H)
    return pad_hamiltonian(H, size=quantum_state_dim)

# ========================
# 🟢 ✅dirac_field関数
# ========================
# jitエラー対策①
def get_dirac_alpha():
    return jnp.array([
        [[0, 1], [1, 0]],
        [[0, -1j], [1j, 0]],
        [[1, 0], [0, -1]]
    ], dtype=jnp.complex64)

# jitエラー対策②
def get_dirac_beta():
    return jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)

# dirac_field処理
@jax.jit
def select_hamiltonian_dirac_field(
    ix, iy, iz, psi_field, Lambda_field, r_field
):
    delta = 1.0

    def gradient_psi(axis):
        shift_pos = jnp.roll(psi_field, -1, axis=axis)[ix, iy, iz]
        shift_neg = jnp.roll(psi_field, +1, axis=axis)[ix, iy, iz]
        return -1j * (shift_pos - shift_neg) / (2 * delta)

    p = jnp.array([jnp.linalg.norm(gradient_psi(a)) for a in range(3)], dtype=jnp.complex64)
    alpha = get_dirac_alpha()
    beta  = get_dirac_beta()

    H = dirac_c * jnp.tensordot(alpha, p, axes=([0],[0])) + dirac_m * dirac_c**2 * beta
    H = jnp.where(jnp.isnan(H) | jnp.isinf(H), 0.0, H)

    # ここでLambdaから状態数取得！
    Lambda = Lambda_field[ix, iy, iz]
    quantum_state_dim = Lambda.shape[-1]
    return pad_hamiltonian(H, size=quantum_state_dim)

# ========================
# 🟢 ✅qed_field関数
# ========================
@jax.jit
def select_hamiltonian_qed_field(
    ix, iy, iz,
    Lambda_field, psi_field,
    A_mu_field, F_mu_nu_field,
    r, gammas  # gammas だけは都度渡す設計OK
):
    # --- 場から値を取得 ---
    Lambda = Lambda_field[ix, iy, iz]
    psi = psi_field[ix, iy, iz]
    A_mu = A_mu_field[ix, iy, iz]
    F_mu_nu = F_mu_nu_field[ix, iy, iz]

    quantum_state_dim = Lambda.shape[-1]

    # --- 各種項の計算（定数はグローバル展開値を参照！）---
    psi_norm = jnp.linalg.norm(psi)
    kinetic_term = jnp.eye(quantum_state_dim, dtype=jnp.complex64) * (psi_norm ** 2)
    mass_term = dirac_m * dirac_c ** 2 * jnp.eye(quantum_state_dim, dtype=jnp.complex64)
    photon_term = qed_beta * jnp.sum(F_mu_nu ** 2) * jnp.eye(quantum_state_dim, dtype=jnp.complex64)

    gamma0 = gammas[0]
    j_mu = psi.conj() @ gamma0 @ psi
    interaction_term = qed_alpha * jnp.real(j_mu * A_mu[0]) * jnp.eye(quantum_state_dim, dtype=jnp.complex64)

    Lambda_term = jnp.trace(Lambda) * jnp.eye(quantum_state_dim, dtype=jnp.complex64) * 0.05

    # --- 距離カットオフもグローバル ---
    grid_pos = jnp.array([ix, iy, iz])
    dists = jnp.sqrt(jnp.sum((r - grid_pos) ** 2, axis=1))
    neighbor_mask = dists < qed_field_r_cut

    # --- ポテンシャル項（マスク付きで結合）---
    potential = jnp.sum(jnp.exp(-dists / structure_length_ref) * neighbor_mask)
    potential_term = potential * jnp.eye(quantum_state_dim, dtype=jnp.complex64)

    # --- 全項合成 ---
    H = kinetic_term + mass_term + photon_term + interaction_term + Lambda_term + potential_term
    H = jnp.where(jnp.isnan(H) | jnp.isinf(H), 0.0, H)
    return H

# ========================
# 🟢 ✅補助関数
# ========================

def initialize_field(grid_size: int, extent: float, quantum_state_dim: int):
    assert quantum_state_dim > 0, "quantum_state_dim must be positive"
    Lambda_field = jnp.zeros((grid_size, grid_size, grid_size, quantum_state_dim, quantum_state_dim), dtype=jnp.complex64)
    psi_field = jnp.zeros((grid_size, grid_size, grid_size, quantum_state_dim), dtype=jnp.complex64)
    Lambda_field = Lambda_field.at[:, :, :, 0, 0].set(1.0)
    psi_field = psi_field.at[:, :, :, 0].set(1.0)
    return Lambda_field, psi_field

def initialize_quantum_state(n_el: int, mode: str, mixed: bool = False):
    qdim = get_quantum_state_dim(mode)
    psi = jnp.zeros((n_el, qdim), dtype=jnp.complex64)
    psi = psi.at[:, 0].set(1.0)
    if not mixed:
        Lambda = jnp.zeros((n_el, qdim, qdim), dtype=jnp.complex64)
        Lambda = Lambda.at[:, 0, 0].set(1.0)
    else:
        Lambda = jnp.tile(jnp.eye(qdim) / qdim, (n_el, 1, 1))
    return psi, Lambda

def compute_rho_T_field(r: jnp.ndarray, grid_size: int, extent: float) -> jnp.ndarray:
    """グリッド上の rho_T 場を計算"""
    grid = jnp.linspace(-extent, extent, grid_size)
    x, y, z = jnp.meshgrid(grid, grid, grid, indexing='ij')
    rho_T_field = jnp.zeros((grid_size, grid_size, grid_size))
    for i in range(r.shape[0]):
        dists = jnp.sqrt((x - r[i, 0])**2 + (y - r[i, 1])**2 + (z - r[i, 2])**2)
        rho_T_field += jnp.exp(-dists / structure_length_ref)
    return jnp.clip(rho_T_field, 1e-5, 10.0)

def get_A(ix, iy, iz, nu, A_mu_field, mode="zero"):
    Nx, Ny, Nz = A_mu_field.shape[:3]
    if mode == "zero":
        ix_ok = (0 <= ix) & (ix < Nx)
        iy_ok = (0 <= iy) & (iy < Ny)
        iz_ok = (0 <= iz) & (iz < Nz)
        in_bounds = ix_ok & iy_ok & iz_ok

        ix = jnp.clip(ix, 0, Nx-1)
        iy = jnp.clip(iy, 0, Ny-1)
        iz = jnp.clip(iz, 0, Nz-1)
        return jnp.where(
            in_bounds,
            A_mu_field[ix, iy, iz, nu],
            0.0
        )
    elif mode == "periodic":
        ix_ = ix % Nx
        iy_ = iy % Ny
        iz_ = iz % Nz
        return A_mu_field[ix_, iy_, iz_, nu]
    else:
        raise ValueError(f"Unknown boundary_mode: {mode}")

def make_A_mu_field(grid_size, crazy_mode=False, pulse_step=0):
    """
    QED格子用のA_mu_fieldを初期化。
    - crazy_mode: Trueならランダムノイズ付き
    - pulse_step: 乱数シード用途（デバッグ・可視化用）
    """
    A_mu = jnp.zeros((grid_size, grid_size, grid_size, 4), dtype=jnp.complex64)
    if crazy_mode:
        # 乱数ノイズで遊びたい場合
        key = jax.random.PRNGKey(pulse_step)
        noise = jax.random.normal(key, A_mu.shape, dtype=jnp.complex64) * 0.01
        A_mu = A_mu + noise
    return A_mu

def compute_f_mu_nu(A_mu_field: jnp.ndarray, ix: int, iy: int, iz: int, mode: str) -> jnp.ndarray:
    delta = 1.0
    F = jnp.zeros((4, 4), dtype=jnp.complex64)
    for mu in range(4):
        for nu in range(4):
            if mu == nu:
                continue
            partial_mu_A_nu = (
                get_A(ix+1, iy, iz, nu, A_mu_field, mode) -
                get_A(ix-1, iy, iz, nu, A_mu_field, mode)
            ) / (2 * delta)
            partial_nu_A_mu = (
                get_A(ix, iy+1, iz, mu, A_mu_field, mode) -
                get_A(ix, iy-1, iz, mu, A_mu_field, mode)
            ) / (2 * delta)
            F = F.at[mu, nu].set(partial_mu_A_nu - partial_nu_A_mu)
    return F

# グリッド用A_mu_field生成
def precompute_F_mu_nu_field(A_mu_field: jnp.ndarray, mode: str) -> jnp.ndarray:
    Nx, Ny, Nz, _ = A_mu_field.shape
    def single(ix, iy, iz):
        return compute_f_mu_nu(A_mu_field, ix, iy, iz, mode)
    vmap_z = jax.vmap(jax.vmap(jax.vmap(single, in_axes=(None, None, 0)), in_axes=(None, 0, None)), in_axes=(0, None, None))
    return vmap_z(jnp.arange(Nx), jnp.arange(Ny), jnp.arange(Nz))  # shape: (Nx, Ny, Nz, 4, 4)

def gamma_matrices(n: int, d: int):
    """
    d次元対応のガンマ行列セットを返す
    n: 返す個数 (空間＋時間成分)
    d: 行列のサイズ（2, 4, ...）
    """
    # 2x2 Pauli行列（d=2用）
    pauli = [
        jnp.array([[0, 1], [1, 0]], dtype=jnp.complex64),
        jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex64),
        jnp.array([[1, 0], [0, -1]], dtype=jnp.complex64)
    ]

    if d == 2:
        return pauli[:n]
    elif d == 4:
        # 標準4x4ディラックガンマ行列
        zero = jnp.zeros((2, 2), dtype=jnp.complex64)
        I = jnp.eye(2, dtype=jnp.complex64)
        sigma_x, sigma_y, sigma_z = pauli
        gamma0 = jnp.block([[I, zero], [zero, -I]])
        gamma1 = jnp.block([[zero, sigma_x], [-sigma_x, zero]])
        gamma2 = jnp.block([[zero, sigma_y], [-sigma_y, zero]])
        gamma3 = jnp.block([[zero, sigma_z], [-sigma_z, zero]])
        return [gamma0, gamma1, gamma2, gamma3][:n]
    elif d > 4:
        # クロネッカー積による再帰構成も設計可能（真に変態的ならここに突っ込もう！）
        raise NotImplementedError("d>4のClifford拡張は要件次第で追加しよう！")
    else:
        raise ValueError(f"サポート外のガンマ行列サイズ: d={d}")

# 例：d=2, d=4
gammas_2 = gamma_matrices(3, 2)  # パウリ
gammas_4 = gamma_matrices(4, 4)  # ディラック標準

# ========================
# 🧠 量子進行方程式の定義
# ========================
@jax.jit
def quantum_evolution_field(Lambda_field, psi_field, H_field, rho_T_field, delta_rhoT):
    grid_shape = Lambda_field.shape[:-2]
    d = Lambda_field.shape[-1]

    # --- kappa（テンション密度逆数）計算・クリップ ---
    kappa = 1.0 / (rho_T_field[..., None, None] + 1e-12)
    kappa = jnp.clip(kappa, -1e6, 1e6)

    # --- ハミルトニアン進化 ---
    commutator = jnp.einsum("...ij,...jk->...ik", H_field, Lambda_field) - \
                 jnp.einsum("...ij,...jk->...ik", Lambda_field, H_field)
    dLambda = -1j * kappa * commutator
    Lambda_field_new = Lambda_field + dLambda * delta_rhoT

    # --- 固有値分解バッチ ---
    def eigh_batch(L):
        eigval, eigvec = jnp.linalg.eigh(L)
        eigval = jnp.nan_to_num(eigval, nan=0.0, posinf=1e6, neginf=-1e6)
        max_idx = jnp.argmax(eigval)
        return eigvec[:, max_idx]

    Lambda_field_flat = Lambda_field_new.reshape(-1, d, d)
    psi_field_new = vmap(eigh_batch)(Lambda_field_flat).reshape(*grid_shape, d)
    Lambda_field_new = Lambda_field_new.reshape(*grid_shape, d, d)

    # --- NaN/Infクリップ ---
    psi_field_new = jnp.nan_to_num(psi_field_new, nan=0.0)
    Lambda_field_new = jnp.nan_to_num(Lambda_field_new, nan=0.0)

    return Lambda_field_new, psi_field_new

@jax.jit
def quantum_evolution(Lambda, psi, H, rho_T, delta_rhoT, compute_grad=False, r=None):
    d = Lambda.shape[0]

    # --- kappaの計算 ---
    kappa = 1.0 / (rho_T + 1e-12)
    # kappaはゼロ化せず、極大クリップ（非物理領域も極端なまま流す）
    kappa = jnp.nan_to_num(kappa, nan=0.0, posinf=1e6, neginf=-1e6)

    # --- ハミルトニアン進化 ---
    commutator = H @ Lambda - Lambda @ H
    dLambda = -1j * kappa * commutator
    Lambda_new = Lambda + dLambda * delta_rhoT

    # --- 固有値分解 ---
    eigenvalues, eigenvectors = jnp.linalg.eigh(Lambda_new)
    # 固有値のnan/infもサルベージ（キャンセルよりclip/nan_to_numで分布を活かす）
    eigenvalues = jnp.nan_to_num(eigenvalues, nan=0.0, posinf=1e6, neginf=-1e6)
    max_idx = jnp.argmax(eigenvalues)
    psi_new = eigenvectors[:, max_idx]

    # --- grad_path（ここは物理分岐そのまま）---
    def grad_path(psi_new_inner):
        sigma = 0.1
        psi_grad = -psi / (sigma**2) * r[0]
        psi_new_grad = psi_new_inner + delta_rhoT * psi_grad
        norm = jnp.linalg.norm(psi_new_grad)
        return jnp.where(norm < 1e-12, psi_new_inner, psi_new_grad / (norm + 1e-12))

    psi_new = jax.lax.cond(
        compute_grad,
        grad_path,
        lambda x: x,
        psi_new
    )

    # --- psi/Lambdaのnan/infもキャンセルではなくnan_to_num ---
    psi_new = jnp.nan_to_num(psi_new, nan=0.0, posinf=1e6, neginf=-1e6)
    Lambda_new = jnp.nan_to_num(Lambda_new, nan=0.0, posinf=1e6, neginf=-1e6)

    return Lambda_new, psi_new

#観測データ（例：DFT軌道、実験波動関数）が得られたら、以下を実装：
#def interpolate_psi_grad(r, data):
    # 例：スプライン補間やニューラルネットワーク
 #   return jax.grad(lambda r: data_interpolator(r))(r)

# ========================
# 🧠 エントロピーの定義
# ========================

@jit
def compute_entanglement_entropy(Lambda):
    eigenvalues = jnp.linalg.eigh(Lambda)[0]
    eigenvalues = jnp.clip(jnp.real(eigenvalues), 1e-8, 1.0)
    return -jnp.sum(eigenvalues * jnp.log(eigenvalues), axis=-1)

@jit
def compute_partial_entropy(Lambda, i):
    # 例えば1粒子部分系
    Lambda_partial = Lambda[jnp.ix_(jnp.array([i]), jnp.array([i]))]
    eigenvalues = jnp.linalg.eigh(Lambda_partial)[0]
    eigenvalues = jnp.clip(jnp.real(eigenvalues), 1e-8, 1.0)
    return -jnp.sum(eigenvalues * jnp.log(eigenvalues))

# ========================
# 🧠 ノイズと観測の定義
# ========================
@jit
def compute_local_noise_scale(lambda_f, sigma_s, delta_lambda, base_scale):
    # 進行テンソルのノルムでゆらぎ増大（ex: モード切替/活性度UP）
    norm_lambda_f = jnp.linalg.norm(lambda_f, axis=1)
    # 同期率が低いほどノイズ大きめ
    sync_factor = 1.0 - jnp.clip(sigma_s, 0.0, 1.0)
    # ΔΛC（進行）の大きさでトリガー
    progress_factor = jnp.clip(delta_lambda / (jnp.max(delta_lambda)+1e-8), 0, 1)
    return base_scale * (1.0 + norm_lambda_f + sync_factor + progress_factor)

def get_global_noise(n_el, strength, key):
    noise = strength * jax.random.normal(key, shape=(n_el,))
    return noise

def get_external_field(step, n_el, key, field_type="magnetic", strength=0.1):
    if field_type == "magnetic":
        field = strength * jax.random.normal(key, (n_el, 3))
    elif field_type == "electric":
        ...
    # 他のfield_type対応
    return field

def random_observe_mask(n_el, observe_prob, key, group=None, region=None):
    # group: クラスタ（例：色/距離/任意分割）、region: 物理空間block ((x_min, x_max), ...)
    if region is not None:
        # region指定時のみその領域を観測
        mask = jnp.zeros(n_el, dtype=bool)
        # region: (start_idx, end_idx)など粒子番号範囲 or 位置条件で作る
        mask = mask.at[region[0]:region[1]].set(True)
        return mask
    if group is not None:
        # group: クラスタごと確率変化 etc
        ...
    # デフォルト：完全ランダム観測
    return jax.random.uniform(key, (n_el,)) < observe_prob

def random_projector(dim, key, noise_scale=0.1):
    """
    dim: 次元（例: 2）
    noise_scale: ノイズの強さ
    key: JAXのPRNGKey
    """
    # 基本は|0⟩⟨0|
    base_proj = jnp.zeros((dim, dim), dtype=jnp.complex64)
    base_proj = base_proj.at[0, 0].set(1.0)

    # ノイズ成分（Hermitianを保つ）
    noise_real = jax.random.normal(key, shape=(dim, dim)) * noise_scale
    noise_imag = jax.random.normal(key, shape=(dim, dim)) * noise_scale
    noise = noise_real + 1j * noise_imag
    # Hermitian化
    noise = (noise + noise.conj().T) / 2

    projector = base_proj + noise
    # 正規化（エルミート性とtrace=1調整したい場合は↓）
    projector = (projector + projector.conj().T) / 2
    projector = projector / jnp.trace(projector)
    return projector

def physical_projector(dim, key, axis_vec=None, noise_scale=0.5):
    """
    物理的な射影行列（軸指定＋微小ノイズ付与）
    """
    # === 軸ベクトルの自動生成 ===
    if axis_vec is None:
        # 単位ベクトルを1成分ランダム選択（単位ベクトル基底射影: |0⟩ or |1⟩ or |2⟩...）
        axis_vec = jnp.zeros((dim,), dtype=jnp.complex64).at[0].set(1.0)
    else:
        axis_vec = jnp.asarray(axis_vec, dtype=jnp.complex64)
        axis_vec = axis_vec / (jnp.linalg.norm(axis_vec) + 1e-12)

    base_proj = jnp.outer(axis_vec, jnp.conj(axis_vec))  # (dim, dim)
    # --- ノイズ ---
    noise_real = jax.random.normal(key, shape=(dim, dim)) * noise_scale
    noise_imag = jax.random.normal(key, shape=(dim, dim)) * noise_scale
    noise = noise_real + 1j * noise_imag
    noise = (noise + noise.conj().T) / 2  # Hermitian

    projector = base_proj + noise
    projector = (projector + projector.conj().T) / 2
    projector = projector / (jnp.trace(projector) + 1e-12)
    return projector.astype(jnp.complex64)

def prepare_projector(Lambda, projector):
    return projector @ Lambda @ projector.conj().T

def make_batched_update_keys(keys, n):
    def split_single_key(base_key):
        key_proj, key_eval = jax.random.split(base_key)
        # return UpdateKeyStruct(projector=key_proj, eigenvector=key_eval)
        return {'projector': key_proj, 'eigenvector': key_eval}
    # vmapでdictのリストになる（JAX的にはdict of arraysで管理OK！）
    return jax.vmap(split_single_key)(keys)

@partial(jax.vmap, in_axes=(0, 0, 0, 0, 0, 0, 0))
def update_particle_vmap(Lambda_i, psi_i, axis_vec, noise_scale, observe_flag, key_proj, key_eval):
    return update_particle(Lambda_i, psi_i, axis_vec, noise_scale, observe_flag, key_proj, key_eval)

def update_particle(Lambda_i, psi_i, axis_vec, noise_scale, observe_flag, key_proj, key_eval):
    alpha = getattr(CONFIG, "alpha_mixing", 0.8)

    # 🎯 引数順注意！（key_projは第2引数）
    projector = physical_projector(Lambda_i.shape[-1], key_proj, axis_vec, noise_scale=noise_scale)
    projector_prepared = prepare_projector(Lambda_i, projector)

    Lambda_new = jax.lax.cond(
        observe_flag,
        lambda _: alpha * measurement_collapse(Lambda_i, projector_prepared) + (1 - alpha) * Lambda_i,
        lambda _: Lambda_i,
        operand=None
    )
    psi_new = jax.lax.cond(
        observe_flag,
        lambda _: random_eigenvector(Lambda_new, key_eval),
        lambda _: psi_i,
        operand=None
    )

    return Lambda_new, psi_new

def random_eigenvector(L, key):
    eigvals, eigvecs = jnp.linalg.eigh(L)
    probs = jnp.abs(eigvals) / (jnp.sum(jnp.abs(eigvals)) + 1e-12)
    idx = jax.random.choice(key, eigvals.shape[0], p=probs)
    return eigvecs[:, idx]

@jit
def measurement_collapse(Lambda, projector):
    # Lambda: (..., N, N), projector: (..., N, N) ← shape統一済みで渡すこと
    Lambda_measured = jnp.einsum("...ij,jk,...kl->...il", Lambda, projector, projector)
    trace = jnp.trace(Lambda_measured, axis1=-2, axis2=-1)
    Lambda_measured = Lambda_measured / (trace[..., None, None] + 1e-12)
    return Lambda_measured

@jit
def add_hamiltonian_noise(H, ham_noise_strength, key):
    dim = H.shape[-1]
    noise_real = jax.random.normal(key, (dim, dim)) * ham_noise_strength
    noise_imag = jax.random.normal(key, (dim, dim)) * ham_noise_strength
    noise = noise_real + 1j * noise_imag
    noise = (noise + noise.conj().T) / 2  # Hermitian
    return H + noise

# ========================
# 🟢 ✅ 干渉変換関数
# ========================
def experiment_to_transaction_params() -> Tuple[Optional[jnp.ndarray], float]:
    experiment_types = CONFIG.experiment_types
    intensities = CONFIG.intensities

    rhoT_total = 0.0

    for exp_type, intensity in zip(experiment_types, intensities):
        if exp_type == "heating":
            T = intensity
            rhoT = compute_rhoT_from_temperature(T)
        elif exp_type == "electric_field":
            E = intensity
            rhoT = 1e-4 * E  # ←桁を上げてみる
        elif exp_type == "pressure":
            P = intensity
            rhoT = 1e-2 * P  # ←同じく
        elif exp_type == "photo_irradiation":
            I = intensity
            rhoT = 1e-1 * I ** 0.9  # ←強度アップ
        elif exp_type == "cooling":
            T = intensity
            rhoT = -0.1 * T
        else:
            raise ValueError(f"Unsupported experiment type: {exp_type}")
        rhoT_total += rhoT

    # correlation boostも同様に強めてOK
    correlation_rhoT_boost = 0.0
    if "heating" in experiment_types and "photo_irradiation" in experiment_types:
        T = intensities[experiment_types.index("heating")]
        I = intensities[experiment_types.index("photo_irradiation")]
        correlation_rhoT_boost += 0.01 * jnp.sqrt(T * I)
    if "heating" in experiment_types and "pressure" in experiment_types:
        T = intensities[experiment_types.index("heating")]
        P = intensities[experiment_types.index("pressure")]
        correlation_rhoT_boost += 1e-3 * jnp.log1p(P * T)
    if "photo_irradiation" in experiment_types and "pressure" in experiment_types:
        I = intensities[experiment_types.index("photo_irradiation")]
        P = intensities[experiment_types.index("pressure")]
        correlation_rhoT_boost += 5e-4 * jnp.sqrt(I * jnp.log1p(P))
    if "electric_field" in experiment_types and "photo_irradiation" in experiment_types:
        E = intensities[experiment_types.index("electric_field")]
        I = intensities[experiment_types.index("photo_irradiation")]
        correlation_rhoT_boost += 1e-4 * jnp.sqrt(E * I)

    rhoT_ext = rhoT_total + correlation_rhoT_boost

    # クリップを大幅に緩和or一時外して、現象の上限を観察
    rhoT_ext = jnp.clip(rhoT_ext, 0.0, 10000.0)
    LambdaF_ext = None

    return LambdaF_ext, rhoT_ext

@jax.jit
def compute_rho_T_quantum(
    Lambda: jnp.ndarray,
    psi: jnp.ndarray,
    key: jax.Array,
    H: jnp.ndarray = None,
    rhoT_ext: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_el = Lambda.shape[0]

    # ① Λノルム
    structure_norms = jnp.linalg.norm(Lambda, axis=(1, 2))

    # ② エントロピー
    entropy = vmap(compute_entanglement_entropy)(Lambda)

    # ③ ψズレ
    psi_norms = jnp.linalg.norm(psi, axis=1)
    psi_deviation = jnp.abs(psi_norms - jnp.mean(psi_norms))

    # ④ Λの非対角成分（＝拍動源テンソル）
    diag = jnp.diagonal(Lambda, axis1=1, axis2=2)
    identity = jnp.eye(Lambda.shape[-1])[None, :, :]
    diag_matrix = identity * diag[:, None, :]
    Lambda_offdiag = Lambda - diag_matrix
    offdiag_norms = jnp.linalg.norm(Lambda_offdiag, axis=(1, 2))

    # ⑤ ψの構造ゆらぎ
    psi_spread = jnp.var(jnp.real(psi), axis=1) + jnp.var(jnp.imag(psi), axis=1)

    # ⑥ スケーリング係数（分散正規化）
    stds = jnp.array([
        jnp.std(structure_norms) + 1e-8,
        jnp.std(entropy) + 1e-8,
        jnp.std(psi_deviation) + 1e-8
    ])
    inv_var_weights = 1.0 / stds
    weights = inv_var_weights / jnp.sum(inv_var_weights)
    alpha, beta, gamma = weights

    # ⑦ rho_T 構成
    rho_T_q = alpha * structure_norms + beta * entropy + gamma * psi_deviation

    # ⑧ s_gen 構成：拍動源ベクトル（Λ・ψの揺らぎにノイズ追加！）
    base_s_gen = 0.5 * offdiag_norms + 0.5 * psi_spread

    # 🌪️ ノイズ注入：Λ³的トポロジカルゆらぎの意図的再導入
    noise = jax.random.uniform(key, shape=(n_el,), minval=0.0, maxval=0.15)
    s_gen = base_s_gen + noise

    # 🔁 正規化（全体拍動テンションの1スケール化）
    s_gen = s_gen / (jnp.max(s_gen) + 1e-8)

    # ⑨ rho_T にも少し意味テンション（s_gen）混入
    rho_T_q += 0.15 * s_gen

    return rho_T_q, s_gen

@jax.jit
def compute_progression_tensor_full_quantum(
    r_current, spins, charges, colors, k_vectors, psi, Lambda, rho_T, c,
    key_psi, key_lambda,
    step: int = 0,
    phase_noise_strength: float = phase_noise_strength
):
    """
    ψ・Λ両主成分の進行テンソル・位相・主方向を同時計算！

    Returns:
        progression_tensors: Tuple[jnp.ndarray (3,), jnp.ndarray (3,)]  # (psi側, Lambda側)
        phases: Tuple[jnp.ndarray (n_el,), jnp.ndarray (n_el,)]         # (psiの位相, Lambdaの位相)
        main_directions: Tuple[int, int]                                 # (psi, Lambda)
    """
    n_el = r_current.shape[0]

    # ======== 共通項 ========
    dipole_vector = compute_dipole_tensor(r_current, charges)
    dipole_strength = jnp.linalg.norm(dipole_vector)
    lap_term = laplacian_term(r_current)
    lap_strength_scaled = jnp.log1p(jnp.sum(lap_term ** 2))
    dists = jnp.linalg.norm(r_current[:, None, :] - r_current[None, :, :], axis=-1)
    mask = (dists < c) & (dists > 1e-5)
    sigma_s, _ = compute_sigma_s_enhanced(spins, charges, colors, k_vectors, dists, mask)
    sigma_s_mean = jnp.mean(sigma_s)
    spread_measure = jnp.mean(jnp.std(r_current, axis=0))

    # ======== ψ 系統スコア ========
    entropy_psi = jnp.max(vmap(compute_entanglement_entropy)(Lambda))
    psi_norms = jnp.linalg.norm(psi, axis=1)
    psi_var = jnp.var(psi_norms)

    SPLIT_SCALE = 6.0   # ← パワフルな分離を見たいならUP!
    SPLIT_ENTROPY_BOOST = jnp.exp(entropy_psi)  # ← 急カーブブーストもOK

    bind_psi  = sigma_s_mean * (1.0 - entropy_psi)
    move_psi  = psi_var + lap_strength_scaled + spread_measure
    split_psi = jnp.maximum(SPLIT_SCALE * dipole_strength * SPLIT_ENTROPY_BOOST * (1.0 - sigma_s_mean), 1e-5)

    pt_psi = jnp.array([bind_psi + 1e-6, move_psi + 1e-6, split_psi])
    pt_psi /= jnp.sum(pt_psi)
    main_dir_psi = jnp.argmax(pt_psi)

    # ======== Λ 系統スコア ========
    purity = jnp.real(jnp.mean(jnp.trace(jnp.matmul(Lambda, Lambda), axis1=1, axis2=2)))
    entropy_lambda = jnp.mean(vmap(compute_entanglement_entropy)(Lambda))
    lambda_spread = jnp.var(jnp.real(Lambda))

    # Λ系も同様のスケーリング補正を適用
    bind_lambda = purity * sigma_s_mean
    move_lambda = lambda_spread + lap_strength_scaled + spread_measure
    split_lambda = SPLIT_SCALE * dipole_strength * (1.0 + entropy_lambda) * (1.0 - sigma_s_mean)

    pt_lambda = jnp.array([bind_lambda + 1e-6, move_lambda + 1e-6, split_lambda + 1e-6])
    pt_lambda /= jnp.sum(pt_lambda)
    main_dir_lambda = jnp.argmax(pt_lambda)

    # ======== 位相成分（＋ノイズ） ========
    phases_psi = jnp.angle(psi[:, 0])
    eigvals = jnp.linalg.eigvals(Lambda)
    phases_lambda = jnp.angle(eigvals[:, 0])

    phases_psi += jax.random.normal(key_psi, shape=phases_psi.shape) * phase_noise_strength
    phases_lambda += jax.random.normal(key_lambda, shape=phases_lambda.shape) * phase_noise_strength

    # ======== 返却 ========
    return (pt_psi, pt_lambda), (phases_psi, phases_lambda), (main_dir_psi, main_dir_lambda)

@jax.jit
def compute_sigma_s_enhanced(
    spins, charges, colors, lambdaF, dists, mask
):
    """
    Λ³トランザクションの構造同期率σₛ（強化版）
    """
    n_el = spins.shape[0]

    # --- スピン一致（完全一致1.0、不一致0.0） ---
    spin_match = (spins[:, None] == spins[None, :]).astype(float)

    # --- 色荷ベクトルの距離による一致度 ---
    dcolor = colors[:, None, :] - colors[None, :, :]
    color_match = jnp.exp(-gamma_color * jnp.sum(dcolor ** 2, axis=-1))

    # --- 電荷の一致度 ---
    dq = charges[:, None] - charges[None, :]
    charge_match = jnp.exp(-gamma_charge * dq ** 2)

    # --- ΛFベクトル（方向）の一致度 ---
    lambdaF_dot = jnp.sum(lambdaF[:, None, :] * lambdaF[None, :, :], axis=-1)
    lambdaF_norm = jnp.linalg.norm(lambdaF, axis=1, keepdims=True)
    lambdaF_angle_cos = safe_divide(lambdaF_dot, lambdaF_norm * lambdaF_norm.T + 1e-8)
    lambdaF_match = jnp.exp(-CONFIG.gamma_lambdaF * (1.0 - lambdaF_angle_cos))

    # --- 距離によるオーバーラップ ---
    local_overlap = jnp.exp(-alpha_distance * dists ** 2)

    # --- 構造同期率σₛ（現象志向の重み合成）---
    sigma_s = (
        CONFIG.w_spin    * spin_match +
        CONFIG.w_color   * color_match +
        CONFIG.w_charge  * charge_match +
        CONFIG.w_dist    * local_overlap +
        CONFIG.w_lambdaF * lambdaF_match
    ) * mask

    # === 正規化なしで物理現象の自然なスケールを保持 ===
    #   （分離・崩壊イベントの“暴走”をそのままシミュレーションに流す）
    # sigma_s = sigma_s / (jnp.max(sigma_s) + 1e-8) ←これは絶対外す！

    # --- 粒子ごとの平均値や各種統計をlog_dictで返す（現象追跡用） ---
    avg_sigma_s = jnp.sum(sigma_s, axis=1) / (jnp.sum(mask, axis=1) + 1e-8)
    log_dict = {
        "sigma_s_mean": jnp.mean(sigma_s),
        "sigma_s_max":  jnp.max(sigma_s),
        "sigma_s_min":  jnp.min(sigma_s),
        "color_match_mean": jnp.mean(color_match),
    }
    for i in range(min(n_el, 10)):
        log_dict[f"sigma_s_{i}"] = avg_sigma_s[i]

    return sigma_s, log_dict

def max_rho_T_dynamic(step: int) -> float:
    """
    シミュレーション進行に応じてrho_Tの最大値を自動調整
    - 例: 初期は高く、進行に応じて徐々に抑制（物理的には“加熱→冷却”にも応用できる！）
    """
    # 例1: 線形減少
    # return float(10.0 - 0.04 * step) if step < 200 else 2.0
    # 例2: 指数減衰
    return float(10.0 * jnp.exp(-step / 80.0) + 1.0)

# ========================
# 🟢 ✅ 量子判定関数
# ========================
def compute_quantum_progression_scores(
    psi,
    Lambda,
    QLambda=None, QLambda_prev=None,
    psi_prev=None, Lambda_prev=None, pca_components=None,
    split_will=None, partial_entropy=None
):
    """
    量子ハミルトニアン現象のテンソル多軸判定（粒子ごと分離もサポート）
    """

    # 1. 波動関数ノルム・バラツキ
    psi_norms = jnp.linalg.norm(psi, axis=1)
    mean_psi_norm = jnp.mean(psi_norms)
    var_psi_norm = jnp.var(psi_norms)

    # 2. 局所化（Variance小さくなったら局在化判定）
    localization = jnp.clip(0.35 - var_psi_norm, 0.0, 1.0)

    # 3. イオン化（ノルムが平均より外れたら逸脱フラグ）
    ionization = jnp.clip(mean_psi_norm - 1.15, 0.0, 1.0)

    # 4. コヒーレンス/デコヒーレンス（局所エントロピーで評価）
    n_el = Lambda.shape[0]
    partial_entropies = vmap(lambda i: compute_partial_entropy(Lambda, i))(jnp.arange(n_el))
    avg_partial_entropy = jnp.mean(partial_entropies)
    max_partial_entropy = jnp.max(partial_entropies)
    purity = jnp.real(jnp.mean(jnp.trace(jnp.matmul(Lambda, Lambda), axis1=1, axis2=2)))
    coherence = jnp.clip(1.0 - avg_partial_entropy, 0.0, 1.0)
    decoherence = jnp.clip(avg_partial_entropy, 0.0, 2.0)

    # 5. 分岐/分離イベント（トポロジカル＋局所分離）
    if (QLambda is not None) and (QLambda_prev is not None):
        splitting = jnp.clip(jnp.abs(QLambda - QLambda_prev), 0.0, 1.0)
    else:
        splitting = jnp.zeros_like(psi_norms)
    splitting_max = float(jnp.max(splitting))
    splitting_mean = float(jnp.mean(splitting))

    # split_will
    if split_will is not None:
        split_will = jnp.asarray(split_will)
        split_will_max = float(jnp.max(split_will))
        split_will_mean = float(jnp.mean(split_will))
    else:
        split_will_max = 0.0
        split_will_mean = 0.0

    #　partial_entropy
    if partial_entropy is not None:
        partial_entropy = jnp.asarray(partial_entropy)
        partial_entropy_max = float(jnp.max(partial_entropy))
        partial_entropy_mean = float(jnp.mean(partial_entropy))
    else:
        partial_entropy_max = float(max_partial_entropy)
        partial_entropy_mean = float(avg_partial_entropy)

    #　main_splittingは現象に即してweightチューニング
    main_splitting = (
        0.3 * splitting_max
        + 0.4 * split_will_max
        + 0.3 * partial_entropy_max
    )

    # 6. 再結合（再同期：局所エントロピーの急減）
    recombination = 0.0
    if (psi_prev is not None) and (Lambda_prev is not None):
        partial_entropy_prev = vmap(lambda i: compute_partial_entropy(Lambda_prev, i))(jnp.arange(n_el))
        entropy_drop_max = float(jnp.max(partial_entropy_prev - partial_entropies))
        recombination = jnp.where(
            entropy_drop_max > 0.05,
            jnp.clip(entropy_drop_max, 0.0, 1.0),
            0.0
        )

    # 7. 結果まとめ
    quantum_progression_tensor = {
        "ionization": float(ionization),
        "localization": float(localization),
        "coherence": float(coherence),
        "decoherence": float(decoherence),
        "splitting_max": splitting_max,
        "splitting_mean": splitting_mean,
        "split_will_max": split_will_max,
        "split_will_mean": split_will_mean,
        "partial_entropy_max": partial_entropy_max,
        "partial_entropy_mean": partial_entropy_mean,
        "main_splitting": main_splitting,
        "recombination": float(recombination),
        "purity": float(purity),
        "splitting_array": splitting,
    }
    return quantum_progression_tensor

# ========================
# 🟢 ✅ ベクトル化関数
# ========================
@jax.jit
def compute_embedding_quantum(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    k_vectors: jnp.ndarray,
    psi: jnp.ndarray,
    Lambda: jnp.ndarray,
    c: float,
    lambda_f: Optional[jnp.ndarray] = None,     # ←進行テンソルだけでOK
    phase: jnp.ndarray = 1.0,
    s_gen: float = 0.0
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    n_el = r.shape[0]
    h = jnp.zeros((n_el, embedding_dim), dtype=jnp.complex64)

    # --- 粒子間距離・マスク ---
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists2 = jnp.sum((r_i - r_j) ** 2, axis=-1)
    dists = jnp.sqrt(dists2) + jnp.eye(n_el) * 1e10
    mask = (dists < c) & (dists > 0)

    # --- スピン整合性・距離減衰 ---
    spin_align = (spins[:, None] == spins[None, :]).astype(float)
    distance_overlap_alpha_local = distance_overlap_alpha * (1.0 + alpha_entropy * s_gen)
    distance_overlap_alpha_local = jnp.clip(distance_overlap_alpha_local, 1e-4, 10.0)
    sigma_s_ij = jnp.exp(-distance_overlap_alpha_local * dists2) * spin_align * mask
    rhoT_ij = (1.0 / (dists + 1e-8)) * mask
    tau_dynamic = tau_base * (1.0 + alpha_entropy * s_gen)
    A_ij = jnp.exp(-dists2 / tau_dynamic) * mask

    # --- 部分エントロピー ---
    partial_entropies = jax.vmap(lambda i: compute_partial_entropy(Lambda, i))(jnp.arange(n_el))
    psi_phase = jnp.angle(jnp.sum(psi * jnp.conj(psi), axis=1))
    quantum_contrib = partial_entropies + 0.1 * psi_phase

    # --- LambdaFベクトル（進行テンソル主成分のみ！ラベル消去） ---
    if lambda_f is None:
        raise ValueError("lambda_f（進行テンソル主成分）を必ず渡してください")
    lambda_f_extended = jnp.tile(lambda_f, embedding_dim // lambda_f.shape[0] + 1)[:embedding_dim]

    # --- フェーズ対応：粒子ごと or スカラー
    if hasattr(phase, "shape") and phase.shape == (n_el,):
        lambda_f_complex_batched = lambda_f_extended.astype(jnp.complex64)[None, :] * phase[:, None].astype(jnp.complex64)
    else:
        phase_scalar = phase if (isinstance(phase, float) or isinstance(phase, complex)) else jnp.mean(jnp.atleast_1d(phase))
        lambda_f_complex_batched = jnp.tile(lambda_f_extended.astype(jnp.complex64) * phase_scalar, (n_el, 1))

    # --- 埋め込みベクトル構築 ---
    def compute_single_embedding(i):
        contrib_scalar = jnp.sum(sigma_s_ij[i] * rhoT_ij[i] * A_ij[i])
        return contrib_scalar * lambda_f_complex_batched[i] + quantum_contrib[i]

    h = jax.vmap(compute_single_embedding)(jnp.arange(n_el))
    return h, lambda_f

# ========================
# 🟢 ✅ フェルミオン関数
# ========================
def propagate_fermion(fermion, external_field=None, dt=0.01, normalize_velocity=False):
    """
    フェルミオンΛ（電子）の1ステップ進行
    - Δt可変でダイナミクスを柔軟化
    - 速度の正規化はoption（現象を殺さない現象志向設計！）
    """
    pos = fermion["position"]
    vel = fermion["velocity"]
    # 外場（例：E場/B場）を加味
    if external_field is not None:
        vel = vel + external_field(pos)
    pos = pos + vel * dt  # Δt可変
    if normalize_velocity:
        # 強制ノルム1は殺し要素なので、option
        vel = vel / (jnp.linalg.norm(vel) + 1e-8)
    # 現象志向なら正規化しない or ゆるくclipするだけ
    fermion["position"] = pos
    fermion["velocity"] = vel
    return fermion

# 量子系
def register_split_fermion_quantum(r_current, spins, charges, lambda_f, idx, psi, Lambda, clip_velocity=None):
    """
    Λ³量子系—現象志向
    - velocityは正規化せず生値で
    - clip_velocity指定時だけ発散防止clip
    """
    velocity = lambda_f
    if clip_velocity is not None:
        norm = jnp.linalg.norm(velocity)
        if norm > clip_velocity:
            velocity = velocity / (norm + 1e-8) * clip_velocity
    fermion_state = {
        "position": r_current[idx],
        "velocity": velocity,
        "charge": charges[idx],
        "spin": spins[idx],
        "psi": psi[idx],
        "Lambda": Lambda[idx],
        "alive": True
    }
    return fermion_state

# 量子系
@jax.jit
def compute_embedding_for_fermion_quantum(
    fermion: dict,
    r_all: jnp.ndarray,
    spins_all: jnp.ndarray,
    charges_all: jnp.ndarray,
    k_vectors_all: jnp.ndarray,
    psi_all: jnp.ndarray,
    Lambda_all: jnp.ndarray,
    c: float,
    s_gen: float = 0.0
) -> jnp.ndarray:
    # --- 位置情報 ---
    r_fermion = fermion["position"][None, :]
    dists2 = jnp.sum((r_all - r_fermion) ** 2, axis=1)
    dists = jnp.sqrt(dists2) + 1e-8
    mask = (dists < c)

    # --- スピン同期率 ---
    spin_align = (spins_all == fermion["spin"]).astype(float)
    distance_overlap_alpha_local = distance_overlap_alpha * (1.0 + alpha_entropy * s_gen)
    distance_overlap_alpha_local = jnp.clip(distance_overlap_alpha_local, 1e-4, 10.0)
    sigma_s = jnp.exp(-distance_overlap_alpha_local * dists2) * spin_align * mask

    # --- テンション密度・Attention ---
    rhoT = (1.0 / dists) * mask
    tau_dynamic = tau_base * (1.0 + alpha_entropy * s_gen)
    A_ij = jnp.exp(-dists2 / tau_dynamic) * mask

    # --- λF展開 ---
    lambda_f = fermion["velocity"]
    lambda_f_extended = jnp.tile(lambda_f, embedding_dim // 3 + 1)[:embedding_dim]
    lambda_f_complex = lambda_f_extended.astype(jnp.complex64)

    # --- 部分エントロピー主導（i番粒子の部分系） ---
    # どのindexが該当フェルミオンか特定する（通常 "index" キーで保持推奨！）
    fermion_idx = fermion.get("index", None)
    if fermion_idx is not None:
        entropy = compute_partial_entropy(Lambda_all, fermion_idx)
    else:
        # fallback: 全成分 or self-Lambda
        entropy = compute_entanglement_entropy(fermion["Lambda"])

    # --- 埋め込み計算 ---
    contrib_scalar = jnp.sum(sigma_s * rhoT * A_ij) + entropy
    h_fermion = contrib_scalar * lambda_f_complex

    return h_fermion

def compute_fermion_embeddings_quantum(
    fermions, r_all, spins_all, charges_all, k_vectors_all, psi_all, Lambda_all, c, s_gen=0.0
):
    # quantum用としてpsi_all, Lambda_allを渡す形に！
    if isinstance(fermions, dict):
        fermion_list = [fermions] if fermions.get("alive", True) else []
    else:
        fermion_list = [f for f in fermions if f["alive"]]
    if len(fermion_list) == 0:
        return None
    embeddings = jnp.stack([
        compute_embedding_for_fermion_quantum(
            fermion=f,
            r_all=r_all,
            spins_all=spins_all,
            charges_all=charges_all,
            k_vectors_all=k_vectors_all,
            psi_all=psi_all,
            Lambda_all=Lambda_all,
            c=c,
            s_gen=s_gen
        )
        for f in fermion_list
    ])
    return embeddings

# ========================
# 🟢 ✅ フェルミオン分離関数
# ========================
@jax.jit
def compute_score_split_map(r_current, spins, charges, colors, lambda_f, c):
    n_el = r_current.shape[0]

    # 距離計算（対角成分を明示的に除外）
    dists = jnp.linalg.norm(r_current[:, None, :] - r_current[None, :, :], axis=-1)
    mask = (dists < c) & (dists > 1e-5)
    mask &= ~jnp.eye(n_el, dtype=bool)

    # λF拡張処理の明確化
    if lambda_f.shape == (3,):
        lambda_f_expanded = jnp.repeat(lambda_f[None, :], n_el, axis=0)
    else:
        lambda_f_expanded = lambda_f

    # σₛ計算（安定化）
    sigma_s, sigma_log = compute_sigma_s_enhanced(spins, charges, colors, lambda_f_expanded, dists, mask)
    avg_sigma_s = jnp.sum(sigma_s * mask, axis=1) / (jnp.sum(mask, axis=1) + 1e-8)

    # テンション密度ρT計算（局所値）
    rho_T_local = jnp.sum(jnp.where(mask, 1.0 / (dists + 1e-8), 0.0), axis=1)

    # 分離スコア計算（スケール調整＆clip処理）
    split_will = jnp.clip(rho_T_local * (1.0 - avg_sigma_s), 0.0, 1.0)

    # ログ出力（主要な統計値のみ）
    log_dict = {
        "split_will_mean": jnp.mean(split_will),
        "split_will_max": jnp.max(split_will),
        "rho_T_local_mean": jnp.mean(rho_T_local),
        "avg_sigma_s_mean": jnp.mean(avg_sigma_s),
    }

    return split_will, log_dict

@jax.jit
def compute_energy_structural(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    c: float,
    direction: int,
    key: jnp.ndarray  # ← デフォルト値無し
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Λ³現象主義ver. Tuned：
    - 構造→進行→現象 の一貫評価
    - dot整合に ReLU & absolute 分離
    - σₛ広範囲対応化（構造の共鳴最大化）
    - lambda_f 構造主導（構造進行と整合）
    """
    n_el = r.shape[0]
    dR = r[:, None, :] - r[None, :, :]
    dists = jnp.sqrt(jnp.sum(dR ** 2, axis=-1)) + 1e-8

    # === テンション密度（距離 + 波数空間）
    rho_T_ij = 1.0 / dists
    apply_kvec = jnp.logical_not(jnp.all(jnp.isclose(k_vectors, 0.0)))
    rho_T_ij += lax.cond(
        apply_kvec,
        lambda kv: jnp.mean(jnp.cos(jnp.tensordot(kv, dR, axes=[[1], [2]])), axis=0) / (dists + 1e-8),
        lambda _: jnp.zeros_like(rho_T_ij),
        k_vectors
    )

    # === 構造共鳴率 σₛ
    dq = charges[:, None] - charges[None, :]
    dcolor = colors[:, None, :] - colors[None, :, :]
    spin_match = jnp.exp(-((spins[:, None] - spins[None, :]) ** 2) / 0.1)
    charge_match = jnp.exp(-CONFIG.gamma_charge * dq ** 2)
    color_match = jnp.exp(-CONFIG.gamma_color * jnp.sum(dcolor ** 2, axis=-1))
    local_overlap = jnp.exp(-CONFIG.alpha_distance * dists ** 2)
    sigma_s_ij = spin_match * charge_match * color_match * local_overlap

    # === 進行ベクトル λF（key→subで分割）
    key, sub = jax.random.split(key)
    noise = jax.random.normal(sub, shape=(3,)) / jnp.sqrt(CONFIG.temp_beta + 1e-9)
    structure_center = jnp.mean(r, axis=0)
    structure_spread = jnp.mean(r - structure_center, axis=0)
    lambda_f = structure_spread + noise

    # === dot整合評価
    dot_raw = jnp.einsum('ijk,k->ij', dR, lambda_f) / (dists + 1e-8)
    dot_relu = jnp.maximum(dot_raw, 0.0)
    dot_abs = jnp.abs(dot_raw)

    # === energy
    energy_forward = rho_T_ij * sigma_s_ij * dot_relu
    energy_total = rho_T_ij * sigma_s_ij * dot_abs

    E_sum = jnp.sum(energy_forward)
    E_abs = jnp.sum(energy_total)
    E_var = jnp.var(energy_total)

    total_energy = E_abs

    return jnp.nan_to_num(total_energy, nan=0.0), energy_total, key

@jax.jit
def compute_rho_T_localized(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    colors: jnp.ndarray,
    idx: int,
    cutoff_radius: float
) -> Tuple[float, float]:
    ri = r[idx]
    dists = jnp.sqrt(jnp.sum((r - ri) ** 2, axis=1))
    mask = (dists < cutoff_radius) & (dists > 0)

    spin_match = (spins[idx] == spins).astype(float)
    dcolor = colors[idx] - colors
    color_norm2 = jnp.sum(dcolor ** 2, axis=-1)
    color_match = jnp.exp(-CONFIG.gamma_color * color_norm2)
    local_overlap = jnp.exp(-CONFIG.alpha_distance * dists ** 2)  # 可変パラメータで暴走調整OK

    sigma_s = spin_match * color_match * local_overlap * mask # クーロン爆発を殺さないゼロ割保険のみ
    rho_T_base = (1.0 / (dists ** 2 + 1e-12)) * mask
    rho_T_est = jnp.sum(rho_T_base)

    S_gen = jnp.sum(1.0 - sigma_s) / (jnp.sum(mask) + 1e-8)

    return rho_T_est, S_gen

# ========================
# 🟢 ✅ サンプリング関数（Λ³乱数key設計フル対応版）
# ========================
@jax.jit
def adaptive_metropolis_sampling_structural_quantum(
    r0: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    psi: jnp.ndarray,
    Lambda: jnp.ndarray,
    c: float,
    direction: int,
    rng_key: jnp.ndarray = jax.random.PRNGKey(0)
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    n_steps = CONFIG.n_steps
    sigma_init = CONFIG.sigma_init
    temp_beta = CONFIG.temp_beta
    target_acceptance = CONFIG.target_acceptance
    target_acceptance_split = CONFIG.target_acceptance_split
    target_acceptance_move = CONFIG.target_acceptance_move
    target_acceptance_bind = CONFIG.target_acceptance_bind
    ema_alpha = CONFIG.ema_alpha
    phase_noise_strength = getattr(CONFIG, "phase_noise_strength", 0.0)

    def safe_divide(a, b, eps=1e-12):
        return a / (b + eps)

    def single_step(state, step):
        (r, sigma, E_curr, key, acc_cnt, r_samples, sigmas,
         progression_tensor_prev_psi, progression_tensor_prev_lambda, acc_rate_ema,
         colors, charges) = state

        # 🔑 必要なだけ乱数キーを分割（8分割!）
        key, k_prop, k_energy, k_color, k_charge, k_sub, k_psi, k_lambda = jax.random.split(key, 8)
        noise = jax.random.normal(k_prop, shape=r.shape)
        r_proposed = jnp.clip(r + sigma * noise, -5.0, 5.0)

        # --- カラー＆チャージノイズ ---
        colors_proposed = colors + color_noise_scale * jax.random.normal(k_color, shape=colors.shape)
        charges_proposed = charges + charge_noise_scale * jax.random.normal(k_charge, shape=charges.shape)
        colors_proposed = jnp.clip(colors_proposed, 0.0, 1.0)
        charges_proposed = jnp.clip(charges_proposed, -1.0, 1.0)

        # --- エネルギー計算 ---
        E_prop, _, key = compute_energy_structural(
            r_proposed, spins, charges_proposed, colors_proposed, k_vectors, c, direction, key=k_energy
        )

        # --- 進行テンソル等（物理計算） ---
        rho_T, _ = compute_rho_T_quantum(Lambda, psi, key=k_sub, rhoT_ext=0.0)
        (progression_tensor_psi, progression_tensor_lambda), \
        (phases_psi_new, phases_lambda_new), \
        (main_dir_psi, main_dir_lambda) = compute_progression_tensor_full_quantum(
            r_proposed, spins, charges_proposed, colors_proposed, k_vectors,
            psi, Lambda, jnp.mean(rho_T), c,
            key_psi=k_psi, key_lambda=k_lambda,
            step=step,
            phase_noise_strength=phase_noise_strength
        )

        # --- acceptance分岐（Λ³主成分選択方式） ---
        target_acc = lax.switch(
            main_dir_psi,
            [lambda _: target_acceptance_bind,
             lambda _: target_acceptance_move,
             lambda _: target_acceptance_split],
            operand=None
        )

        acc_ratio = safe_divide(E_curr, E_prop)
        accept = jax.random.uniform(k_prop) < acc_ratio
        r_new = lax.select(accept, r_proposed, r)
        E_new = lax.select(accept, E_prop, E_curr)
        colors_new = lax.select(accept, colors_proposed, colors)
        charges_new = lax.select(accept, charges_proposed, charges)
        acc_cnt += accept

        r_samples = r_samples.at[step + 1].set(r_new)
        sigmas = sigmas.at[step + 1].set(sigma)

        def _update_sigma(args):
            acc_cnt, sigma, acc_rate_ema = args
            acc_rate = acc_cnt / 100.0
            acc_rate_ema_new = ema_alpha * acc_rate_ema + (1 - ema_alpha) * acc_rate
            factor = lax.cond(acc_rate_ema_new > target_acc, lambda _: 1.1, lambda _: 0.9, operand=None)
            scale = 1.0 / (1.0 + step / n_steps)
            sigma_new = jnp.clip(sigma * (1 + (factor - 1) * scale), 0.05, 2.0)
            return sigma_new, acc_rate_ema_new

        update_cond = (step + 1) % 100 == 0
        sigma, acc_rate_ema = lax.cond(
            update_cond,
            _update_sigma,
            lambda args: (args[1], args[2]),
            (acc_cnt, sigma, acc_rate_ema)
        )

        return (
            r_new, sigma, E_new, key, acc_cnt, r_samples, sigmas,
            progression_tensor_psi, progression_tensor_lambda, acc_rate_ema,
            colors_new, charges_new
        ), None

    # --- 初期状態準備 ---
    acc_rate_ema_init = 0.5
    r_samples = jnp.zeros((n_steps + 1, *r0.shape))
    r_samples = r_samples.at[0].set(r0)
    sigmas = jnp.zeros(n_steps + 1)
    sigmas = sigmas.at[0].set(sigma_init)

    # 🔑 初期化もすべて分割keyを明示管理
    rng_key, k0_energy, k0_sub, k0_psi, k0_lambda = jax.random.split(rng_key, 5)
    E0, _, rng_key = compute_energy_structural(
        r0, spins, charges, colors, k_vectors, c, direction, key=k0_energy
    )

    rho_T, _ = compute_rho_T_quantum(Lambda, psi, key=k0_sub, rhoT_ext=0.0)

    (progression_tensor_init_psi, progression_tensor_init_lambda), \
    (phase_psi_init, phase_lambda_init), \
    (main_dir_psi_init, main_dir_lambda_init) = compute_progression_tensor_full_quantum(
        r0, spins, charges, colors, k_vectors,
        psi, Lambda, jnp.mean(rho_T), c,
        key_psi=k0_psi, key_lambda=k0_lambda,
        step=0,
        phase_noise_strength=phase_noise_strength
    )

    init_state = (
        r0, sigma_init, E0, rng_key, 0, r_samples, sigmas,
        progression_tensor_init_psi, progression_tensor_init_lambda, acc_rate_ema_init,
        colors, charges
    )

    final_state, _ = lax.scan(single_step, init_state, jnp.arange(n_steps))
    (_, _, _, _, _, r_samples, sigmas, _, _, _, colors_final, charges_final) = final_state

    return r_samples, sigmas, colors_final, charges_final

# ========================
# 🟢 ✅ スピン関数
# ========================
def apply_spin_quench(spins, factor, cooldown_level=1.0):
    quench_amount = factor * cooldown_level
    return spins * (1.0 - quench_amount)

@jax.jit
def randomize_spins(
    spins: jnp.ndarray,
    key: jnp.ndarray,
    pressure_clip: float = 10.0,
    prob_clip: float = 1.5
) -> jnp.ndarray:

    # 1. 温度/圧力パラメータ取得
    T = CONFIG.intensities[CONFIG.experiment_types.index("heating")] if "heating" in CONFIG.experiment_types else 0.0
    flip_prob = compute_spin_flip_probability(T)

    if hasattr(CONFIG, 'cooldown_target_on') and hasattr(CONFIG, 'cooldown_ewma_alpha'):
        cooldown_level = CONFIG.cooldown_target_on
        flip_prob *= (1.0 - cooldown_level / CONFIG.cooldown_target_on)

    if "pressure" in CONFIG.experiment_types:
        P = CONFIG.intensities[CONFIG.experiment_types.index("pressure")]
        flip_prob += 0.005 * jnp.clip(jnp.log1p(P), 0.0, pressure_clip)

    flip_prob = jnp.clip(flip_prob, 0.0, prob_clip)

    # 2. 乱数マスク生成（keyは外部splitして渡す！）
    flip_mask = jax.random.bernoulli(key, p=jnp.minimum(flip_prob, 1.0), shape=spins.shape)
    flipped_spins = spins * jnp.where(flip_mask, -1, 1)
    return flipped_spins

@jit
def compute_dipole_tensor(
    r: jnp.ndarray,
    charges: jnp.ndarray
) -> jnp.ndarray:

    return jnp.sum(r * charges[:, None], axis=0)

# ラプラシアンテンソルを用いた局所構造テンション評価
@jit
def laplacian_term(r: jnp.ndarray) -> jnp.ndarray:
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

@jit
def compute_dipole_vector(r: np.ndarray, charges: np.ndarray) -> np.ndarray:
    return np.sum(r * charges[:, None], axis=0)

def update_k_vectors(
    r,
    method=None,
    charges=None,
    norm_clip: float = None
):
    n_el = r.shape[0]
    method = method or k_vector_update_method

    if method == "center":
        center = np.mean(r, axis=0)
        k_vectors = r - center

    elif method == "dipole":
        if charges is None:
            charges = np.ones(n_el)
        dipole = np.sum(r * charges[:, None], axis=0)
        k_vectors = np.tile(dipole, (n_el, 1))

    elif method == "laplacian":
        lap = laplacian_term(jnp.array(r))
        k_vectors = np.array(lap)

    elif method == "dipole+laplacian":
        if charges is None:
            charges = np.ones(n_el)
        dipole = np.sum(r * charges[:, None], axis=0)
        lap = laplacian_term(jnp.array(r))
        k_vectors = np.array(lap) + np.tile(dipole, (n_el, 1))

    else:
        raise ValueError(f"Unknown method: {method}")

    if norm_clip is not None:
        norm = np.linalg.norm(k_vectors, axis=1, keepdims=True)
        k_vectors = np.where(norm > norm_clip, k_vectors / (norm + 1e-8) * norm_clip, k_vectors)

    return k_vectors

# ========================
# 🟢 ✅ カラーチャージ関数
# ========================
def detect_events(
    metrics,
    threshold_ionization,
    threshold_redox,
    threshold_excitation,
    threshold_proton_hop,
    step=0,  # stepを引数で受け取る
    initial_relax_steps=5,  # 初期現象抑止ステップ数（自由に調整OK）
    relax_scale=15.0,        # 閾値を何倍にするか
):
    # --- 初期数stepは閾値を大幅UP ---
    # stepが初期なら閾値を上げて現象発火しにくくする
    th_ion = threshold_ionization
    th_red = threshold_redox
    th_exc = threshold_excitation
    th_pro = threshold_proton_hop

    if step < initial_relax_steps:
        th_ion *= relax_scale
        th_red *= relax_scale
        th_exc *= relax_scale
        th_pro *= relax_scale

    ionization_mask = metrics["split_will"] > th_ion
    redox_mask      = metrics["charge_transfer"] > th_red
    excitation_mask = metrics["excitation_entropy"] > th_exc
    proton_mask     = metrics["proton_hop_score"] > th_pro

    return ionization_mask, redox_mask, excitation_mask, proton_mask


@jit
def compute_excitation_entropy(Lambda, psi):
    n_el = Lambda.shape[0]
    # Lambda, psiどちらも使いたい場合はvmap2引数対応もOK
    return jax.vmap(lambda idx: compute_partial_entropy(Lambda, idx))(jnp.arange(n_el))

@jit
def compute_charge_transfer_score(charges, r_current, c):
    # 単純な例：近接粒子とのcharge差が大きい＝電子移動活性
    n_el = charges.shape[0]
    dR = r_current[:, None, :] - r_current[None, :, :]
    dists = jnp.sqrt(jnp.sum(dR ** 2, axis=-1)) + jnp.eye(n_el) * 1e10
    neighbor_mask = (dists < c) & (dists > 0)
    # 各粒子の周囲へのcharge変動（合計）
    transfer_score = jnp.sum(jnp.abs(charges[:, None] - charges[None, :]) * neighbor_mask, axis=1)
    return transfer_score

@jit
def compute_proton_movement(proton_mask, proton_move_delta, key):
    # プロトン移動が起きる粒子だけ新しい座標 or +α
    n = len(proton_mask)
    # 仮：ランダム方向に微小シフト
    random_shift = proton_move_delta * jax.random.normal(key, (n, 3))
    move_vec = jnp.where(proton_mask[:, None], random_shift, 0.0)
    return move_vec

@jit
def compute_proton_hop_score(r_current, Lambda):
    # 仮：隣接とのLambda変動 or 局所距離差分など
    n = r_current.shape[0]
    dR = r_current[:, None, :] - r_current[None, :, :]
    dists = jnp.sqrt(jnp.sum(dR**2, axis=-1)) + jnp.eye(n) * 1e10
    neighbor_mask = (dists < 1.5) & (dists > 0)
    hop_score = jnp.sum(neighbor_mask, axis=1) * 0.1  # シンプル例
    return hop_score


@jit
def apply_event_updates(
    charges, colors,
    ionization_mask, redox_mask, excitation_mask, proton_mask,
    redox_delta, proton_move_delta, key
):
    """
    判定マスクに従って物理状態(charges, colors)と励起・プロトン情報を更新
    """
    # イオン化イベント（例：電子放出）
    charges = jnp.where(ionization_mask, charges - 1.0, charges)
    colors = jnp.where(ionization_mask[:, None], ionized_color_update(colors), colors)

    # 酸化還元イベント（例：電子受け渡し）
    charges = jnp.where(redox_mask, charges + redox_delta, charges)

    # 励起イベント（状態フラグ）
    excitation_flags = jnp.where(excitation_mask, 1, 0)

    # プロトン移動イベント（座標更新ロジック）
    key, key_proton, key_other = jax.random.split(key, 3)
    proton_coords = compute_proton_movement(proton_mask, proton_move_delta, key)

    return charges, colors, excitation_flags, proton_coords


@jit
def ionized_color_update(original_colors):
    # 各粒子ごとカラーを80%に減衰（例）
    return original_colors * 0.8

# ========================
# 🟢 ✅ トポロジカル保存則
# ========================

def auto_compute_topological_charge(Lambda_field):
    # 場のshape次元数によって自動切替
    field_shape = Lambda_field.shape

    # 2Dスカラー場
    if len(field_shape) == 2:
        return compute_topological_charge(Lambda_field)

    # 3Dスカラー場（稀）
    elif len(field_shape) == 3 and field_shape[-1] != field_shape[-2]:
        return compute_topological_charge_3d(Lambda_field)

    # 2Dテンソル場
    elif len(field_shape) == 3 and field_shape[-1] == field_shape[-2]:
        # 2Dテンソル場（[X,Y,2,2]など）→ 1枚1枚Q計算して合計/平均
        Qs = [compute_topological_charge(Lambda_field[:, :, i, i]) for i in range(field_shape[-1])]
        return np.mean(Qs)  # またはnp.sum(Qs)

    # 3Dテンソル場（通常）
    elif len(field_shape) == 5 and field_shape[-1] == field_shape[-2]:
        return compute_topological_charge_3d(Lambda_field)

    else:
        raise ValueError(f"Unknown field shape for topological charge: {field_shape}")

def generate_Lambda_field(r, charges, grid_size, grid_extent, sigma, phases=None):
    """
    r: (N, 3) 粒子位置
    charges: (N,) 粒子ごとの重み（密度）
    grid_size: 格子サイズ
    grid_extent: 格子の空間幅
    sigma: 拡がり（ガウス分布幅）
    phases: (N,) 粒子ごとの位相（省略可。Noneなら全0）
    """
    field = np.zeros((grid_size, grid_size), dtype=np.complex64)
    xs = np.linspace(-grid_extent, grid_extent, grid_size)
    ys = np.linspace(-grid_extent, grid_extent, grid_size)
    if phases is None:
        phases = np.zeros(r.shape[0])
    for i in range(r.shape[0]):
        x0, y0 = r[i, 0], r[i, 1]
        ρ = charges[i]
        θ = phases[i]
        # 2Dガウス分布（滑らか密度配置＋位相つき）
        for ix, x in enumerate(xs):
            for iy, y in enumerate(ys):
                dist2 = (x - x0) ** 2 + (y - y0) ** 2
                amp = ρ * np.exp(-dist2 / (2 * sigma ** 2))
                field[ix, iy] += amp * np.exp(1j * θ)
    return field

def compute_topological_charge(Lambda_field):
    """
    2D格子場ΛのトポロジカルチャージQΛを数値的に積分
    例: 1周ループ上のΔarg(Λ)の合計で巻き数算出
    """
    phase = np.angle(Lambda_field)
    # 1格子外周ループを取る（境界部分だけ抽出）
    # 巻き数sum(Δarg) / 2π でトポロジカル保存量
    loop_phase = phase[0, :]  # 上端
    loop_phase = np.concatenate([
        phase[0, :],  # 上端
        phase[1:, -1],  # 右端
        phase[-1, ::-1],  # 下端（逆順）
        phase[-2:0:-1, 0],  # 左端（逆順）
    ])
    delta = np.diff(np.unwrap(loop_phase))
    QLambda = np.round(np.sum(delta) / (2 * np.pi))
    return QLambda

def compute_topological_charge_3d(Lambda_field_3d):
    phase = jnp.angle(jnp.trace(Lambda_field_3d, axis1=-2, axis2=-1))
    def finite_diff(arr, axis): return (jnp.roll(arr, -1, axis=axis) - jnp.roll(arr, 1, axis=axis)) / 2.0
    grad_x = finite_diff(phase, axis=0)
    grad_y = finite_diff(phase, axis=1)
    grad_z = finite_diff(phase, axis=2)
    curl_x = finite_diff(grad_z, axis=1) - finite_diff(grad_y, axis=2)
    curl_y = finite_diff(grad_x, axis=2) - finite_diff(grad_z, axis=0)
    curl_z = finite_diff(grad_y, axis=0) - finite_diff(grad_x, axis=1)
    curl_mag = jnp.sqrt(curl_x**2 + curl_y**2 + curl_z**2)
    Q = jnp.sum(curl_mag) / (2 * jnp.pi)
    return jnp.round(Q)

# ========================
# 🟢 ✅ 構造履歴関数
# ========================
class MinimalBlock:
    def __init__(self, index, step, particle_id, data, previous_hash,
                 position=None, spin=None, charge=None, color=None, Lambda=None,
                 lambda_F=None, lambda_F_psi=None, lambda_F_Lambda=None,
                 rho_T=None, sigma_s=None, divergence=None):
        self.index = index
        self.step = step
        self.particle_id = particle_id
        self.data = data
        self.previous_hash = previous_hash
        self.hash = self.hashing()
        self.position = position
        self.spin = spin
        self.charge = charge
        self.color = color
        self.Lambda = Lambda
        self.lambda_F = lambda_F
        self.lambda_F_psi = lambda_F_psi
        self.lambda_F_Lambda = lambda_F_Lambda
        self.rho_T = rho_T
        self.sigma_s = sigma_s
        self.divergence = divergence

    def hashing(self):
        key = hashlib.sha256()
        # 主要プロパティのみhash
        key.update(str(self.index).encode('utf-8'))
        key.update(str(self.step).encode('utf-8'))
        key.update(str(self.particle_id).encode('utf-8'))
        key.update(str(self.data).encode('utf-8'))
        key.update(str(self.previous_hash).encode('utf-8'))
        return key.hexdigest()

class MinimalTransactionChain:
    def __init__(self, particle_id):
        self.particle_id = particle_id
        self.blocks = [self.get_genesis_block()]

    def get_genesis_block(self):
        return MinimalBlock(
            index=0, step=0, particle_id=self.particle_id,
            data="Genesis", previous_hash="0",
            position=np.zeros(3), spin=0, charge=0, color=0, Lambda=np.zeros((2,2)),
            lambda_F=np.zeros(3),
            lambda_F_psi=np.zeros(3),
            lambda_F_Lambda=np.zeros(3),
            rho_T=1.0, sigma_s=1.0, divergence=0.0
        )

    def add_block(self, step, position, spin, charge, color, Lambda,
                  lambda_F=None, lambda_F_psi=None, lambda_F_Lambda=None,
                  rho_T=None, sigma_s=None, divergence=None):
        prev_block = self.blocks[-1]
        data = {
            "step": step,
            "position": position.tolist(),
            "spin": spin,
            "charge": charge,
            "color": color,
            "Lambda": Lambda.tolist() if Lambda is not None else None,
            "lambda_F": lambda_F.tolist() if lambda_F is not None else None,
            "lambda_F_psi": lambda_F_psi.tolist() if lambda_F_psi is not None else None,
            "lambda_F_Lambda": lambda_F_Lambda.tolist() if lambda_F_Lambda is not None else None,
            "rho_T": rho_T,
            "sigma_s": sigma_s,
            "divergence": divergence,
        }
        new_block = MinimalBlock(
            index=len(self.blocks), step=step, particle_id=self.particle_id,
            data=data, previous_hash=prev_block.hash,
            position=position, spin=spin, charge=charge, color=color,
            Lambda=Lambda, lambda_F=lambda_F,
            lambda_F_psi=lambda_F_psi, lambda_F_Lambda=lambda_F_Lambda,
            rho_T=rho_T, sigma_s=sigma_s, divergence=divergence
        )
        self.blocks.append(new_block)
        return new_block

    def get_chain(self):
        return self.blocks

    def last_divergence(self):
        return self.blocks[-1].divergence

    def verify(self):
        for i in range(1, len(self.blocks)):
            if self.blocks[i].previous_hash != self.blocks[i-1].hash:
                print(f"Block {i} hash mismatch!")
                return False
        return True

# ========================
# 🟢 ✅ ビジュアライズ関数
# ========================
def visualize_lambda_f_with_crystal_axis(
    lambda_f: np.ndarray,
    step: int,
    r: jnp.ndarray,
    dipole_vector: np.ndarray = None,
    filename: str = "lambda_f_visualization.html"
):
    fig = go.Figure()

    # ΛFベクトルの描画（最新のみ！）
    fig.add_trace(go.Scatter3d(
        x=[0, lambda_f[0]], y=[0, lambda_f[1]], z=[0, lambda_f[2]],
        mode='lines+markers',
        line=dict(color='blue', width=6),
        marker=dict(size=6),
        name=f'ΛF vector'
    ))

    # ΛFベクトルの描画
    fig.add_trace(go.Scatter3d(
        x=[0, lambda_f[0]], y=[0, lambda_f[1]], z=[0, lambda_f[2]],
        mode='lines+markers',
        line=dict(color='blue', width=6),
        marker=dict(size=6),
        name=f'ΛF vector (Step {step})'
    ))

    # Cp環の法線ベクトル（z軸）
    cp_normal = jnp.array([0.0, 0.0, 1.0])
    fig.add_trace(go.Scatter3d(
        x=[0, cp_normal[0]], y=[0, cp_normal[1]], z=[0, cp_normal[2]],
        mode='lines',
        line=dict(color='red', width=4, dash='dash'),
        name='Cp ring normal'
    ))

    # Dipoleベクトル（オプション）
    if dipole_vector is not None:
        fig.add_trace(go.Scatter3d(
            x=[0, float(dipole_vector[0])],
            y=[0, float(dipole_vector[1])],
            z=[0, float(dipole_vector[2])],
            mode='lines+markers',
            line=dict(color='orange', width=4),
            marker=dict(size=4),
            name='Dipole Vector'
        ))

    # ラベルの描画（Bind / Move / Split）
    fig.add_trace(go.Scatter3d(
        x=[lambda_f[0]], y=[lambda_f[1]], z=[lambda_f[2]],
        mode='text',
        text=[f"B:{lambda_f[0]:.2f}<br>M:{lambda_f[1]:.2f}<br>S:{lambda_f[2]:.2f}"],
        textposition='top center',
        showlegend=False
    ))

    # レイアウト設定
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

    # 保存
    fig.write_html(filename)

def analyze_lambda_f(lambda_f_matrix: list, log_to_wandb: bool = True, step: int = None):
    if not lambda_f_matrix or len(lambda_f_matrix) < 2:
        print("Warning: Not enough samples for PCA! Skipping PCA analysis and plot.")
        if log_to_wandb:
            wandb.log({"warning": "lambda_f_matrix is empty or insufficient", "step": step or 0})
        return

    data = np.stack(lambda_f_matrix)
    pca = PCA(n_components=2)
    transformed = pca.fit_transform(data)

    print("Principal components of transaction modes (PCA):")
    print(pca.components_)
    print("Explained variance ratio:", pca.explained_variance_ratio_)

    # --- 可視化（plotをwandbに画像として送る） ---
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    ax1.plot(transformed[:, 0], transformed[:, 1], '-o', label="ΛF trajectory", color='b')
    ax1.set_title("ΛF Transaction Mode Trajectory (PCA)")
    ax1.set_xlabel("PC1")
    ax1.set_ylabel("PC2")
    ax1.legend()
    ax1.grid(True)
    fig1.tight_layout()

    fig2, ax2 = plt.subplots(figsize=(6, 3))
    ax2.bar([1, 2], pca.explained_variance_ratio_, color='c')
    ax2.set_xlabel('Principal Components')
    ax2.set_ylabel('Explained Variance Ratio')
    ax2.set_title('Explained Variance by Each Component')
    ax2.set_xticks([1, 2])
    fig2.tight_layout()

    if log_to_wandb:
        log_data = {
            "LambdaF_PCA_Trajectory": wandb.Image(fig1),
            "LambdaF_PCA_Variance": wandb.Image(fig2),
            "LambdaF_PCA_PC1": pca.components_[0].tolist(),
            "LambdaF_PCA_PC2": pca.components_[1].tolist(),
            "LambdaF_PCA_Variance_Ratio": pca.explained_variance_ratio_.tolist(),
        } if step is None else {
            f"LambdaF_PCA_Trajectory/step_{step}": wandb.Image(fig1),
            f"LambdaF_PCA_Variance/step_{step}": wandb.Image(fig2),
            f"LambdaF_PCA_PC1/step_{step}": pca.components_[0].tolist(),
            f"LambdaF_PCA_PC2/step_{step}": pca.components_[1].tolist(),
            f"LambdaF_PCA_Variance_Ratio/step_{step}": pca.explained_variance_ratio_.tolist(),
        }
        wandb.log(log_data)

# ========================
# 🟢 ✅ 量子実行関数
# ========================
def lambda3_fire_vmc_quantum(
    r: jnp.ndarray,
    spins: jnp.ndarray,
    charges: jnp.ndarray,
    colors: jnp.ndarray,
    k_vectors: jnp.ndarray,
    psi: jnp.ndarray,
    Lambda: jnp.ndarray,
    lambdaF_ext: Optional[jnp.ndarray] = None,
    rhoT_ext: float = 0.0,
    adaptive_stop: bool = False,
    fermion_list: Optional[List[dict]] = None,
    key: Optional[jax.Array] = None,
) -> Tuple[jnp.ndarray, List[jnp.ndarray]]:

    global key_global
    # --- state_dimの決定 ---
    quantum_state_dim = get_quantum_state_dim(HAMILTONIAN_MODE)

    # === 入力検証 ===
    n_el = r.shape[0]
    assert spins.shape   == (n_el,)
    assert charges.shape == (n_el,)
    assert colors.shape  == (n_el, 3)
    assert k_vectors.shape == (n_el, 3)
    assert psi.shape     == (n_el, quantum_state_dim)
    assert Lambda.shape  == (n_el, quantum_state_dim, quantum_state_dim)

    # === WandB & 状態履歴 初期化 ===
    wandb.init(project=project_name, config={"n_steps": n_steps, "mode": "quantum"})

    r_current = r
    psi_current = psi
    Lambda_current = Lambda
    lambda_f_history, lambda_f_matrix = [], []
    dipole_history = []
    score_bind_history, score_move_history, score_split_history = [], [], []
    energy_history = []
    QLambda_psi_history = []
    QLambda_Lambda_history = []
    QLambda_psi_prev = None
    QLambda_Lambda_prev = None
    lambda_f_psi_history = []     # ← ψ由来進行ベクトル
    lambda_f_Lambda_history = []
    cooldown_level = 0.0
    cooldown_triggered = False

    # 🟢 各粒子ごとブロックチェーン
    chains = [MinimalTransactionChain(particle_id=i) for i in range(n_el)]
    QLambda_prev = None
    psi_prev = None
    Lambda_prev = None

    # --- 状態のコピー ---
    r_current, psi_current, Lambda_current = r, psi, Lambda
    lambda_f_history, lambda_f_matrix = [], []
    energy_history, dipole_history = [], []

    # === ガンマ行列と境界条件 ===
    gammas = jnp.stack(gamma_matrices(4, quantum_state_dim))
    boundary_mode  = getattr(CONFIG, "boundary_mode", "zero")

    # === 量子場の初期化 ===
    Lambda_field = psi_field = None
    A_mu_field = F_mu_nu_field = None

    if HAMILTONIAN_MODE in ["heisenberg", "huckel", "hubbard", "custom", "dirac"]:
        psi_current, Lambda_current = initialize_quantum_state(n_el, HAMILTONIAN_MODE)
    elif HAMILTONIAN_MODE == "dirac_field":
        Lambda_field, psi_field = initialize_field(grid_size, grid_extent, quantum_state_dim)
    elif HAMILTONIAN_MODE == "qed_field":
        Lambda_field, psi_field = initialize_field(grid_size, grid_extent, quantum_state_dim)
        A_mu_field = make_A_mu_field(grid_size, crazy_mode=False, pulse_step=0)
        F_mu_nu_field = precompute_F_mu_nu_field(A_mu_field, mode=boundary_mode)

    # === メッシュグリッド ===
    coords = jnp.linspace(-grid_extent, grid_extent, grid_size)
    x, y, z = jnp.meshgrid(coords, coords, coords, indexing='ij')
    r_field = jnp.stack([x, y, z], axis=-1)

    # === 外部パラメータ ===
    if rhoT_ext == 0.0:
        _, rhoT_ext = experiment_to_transaction_params()

    # === 差分系の初期化 ===
    r_prev = r_current.copy()
    Lambda_prev = Lambda_current.copy()
    spins_prev = spins.copy()
    charges_prev = charges.copy()
    colors_prev = colors.copy()

    # =========================================================
    # 🔁 メインループ
    # =========================================================
    measurement_events = 0

    try:
        for step in range(1, n_steps):
            # 🔑 stepごとに必要keyを全部分割しておく（この例では5つ！）
            key, sub_rhoT, sub_prog_psi, sub_prog_lambda, sub_event = jax.random.split(key, 5)

            # ① 動的カットオフ・k ベクトル更新
            c = compute_dynamic_cutoff(r_current, step, T, I)
            print(f"step={step}, r_min={np.min(r_current):.3f}, r_max={np.max(r_current):.3f}, dynamic_radius={c:.3f}")
            k_vectors = update_k_vectors(np.array(r_current),
                                        method=CONFIG.k_vector_update_method,
                                        charges=np.array(charges))

            # ② rho_T
            rho_T_raw, s_gen = compute_rho_T_quantum(Lambda_current, psi_current, key=sub_rhoT, rhoT_ext=rhoT_ext)
            rho_T = jnp.clip(rho_T_raw, 1e-5, max_rho_T_dynamic(step))
            print(f"[rho_T_quantum] step={step}  rho_T_raw={rho_T_raw}  rho_T_clipped={rho_T}  s_gen={s_gen}")

            # ③ 進行テンソル（方向スコア）: keyを2つに分割して渡す
            (progression_tensor_psi, progression_tensor_lambda), \
            (phases_psi, phases_lambda), \
            (main_dir_psi, main_dir_lambda) = compute_progression_tensor_full_quantum(
                r_current, spins, charges, colors, k_vectors,
                psi_current, Lambda_current, jnp.mean(rho_T), c,
                key_psi=sub_prog_psi,
                key_lambda=sub_prog_lambda,
                step=step,
                phase_noise_strength=phase_noise_strength
            )

            lambda_f = progression_tensor_psi / (jnp.sum(progression_tensor_psi) + 1e-12)
            lambda_f_Lambda = progression_tensor_lambda / (jnp.sum(progression_tensor_lambda) + 1e-12)
            lambda_f_diff = jnp.abs(lambda_f - lambda_f_Lambda)

            # 🟢 split_will/現象スコア計算（変わらず）
            split_will, _ = compute_score_split_map(
                r_current, spins, charges, colors, lambda_f, c
            )
            avg_split_score = float(jnp.mean(split_will))
            max_split_score = float(jnp.max(split_will))
            split_idx = int(jnp.argmax(split_will)) if len(split_will) > 0 else None

            # スコア類計算
            charge_transfer_score = compute_charge_transfer_score(charges, r_current, c)
            excitation_entropy = compute_excitation_entropy(Lambda_current, psi_current)
            proton_hop_score = compute_proton_hop_score(r_current, Lambda_current)

            metrics = {
                "split_will": split_will,
                "charge_transfer": charge_transfer_score,
                "excitation_entropy": excitation_entropy,
                "proton_hop_score": proton_hop_score,
            }

            # ② 判定
            ionization_mask, redox_mask, excitation_mask, proton_mask = detect_events(
                metrics,
                CONFIG.threshold_ionization,
                CONFIG.threshold_redox,
                CONFIG.threshold_excitation,
                CONFIG.threshold_proton_hop,
                step=step,  # ←渡す
            )

            # ③ 状態更新：ここもsub_eventでkeyを渡す（内部でさらにsplitしてOK）
            charges, colors, excitation_flags, proton_coords = apply_event_updates(
                charges, colors,
                ionization_mask, redox_mask, excitation_mask, proton_mask,
                redox_delta, proton_move_delta, key=sub_event
            )

            # 4. 追加の出力や記録
            metrics.update({
                "excitation_flags": excitation_flags,
                "proton_coords": proton_coords
            })

            # ---- 結果をprintで可視化 ----
            print(f"\n=== [Step {step}] Event Summary ===")
            for i in range(len(charges)):
                msg = f"[Particle {i}] "
                if ionization_mask[i]:
                    msg += "IONIZED! "
                if redox_mask[i]:
                    msg += "REDOX! "
                if excitation_mask[i]:
                    msg += "EXCITED! "
                if proton_mask[i]:
                    msg += "PROTON HOP! "
                msg += f"charge={charges[i]:.2f}, color={colors[i]}, excitation={excitation_flags[i]}, proton_move={proton_coords[i]}"
                print(msg)
            print(f"metrics (split/max): {np.max(metrics['split_will']):.3f}, (redox/max): {np.max(metrics['charge_transfer']):.3f}, (excitation/max): {np.max(metrics['excitation_entropy']):.3f}, (proton/max): {np.max(metrics['proton_hop_score']):.3f}")

            # （任意）記録/可視化用
            #record_results(step, metrics, r_current, spins, charges, colors, psi_current, Lambda_current)

            # =========================================================
            # ④ ハミルトニアン計算 & 量子進行
            # =========================================================
            key_global, subkey = jax.random.split(key_global)
            if HAMILTONIAN_MODE == "qed_field":
                rho_T_field = compute_rho_T_field(r_current, grid_size, grid_extent)

                def ham_for_idx(idx, key):
                    ix = (idx // (grid_size * grid_size)) % grid_size
                    iy = (idx // grid_size) % grid_size
                    iz = idx % grid_size
                    H = get_hamiltonian(
                        ix=ix, iy=iy, iz=iz,
                        Lambda_field=Lambda_field, psi_field=psi_field,
                        A_mu_field=A_mu_field, F_mu_nu_field=F_mu_nu_field,
                        r=r_current, gammas=gammas
                    )
                    return add_hamiltonian_noise(H, CONFIG.ham_noise_strength, key)

                field_size = grid_size ** 3
                # 🟢 subkeyから必要数だけsplitして全field siteのkey配列をつくる
                keys = jax.random.split(subkey, field_size)
                H_field = jax.vmap(ham_for_idx)(jnp.arange(field_size), keys).reshape(
                    grid_size, grid_size, grid_size, quantum_state_dim, quantum_state_dim
                )
                Lambda_field, psi_field = quantum_evolution_field(
                    Lambda_field, psi_field, H_field, rho_T_field, delta_rhoT
                )
                Lambda_current = jnp.stack([Lambda_field[0, 0, 0]] * n_el)
                psi_current    = jnp.stack([psi_field[0, 0, 0]] * n_el)

            elif HAMILTONIAN_MODE == "dirac_field":
                rho_T_field = compute_rho_T_field(r_current, grid_size, grid_extent)

                def ham_for_idx(idx, key):
                    ix = (idx // (grid_size ** 2)) % grid_size
                    iy = (idx // grid_size) % grid_size
                    iz = idx % grid_size
                    H = get_hamiltonian(
                        ix=ix, iy=iy, iz=iz,
                        psi_field=psi_field, Lambda_field=Lambda_field, r_field=r_field
                    )
                    return add_hamiltonian_noise(H, CONFIG.ham_noise_strength, key)

                field_size = grid_size ** 3
                keys = jax.random.split(subkey, field_size)
                H_field = jax.vmap(ham_for_idx)(jnp.arange(field_size), keys).reshape(
                    grid_size, grid_size, grid_size, quantum_state_dim, quantum_state_dim
                )
                Lambda_current = jnp.stack([Lambda_field[0, 0, 0]] * n_el)
                psi_current    = jnp.stack([psi_field[0, 0, 0]] * n_el)

            elif HAMILTONIAN_MODE == "dirac":
                indices = jnp.arange(n_el)
                keys = jax.random.split(subkey, n_el)
                H = jax.vmap(lambda i, k: add_hamiltonian_noise(
                    get_hamiltonian(
                        i=i, r=r_current, Lambda=Lambda_current, psi=psi_current,
                        identity_ids=jnp.zeros(n_el, dtype=jnp.int32)
                    ), CONFIG.ham_noise_strength, k)
                )(indices, keys)
                Lambda_current, psi_current = jax.vmap(
                    lambda L, p, h, rt, r_, cg: quantum_evolution(
                        L, p, h, rt, delta_rhoT, cg, r=r_
                    )
                )(Lambda_current, psi_current, H, rho_T, r_current,
                  jnp.full(n_el, True, dtype=jnp.bool_))
                default_psi = jnp.zeros_like(psi_current).at[:, 0].set(1.0)
                psi_current = jnp.where(jnp.isnan(psi_current) | jnp.isinf(psi_current), default_psi, psi_current)

            else:
                compute_grad = False
                indices = jnp.arange(n_el)
                keys = jax.random.split(subkey, n_el)
                H = jax.vmap(lambda i, k: add_hamiltonian_noise(
                    get_hamiltonian(
                        i=i, r=r_current, Lambda=Lambda_current, psi=psi_current,
                        identity_ids=jnp.zeros(n_el, dtype=jnp.int32)
                    ), CONFIG.ham_noise_strength, k)
                )(indices, keys)
                Lambda_current, psi_current = jax.vmap(
                    lambda L, p, h, rt, r_, cg: quantum_evolution(
                        L, p, h, rt, delta_rhoT, cg, r=r_
                    )
                )(Lambda_current, psi_current, H, rho_T, r_current,
                  jnp.full(n_el, compute_grad, dtype=jnp.bool_))
                default_psi = jnp.zeros_like(psi_current).at[:, 0].set(1.0)
                psi_current = jnp.where(jnp.isnan(psi_current) | jnp.isinf(psi_current), default_psi, psi_current)

            # 正規化
            psi_current = psi_current / (jnp.linalg.norm(psi_current, axis=1, keepdims=True) + 1e-12)

            # スピンフリップ
            if step % spin_flip_interval == 0:
                key_global, k_spins = jax.random.split(key_global)
                spins = randomize_spins(spins, key=k_spins)

            # 2. LambdaF 計算
            key, key_psi = jax.random.split(key)
            key_psi, key_lambda = jax.random.split(key_psi)

            (progression_tensor_psi, progression_tensor_lambda), \
            (phases_psi, phases_lambda), \
            (main_dir_psi, main_dir_lambda) = compute_progression_tensor_full_quantum(
                r_current, spins, charges, colors, k_vectors,
                psi_current, Lambda_current, jnp.mean(rho_T), c,
                key_psi=key_psi,
                key_lambda=key_lambda,
                step=step,
                phase_noise_strength=phase_noise_strength
            )

            # --- 標準系: ψ系の進行ベクトルをlambda_fとする ---
            lambda_f = progression_tensor_psi / (jnp.sum(progression_tensor_psi) + 1e-12)

            # --- 必要ならLambda系や差分も計算 ---
            lambda_f_Lambda = progression_tensor_lambda / (jnp.sum(progression_tensor_lambda) + 1e-12)
            lambda_f_diff = jnp.abs(lambda_f - lambda_f_Lambda)

            # ψ系phaseを標準に使うなら
            phases = phases_psi

            # Λ系のphaseを別に使いたい場合
            phases_lambda_val = phases_lambda

            # 🟢 LambdaFを“進行テンソル主成分”として渡す
            h, _ = compute_embedding_quantum(
                r_current, spins, charges, k_vectors, psi_current, Lambda_current, c,
                lambda_f=lambda_f,    # ここに進行テンソルをそのまま！
                phase=phases_psi,          # phaseもtensorからそのまま流用
                s_gen=s_gen
            )

            # ==== 3. 現象進行テンソル・差分の計算 ====
            if step > 0:
                # 前stepからの差分（現象ベクトル化）
                delta_r = jnp.linalg.norm(r_current - r_prev, axis=1)          # (n_el,)
                delta_Lambda = jnp.linalg.norm(Lambda_current - Lambda_prev, axis=(1,2))  # (n_el,)
                delta_spin = jnp.abs(spins - spins_prev)                       # (n_el,)
                delta_charge = jnp.abs(charges - charges_prev)
                delta_color = jnp.linalg.norm(colors - colors_prev, axis=1)

                # 複合テンソル現象スコア例（最大・平均・合成など）
                max_delta_r = float(jnp.max(delta_r))
                avg_delta_r = float(jnp.mean(delta_r))
                max_delta_Lambda = float(jnp.max(delta_Lambda))
                avg_delta_Lambda = float(jnp.mean(delta_Lambda))
                # ...他も同様に出せる

                # 任意の「現象しきい値」でトリガー
                PHENOMENON_DELTA_R_THRESH = 5.00
                PHENOMENON_DELTA_LAMBDA_THRESH = 5.00
                cooling_needed = (max_delta_r > PHENOMENON_DELTA_R_THRESH) or (max_delta_Lambda > PHENOMENON_DELTA_LAMBDA_THRESH)
                cooling_reason = (
                    f"max_delta_r({max_delta_r:.4f}) > {PHENOMENON_DELTA_R_THRESH:.2f} OR "
                    f"max_delta_Lambda({max_delta_Lambda:.4f}) > {PHENOMENON_DELTA_LAMBDA_THRESH:.2f}"
                )
            else:
                cooling_needed = False
                cooling_reason = "step==0 (skip delta check)"

            # ==== 4. クーリング判定・処理 ====
            if step > (warmup_step + warmup_buffer):
                cooldown_target = cooldown_target_on if cooling_needed else cooldown_target_off
                cooldown_level = (1 - cooldown_ewma_alpha) * cooldown_level + cooldown_ewma_alpha * cooldown_target

                print(f"[COOLING CHECK] Step {step}: cooling_needed={cooling_needed} | {cooling_reason}")

                if cooling_needed and not cooldown_triggered:
                    print(f"[COOLING TRIGGER] Step {step} - spins quenched! {cooling_reason}")
                    spins = apply_spin_quench(spins, spin_quench_factor)
                    cooldown_triggered = True

                rhoT_ext = 0.0 if cooling_needed else experiment_to_transaction_params()[1]

            # ==== 進行履歴の保存 ====
            r_prev = r_current
            Lambda_prev = Lambda_current
            spins_prev = spins
            charges_prev = charges
            colors_prev = colors
            # ==== 進行定義 ====
            direction = main_dir_psi

            # エネルギー
            energy, energy_matrix, key_global = compute_energy_structural(
                r_current, spins, charges, colors, k_vectors, c, direction, key=key_global
            )
            if len(energy_history) >= CONFIG.ema_energy_window:
                energy = CONFIG.ema_energy_current_weight * energy + \
                        CONFIG.ema_energy_history_weight * jnp.mean(jnp.array(energy_history[-CONFIG.ema_energy_window:]))
            energy_history.append(float(energy))

            # ===========================================================
            # 🏹 QLambda（トポロジカルチャージ）進行・ロギング（Phaseタプル対応）
            # ===========================================================
            # 1. 進行テンソル＋Phase（ψ・Λ主成分の2種）を取得
            key, key_psi = jax.random.split(key)
            key_psi, key_lambda = jax.random.split(key_psi)

            (progression_tensor_psi, progression_tensor_lambda), \
            (phases_psi, phases_lambda), \
            (main_direction_psi, main_direction_lambda) = compute_progression_tensor_full_quantum(
                r_current, spins, charges, colors, k_vectors,
                psi_current, Lambda_current, jnp.mean(rho_T), c,
                key_psi=key_psi,
                key_lambda=key_lambda,
                step=step,
                phase_noise_strength=phase_noise_strength
            )
            # 2. Lambda_field（psi, Λ主成分）分岐生成
            if HAMILTONIAN_MODE in ["dirac_field", "qed_field"]:
                # 格子場モードは“Lambda_field”を直接両方に使う
                Lambda_field_psi = Lambda_field
                Lambda_field_Lambda = Lambda_field
            else:
                # 粒子・分子系は「generate_Lambda_field」で位相源切替
                charges = spins + 1.0
                Lambda_field_psi = generate_Lambda_field(
                    np.array(r_current), np.array(charges),
                    CONFIG.grid_size, CONFIG.grid_extent,
                    sigma=CONFIG.sigma,
                    phases=phases_psi                 # ψ由来
                )
                Lambda_field_Lambda = generate_Lambda_field(
                    np.array(r_current), np.array(charges),
                    CONFIG.grid_size, CONFIG.grid_extent,
                    sigma=CONFIG.sigma,
                    phases=phases_lambda              # Λ主成分由来
                )

            # 3. QΛ両方計算
            QLambda_psi = auto_compute_topological_charge(Lambda_field_psi)
            QLambda_Lambda = auto_compute_topological_charge(Lambda_field_Lambda)

            # 4. QΛ履歴管理
            if 'QLambda_psi_history' not in locals():
                QLambda_psi_history = []
            if 'QLambda_Lambda_history' not in locals():
                QLambda_Lambda_history = []
            QLambda_psi_history.append(float(QLambda_psi))
            QLambda_Lambda_history.append(float(QLambda_Lambda))

            # 5. EMA計算（任意）
            if len(QLambda_psi_history) >= CONFIG.ema_score_window:
                QLambda_psi_ema = (
                    CONFIG.ema_score_current_weight * QLambda_psi +
                    CONFIG.ema_score_history_weight * np.mean(QLambda_psi_history[-CONFIG.ema_score_window:])
                )
            else:
                QLambda_psi_ema = QLambda_psi

            if len(QLambda_Lambda_history) >= CONFIG.ema_score_window:
                QLambda_Lambda_ema = (
                    CONFIG.ema_score_current_weight * QLambda_Lambda +
                    CONFIG.ema_score_history_weight * np.mean(QLambda_Lambda_history[-CONFIG.ema_score_window:])
                )
            else:
                QLambda_Lambda_ema = QLambda_Lambda

            # 6. QΛジャンプ閾値判定・ロギング
            Q_JUMP_THRESHOLD = getattr(CONFIG, "q_lambda_jump_threshold", 0.5)

            # --- psi-QΛジャンプ ---
            delta_Q_psi = 0.0
            jump_detected_psi = False
            if QLambda_psi_prev is not None:
                delta_Q_psi = QLambda_psi - QLambda_psi_prev
                if abs(delta_Q_psi) >= Q_JUMP_THRESHOLD:
                    jump_detected_psi = True
                    wandb.log({
                        "Q_Lambda_psi_jump": float(delta_Q_psi),
                        "step": step,
                        "transaction_status": status if 'status' in locals() else ""
                    })
                    print(f"[Q_JUMP][psi] Step {step}: ΔQ = {delta_Q_psi:.4f} (from {QLambda_psi_prev:.4f} → {QLambda_psi:.4f})")
            QLambda_psi_prev = QLambda_psi

            # --- Λ-QΛジャンプ ---
            delta_Q_Lambda = 0.0
            jump_detected_Lambda = False
            if QLambda_Lambda_prev is not None:
                delta_Q_Lambda = QLambda_Lambda - QLambda_Lambda_prev
                if abs(delta_Q_Lambda) >= Q_JUMP_THRESHOLD:
                    jump_detected_Lambda = True
                    wandb.log({
                        "Q_Lambda_Lambda_jump": float(delta_Q_Lambda),
                        "step": step,
                        "transaction_status": status if 'status' in locals() else ""
                    })
                    print(f"[Q_JUMP][Lambda] Step {step}: ΔQ = {delta_Q_Lambda:.4f} (from {QLambda_Lambda_prev:.4f} → {QLambda_Lambda:.4f})")
            QLambda_Lambda_prev = QLambda_Lambda

            # 7. 局所位相ジャンプ検出（psi/Λ主成分）
            if 'phase_psi_prev' in locals():
                delta_phases_psi = np.angle(np.exp(1j * (phase_psi - phase_psi_prev)))
                wandb.log({
                    "atomic_phase_jumps_psi": delta_phases_psi.tolist(),
                    "step": step
                })
            phases_psi_prev = phases_psi.copy()

            if 'phase_lambda_prev' in locals():
                delta_phases_lambda = np.angle(np.exp(1j * (phase_lambda - phase_lambda_prev)))
                wandb.log({
                    "atomic_phase_jumps_Lambda": delta_phases_lambda.tolist(),
                    "step": step
                })
            phases_lambda_prev = phases_lambda.copy()

            # ========================
            # 🟢 ✅ 量子測定ブロック
            # ========================
            if HAMILTONIAN_MODE in ["dirac_field", "qed_field"]:
                # --- テンソル場（空間grid系） ---
                field_shape = Lambda_field.shape[:-2]
                n_sites = np.prod(field_shape)
                dim = Lambda_field.shape[-1]

                Lambda_flat = Lambda_field.reshape(n_sites, dim, dim)
                psi_flat = psi_field.reshape(n_sites, dim) if psi_field is not None else jnp.zeros((n_sites, dim), dtype=jnp.complex64)

                # 【1】spins_fieldのshapeをn_sitesに統一して管理
                if 'spins_field' not in locals():
                    spins_field = jnp.zeros(n_sites, dtype=jnp.float32)  # 必要なら初期値でOK

                # 🔑 必要なkeyをn_sitesごとに分割
                site_keys = jax.random.split(subkey, n_sites)  # shape = (n_sites,)

                # (A) ノイズ・観測パラメータ（vmapでkeyを使う！）
                local_noise_scale = jnp.full(n_sites, CONFIG.noise_scale)
                global_noise = jax.vmap(lambda k: get_global_noise(1, strength=CONFIG.global_noise_strength, key=k))(site_keys).squeeze()
                total_noise = local_noise_scale + global_noise

                # (B) 外部フィールド生成・部分観測
                axis_vecs = (
                    jax.vmap(lambda k: jax.random.normal(k, shape=(dim,)))(site_keys) +
                    1j * jax.vmap(lambda k: jax.random.normal(k, shape=(dim,)))(site_keys)
                )
                axis_vecs = axis_vecs / (jnp.linalg.norm(axis_vecs, axis=1, keepdims=True) + 1e-12)
                observe_mask = jax.vmap(lambda k: random_observe_mask(1, CONFIG.observe_prob, k))(site_keys).squeeze()

                # spins_field用ノイズ生成＆加算（n_sites進行）
                external_field = 0.1 * jnp.sin(0.1 * step) * jax.vmap(lambda k: jax.random.normal(k, shape=(1,)))(site_keys).squeeze()
                spins_field = spins_field + external_field

                random_field_strength = 0.05 * jax.vmap(lambda k: jax.random.normal(k, shape=(1,)))(site_keys).squeeze()
                spins_field = spins_field + random_field_strength

                # 🔑 key構造生成
                update_keys = make_batched_update_keys(site_keys, n_sites)  # siteごと

                # ✅ Lambda/ψ更新
                Lambda_flat, psi_flat = update_particle_vmap(
                    Lambda_flat, psi_flat, axis_vecs, total_noise, observe_mask,
                    update_keys['projector'], update_keys['eigenvector']
                )

                Lambda_field = Lambda_flat.reshape(field_shape + (dim, dim))
                psi_field = psi_flat.reshape(field_shape + (dim,))

                # 代表点転写（粒子系への縮約: n_el点ぶん）
                Lambda_current = jnp.stack([Lambda_field[0, 0, 0] for _ in range(n_el)])
                psi_current    = jnp.stack([psi_field[0, 0, 0] for _ in range(n_el)])
                spins         = jnp.stack([spins_field[0] for _ in range(n_el)])

            else:
                # --- 粒子・分子系（Dirac含む） ---
                dim = Lambda_current.shape[-1]
                n_el = Lambda_current.shape[0]

                particle_keys = jax.random.split(subkey, n_el)  # shape = (n_el,)

                # (A) 進行テンソル・同期率・進行量の算出
                lambda_f = progression_tensor
                sigma_s = compute_sigma_s_enhanced(spins, charges, colors, k_vectors, dists, mask)
                delta_lambda = jnp.linalg.norm(Lambda_current - Lambda_prev, axis=(1,2))

                # (B) ノイズ強度動的算出（keyごとに生成）
                local_noise_scale = compute_local_noise_scale(lambda_f, sigma_s, delta_lambda, base_scale=CONFIG.noise_scale)
                global_noise = jax.vmap(lambda k: get_global_noise(1, strength=CONFIG.global_noise_strength, key=k))(particle_keys).squeeze()
                total_noise = local_noise_scale + global_noise

                # (C) 部分観測・軸ベクトル生成もkeyごと
                observe_mask = jax.vmap(lambda k: random_observe_mask(1, CONFIG.observe_prob, k))(particle_keys).squeeze()
                axis_vecs = (
                    jax.vmap(lambda k: jax.random.normal(k, shape=(dim,)))(particle_keys) +
                    1j * jax.vmap(lambda k: jax.random.normal(k, shape=(dim,)))(particle_keys)
                )
                axis_vecs = axis_vecs / (jnp.linalg.norm(axis_vecs, axis=1, keepdims=True) + 1e-12)

                # 🔑 key構造生成
                update_keys = make_batched_update_keys(particle_keys, n_el)

                # ✅ Lambda/ψ更新
                Lambda_current, psi_current = update_particle_vmap(
                    Lambda_current, psi_current, axis_vecs, total_noise, observe_mask,
                    update_keys['projector'], update_keys['eigenvector']
                )
                # spinsは(n_el,)のままでOK

            measurement_events += 1

            # エントロピー
            entropy_per_particle = vmap(compute_entanglement_entropy)(Lambda_current)
            entropy = jnp.mean(entropy_per_particle)
            entropy = jnp.where(jnp.isnan(entropy), 0.0, entropy)

            # 部分系エントロピー
            partial_entropies = vmap(lambda i: compute_partial_entropy(Lambda_current, i))(jnp.arange(Lambda_current.shape[0]))
            partial_entropy_mean = jnp.mean(partial_entropies)
            partial_entropy_max = jnp.max(partial_entropies)

            # フェルミオン登録
            for i in range(n_el):
                if split_will[i] > split_threshold:
                    new_fermion = register_split_fermion_quantum(
                        r_current, spins, charges, lambda_f, i, psi_current, Lambda_current
                    )
                    fermion_list.append(new_fermion)
                    particle_label = ["Fe", "C1", "C2", "C3", "C4", "Cl", "C6", "C7", "C8", "C9"][i]
                    wandb.log({f"fermi_split/{particle_label}": float(split_will[i])})

            # フェルミオン埋め込み
            fermion_embeddings = compute_fermion_embeddings_quantum(
                fermion_list, r_current, spins, charges, k_vectors, psi_current, Lambda_current, c, s_gen
            )
            fermion_h_mean_norm = None
            if fermion_embeddings is not None:
                fermion_h_mean_norm = float(jnp.mean(jnp.linalg.norm(fermion_embeddings, axis=1)))

            # フェルミオン進行
            fermion_list = [
                propagate_fermion(f) for f in fermion_list
                if isinstance(f, dict) and f.get("alive", False)
            ]

            # 量子版メトロポリスサンプリング
            key_global, k_sampling = jax.random.split(key_global)
            r_samples, sigmas, colors_final, charges_final = adaptive_metropolis_sampling_structural_quantum(
                r_current, spins, charges, colors, k_vectors,
                psi_current, Lambda_current, c, direction,
                rng_key=k_sampling
            )

            r_current = r_samples[-1] + 0.002 * laplacian_term(r_current)

            # 分離イベント（Splitモードの特別分岐：一番分離スコア高い粒子）
            if direction == 2:
                split_idx = int(jnp.argmax(split_will))
                new_fermion = register_split_fermion_quantum(
                    r_current, spins, charges, lambda_f, split_idx, psi_current, Lambda_current
                )
                fermion_list.append(new_fermion)

            # 全粒子分離閾値チェック（複数分離発生現象）
            for i in range(n_el):
                if split_will[i] > split_threshold:
                    new_fermion = register_split_fermion_quantum(
                        r_current, spins, charges, lambda_f, i, psi_current, Lambda_current
                    )
                    fermion_list.append(new_fermion)

            # --- LambdaF進行スコア（psi/Lambda系統の両方取得）---
            key, k_psi = jax.random.split(key)
            k_psi, k_lambda = jax.random.split(k_psi)

            (progression_tensor_psi, progression_tensor_lambda), \
            (phases_psi, phases_lambda), \
            (main_direction_psi, main_direction_lambda) = compute_progression_tensor_full_quantum(
                r_current, spins, charges, colors, k_vectors,
                psi_current, Lambda_current, jnp.mean(rho_T), c,
                key_psi=k_psi,
                key_lambda=k_lambda,
                step=step,
                phase_noise_strength=phase_noise_strength
            )

            lambda_f_psi    = progression_tensor_psi / (jnp.sum(progression_tensor_psi) + 1e-12)
            lambda_f_Lambda = progression_tensor_lambda / (jnp.sum(progression_tensor_lambda) + 1e-12)

            # --- LambdaF履歴（**分離管理**）---
            lambda_f_psi_history.append(np.array(lambda_f_psi))
            lambda_f_Lambda_history.append(np.array(lambda_f_Lambda))

            # --- チェーン登録（ここはどちらか選択、あるいは両方記録してもOK！）---
            for i in range(n_el):
                prev_block = chains[i].blocks[-1]
                divergence = np.linalg.norm(r_current[i] - prev_block.position) if step > 0 else 0.0

                # 登録するlambda_fを選択
                local_lambda_f_psi = lambda_f_psi[i] if lambda_f_psi.ndim > 1 else lambda_f_psi
                local_lambda_f_Lambda = lambda_f_Lambda[i] if lambda_f_Lambda.ndim > 1 else lambda_f_Lambda

                # 例：両方記録する
                chains[i].add_block(
                    step=step,
                    position=np.array(r_current[i]),
                    spin=int(spins[i]),
                    charge=float(charges[i]),
                    color=np.array(colors[i]).tolist(),
                    Lambda=np.array(Lambda_current[i]),
                    lambda_F=None,  # 使わないときはNoneでOK
                    lambda_F_psi=np.array(local_lambda_f_psi),     # psi側
                    lambda_F_Lambda=np.array(local_lambda_f_Lambda),  # Lambda側
                    rho_T=float(rho_T[i]),
                    sigma_s=float(sigma_s[i]) if 'sigma_s' in locals() else None,
                    divergence=float(divergence)
                )

            # --- チェーン全体チェック ---
            for ch in chains:
                assert ch.verify()

            # --- 量子状態スコア（psi/Λ主成分両方）---
            quantum_scores_psi = compute_quantum_progression_scores(
                psi_current, Lambda_current,
                QLambda=QLambda_psi,
                QLambda_prev=QLambda_psi_prev,
                psi_prev=psi_prev,
                Lambda_prev=Lambda_prev,
                split_will=split_will,
                partial_entropy=partial_entropies
            )

            quantum_scores_Lambda = compute_quantum_progression_scores(
                psi_current, Lambda_current,
                QLambda=QLambda_Lambda,
                QLambda_prev=QLambda_Lambda_prev,
                psi_prev=psi_prev,
                Lambda_prev=Lambda_prev,
                split_will=split_will,
                partial_entropy=partial_entropies
            )

            # --- デバッグ用プリント（psi／Lambda両方を明示表示）---
            for label, pt in {
                "psi": progression_tensor_psi,
                "Lambda": progression_tensor_lambda
            }.items():
                print(f"=== [Step {step}] Λ³進行テンソル ({label}) ===")
                print(f"bind={float(pt[0]):.4f}, move={float(pt[1]):.4f}, split={float(pt[2]):.4f}")
                quantum_scores = quantum_scores_psi if label == "psi" else quantum_scores_Lambda
                for k, v in quantum_scores.items():
                    if k == "splitting_array":
                        splitting_arr = np.asarray(v)
                        print(f"quantum/{label}/splitting_array: {splitting_arr.round(4)}")
                    else:
                        print(f"quantum/{label}/{k}: {float(v):.4f}")


            # ---- wandb 記録セクション（インラインで） ----
            dipole_vector = compute_dipole_tensor(r_current, charges)
            dipole_magnitude = float(jnp.linalg.norm(dipole_vector))
            dipole_history.append(dipole_magnitude)
            dipole_delta = (
                jnp.abs(dipole_magnitude - jnp.mean(jnp.array(dipole_history[-ema_energy_window:])))
                if len(dipole_history) >= ema_energy_window else 0.0
            )

            reward = float(-jnp.tanh(jnp.abs(jnp.mean(rho_T_raw) - 1.0)) - 0.1 * jnp.tanh(dipole_delta))

            try:
                delta_Q_val = float(delta_Q) if direction == 2 and QLambda_prev is not None else 0.0
            except Exception:
                delta_Q_val = 0.0

            eigenvalues = vmap(jnp.linalg.eigvalsh)(Lambda_current)
            psi_norms = jnp.linalg.norm(psi_current, axis=1)
            entanglement_entropy = vmap(compute_entanglement_entropy)(Lambda_current)

            partial_entropies = vmap(lambda i: compute_partial_entropy(Lambda_current, i))(jnp.arange(Lambda_current.shape[0])) if Lambda_current.shape[0] > 0 else jnp.array([0.0])
            partial_entropy_max = float(jnp.max(partial_entropies))
            partial_entropy_mean = float(jnp.mean(partial_entropies))

            split_will, _ = compute_score_split_map(r_current, spins, charges, colors, lambda_f, c) if 'split_will' not in locals() else (split_will, None)
            avg_split_score = float(jnp.mean(split_will))
            max_split_score = float(jnp.max(split_will))
            split_idx = int(jnp.argmax(split_will)) if len(split_will) > 0 else None

            wandb_log_data = {
                "step": step,

                # === Λ³量子進行テンソル現象 ===
                "quantum/bind": float(progression_tensor_psi[0]),
                "quantum/move": float(progression_tensor_psi[1]),
                "quantum/split": float(progression_tensor_psi[2]),

                # === 埋め込み空間進行（LambdaF） ===
                "lambda_f": np.array(lambda_f, dtype=np.float32),
                "lambdaF_history": wandb.Table(data=lambda_f_history, columns=[f"λF_dim{i+1}" for i in range(len(lambda_f))]) if lambda_f_history else None,

                # === 物理観測量（エネルギー/エントロピー/トポロジカル） ===
                "ΛScalar_energy": float(energy),
                "energy_hist": wandb.Histogram(energy_history[-len(rho_T_raw):]) if energy_history else None,
                "entropy": float(entropy),
                "entropy_max": float(jnp.max(entropy_per_particle)),
                "entropy_min": float(jnp.min(entropy_per_particle)),
                "entropy_mean": float(jnp.mean(entropy_per_particle)),
                "partial_entropy_max": float(partial_entropy_max),
                "partial_entropy_mean": float(partial_entropy_mean),
                "QLambda_psi": float(QLambda_psi),
                "QLambda_Lambda": float(QLambda_Lambda),
                "QLambda_psi_jump": float(delta_Q_psi) if 'delta_Q_psi' in locals() else 0.0,
                "QLambda_Lambda_jump": float(delta_Q_Lambda) if 'delta_Q_Lambda' in locals() else 0.0,

                # === 局所現象進行（分離・分岐強度/現象差分など） ===
                "split_will_max": float(jnp.max(split_will)),
                "split_will_avg": float(jnp.mean(split_will)),
                "max_delta_r": float(max_delta_r) if 'max_delta_r' in locals() else 0.0,
                "avg_delta_r": float(avg_delta_r) if 'avg_delta_r' in locals() else 0.0,
                "max_delta_Lambda": float(max_delta_Lambda) if 'max_delta_Lambda' in locals() else 0.0,
                "avg_delta_Lambda": float(avg_delta_Lambda) if 'avg_delta_Lambda' in locals() else 0.0,
                "split_idx": int(split_idx) if 'split_idx' in locals() else None,

                # === テンション・同期・ダイバージェンスなどΛ³観測値 ===
                "rho_T": float(jnp.mean(rho_T_raw)),
                "rho_T_distribution": wandb.Histogram(rho_T_raw),
                "sigma_s": float(sigma_s_value) if 'sigma_s_value' in locals() else None,
                "transaction_divergence": float(divergence) if 'divergence' in locals() else 0.0,
                "transaction_status": str(status) if 'status' in locals() else "",

                # === フェルミオン現象 ===
                "n_fermions": len(fermion_list),
                "fermion_psi_abs": wandb.Histogram([np.abs(f["psi"]) for f in fermion_list]) if fermion_list else None,
                "fermion_Lambda_eigen_abs": wandb.Histogram([np.abs(np.linalg.eigvals(f["Lambda"])) for f in fermion_list]) if fermion_list else None,

                # === Dipole etc. ===
                "dipole_magnitude": float(dipole_magnitude) if 'dipole_magnitude' in locals() else 0.0,
                "dipole_delta": float(dipole_delta) if 'dipole_delta' in locals() else 0.0,
                "dipole_history": wandb.Histogram(dipole_history) if dipole_history else None,

                # === 報酬 ===
                "reward": float(reward) if 'reward' in locals() else 0.0,

                # === 3D配置 ===
                "electron_positions": wandb.Object3D({
                    "type": "lidar/beta",
                    "points": np.array(r_current, dtype=np.float32)
                }),

                # === 測定イベント・進行 ===
                "measurement_events": int(measurement_events) if 'measurement_events' in locals() else 0,
                "step": step
            }

            # --- 進行テンソルラベル（判定ラベルquantum_scores）を追加 ---
            wandb_log_data.update({
                f"quantum/{k}": (np.asarray(v) if hasattr(v, "shape") and v.ndim > 0 else float(v))
                for k, v in quantum_scores.items()
            })

            wandb.log(wandb_log_data)

            # Visualization every 100 steps
            if step % 100 == 0:
                visualize_lambda_f_with_crystal_axis(lambda_f_psi, step, r_current, dipole_vector=dipole_vector)

            lambda_f_matrix.append(np.array(lambda_f))

    except Exception as e:
        context = {
            "step": step,
            "r_shape": str(r_current.shape),
            "psi_shape": str(psi_current.shape),
            "Lambda_shape": str(Lambda_current.shape),
            "rho_T_shape": str(rho_T.shape)
        }
        log_error(step, "quantum_main_loop", e, context)

    # === 最終処理 ===
    analyze_lambda_f(lambda_f_matrix, step=n_steps)
    wandb.finish()

    return r_current, lambda_f_matrix

def setup_chloroferrocene_quantum(
    quantum_state_dim,
    key,
    mode="default",
    noise_scale=0.01,
    asymmetry=0.0,
    with_noise=True,
):
    n_el = 10

    # === 1. 初期座標（クロロフェロセン骨格） ===
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
        [-3.0, 0.0, -2.05],     # C9
    ], dtype=jnp.float32)

    # === 2. 幾何学的非対称性の導入 ===
    if asymmetry > 0.0:
        key, subkey = jax.random.split(key)
        r = r.at[1:].add(asymmetry * jax.random.normal(subkey, shape=r[1:].shape))

    # === 3. ホワイトノイズ注入（with_noise に基づく） ===
    if with_noise and noise_scale > 0.0:
        key, subkey = jax.random.split(key)
        r += noise_scale * jax.random.normal(subkey, shape=r.shape)

    # === 4. スピン初期化 ===
    key, subkey = jax.random.split(key)
    spins = jax.random.uniform(subkey, shape=(n_el,), minval=-1.0, maxval=1.0)

    # === 5. チャージ・カラー・運動量ベクトル ===
    charges = jnp.ones(n_el, dtype=jnp.float32)
    colors = jnp.array([
        [0.8, 0.0, 0.0],  # Fe
        [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2],  # C1-4
        [0.1, 0.1, 0.5],  # Cl
        [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2], [0.2, 0.2, 0.2]   # C6-9
    ], dtype=jnp.float32)
    k_vectors = jnp.array([[1.0, 0.0, 0.0]] * n_el, dtype=jnp.float32)

    # === 6. 量子状態 ψ の初期化（ノイズ付き） ===
    key, subkey1 = jax.random.split(key)
    key, subkey2 = jax.random.split(key)
    psi = jax.random.normal(subkey1, (n_el, quantum_state_dim), dtype=jnp.complex64)
    psi += 0.2 * jax.random.normal(subkey2, (n_el, quantum_state_dim), dtype=jnp.complex64)
    psi /= jnp.linalg.norm(psi, axis=1, keepdims=True) + 1e-8

    # === 7. 構造テンソル Λ 初期化（ψ の外積＋アイデンティティ微小補正）===
    Lambda = jax.vmap(lambda p: jnp.outer(p, jnp.conj(p)) + 0.05 * jnp.eye(quantum_state_dim))(psi)

    return r, spins, charges, colors, k_vectors, psi, Lambda, key

# ========================
# 🟢 ✅実行関数
# ========================
if __name__ == "__main__":
    config = Lambda3Fire_tamaki_Config()
    fermion_list = []
    lambdaF_ext, rhoT_ext = experiment_to_transaction_params()
    HAMILTONIAN_MODE = config.HAMILTONIAN_MODE
    quantum_state_dim = get_quantum_state_dim(HAMILTONIAN_MODE)
    key = key_global

    # 1. setup
    key, subkey = jax.random.split(key)
    r, spins, charges, colors, k_vectors, psi, Lambda, key = setup_chloroferrocene_quantum(
        quantum_state_dim,
        key=subkey,
        mode="default",
        noise_scale=0.01,
        asymmetry=0.0,
        with_noise=True,
    )

    # 2. mainloop
    key, subkey = jax.random.split(key)
    r_final, lambda_f_matrix = lambda3_fire_vmc_quantum(
        r, spins, charges, colors, k_vectors, psi, Lambda,
        lambdaF_ext=lambdaF_ext, rhoT_ext=rhoT_ext,
        adaptive_stop=True,
        fermion_list=fermion_list,
        key=subkey
    )

    print(f"Final electron positions for chloroferrocene:\n{r_final}")

