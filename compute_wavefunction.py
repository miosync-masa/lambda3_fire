@jit
def compute_wavefunction(r, h, params=None):
    """
    波動関数 Ψ(r) を計算（簡略化版）。
    Args:
        r: 電子の位置 (n_el, 3)
        h: 埋め込み (n_el, embedding_dim)
    Returns:
        psi: 波動関数の値
    """
    n_el = r.shape[0]
    
    # 軌道行列 Φ_d(r) の構築（簡略化）
    Phi = jnp.zeros((N_DET, n_el, n_el), dtype=jnp.complex64)
    for d in range(N_DET):
        # ダミー：h を基に軌道行列を構築（実際には線形変換などが必要）
        Phi_d = jnp.eye(n_el) * h  # 簡略化（実際には h から軌道を計算）
        Phi = Phi.at[d].set(Phi_d)
    
    # 行列式の計算
    dets = jnp.array([jnp.linalg.det(Phi[d]) for d in range(N_DET)])
    
    # Jastrow 因子の計算（簡略化）
    J = 1.0  # ダミー（実際には r, h を用いて計算）
    
    # 波動関数 Ψ(r)
    psi = J * jnp.sum(dets)
    return psi

@jit
def compute_energy(r, psi, params=None):
    """
    エネルギー E を計算（簡略化版）。
    Args:
        r: 電子の位置 (n_el, 3)
        psi: 波動関数の値
    Returns:
        energy: エネルギー
    """
    # 局所エネルギー E_L(r) の計算（ダミー）
    kinetic_energy = 1.0  # ダミー（実際にはラプラシアン計算が必要）
    potential_energy = 1.0  # ダミー（実際にはポテンシャル項を計算）
    E_L = kinetic_energy + potential_energy
    
    # 変分エネルギー（簡略化）
    energy = E_L * jnp.abs(psi) ** 2  # 実際にはサンプリングが必要
    return energy

# VMC ステップの更新
def lambda3_fire_vmc_step(r, spins, charges, k_vectors, params=None):
    """
    $\Lambda^3$-FiRE の VMC ステップ（更新版）。
    """
    c = compute_dynamic_cutoff(r)
    rho_T = compute_rho_T(r, spins, c, E_density=1.0, theta=0.0, d0=1.0)
    direction, phase = compute_transaction_direction(r, spins, charges, k_vectors, rho_T, c)
    h, lambda_f = compute_embedding(r, spins, charges, k_vectors, c, direction, phase)
    
    # 次のステップ：波動関数とエネルギーの計算
    psi = compute_wavefunction(r, h)
    energy = compute_energy(r, psi)
    
    return h, c, direction, rho_T, lambda_f, psi, energy
