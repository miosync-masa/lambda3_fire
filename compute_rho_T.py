@jit
def compute_rho_T(r, spins, c, beta_entropy=0.1, gamma=GAMMA, E_density=1.0, theta=0.0, d0=1.0):
    """
    rho_T を計算（エントロピー生成トリガーを含む）。
    Args:
        r: 電子の位置 (n_el, 3)
        spins: スピン (n_el,)
        c: カットオフ
        beta_entropy: エントロピー寄与の重み
        gamma: エントロピー生成スケール
        E_density: エネルギー密度（オプション）
        theta: 衝突角度（オプション）
        d0: 基準距離（オプション）
    Returns:
        rho_T: 意味テンション密度
    """
    n_el = r.shape[0]
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    dists = dists + jnp.eye(n_el) * 1e10
    
    # 基本的な rho_T（エネルギー密度と距離依存）
    rho_T_base = E_density * (d0 / dists) ** 2 * jnp.cos(theta)
    rho_T_base = jnp.sum(rho_T_base, where=dists < c) / n_el
    
    # スピン不整合に基づくエントロピー生成
    spin_match = (spins[:, None] == spins[None, :]).astype(float) * 0.5 + 0.5
    mask = (dists < c) & (dists > 0)
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * spin_match * mask
    S_gen = gamma * jnp.sum(1.0 - sigma_s_ij, where=mask) / jnp.sum(mask)
    
    # rho_T にエントロピー生成項を追加
    rho_T = rho_T_base + beta_entropy * S_gen
    return rho_T
