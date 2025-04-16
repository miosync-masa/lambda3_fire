@jit
def compute_transaction_direction(r, spins, charges, rho_T, c):
    # r: 電子の位置 (n_el, 3), spins: スピン (n_el,), charges: 電荷 (n_el,)
    n_el = r.shape[0]
    
    # 距離と近傍マスク
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    mask = (dists < c) & (dists > 0)
    
    # sigma_s の計算（結合・運動・分離）
    sigma_s_bind = jnp.zeros((n_el, n_el))
    sigma_s_move = jnp.zeros((n_el, n_el))
    sigma_s_split = jnp.zeros((n_el, n_el))
    
    # スピンと電荷の整合性
    spin_match = (spins[:, None] == spins[None, :]).astype(float) * 0.5 + 0.5
    charge_match = charges[:, None] * charges[None, :] / (dists + 1e-10)
    
    # 各方向の sigma_s
    base_sigma_s = jnp.exp(-0.1 * dists ** 2) * mask
    sigma_s_bind = base_sigma_s * spin_match * charge_match  # 結合：強い整合
    sigma_s_move = base_sigma_s * 0.5  # 運動：中程度の共鳴
    sigma_s_split = base_sigma_s * (1.0 - spin_match)  # 分離：不整合
    
    # 平均 sigma_s
    sigma_s_bind = jnp.sum(sigma_s_bind) / jnp.sum(mask)
    sigma_s_move = jnp.sum(sigma_s_move) / jnp.sum(mask)
    sigma_s_split = jnp.sum(sigma_s_split) / jnp.sum(mask)
    
    # 分岐スコア
    scores = jnp.array([sigma_s_bind * rho_T, sigma_s_move * rho_T, sigma_s_split * rho_T])
    direction = jnp.argmax(scores)  # 0: 結合, 1: 運動, 2: 分離
    
    return direction

# トランザクションの適用
@jit
def apply_transaction(r, direction, rho_T, c):
    # 簡単な例：結合トランザクションで埋め込みを更新
    if direction == 0:  # 結合
        # 近傍電子とのトランザクションを加算
        # （実際には h_i(r) の計算に組み込む）
        pass
    return r

# 例：トランザクション方向の計算
spins = jnp.array([1, -1, 1])
charges = jnp.array([1.0, 1.0, 1.0])
direction = compute_transaction_direction(r, spins, charges, rho_T=1.0, c=3.0)
print(f"Transaction direction: {direction}")
