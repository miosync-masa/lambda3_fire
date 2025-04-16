@jit
def compute_embedding(r, c, direction):
    # FiRE の埋め込み h_i(r) を動的 c で計算
    n_el = r.shape[0]
    h = jnp.zeros((n_el, embedding_dim))  # ダミー埋め込み
    
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    mask = (dists < c) & (dists > 0)
    
    # 近傍電子との相互作用（結合トランザクションの例）
    for i in range(n_el):
        neighbors = jnp.where(mask[i])[0]
        for j in neighbors:
            delta_lambda_c = 1.0 / (dists[i, j] + 1e-10)  # rho_T の簡略化
            h = h.at[i].add(delta_lambda_c)  # 埋め込みに加算
    return h

# VMC ループ内での統合
def vmc_step(r, spins, charges, params):
    rho_T = compute_rho_T(r)  # テンション密度の計算（簡略化）
    c = compute_dynamic_cutoff(r, rho_T0=rho_T)
    direction = compute_transaction_direction(r, spins, charges, rho_T, c)
    h = compute_embedding(r, c, direction)
    # 以降、FiRE の波動関数計算（det, Jastrow など）に h を使用
    return h

# 実行例
h = vmc_step(r, spins, charges, params=None)
print(f"Embedding: {h}")
