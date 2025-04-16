# トランザクション方向ベクトルの定義
LAMBDA_F_BIND = jnp.array([1.0, 0.0, 0.0])  # 結合方向
LAMBDA_F_MOVE = jnp.array([0.0, 1.0, 0.0])  # 運動方向
LAMBDA_F_SPLIT = jnp.array([0.0, 0.0, 1.0])  # 分離方向

@jit
def compute_embedding(r, spins, charges, k_vectors, c, direction, phase):
    """
    電子 i の埋め込み h_i(r) を計算。
    Args:
        r: 電子の位置 (n_el, 3)
        spins: スピン (n_el,)
        charges: 電荷 (n_el,)
        k_vectors: 運動量ベクトル (n_el, 3)
        c: カットオフ
        direction: トランザクション方向
        phase: フェーズ因子 (n_el,)
    Returns:
        h: 埋め込み (n_el, embedding_dim)
        lambda_f: 選択された方向テンソル (3,)
    """
    n_el = r.shape[0]
    h = jnp.zeros((n_el, EMBEDDING_DIM), dtype=jnp.complex64)
    
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))
    mask = (dists < c) & (dists > 0)
    
    # トランザクションの適用
    delta_lambda_c = 1.0 / (dists + 1e-10) * mask  # rho_T の簡略化
    delta_lambda_c = delta_lambda_c * phase[:, None]  # フェーズ因子の適用
    
    # 方向テンソルの選択
    lambda_f = jnp.where(direction == 0, LAMBDA_F_BIND,
                         jnp.where(direction == 1, LAMBDA_F_MOVE, LAMBDA_F_SPLIT))
    
    # 方向に応じた重み（lambda_f の大きさに基づく）
    weight_factor = jnp.linalg.norm(lambda_f)  # ベクトルのノルム
    weights = jnp.ones_like(delta_lambda_c) * weight_factor
    
    # 埋め込みの更新
    for i in range(n_el):
        h_i = jnp.sum(delta_lambda_c[i] * weights[i], axis=0)
        h = h.at[i].set(h_i)
    
    return h, lambda_f
