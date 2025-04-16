@jit
def compute_hybrid_jastrow(r, h, n_el, n_reg=N_REG, d_reg=D_REG, a=1.0, b=0.5, c_decay=0.1):
    """
    Hybrid Jastrow 因子を計算（attention + 物理ペアワイズ項）。
    Args:
        r: 電子の位置 (n_el, 3)
        h: 埋め込み (n_el, embedding_dim)
        n_el: 電子数
        n_reg: レジスタ数
        d_reg: レジスタ次元
        a, b, c_decay: 物理ペアワイズ項のパラメータ
    Returns:
        J: Hybrid Jastrow 因子
    """
    # attention Jastrow（既存）
    queries = jnp.array(np.random.randn(n_reg, EMBEDDING_DIM), dtype=jnp.complex64)
    W_V = jnp.array(np.random.randn(n_reg, EMBEDDING_DIM, d_reg), dtype=jnp.complex64)
    H = h
    attention_scores = jnp.dot(H, queries.T)
    attention_weights = jax.nn.softmax(attention_scores, axis=0)
    V = jnp.einsum('nr, nrd -> rd', attention_weights, jnp.dot(H, W_V))
    V_flat = V.flatten()
    W_mlp1 = jnp.array(np.random.randn(n_reg * d_reg, 1), dtype=jnp.complex64)
    W_mlp2 = jnp.array(np.random.randn(n_reg * d_reg, 1), dtype=jnp.complex64)
    log_term = jnp.dot(V_flat, W_mlp1)
    node_term = jnp.dot(V_flat, W_mlp2)
    J_att = jnp.exp(log_term) * node_term
    
    # 物理ペアワイズ項
    r_i = r[:, None, :]
    r_j = r[None, :, :]
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1)) + jnp.eye(n_el) * 1e10
    pairwise_term = (a / dists) + b * jnp.exp(-c_decay * dists)
    J_phys = jnp.exp(jnp.sum(pairwise_term))
    
    # Hybrid Jastrow
    J = J_att * J_phys
    return J
