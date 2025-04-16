@jit
def generate_orbitals(h, n_el, n_det=N_DET, embedding_dim=EMBEDDING_DIM):
    """
    埋め込み h から軌道行列 Φ_d を生成。
    Args:
        h: 埋め込み (n_el, embedding_dim)
        n_el: 電子数
        n_det: 行列式の数
        embedding_dim: 埋め込み次元
    Returns:
        Phi: 軌道行列 (n_det, n_el, n_el)
    """
    # 線形変換 (h -> hidden)
    W = jnp.array(np.random.randn(embedding_dim, embedding_dim), dtype=jnp.complex64)  # ダミー重み
    hidden = jnp.dot(h, W)  # (n_el, embedding_dim)
    
    # 活性化関数 (tanh)
    hidden = jnp.tanh(hidden)
    
    # 軌道への変換 (hidden -> orbital)
    W_orbital = jnp.array(np.random.randn(embedding_dim, n_el), dtype=jnp.complex64)  # ダミー重み
    orbitals = jnp.dot(hidden, W_orbital)  # (n_el, n_el)
    
    # 各行列式ごとに Φ_d を生成（簡略化：同一軌道を複製）
    Phi = jnp.stack([orbitals for _ in range(n_det)], axis=0)  # (n_det, n_el, n_el)
    return Phi
