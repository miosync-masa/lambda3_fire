@jit
def compute_laplacian_psi(r, h, Phi, J):
    """
    ラプラシアン ∇²Ψ を計算。
    Args:
        r: 電子の位置 (n_el, 3)
        h: 埋め込み (n_el, embedding_dim)
        Phi: 軌道行列 (n_det, n_el, n_el)
        J: Jastrow 因子
    Returns:
        laplacian_psi: ∇²Ψ
    """
    def psi_fn(r_flat):
        r = r_flat.reshape(-1, 3)
        h_new, _ = compute_embedding(r, spins, charges, k_vectors, c, direction, phase)
        Phi_new = generate_orbitals(h_new, n_el)
        J_new = compute_attention_jastrow(h_new, n_el)
        dets = jnp.array([jnp.linalg.det(Phi_new[d]) for d in range(N_DET)])
        return J_new * jnp.sum(dets)
    
    # ラプラシアンの計算
    r_flat = r.flatten()
    hessian = jax.hessian(psi_fn)(r_flat)  # ヘシアン (n_el*3, n_el*3)
    laplacian_psi = jnp.trace(hessian)  # ラプラシアン（トレース）
    return laplacian_psi
