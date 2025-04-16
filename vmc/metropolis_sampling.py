@jit
def metropolis_sampling(r, h, Phi, J, n_steps=1000, sigma=0.1):
    """
    Metropolis-Hastings によるサンプリング。
    Args:
        r: 初期電子位置 (n_el, 3)
        h: 埋め込み
        Phi: 軌道行列
        J: Jastrow 因子
        n_steps: サンプリングステップ数
        sigma: 提案分布の標準偏差
    Returns:
        r_samples: サンプリングされた位置 (n_steps, n_el, 3)
    """
    r_samples = [r]
    current_psi = J * jnp.sum(jnp.array([jnp.linalg.det(Phi[d]) for d in range(N_DET)]))
    current_prob = jnp.abs(current_psi) ** 2
    
    key = jax.random.PRNGKey(0)
    for _ in range(n_steps):
        key, subkey = jax.random.split(key)
        # 提案位置（ガウス分布）
        r_proposed = r + jax.random.normal(subkey, shape=r.shape) * sigma
        h_new, _ = compute_embedding(r_proposed, spins, charges, k_vectors, c, direction, phase)
        Phi_new = generate_orbitals(h_new, r.shape[0])
        J_new = compute_attention_jastrow(h_new, r.shape[0])
        proposed_psi = J_new * jnp.sum(jnp.array([jnp.linalg.det(Phi_new[d]) for d in range(N_DET)]))
        proposed_prob = jnp.abs(proposed_psi) ** 2
        
        # 受理確率
        acceptance_ratio = proposed_prob / current_prob
        if jax.random.uniform(subkey) < acceptance_ratio:
            r = r_proposed
            current_prob = proposed_prob
        r_samples.append(r)
    
    return jnp.stack(r_samples)
