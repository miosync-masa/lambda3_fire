@jit
def adaptive_metropolis_sampling(r, h, Phi, J, n_steps=1000, sigma_init=0.1, target_acceptance=0.5):
    """
    Adaptive Metropolis-Hastings によるサンプリング。
    Args:
        r: 初期電子位置 (n_el, 3)
        h: 埋め込み
        Phi: 軌道行列
        J: Jastrow 因子
        n_steps: サンプリングステップ数
        sigma_init: 初期提案分布の標準偏差
        target_acceptance: 目標受理率
    Returns:
        r_samples: サンプリングされた位置 (n_steps, n_el, 3)
        sigmas: 各ステップの sigma (n_steps,)
    """
    r_samples = [r]
    sigmas = [sigma_init]
    sigma = sigma_init
    acceptance_count = 0
    
    current_psi = J * jnp.sum(jnp.array([jnp.linalg.det(Phi[d]) for d in range(N_DET)]))
    current_prob = jnp.abs(current_psi) ** 2
    
    key = jax.random.PRNGKey(0)
    for step in range(n_steps):
        key, subkey = jax.random.split(key)
        r_proposed = r + jax.random.normal(subkey, shape=r.shape) * sigma
        h_new, _ = compute_embedding(r_proposed, spins, charges, k_vectors, c, direction, phase)
        Phi_new = generate_orbitals(h_new, r.shape[0])
        J_new = compute_hybrid_jastrow(r_proposed, h_new, r.shape[0])
        proposed_psi = J_new * jnp.sum(jnp.array([jnp.linalg.det(Phi_new[d]) for d in range(N_DET)]))
        proposed_prob = jnp.abs(proposed_psi) ** 2
        
        acceptance_ratio = proposed_prob / current_prob
        if jax.random.uniform(subkey) < acceptance_ratio:
            r = r_proposed
            current_prob = proposed_prob
            acceptance_count += 1
        r_samples.append(r)
        
        # 受理率に基づく sigma の調整（100ステップごとに更新）
        if (step + 1) % 100 == 0:
            acceptance_rate = acceptance_count / 100
            if acceptance_rate < target_acceptance:
                sigma *= 0.9  # 受理率が低い場合、sigma を小さく
            else:
                sigma *= 1.1  # 受理率が高い場合、sigma を大きく
            acceptance_count = 0
        sigmas.append(sigma)
    
    return jnp.stack(r_samples), jnp.array(sigmas)
