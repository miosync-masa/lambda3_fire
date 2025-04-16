from tensorboardX import SummaryWriter

def lambda3_fire_vmc_with_visualization(r, spins, charges, k_vectors, n_steps=1000, log_dir="logs"):
    """
    VMC ステップを可視化付きで実行。
    """
    writer = SummaryWriter(log_dir)
    r_current = r
    for step in range(n_steps):
        # VMC ステップ
        c = compute_dynamic_cutoff(r_current)
        rho_T = compute_rho_T(r_current, spins, c)
        direction, phase = compute_transaction_direction(r_current, spins, charges, k_vectors, rho_T, c)
        h, lambda_f = compute_embedding(r_current, spins, charges, k_vectors, c, direction, phase)
        Phi = generate_orbitals(h, r.shape[0])
        J = compute_attention_jastrow(h, r.shape[0])
        psi = J * jnp.sum(jnp.array([jnp.linalg.det(Phi[d]) for d in range(N_DET)]))
        
        # サンプリング
        r_samples = metropolis_sampling(r_current, h, Phi, J, n_steps=1)
        r_current = r_samples[-1]
        
        # 可視化
        writer.add_scalar("rho_T", rho_T, step)
        writer.add_scalar("Psi", jnp.abs(psi), step)
        writer.add_histogram("lambda_f", lambda_f, step)
    
    writer.close()
    return r_current

# 実行例
if __name__ == "__main__":
    r = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    spins = jnp.array([1, -1, 1], dtype=jnp.float32)
    charges = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    k_vectors = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], dtype=jnp.float32)
    
    r_final = lambda3_fire_vmc_with_visualization(r, spins, charges, k_vectors)
    print(f"Final electron positions: {r_final}")
