import wandb

def lambda3_fire_vmc_with_wandb(r, spins, charges, k_vectors, n_steps=1000, project_name="lambda3-fire"):
    """
    VMC ステップを wandb で可視化。
    """
    wandb.init(project=project_name, config={"n_steps": n_steps})
    r_current = r
    lambda_f_history = []
    
    for step in range(n_steps):
        c = compute_dynamic_cutoff(r_current)
        rho_T = compute_rho_T(r_current, spins, c)
        direction, phase = compute_transaction_direction(r_current, spins, charges, k_vectors, rho_T, c)
        h, lambda_f = compute_embedding(r_current, spins, charges, k_vectors, c, direction, phase)
        Phi = generate_orbitals(h, r.shape[0])
        J = compute_hybrid_jastrow(r_current, h, r.shape[0])
        psi = J * jnp.sum(jnp.array([jnp.linalg.det(Phi[d]) for d in range(N_DET)]))
        
        r_samples, sigmas = adaptive_metropolis_sampling(r_current, h, Phi, J, n_steps=1)
        r_current = r_samples[-1]
        lambda_f_history.append(lambda_f)
        
        # wandb で記録
        wandb.log({
            "rho_T": float(rho_T),
            "Psi": float(jnp.abs(psi)),
            "lambda_f_bind": float(lambda_f[0]),
            "lambda_f_move": float(lambda_f[1]),
            "lambda_f_split": float(lambda_f[2]),
            "sigma": float(sigmas[-1])
        })
        
        # 3D可視化（ステップ2）を統合
        if step % 100 == 0:
            visualize_lambda_f(np.array(lambda_f_history), step)
    
    wandb.finish()
    return r_current

# 実行例
if __name__ == "__main__":
    r = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    spins = jnp.array([1, -1, 1], dtype=jnp.float32)
    charges = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    k_vectors = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], dtype=jnp.float32)
    
    r_final = lambda3_fire_vmc_with_wandb(r, spins, charges, k_vectors)
    print(f"Final electron positions: {r_final}")
