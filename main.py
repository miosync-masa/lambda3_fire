# テスト用のダミーデータ
if __name__ == "__main__":
    r = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=jnp.float32)
    spins = jnp.array([1, -1, 1], dtype=jnp.float32)
    charges = jnp.array([1.0, 1.0, 1.0], dtype=jnp.float32)
    k_vectors = jnp.array([[0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]], dtype=jnp.float32)

    # VMC ステップの実行（更新版）
    h, c, direction, rho_T, lambda_f, psi, energy = lambda3_fire_vmc_step(r, spins, charges, k_vectors)
    
    print(f"Dynamic cutoff: {c}")
    print(f"rho_T (with entropy): {rho_T}")
    print(f"Transaction direction (0: bind, 1: move, 2: split): {direction}")
    print(f"Lambda_F direction tensor: {lambda_f}")
    print(f"Embedding (first electron): {h[0]}")
    print(f"Wavefunction psi: {psi}")
    print(f"Energy: {energy}")
