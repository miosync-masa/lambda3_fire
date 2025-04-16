import jax.numpy as jnp
from jax import jit

# 動的カットオフの計算
@jit
def compute_dynamic_cutoff(r, c0=3.0, rho_T0=1.0, alpha=0.5, beta=0.3):
    # r: 電子の位置 (n_el, 3)
    n_el = r.shape[0]
    
    # 電子間距離の計算
    r_i = r[:, None, :]  # (n_el, 1, 3)
    r_j = r[None, :, :]  # (1, n_el, 3)
    dists = jnp.sqrt(jnp.sum((r_i - r_j) ** 2, axis=-1))  # (n_el, n_el)
    dists = dists + jnp.eye(n_el) * 1e10  # 対角成分を除外
    
    # rho_T の計算（距離の逆数和）
    rho_T = jnp.sum(1.0 / dists, where=dists < 10.0) / n_el
    
    # sigma_s の計算（近傍電子との共鳴度）
    mask = dists < c0  # 初期カットオフで近傍を仮定
    sigma_s_ij = jnp.exp(-0.1 * dists ** 2) * mask  # ガウス型共鳴度
    sigma_s = jnp.sum(sigma_s_ij) / jnp.sum(mask)
    
    # 動的カットオフの計算
    c = c0 * (rho_T / rho_T0) ** alpha * sigma_s ** beta
    return c

# 例：電子位置 r からカットオフを計算
r = jnp.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # ダミーデータ
c = compute_dynamic_cutoff(r)
print(f"Dynamic cutoff: {c}")
