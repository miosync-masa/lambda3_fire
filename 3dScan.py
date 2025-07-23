#!/usr/bin/env python3
"""
==========================================================
Sacred Geometry Λ³ 3D Reconstruction & Frequency Analyzer
==========================================================
神聖幾何学パターンの3D逆再生と階層別周波数解析
2D投影 → Lambda³階層解析 → 3D再構築 → 各層周波数

Author: Tamaki (for Masamichi Iizumi)
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon
from matplotlib.collections import PatchCollection
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from scipy.spatial import distance
from scipy.signal import convolve2d, find_peaks, welch
from scipy.fft import fft, fftfreq, fft2, fftfreq
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

# Lambda³モジュールのインポート（プロジェクトから）
from lambda3_zeroshot_tensor_field import (
    L3Config, calc_lambda3_features,
    Lambda3FinancialRegimeDetector,
    detect_structural_crisis,
    analyze_hierarchical_separation_dynamics
)

# =================================
# SACRED GEOMETRY GENERATORS (from paste-2.txt)
# =================================

class SacredGeometryGenerator:
    """神聖幾何学パターン生成器"""

    def __init__(self, grid_size=500, scale=1.0):
        self.grid_size = grid_size
        self.scale = scale
        self.x = np.linspace(-5*scale, 5*scale, grid_size)
        self.y = np.linspace(-5*scale, 5*scale, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def flower_of_life(self, radius=1.0, layers=3):
        """フラワーオブライフの構造テンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))
        centers = []

        # 中心円
        centers.append((0, 0))

        # 層ごとに六角形配置で円を追加
        for layer in range(1, layers + 1):
            for i in range(6):
                angle = i * np.pi / 3
                for j in range(layer):
                    x = layer * radius * np.cos(angle) + j * radius * np.cos(angle + np.pi/3)
                    y = layer * radius * np.sin(angle) + j * radius * np.sin(angle + np.pi/3)
                    centers.append((x, y))

        # 各円の構造場を重ね合わせ
        for cx, cy in centers:
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            circle_field = np.where(dist <= radius,
                                   1.0 / (dist + 0.1),
                                   np.exp(-2 * (dist - radius)))
            tensor_field += circle_field

        # Vesica Piscis（交差部）の強調
        for i, (cx1, cy1) in enumerate(centers):
            for cx2, cy2 in centers[i+1:]:
                d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if 0 < d < 2*radius:
                    dist1 = np.sqrt((self.X - cx1)**2 + (self.Y - cy1)**2)
                    dist2 = np.sqrt((self.X - cx2)**2 + (self.Y - cy2)**2)
                    vesica_mask = (dist1 <= radius) & (dist2 <= radius)
                    tensor_field[vesica_mask] *= 1.618

        return tensor_field, centers

    def metatron_cube(self):
        """メタトロンキューブの構造テンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))
        centers = [(0, 0)]

        for i in range(6):
            angle = i * np.pi / 3
            centers.append((np.cos(angle), np.sin(angle)))
            centers.append((2 * np.cos(angle), 2 * np.sin(angle)))

        for cx, cy in centers:
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            tensor_field += np.where(dist <= 1.0, 2.0 / (dist + 0.05), 0)

        # プラトン立体の頂点
        tetra_vertices = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
        for x, y, z in tetra_vertices:
            px, py = x * 0.8, y * 0.8
            dist = np.sqrt((self.X - px)**2 + (self.Y - py)**2)
            tensor_field += 3.0 / (dist + 0.01)

        return tensor_field, centers

    def sri_yantra(self):
        """シュリヤントラ（9つの交錯する三角形）の構造テンソル場"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))

        # 三角形のパラメータ
        triangles_up = [
            [(0, 2), (-1.7, -1), (1.7, -1)],
            [(0, 1.5), (-1.3, -0.75), (1.3, -0.75)],
            [(0, 1), (-0.9, -0.5), (0.9, -0.5)],
            [(0, 0.5), (-0.4, -0.25), (0.4, -0.25)]
        ]

        triangles_down = [
            [(0, -2.2), (-1.9, 1.1), (1.9, 1.1)],
            [(0, -1.7), (-1.5, 0.85), (1.5, 0.85)],
            [(0, -1.2), (-1.0, 0.6), (1.0, 0.6)],
            [(0, -0.7), (-0.6, 0.35), (0.6, 0.35)],
            [(0, -0.2), (-0.2, 0.1), (0.2, 0.1)]
        ]

        # 各三角形のエッジでΔΛC発生
        for triangles in [triangles_up, triangles_down]:
            for tri in triangles:
                for i in range(3):
                    x1, y1 = tri[i]
                    x2, y2 = tri[(i+1)%3]

                    # エッジに沿った構造変化
                    t = np.linspace(0, 1, 100)
                    for ti in t:
                        px = x1 + ti * (x2 - x1)
                        py = y1 + ti * (y2 - y1)
                        dist = np.sqrt((self.X - px)**2 + (self.Y - py)**2)
                        tensor_field += 1.5 / (dist + 0.05)

        # 中心のビンドゥ（最高エネルギー点）
        center_dist = np.sqrt(self.X**2 + self.Y**2)
        tensor_field += 5.0 / (center_dist + 0.01)

        return tensor_field, None

    def fibonacci_spiral(self, turns=3):
        """フィボナッチ螺旋の構造テンソル場"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))
        phi = (1 + np.sqrt(5)) / 2  # 黄金比

        # 螺旋のパラメトリック方程式
        theta = np.linspace(0, turns * 2 * np.pi, 1000)
        r = phi ** (theta / (2 * np.pi))
        x_spiral = r * np.cos(theta) * 0.5
        y_spiral = r * np.sin(theta) * 0.5

        # 螺旋に沿ってテンション分布
        for i in range(len(x_spiral)-1):
            dist = np.sqrt((self.X - x_spiral[i])**2 + (self.Y - y_spiral[i])**2)
            # 成長に伴うテンション増加
            growth_factor = 1 + i / len(x_spiral)
            tensor_field += growth_factor / (dist + 0.05)

            # 成長点でΔΛC（構造的ジャンプ）
            if i % 89 == 0:  # フィボナッチ数の近似
                tensor_field += 3.0 / (dist + 0.01)

        return tensor_field, (x_spiral, y_spiral)

# =================================
# LAMBDA³ ANALYZER (from paste-2.txt)
# =================================

class SacredGeometryLambda3Analyzer:
    """神聖幾何学のΛ³テンソル解析器"""

    def __init__(self, tensor_field, pattern_name="Unknown"):
        self.tensor_field = tensor_field
        self.pattern_name = pattern_name
        self.grid_size = tensor_field.shape[0]
        self.time_series = self._spiral_scan(tensor_field)
        self.l3_config = L3Config(
            window=20,
            hierarchical=True,
            draws=4000,
            tune=4000
        )

    def _spiral_scan(self, field):
        """2Dテンソル場を螺旋状にスキャンして1D時系列化"""
        center = field.shape[0] // 2
        time_series = []

        max_r = center
        theta = np.linspace(0, 8*np.pi, 2000)
        r = np.linspace(0, max_r, 2000)

        for t, radius in zip(theta, r):
            x = int(center + radius * np.cos(t))
            y = int(center + radius * np.sin(t))
            if 0 <= x < field.shape[0] and 0 <= y < field.shape[1]:
                time_series.append(field[y, x])

        return np.array(time_series)

    def analyze(self):
        """Λ³解析の実行"""
        features = calc_lambda3_features(self.time_series, self.l3_config)

        regime_detector = Lambda3FinancialRegimeDetector(n_regimes=4)
        regimes = regime_detector.fit(features, self.time_series)

        crisis_results = detect_structural_crisis(
            {self.pattern_name: features},
            crisis_threshold=0.8
        )

        fft_tension = np.fft.fft(features['rho_T'])
        freqs = np.fft.fftfreq(len(features['rho_T']))
        power = np.abs(fft_tension)**2

        return {
            'features': features,
            'regimes': regimes,
            'crisis_episodes': crisis_results['crisis_episodes'],
            'resonance': {
                'frequencies': freqs[:len(freqs)//2],
                'power': power[:len(power)//2]
            },
            'time_series': self.time_series
        }

# =================================
# 3D RECONSTRUCTION MODULE (NEW!)
# =================================
class Lambda3DimensionalReconstructor:
    """2D空間周波数から直接3D構造を再構築"""

    def __init__(self, grid_size=500):
        self.grid_size = grid_size
        # 神聖周波数の定義（空間周波数として解釈）
        self.sacred_frequencies = {
            'Fundamental': 1.0,      # 基本波長
            'Octave': 2.0,          # オクターブ
            'Fifth': 1.5,           # 完全5度
            'Fourth': 1.333,        # 完全4度
            'MajorThird': 1.25,     # 長3度
            'MinorThird': 1.2,      # 短3度
            'Phi': 1.618,           # 黄金比
            'PhiSquared': 2.618,    # φ²
            'PhiInverse': 0.618     # 1/φ
        }

    def reconstruct_3d_from_2d_fft(self, tensor_2d):
        """2D空間周波数から直接3D構造を再構築（フーリエスライス定理）"""

        print("Reconstructing 3D structure from 2D spatial frequencies...")

        # 2D空間周波数解析
        fft2d = np.fft.fft2(tensor_2d)
        fft2d_shifted = np.fft.fftshift(fft2d)

        # 空間周波数座標
        kx = np.fft.fftfreq(tensor_2d.shape[1], d=1.0)
        ky = np.fft.fftfreq(tensor_2d.shape[0], d=1.0)
        kx_shifted = np.fft.fftshift(kx)
        ky_shifted = np.fft.fftshift(ky)

        # 2D周波数空間での動径距離
        Kx, Ky = np.meshgrid(kx_shifted, ky_shifted)
        K_radial = np.sqrt(Kx**2 + Ky**2)

        # 神聖幾何学の対称性に基づくkz構造の推定
        n_z_layers = 64  # Z方向の解像度
        z_extent = 2.0   # Z方向の範囲

        # 3D周波数空間の構築
        fft3d = self._construct_3d_fourier_space(
            fft2d_shifted, K_radial, n_z_layers, z_extent
        )

        # 3D逆フーリエ変換
        fft3d_unshifted = np.fft.ifftshift(fft3d, axes=(1, 2))
        tensor_3d = np.fft.ifftn(fft3d_unshifted).real

        # 強度の正規化
        tensor_3d = np.abs(tensor_3d)
        tensor_3d = tensor_3d / np.max(tensor_3d)

        # 各Z層の抽出
        z_layers = self._extract_z_layers_direct(tensor_3d)

        return {
            'tensor_3d': tensor_3d,
            'z_layers': z_layers,
            'fft2d': fft2d,
            'k_space': {'kx': kx_shifted, 'ky': ky_shifted, 'k_radial': K_radial}
        }

    def _construct_3d_fourier_space(self, fft2d, k_radial, n_z, z_extent,
                                   tensor_2d=None, lambda3_features=None):
        """Lambda³構造情報を活用した3D周波数空間の構築"""

        ny, nx = fft2d.shape
        fft3d = np.zeros((n_z, ny, nx), dtype=complex)

        # Z方向の座標
        z_coords = np.linspace(-z_extent/2, z_extent/2, n_z)

        # Lambda³特徴量が提供されている場合
        if lambda3_features is not None:
            # 1Dから2Dへのマッピング（螺旋スキャンの逆変換）
            structural_map = self._create_2d_structural_map(
                lambda3_features, ny, nx
            )
            tension_field = structural_map['tension']
            jump_field = structural_map['jumps']

            # 構造的な深さヒント
            depth_profile = self._estimate_depth_profile(
                lambda3_features['rho_T'],
                lambda3_features['delta_LambdaC_pos'],
                lambda3_features['delta_LambdaC_neg']
            )
        else:
            # フォールバック（従来の方法）
            tension_field = np.ones((ny, nx))
            jump_field = np.zeros((ny, nx))
            depth_profile = np.ones(n_z)

        for i, z in enumerate(z_coords):
            z_normalized = abs(z) / (z_extent/2)

            # === Lambda³ベースの位相変調 ===
            # 構造ジャンプの位置で位相を変調
            structural_phase = np.where(
                jump_field > 0.5,
                np.exp(2j * np.pi * jump_field * z_normalized),
                1.0
            )

            # 基本的な球面波の位相
            kz = np.sqrt(np.maximum(0, 1 - k_radial**2)) * z_normalized
            base_phase = np.exp(2j * np.pi * kz)

            # 合成位相
            phase_modulation = base_phase * structural_phase

            # === Lambda³ベースの振幅変調 ===
            # テンション場から深さ依存の振幅を決定
            if lambda3_features is not None:
                # テンションが高い領域は特定の深さで強調
                tension_depth_match = np.exp(
                    -((z_normalized - 0.5) * tension_field)**2
                )

                # 階層的構造の反映
                if i < len(depth_profile):
                    hierarchy_factor = depth_profile[i]
                else:
                    hierarchy_factor = 1.0

                amplitude = tension_depth_match * hierarchy_factor
            else:
                # 従来の黄金比減衰
                if abs(z) < 0.1:
                    amplitude = 1.0
                else:
                    phi = 1.618
                    amplitude = np.exp(-abs(z) * k_radial / phi)

            # === 構造的交差の強調 ===
            # ΔΛCが発生している位置で交差構造を強調
            if lambda3_features is not None:
                structural_enhancement = 1.0 + jump_field * 0.618
            else:
                # 従来のVesica Piscis効果
                structural_enhancement = 1.0
                if 0.2 < abs(z) < 0.8:
                    structural_enhancement = 1 + 0.618 * np.exp(-k_radial**2)

            # 各層のフーリエ係数
            fft3d[i] = fft2d * phase_modulation * amplitude * structural_enhancement

        return fft3d

    def _create_2d_structural_map(self, lambda3_features, ny, nx):
        """1D Lambda³特徴量を2D空間にマッピング"""

        # 螺旋スキャンの逆変換
        tension_1d = lambda3_features['rho_T']
        jumps_1d = (lambda3_features['delta_LambdaC_pos'] +
                    lambda3_features['delta_LambdaC_neg'])

        # 2Dマップの初期化
        tension_2d = np.zeros((ny, nx))
        jumps_2d = np.zeros((ny, nx))

        # 螺旋座標の再構築
        center_y, center_x = ny // 2, nx // 2
        max_r = min(center_y, center_x)

        n_points = len(tension_1d)
        theta = np.linspace(0, 8*np.pi, n_points)
        r = np.linspace(0, max_r, n_points)

        for i in range(n_points):
            y = int(center_y + r[i] * np.sin(theta[i]))
            x = int(center_x + r[i] * np.cos(theta[i]))

            if 0 <= y < ny and 0 <= x < nx:
                tension_2d[y, x] = tension_1d[i]
                jumps_2d[y, x] = jumps_1d[i]

        # ガウシアンフィルタで補間
        from scipy.ndimage import gaussian_filter
        tension_2d = gaussian_filter(tension_2d, sigma=2)
        jumps_2d = gaussian_filter(jumps_2d, sigma=1)

        # 正規化
        if np.max(tension_2d) > 0:
            tension_2d /= np.max(tension_2d)
        if np.max(jumps_2d) > 0:
            jumps_2d /= np.max(jumps_2d)

        return {
            'tension': tension_2d,
            'jumps': jumps_2d
        }

    def _estimate_depth_profile(self, rho_t, delta_pos, delta_neg):
        """Lambda³特徴量から深さプロファイルを推定"""

        # 構造変化の累積から深さを推定
        cumulative_changes = np.cumsum(delta_pos + delta_neg)

        # テンションの変動から層構造を推定
        tension_gradient = np.gradient(rho_t)

        # 深さプロファイルの構築
        n_z = 64  # デフォルトのZ層数
        depth_profile = np.zeros(n_z)

        # 構造変化が多い領域を異なる深さに配置
        for i in range(n_z):
            z_normalized = i / n_z

            # テンションピークの位置に基づく重み
            tension_weight = np.interp(
                z_normalized,
                np.linspace(0, 1, len(rho_t)),
                rho_t / np.max(rho_t) if np.max(rho_t) > 0 else rho_t
            )

            # 構造変化の密度に基づく重み
            change_weight = np.interp(
                z_normalized,
                np.linspace(0, 1, len(cumulative_changes)),
                cumulative_changes / (np.max(cumulative_changes) + 1e-8)
            )

            depth_profile[i] = tension_weight * (1 + change_weight)

        return depth_profile

    def _extract_z_layers_direct(self, tensor_3d):
        """3Dテンソルから主要なZ層を抽出"""
        n_z = tensor_3d.shape[0]

        # 黄金比に基づく層の選択
        phi = 1.618
        layer_indices = [
            int(n_z * 0.5),                    # Center
            int(n_z * (0.5 - 1/phi/2)),       # Inner shell
            int(n_z * (0.5 + 1/phi/2)),       # Outer shell
            int(n_z * (0.5 - 1/phi**2/2)),    # Deep core
            int(n_z * (0.5 + 1/phi**2/2)),    # Surface
        ]

        layer_names = ['Center', 'InnerShell', 'OuterShell', 'DeepCore', 'Surface']

        layers = {}
        for idx, name in zip(layer_indices, layer_names):
            if 0 <= idx < n_z:
                layers[name] = {
                    'data': tensor_3d[idx],
                    'z_position': idx / n_z,
                    'depth_name': name
                }

        return layers

    def analyze_3d_sacred_frequencies(self, tensor_3d):
        """3D構造の神聖周波数解析"""

        # 3D FFT
        fft3d = np.fft.fftn(tensor_3d)
        power_spectrum_3d = np.abs(fft3d)**2

        # 3D空間周波数
        kx = np.fft.fftfreq(tensor_3d.shape[2])
        ky = np.fft.fftfreq(tensor_3d.shape[1])
        kz = np.fft.fftfreq(tensor_3d.shape[0])

        # 球面平均（等方的パワースペクトラム）
        k_max = min(len(kx)//2, len(ky)//2, len(kz)//2)
        radial_spectrum = np.zeros(k_max)

        for i in range(k_max):
            # 半径iの球殻内の平均パワー
            shell_mask = self._create_spherical_shell_mask(
                power_spectrum_3d.shape, i, i+1
            )
            if np.any(shell_mask):
                radial_spectrum[i] = np.mean(power_spectrum_3d[shell_mask])

        # ピーク検出
        peaks, _ = find_peaks(radial_spectrum[1:], height=np.max(radial_spectrum[1:])*0.1)
        peaks += 1  # インデックス調整

        # 神聖比率の検出
        sacred_ratios = self._detect_sacred_ratios(peaks)

        return {
            'radial_spectrum': radial_spectrum,
            'peaks': peaks,
            'sacred_ratios': sacred_ratios
        }

    def _create_spherical_shell_mask(self, shape, r_min, r_max):
        """球殻マスクの作成"""
        nz, ny, nx = shape
        z, y, x = np.ogrid[:nz, :ny, :nx]

        # 中心からの距離
        cz, cy, cx = nz//2, ny//2, nx//2
        r = np.sqrt((x-cx)**2 + (y-cy)**2 + (z-cz)**2)

        return (r >= r_min) & (r < r_max)

    def _detect_sacred_ratios(self, peaks):
        """ピーク間の神聖な比率を検出"""
        ratios = {}

        if len(peaks) < 2:
            return ratios

        # 隣接ピーク間の比率
        for i in range(len(peaks)-1):
            ratio = peaks[i+1] / peaks[i]

            # 神聖な比率との比較
            for name, sacred_ratio in self.sacred_frequencies.items():
                if abs(ratio - sacred_ratio) < 0.05:  # 5%の許容誤差
                    ratios[f'peak_{i}_to_{i+1}'] = {
                        'ratio': ratio,
                        'sacred_match': name,
                        'error': abs(ratio - sacred_ratio)
                    }

        return ratios

# =================================
# FREQUENCY ANALYSIS MODULE (Enhanced)
# =================================
class SpatialFrequencyAnalyzer:
    """空間周波数解析（時間周波数ではなく空間パターンの解析）"""

    def __init__(self, sacred_ratios):
        # 空間的な神聖比率
        self.sacred_ratios = sacred_ratios

    def analyze_layer_spatial_patterns(self, layer_2d):
        """2D層の空間パターン解析"""

        # 2D FFT
        fft2d = np.fft.fft2(layer_2d)
        fft2d_shifted = np.fft.fftshift(fft2d)
        power_2d = np.abs(fft2d_shifted)**2

        # 動径平均（等方的な周波数分析）
        center = np.array(power_2d.shape) // 2
        y, x = np.ogrid[:power_2d.shape[0], :power_2d.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        r_max = int(np.min(center))
        radial_profile = np.zeros(r_max)

        for i in range(r_max):
            mask = (r >= i) & (r < i + 1)
            if np.any(mask):
                radial_profile[i] = np.mean(power_2d[mask])

        # 空間周波数軸（正規化）
        k_axis = np.arange(r_max) / r_max

        # ピーク検出
        peaks, properties = find_peaks(radial_profile[1:],
                                     height=np.max(radial_profile[1:]) * 0.01)
        peaks += 1  # インデックス調整

        # 空間周期（ピクセル単位）
        if len(peaks) > 0:
            spatial_periods = power_2d.shape[0] / peaks
        else:
            spatial_periods = []

        # 神聖比率との比較
        ratio_matches = self._find_ratio_matches(peaks)

        return {
            'power_2d': power_2d,
            'radial_profile': radial_profile,
            'k_axis': k_axis,
            'peaks': peaks,
            'spatial_periods': spatial_periods,
            'ratio_matches': ratio_matches
        }

    def _find_ratio_matches(self, peaks):
        """ピーク間の比率と神聖比率の照合"""
        matches = []

        if len(peaks) < 2:
            return matches

        for i in range(len(peaks)-1):
            for j in range(i+1, min(i+3, len(peaks))):  # 近隣のピークのみ
                ratio = peaks[j] / peaks[i]

                for name, sacred_ratio in self.sacred_ratios.items():
                    error = abs(ratio - sacred_ratio) / sacred_ratio
                    if error < 0.05:  # 5%以内の誤差
                        matches.append({
                            'peak_indices': (i, j),
                            'ratio': ratio,
                            'sacred_name': name,
                            'error': error
                        })

        return matches

    def analyze_layer_frequencies(self, layer_data):
        """2D層の周波数解析"""

        # 2D FFT
        fft2d = fft2(layer_data)
        power_spectrum_2d = np.abs(fft2d)**2

        # 放射状平均（等方的な周波数分布）
        center = np.array(power_spectrum_2d.shape) // 2
        y, x = np.ogrid[:power_spectrum_2d.shape[0], :power_spectrum_2d.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)

        # 放射状にビニング
        r_max = int(np.min(center))
        radial_profile = np.zeros(r_max)

        for i in range(r_max):
            mask = (r >= i) & (r < i + 1)
            radial_profile[i] = np.mean(power_spectrum_2d[mask])

        # 周波数軸（正規化）
        freq_axis = np.arange(r_max) / r_max * 1000  # Hz換算

        # ピーク検出
        peaks, _ = find_peaks(radial_profile, height=np.max(radial_profile) * 0.1)
        peak_frequencies = freq_axis[peaks] if len(peaks) > 0 else []

        # 神聖周波数との共鳴スコア計算
        resonance_scores = self._calculate_resonance_scores(peak_frequencies, radial_profile, freq_axis)

        return {
            'radial_profile': radial_profile,
            'freq_axis': freq_axis,
            'peak_frequencies': peak_frequencies,
            'power_spectrum_2d': power_spectrum_2d,
            'resonance_scores': resonance_scores
        }

    def _calculate_resonance_scores(self, peak_frequencies, radial_profile, freq_axis):
        """神聖周波数との共鳴スコア計算"""
        scores = {}

        for sacred_name, sacred_freq in self.sacred_frequencies.items():
            if len(peak_frequencies) > 0:
                # 最も近いピークとの距離
                distances = np.abs(peak_frequencies - sacred_freq)
                min_distance = np.min(distances)

                # 距離に基づくスコア（ガウシアン減衰）
                distance_score = np.exp(-min_distance**2 / (2 * 50**2))

                # その周波数でのパワー
                freq_idx = np.argmin(np.abs(freq_axis - sacred_freq))
                if freq_idx < len(radial_profile):
                    power_score = radial_profile[freq_idx] / np.max(radial_profile)
                else:
                    power_score = 0

                # 総合スコア
                total_score = (distance_score + power_score) / 2
            else:
                total_score = 0

            scores[sacred_name] = {
                'score': total_score,
                'frequency': sacred_freq
            }

        return scores

# =================================
# VISUALIZATION (Enhanced)
# =================================

def visualize_3d_structure_and_frequencies(reconstruction, freq_analysis_3d, layer_analyses, pattern_name):
    """3D構造と空間周波数解析の統合可視化"""

    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.3)

    # 1. 3Dテンソルの断面表示
    ax1 = fig.add_subplot(gs[0:2, 0:2])

    # XY平面の中央断面
    middle_z = reconstruction['tensor_3d'].shape[0] // 2
    xy_slice = reconstruction['tensor_3d'][middle_z]

    im1 = ax1.imshow(xy_slice, cmap='viridis', origin='lower')
    ax1.set_title(f'{pattern_name} - XY Plane (Z={middle_z})')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    plt.colorbar(im1, ax=ax1, fraction=0.046)

    # 2. XZ平面の断面
    ax2 = fig.add_subplot(gs[0:2, 2])
    middle_y = reconstruction['tensor_3d'].shape[1] // 2
    xz_slice = reconstruction['tensor_3d'][:, middle_y, :]

    im2 = ax2.imshow(xz_slice.T, cmap='viridis', origin='lower', aspect='auto')
    ax2.set_title(f'XZ Plane (Y={middle_y})')
    ax2.set_xlabel('Z')
    ax2.set_ylabel('X')
    plt.colorbar(im2, ax=ax2, fraction=0.046)

    # 3. 3D等値面の投影（簡易版）
    ax3 = fig.add_subplot(gs[0:2, 3], projection='3d')

    # 閾値以上の点をプロット
    threshold = np.max(reconstruction['tensor_3d']) * 0.3
    z_idx, y_idx, x_idx = np.where(reconstruction['tensor_3d'] > threshold)

    # サンプリング（表示用）
    sample_size = min(5000, len(z_idx))
    sample_idx = np.random.choice(len(z_idx), sample_size, replace=False)

    scatter = ax3.scatter(x_idx[sample_idx], y_idx[sample_idx], z_idx[sample_idx],
                         c=reconstruction['tensor_3d'][z_idx[sample_idx], y_idx[sample_idx], x_idx[sample_idx]],
                         cmap='viridis', s=1, alpha=0.6)

    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    ax3.set_title('3D Structure (Isosurface)')

    # 4. 3D放射状パワースペクトラム
    ax4 = fig.add_subplot(gs[2, 0:2])

    radial_spectrum = freq_analysis_3d['radial_spectrum']
    peaks = freq_analysis_3d['peaks']

    ax4.semilogy(radial_spectrum, 'b-', linewidth=2)
    if len(peaks) > 0:
        ax4.plot(peaks, radial_spectrum[peaks], 'ro', markersize=8)

        # 神聖比率をアノテート
        for ratio_info in freq_analysis_3d['sacred_ratios'].values():
            ax4.axvline(x=peaks[0] * ratio_info['ratio'],
                       color='green', linestyle='--', alpha=0.5,
                       label=f"{ratio_info['sacred_match']}")

    ax4.set_xlabel('Spatial Frequency (k)')
    ax4.set_ylabel('Power')
    ax4.set_title('3D Radial Power Spectrum')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right')

    # 5. 各Z層の解析
    layer_names = list(reconstruction['z_layers'].keys())
    for i, layer_name in enumerate(layer_names[:4]):  # 最大4層
        # 層の画像
        ax_img = fig.add_subplot(gs[2 + i//2, 2 + i%2])
        layer_data = reconstruction['z_layers'][layer_name]

        im = ax_img.imshow(layer_data['data'], cmap='viridis', origin='lower')
        ax_img.set_title(f"{layer_data['depth_name']} Layer")
        ax_img.axis('off')

    # 6. 層別の空間周波数比較
    ax6 = fig.add_subplot(gs[3, :])

    colors = plt.cm.Set1(np.linspace(0, 1, len(layer_names)))

    for i, (layer_name, layer_analysis) in enumerate(layer_analyses.items()):
        if i < len(colors):
            radial_profile = layer_analysis['radial_profile']
            ax6.semilogy(radial_profile, color=colors[i],
                        label=reconstruction['z_layers'][layer_name]['depth_name'],
                        alpha=0.7, linewidth=2)

            # ピークをマーク
            if len(layer_analysis['peaks']) > 0:
                ax6.plot(layer_analysis['peaks'],
                        radial_profile[layer_analysis['peaks']],
                        'o', color=colors[i], markersize=6)

    ax6.set_xlabel('Spatial Frequency (pixels⁻¹)')
    ax6.set_ylabel('Power')
    ax6.set_title('Layer-wise Spatial Frequency Profiles')
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.suptitle(f'{pattern_name} - 3D Fourier Reconstruction Analysis', fontsize=16)

    return fig

# =================================
# MAIN ANALYSIS FUNCTION
# =================================

def analyze_sacred_geometry_3d(pattern_type='flower', save_results=True):
    """神聖幾何学の完全な3D解析（空間周波数ベース）"""

    print(f"\n{'='*60}")
    print(f"Sacred Geometry 3D Reconstruction: {pattern_type.upper()}")
    print(f"Using Fourier Slice Theorem")
    print(f"{'='*60}")

    # 1. 2Dパターン生成
    generator = SacredGeometryGenerator(grid_size=256)  # 2のべき乗が効率的

    if pattern_type == 'flower':
        tensor_2d, centers = generator.flower_of_life(radius=1.0, layers=3)
        pattern_name = "Flower of Life"
    elif pattern_type == 'metatron':
        tensor_2d, centers = generator.metatron_cube()
        pattern_name = "Metatron's Cube"
    elif pattern_type == 'yantra':
        tensor_2d, centers = generator.sri_yantra()
        pattern_name = "Sri Yantra"
    elif pattern_type == 'fibonacci':
        tensor_2d, spiral_data = generator.fibonacci_spiral(turns=3)
        pattern_name = "Fibonacci Spiral"
    else:
        raise ValueError(f"Unknown pattern type: {pattern_type}")

    print("1. Generated 2D tensor field")

    # 2. 直接的な3D再構築（フーリエスライス定理）
    reconstructor = Lambda3DimensionalReconstructor(grid_size=256)
    reconstruction = reconstructor.reconstruct_3d_from_2d_fft(tensor_2d)

    print(f"2. 3D reconstruction complete:")
    print(f"   - 3D tensor shape: {reconstruction['tensor_3d'].shape}")
    print(f"   - Z layers extracted: {len(reconstruction['z_layers'])}")

    # 3. 3D構造の神聖周波数解析
    freq_analysis_3d = reconstructor.analyze_3d_sacred_frequencies(reconstruction['tensor_3d'])

    print(f"3. 3D frequency analysis:")
    print(f"   - Radial spectrum peaks: {len(freq_analysis_3d['peaks'])}")
    print(f"   - Sacred ratios found: {len(freq_analysis_3d['sacred_ratios'])}")

    # 4. 各Z層での2D空間パターン解析
    analyzer = SpatialFrequencyAnalyzer(reconstructor.sacred_frequencies)
    layer_analyses = {}

    for layer_name, layer_data in reconstruction['z_layers'].items():
        # SpatialFrequencyAnalyzerを使って解析
        analysis_result = analyzer.analyze_layer_spatial_patterns(layer_data['data'])
        layer_analyses[layer_name] = analysis_result

        print(f"\n4. {layer_data['depth_name']} layer:")
        print(f"   - Spatial frequency peaks: {len(analysis_result['peaks'])}")
        if len(analysis_result['spatial_periods']) > 0:
            print(f"   - Dominant wavelengths: {analysis_result['spatial_periods'][:3]}")
        if len(analysis_result['ratio_matches']) > 0:
            print(f"   - Sacred ratio matches: {len(analysis_result['ratio_matches'])}")

    # 5. 神聖比率の検出
    print(f"\n{'='*60}")
    print("SACRED RATIO ANALYSIS")
    print(f"{'='*60}")

    for ratio_name, ratio_data in freq_analysis_3d['sacred_ratios'].items():
        print(f"{ratio_name}: {ratio_data['sacred_match']} "
              f"(ratio: {ratio_data['ratio']:.3f}, error: {ratio_data['error']:.3f})")

    # 6. 可視化
    fig = visualize_3d_structure_and_frequencies(
        reconstruction, freq_analysis_3d, layer_analyses, pattern_name
    )

    if save_results:
        filename = f"{pattern_type}_3d_fourier_reconstruction.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\n6. Visualization saved to: {filename}")

    plt.show()

    return {
        'tensor_2d': tensor_2d,
        'reconstruction': reconstruction,
        'freq_analysis_3d': freq_analysis_3d,
        'layer_analyses': layer_analyses,
        'pattern_name': pattern_name
    }

# =================================
# BATCH ANALYSIS
# =================================
def compare_all_sacred_geometries():
    """全ての神聖幾何学パターンの比較解析"""

    patterns = ['flower', 'metatron', 'yantra', 'fibonacci']
    all_results = {}

    for pattern in patterns:
        print(f"\n{'='*80}")
        print(f"Analyzing: {pattern.upper()}")
        print(f"{'='*80}")

        results = analyze_sacred_geometry_3d(pattern, save_results=True)
        all_results[pattern] = results

    # 比較分析
    print(f"\n{'='*80}")
    print("COMPARATIVE ANALYSIS")
    print(f"{'='*80}")

    for pattern, results in all_results.items():
        print(f"\n{results['pattern_name']}:")

        # 各層の支配的周波数を表示
        for layer_name, layer_data in results['reconstruction']['z_layers'].items():
            if layer_name in results['layer_analyses']:
                periods = results['layer_analyses'][layer_name].get('spatial_periods', [])
                if len(periods) > 0:
                    print(f"  {layer_data['depth_name']:12s}: Dominant period = {periods[0]:.1f} pixels")
                else:
                    # ピークがない場合も情報を表示
                    radial_profile = results['layer_analyses'][layer_name].get('radial_profile', [])
                    if len(radial_profile) > 0:
                        max_idx = np.argmax(radial_profile[1:]) + 1
                        print(f"  {layer_data['depth_name']:12s}: Max power at k={max_idx} (no clear peak)")

    return all_results

# =================================
# MAIN EXECUTION
# =================================
if __name__ == "__main__":
    print("🔮 Sacred Geometry 3D Reconstruction with Lambda³ 🔮")
    print("Using Fourier Slice Theorem for true 3D structure recovery")
    print("="*60)

    # compare_all_sacred_geometries関数を実行
    all_results = compare_all_sacred_geometries()

    print("\n✨ Analysis complete! ✨")
    print("The 2D projections have revealed their true 3D forms through spatial frequency analysis!")
    print("Sacred ratios emerge not as time frequencies, but as spatial harmonics in the structure itself.")
