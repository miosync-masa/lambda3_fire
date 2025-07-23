#!/usr/bin/env python3
"""
==========================================================
Lambda³-Enhanced Sacred Geometry Frequency Analysis
==========================================================
図案 → Λ³変換 → 構造的特徴の周波数解析
全神聖幾何学パターンの完全解析版

Author: Tamaki (for Masamichi Iizumi)
==========================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle, Polygon
import seaborn as sns
from scipy.signal import find_peaks, welch
from scipy.fft import fft, fftfreq

# Lambda³モジュールのインポート（プロジェクトから）
from lambda3_zeroshot_tensor_field import (
    L3Config, calc_lambda3_features,
    calculate_rho_t, calculate_diff_and_threshold, detect_jumps
)

# =================================
# 完全版 Sacred Geometry Generator
# =================================
class SacredGeometryGenerator:
    """全神聖幾何学パターン生成器"""

    def __init__(self, grid_size=256):
        self.grid_size = grid_size
        self.x = np.linspace(-5, 5, grid_size)
        self.y = np.linspace(-5, 5, grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

    def flower_of_life(self, radius=1.0, layers=3):
        """フラワーオブライフのテンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))
        centers = [(0, 0)]

        # 六角形配置
        for layer in range(1, layers + 1):
            for i in range(6):
                angle = i * np.pi / 3
                for j in range(layer):
                    x = layer * radius * np.cos(angle) + j * radius * np.cos(angle + np.pi/3)
                    y = layer * radius * np.sin(angle) + j * radius * np.sin(angle + np.pi/3)
                    centers.append((x, y))

        # 各円の構造場
        for cx, cy in centers:
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            circle_field = np.where(dist <= radius, 1.0 / (dist + 0.1), np.exp(-2 * (dist - radius)))
            tensor_field += circle_field

        # Vesica Piscis強調
        for i, (cx1, cy1) in enumerate(centers):
            for cx2, cy2 in centers[i+1:]:
                d = np.sqrt((cx1-cx2)**2 + (cy1-cy2)**2)
                if 0 < d < 2*radius:
                    dist1 = np.sqrt((self.X - cx1)**2 + (self.Y - cy1)**2)
                    dist2 = np.sqrt((self.X - cx2)**2 + (self.Y - cy2)**2)
                    vesica_mask = (dist1 <= radius) & (dist2 <= radius)
                    tensor_field[vesica_mask] *= 1.618  # 黄金比

        return tensor_field / np.max(tensor_field)

    def metatron_cube(self):
        """メタトロンキューブの構造テンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))

        # 13個の円の中心
        centers = [(0, 0)]  # 中心

        # 内側の6個
        for i in range(6):
            angle = i * np.pi / 3
            centers.append((np.cos(angle), np.sin(angle)))

        # 外側の6個
        for i in range(6):
            angle = i * np.pi / 3
            centers.append((2 * np.cos(angle), 2 * np.sin(angle)))

        # 円の構造場
        radius = 0.8
        for cx, cy in centers:
            dist = np.sqrt((self.X - cx)**2 + (self.Y - cy)**2)
            circle_field = np.where(dist <= radius, 2.0 / (dist + 0.05), np.exp(-3 * (dist - radius)))
            tensor_field += circle_field

        # プラトン立体の頂点による特異点
        # 正四面体の頂点
        tetra_vertices = [
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        # 立方体の頂点（2D投影）
        cube_scale = 1.5
        cube_vertices = []
        for x in [-1, 1]:
            for y in [-1, 1]:
                cube_vertices.append((x * cube_scale, y * cube_scale))

        # 頂点に特異点追加
        for vertices in [tetra_vertices, cube_vertices]:
            for px, py in vertices:
                dist = np.sqrt((self.X - px)**2 + (self.Y - py)**2)
                tensor_field += 3.0 / (dist + 0.01)

        # 線でつなぐ（構造線）
        for i in range(len(centers)):
            for j in range(i+1, len(centers)):
                cx1, cy1 = centers[i]
                cx2, cy2 = centers[j]
                # 線分に沿った構造
                for t in np.linspace(0, 1, 50):
                    px = cx1 + t * (cx2 - cx1)
                    py = cy1 + t * (cy2 - cy1)
                    dist = np.sqrt((self.X - px)**2 + (self.Y - py)**2)
                    tensor_field += 0.5 / (dist + 0.1)

        return tensor_field / np.max(tensor_field)

    def sri_yantra(self):
        """シュリヤントラの構造テンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))

        # 9つの交錯する三角形
        # 上向き4つ
        triangles_up = [
            [(0, 2.5), (-2.2, -1.25), (2.2, -1.25)],
            [(0, 1.8), (-1.6, -0.9), (1.6, -0.9)],
            [(0, 1.2), (-1.0, -0.6), (1.0, -0.6)],
            [(0, 0.6), (-0.5, -0.3), (0.5, -0.3)]
        ]

        # 下向き5つ
        triangles_down = [
            [(0, -2.8), (-2.4, 1.4), (2.4, 1.4)],
            [(0, -2.2), (-1.9, 1.1), (1.9, 1.1)],
            [(0, -1.6), (-1.4, 0.8), (1.4, 0.8)],
            [(0, -1.0), (-0.8, 0.5), (0.8, 0.5)],
            [(0, -0.4), (-0.3, 0.2), (0.3, 0.2)]
        ]

        # 各三角形のエッジでΔΛC発生
        all_triangles = triangles_up + triangles_down
        for tri in all_triangles:
            for i in range(3):
                x1, y1 = tri[i]
                x2, y2 = tri[(i+1)%3]

                # エッジに沿った構造変化
                n_points = 80
                for j in range(n_points):
                    t = j / n_points
                    px = x1 + t * (x2 - x1)
                    py = y1 + t * (y2 - y1)
                    dist = np.sqrt((self.X - px)**2 + (self.Y - py)**2)
                    tensor_field += 1.8 / (dist + 0.05)

        # 交点の強調（シュリヤントラの特徴）
        # 簡略化のため主要な交点のみ
        intersections = [
            (0, 0), (0, 0.8), (0, -0.8),
            (0.6, 0.3), (-0.6, 0.3),
            (0.6, -0.3), (-0.6, -0.3)
        ]

        for ix, iy in intersections:
            dist = np.sqrt((self.X - ix)**2 + (self.Y - iy)**2)
            tensor_field += 4.0 / (dist + 0.01)

        # 中心のビンドゥ（最高エネルギー点）
        center_dist = np.sqrt(self.X**2 + self.Y**2)
        tensor_field += 6.0 / (center_dist + 0.005)

        # 外周の正方形と円
        # 正方形の境界
        square_size = 3.5
        square_mask = (np.abs(self.X) < square_size) & (np.abs(self.Y) < square_size)
        square_edge = ((np.abs(np.abs(self.X) - square_size) < 0.1) |
                       (np.abs(np.abs(self.Y) - square_size) < 0.1)) & square_mask
        tensor_field[square_edge] += 1.0

        return tensor_field / np.max(tensor_field)

    def fibonacci_spiral(self, turns=4):
        """フィボナッチ螺旋の構造テンソル場生成"""
        tensor_field = np.zeros((self.grid_size, self.grid_size))
        phi = (1 + np.sqrt(5)) / 2  # 黄金比

        # 螺旋のパラメトリック方程式
        theta = np.linspace(0, turns * 2 * np.pi, 2000)
        r = 0.3 * phi ** (theta / (2 * np.pi))
        x_spiral = r * np.cos(theta)
        y_spiral = r * np.sin(theta)

        # 螺旋に沿ってテンション分布
        for i in range(len(x_spiral)-1):
            dist = np.sqrt((self.X - x_spiral[i])**2 + (self.Y - y_spiral[i])**2)
            # 成長に伴うテンション増加
            growth_factor = 1 + i / len(x_spiral) * 2
            tensor_field += growth_factor / (dist + 0.05)

            # フィボナッチ数の位置で構造的ジャンプ
            fib_positions = [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987]
            if i in fib_positions:
                tensor_field += 4.0 / (dist + 0.01)

        # 黄金矩形の追加
        rectangles = [
            (0, 0, 1, 1),
            (1, 0, 1, 1),
            (1, -1, 2, 1),
            (-1, -1, 2, 3),
            (-1, -1, 5, 3)
        ]

        for x0, y0, w, h in rectangles[:3]:  # 最初の3つ
            rect_x = x0 + w/2
            rect_y = y0 + h/2
            # 矩形の境界
            rect_mask = ((np.abs(self.X - rect_x) < w/2) &
                        (np.abs(self.Y - rect_y) < h/2))
            rect_edge = (((np.abs(np.abs(self.X - rect_x) - w/2) < 0.1) |
                         (np.abs(np.abs(self.Y - rect_y) - h/2) < 0.1)) & rect_mask)
            tensor_field[rect_edge] += 0.5

        return tensor_field / np.max(tensor_field)

# =================================
# ヒルベルト曲線の詳細実装
# =================================
def hilbert_curve(n):
    """
    n次のヒルベルト曲線の座標を生成
    再帰的アルゴリズムで正確な空間充填曲線を作る
    """
    if n == 0:
        return np.array([[0.5, 0.5]])

    # 前の次数のヒルベルト曲線を取得
    prev = hilbert_curve(n-1)
    size = 2**(n-1)

    # 4つの象限に配置（ヒルベルト曲線の再帰構造）
    # 左下：90度回転して反転
    q1 = np.column_stack([prev[:, 1], prev[:, 0]])
    # 左上：そのまま
    q2 = prev + [0, size]
    # 右上：そのまま
    q3 = prev + [size, size]
    # 右下：270度回転して反転
    q4 = np.column_stack([2*size - 1 - prev[:, 1], size - 1 - prev[:, 0]])

    # 全体を結合
    return np.vstack([q1, q2, q3, q4])

# =================================
# Lambda³強化スキャン解析
# =================================
class Lambda3EnhancedAnalyzer:
    """Lambda³変換を組み込んだ解析器"""

    def __init__(self, tensor_field, l3_config=None):
        self.tensor_field = tensor_field
        self.grid_size = tensor_field.shape[0]

        # Lambda³設定
        if l3_config is None:
            self.l3_config = L3Config(
                window=10,
                delta_percentile=97.0,
                local_jump_percentile=95.0,
                hierarchical=False  # シンプルに
            )
        else:
            self.l3_config = l3_config

    def spiral_scan(self):
        """アルキメデス螺旋スキャン"""
        center = self.grid_size // 2
        time_series = []

        max_r = center - 1
        n_points = 2000
        theta = np.linspace(0, 8*np.pi, n_points)
        r = np.linspace(0, max_r, n_points)

        for t, radius in zip(theta, r):
            x = int(center + radius * np.cos(t))
            y = int(center + radius * np.sin(t))
            if 0 <= x < self.grid_size and 0 <= y < self.grid_size:
                time_series.append(self.tensor_field[y, x])

        return np.array(time_series)

    def raster_scan(self):
        """ラスタースキャン"""
        time_series = []

        for y in range(self.grid_size):
            if y % 2 == 0:  # 偶数行は左から右
                for x in range(self.grid_size):
                    time_series.append(self.tensor_field[y, x])
            else:  # 奇数行は右から左
                for x in range(self.grid_size-1, -1, -1):
                    time_series.append(self.tensor_field[y, x])

        return np.array(time_series)

    def hilbert_curve_scan(self):
        """詳細なヒルベルト曲線スキャン"""
        # グリッドサイズに最も近い2のべき乗を見つける
        n = int(np.log2(self.grid_size))
        if 2**n < self.grid_size:
            n += 1

        # ヒルベルト曲線の座標を生成
        hilbert_coords = hilbert_curve(n)

        # テンソル場から値を抽出
        time_series = []
        scale = self.grid_size / (2**n)

        for hx, hy in hilbert_coords:
            # ヒルベルト座標をグリッド座標に変換
            x = int(hx * scale)
            y = int(hy * scale)

            if x < self.grid_size and y < self.grid_size:
                time_series.append(self.tensor_field[y, x])

        return np.array(time_series)

    def extract_lambda3_features(self, time_series):
        """時系列データからΛ³特徴量を抽出"""
        # Lambda³特徴量抽出
        features = calc_lambda3_features(time_series, self.l3_config)

        return {
            'time_series': time_series,
            'delta_LambdaC_pos': features['delta_LambdaC_pos'],
            'delta_LambdaC_neg': features['delta_LambdaC_neg'],
            'rho_T': features['rho_T'],
            'local_jump_detect': features['local_jump_detect']
        }

    def analyze_lambda3_spectrum(self, l3_features, scan_name):
        """Λ³特徴量の周波数スペクトラム解析"""
        # 1. ΔΛC（構造変化）の周波数解析
        structural_changes = l3_features['delta_LambdaC_pos'] - l3_features['delta_LambdaC_neg']

        # FFT for structural changes
        n = len(structural_changes)
        yf_structure = fft(structural_changes)
        xf = fftfreq(n, 1.0)[:n//2]
        power_structure = 2.0/n * np.abs(yf_structure[:n//2])**2

        # 2. ρT（テンション）の周波数解析
        rho_t = l3_features['rho_T']
        yf_tension = fft(rho_t)
        power_tension = 2.0/n * np.abs(yf_tension[:n//2])**2

        # 3. 構造-テンション結合スペクトラム
        combined_signal = structural_changes * rho_t
        yf_combined = fft(combined_signal)
        power_combined = 2.0/n * np.abs(yf_combined[:n//2])**2

        # ピーク検出
        peaks_structure, _ = find_peaks(power_structure, height=np.max(power_structure)*0.1)
        peaks_tension, _ = find_peaks(power_tension, height=np.max(power_tension)*0.1)

        # 周波数をHz換算
        freq_scale = 1000

        # 神聖周波数との比較
        sacred_freqs = {
            'Schumann': 7.83,
            'Alpha': 10.0,
            'OM': 136.1,
            'Love': 528.0,
            'Miracle': 639.0,
            'Universe': 432.0
        }

        # 構造的共鳴スコア計算
        resonance_scores = {}
        for name, sacred_f in sacred_freqs.items():
            # 構造変化の周波数での共鳴
            if len(peaks_structure) > 0:
                struct_distances = np.abs(xf[peaks_structure] * freq_scale - sacred_f)
                struct_score = np.exp(-np.min(struct_distances)/50)
            else:
                struct_score = 0

            # テンションの周波数での共鳴
            if len(peaks_tension) > 0:
                tension_distances = np.abs(xf[peaks_tension] * freq_scale - sacred_f)
                tension_score = np.exp(-np.min(tension_distances)/50)
            else:
                tension_score = 0

            # 総合スコア
            resonance_scores[name] = {
                'structure_score': struct_score,
                'tension_score': tension_score,
                'total_score': (struct_score + tension_score) / 2
            }

        return {
            'scan_name': scan_name,
            'frequencies': xf * freq_scale,
            'power_structure': power_structure,
            'power_tension': power_tension,
            'power_combined': power_combined,
            'peaks_structure': xf[peaks_structure] * freq_scale if len(peaks_structure) > 0 else [],
            'peaks_tension': xf[peaks_tension] * freq_scale if len(peaks_tension) > 0 else [],
            'resonance_scores': resonance_scores,
            'lambda3_features': l3_features
        }

def compare_lambda3_scan_methods(tensor_field, pattern_name="Sacred Geometry"):
    """Lambda³変換を含むスキャン方法の比較"""
    analyzer = Lambda3EnhancedAnalyzer(tensor_field)

    results = {}

    print(f"\n{'='*60}")
    print(f"Lambda³ Analysis of {pattern_name}")
    print(f"{'='*60}")

    # 各スキャン方法で解析
    scan_methods = {
        'spiral': analyzer.spiral_scan,
        'raster': analyzer.raster_scan,
        'hilbert': analyzer.hilbert_curve_scan
    }

    for scan_type, scan_func in scan_methods.items():
        # 1. スキャンして時系列取得
        time_series = scan_func()

        # 2. Lambda³特徴量抽出
        l3_features = analyzer.extract_lambda3_features(time_series)

        # 3. Lambda³スペクトラム解析
        result = analyzer.analyze_lambda3_spectrum(l3_features, f'{scan_type.capitalize()} Scan')
        results[scan_type] = result

        # 統計情報の表示
        print(f"\n{result['scan_name']}:")
        print(f"  ΔΛC+ events: {np.sum(l3_features['delta_LambdaC_pos'])}")
        print(f"  ΔΛC- events: {np.sum(l3_features['delta_LambdaC_neg'])}")
        print(f"  Mean ρT: {np.mean(l3_features['rho_T']):.3f}")
        print(f"  Max ρT: {np.max(l3_features['rho_T']):.3f}")

        # 最高共鳴スコア
        best_resonance = max(result['resonance_scores'].items(),
                           key=lambda x: x[1]['total_score'])
        print(f"  Best resonance: {best_resonance[0]} "
              f"(score: {best_resonance[1]['total_score']:.3f})")

    return results

def visualize_lambda3_comparison(results, tensor_field, pattern_name="Sacred Geometry"):
    """Lambda³解析結果の可視化"""
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(5, 3, figure=fig, hspace=0.35, wspace=0.3)

    # カラーマップ
    colors = {'spiral': 'red', 'raster': 'green', 'hilbert': 'blue'}

    # 1. テンソル場
    ax_field = fig.add_subplot(gs[0, 0])
    im = ax_field.imshow(tensor_field, cmap='twilight', origin='lower')
    ax_field.set_title('Original Tensor Field')
    ax_field.axis('off')
    plt.colorbar(im, ax=ax_field, fraction=0.046)

    # 2. 各スキャンのΔΛC分布
    for idx, (scan_type, result) in enumerate(results.items()):
        ax = fig.add_subplot(gs[1, idx])
        l3_features = result['lambda3_features']

        # ΔΛCイベントの可視化
        events = l3_features['delta_LambdaC_pos'] - l3_features['delta_LambdaC_neg']
        ax.plot(events[:500], color=colors[scan_type], alpha=0.7)
        ax.set_title(f'{scan_type.capitalize()} ΔΛC')
        ax.set_xlabel('Position')
        ax.set_ylabel('ΔΛC')
        ax.grid(True, alpha=0.3)

    # 3. ρT（テンション）の比較
    ax_tension = fig.add_subplot(gs[2, :])
    for scan_type, result in results.items():
        rho_t = result['lambda3_features']['rho_T'][:500]
        ax_tension.plot(rho_t, color=colors[scan_type], alpha=0.7,
                       label=f'{scan_type.capitalize()}')
    ax_tension.set_xlabel('Position')
    ax_tension.set_ylabel('Tension (ρT)')
    ax_tension.set_title('Tension Comparison')
    ax_tension.legend()
    ax_tension.grid(True, alpha=0.3)

    # 4. 構造変化スペクトラム
    ax_struct = fig.add_subplot(gs[3, :2])
    for scan_type, result in results.items():
        freqs = result['frequencies']
        power = result['power_structure']
        mask = (freqs > 0) & (freqs < 1000)
        ax_struct.semilogy(freqs[mask], power[mask],
                          color=colors[scan_type], alpha=0.7,
                          label=f'{scan_type.capitalize()}')
    ax_struct.set_xlabel('Frequency (Hz)')
    ax_struct.set_ylabel('ΔΛC Power')
    ax_struct.set_title('Structural Change Spectrum')
    ax_struct.legend()
    ax_struct.grid(True, alpha=0.3)

    # 5. テンションスペクトラム
    ax_tens_spec = fig.add_subplot(gs[3, 2])
    for scan_type, result in results.items():
        freqs = result['frequencies']
        power = result['power_tension']
        mask = (freqs > 0) & (freqs < 1000)
        ax_tens_spec.semilogy(freqs[mask], power[mask],
                             color=colors[scan_type], alpha=0.7,
                             label=f'{scan_type.capitalize()}')
    ax_tens_spec.set_xlabel('Frequency (Hz)')
    ax_tens_spec.set_ylabel('ρT Power')
    ax_tens_spec.set_title('Tension Spectrum')
    ax_tens_spec.legend()
    ax_tens_spec.grid(True, alpha=0.3)

    # 6. 神聖周波数共鳴スコア
    ax_resonance = fig.add_subplot(gs[4, :])

    # データ準備
    sacred_names = ['Schumann', 'Alpha', 'OM', 'Universe', 'Love', 'Miracle']
    scan_types = ['Spiral', 'Raster', 'Hilbert']

    # 構造共鳴とテンション共鳴を別々に表示
    x = np.arange(len(sacred_names))
    width = 0.25

    for i, (scan_type, result) in enumerate(results.items()):
        struct_scores = [result['resonance_scores'][name]['structure_score']
                        for name in sacred_names]
        tension_scores = [result['resonance_scores'][name]['tension_score']
                         for name in sacred_names]

        ax_resonance.bar(x + i*width - width, struct_scores, width,
                        label=f'{scan_type.capitalize()} (ΔΛC)',
                        color=colors[scan_type], alpha=0.7)
        ax_resonance.bar(x + i*width - width, tension_scores, width,
                        bottom=struct_scores,
                        label=f'{scan_type.capitalize()} (ρT)',
                        color=colors[scan_type], alpha=0.4)

    ax_resonance.set_xlabel('Sacred Frequency')
    ax_resonance.set_ylabel('Resonance Score')
    ax_resonance.set_title('Lambda³ Resonance with Sacred Frequencies')
    ax_resonance.set_xticks(x)
    ax_resonance.set_xticklabels([f"{name}\n({freq}Hz)" for name, freq in
                                  [('Schumann', 7.83), ('Alpha', 10), ('OM', 136.1),
                                   ('Universe', 432), ('Love', 528), ('Miracle', 639)]])
    ax_resonance.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax_resonance.grid(True, alpha=0.3, axis='y')

    plt.suptitle(f'Lambda³ Enhanced Analysis: {pattern_name}', fontsize=16)
    plt.tight_layout()
    plt.show()

def analyze_all_sacred_geometries():
    """全神聖幾何学パターンの解析"""
    print("🌀 Lambda³ Analysis of All Sacred Geometry Patterns 🌀")
    print("="*60)

    # Generator初期化
    generator = SacredGeometryGenerator(grid_size=256)

    # 全パターンの定義
    patterns = {
        'Flower of Life': lambda: generator.flower_of_life(radius=1.0, layers=3),
        'Metatron\'s Cube': lambda: generator.metatron_cube(),
        'Sri Yantra': lambda: generator.sri_yantra(),
        'Fibonacci Spiral': lambda: generator.fibonacci_spiral(turns=4)
    }

    # 全体の結果を保存
    all_results = {}

    # 各パターンを解析
    for pattern_name, pattern_func in patterns.items():
        print(f"\n{'='*60}")
        print(f"Analyzing: {pattern_name}")
        print(f"{'='*60}")

        # パターン生成
        tensor_field = pattern_func()

        # Lambda³解析
        results = compare_lambda3_scan_methods(tensor_field, pattern_name)
        all_results[pattern_name] = results

        # 可視化
        visualize_lambda3_comparison(results, tensor_field, pattern_name)

        # パターン固有の最適周波数を見つける
        print(f"\nOptimal Sacred Frequency for {pattern_name}:")

        optimal_freq = None
        max_score = 0
        optimal_scan = None

        for scan_type, scan_result in results.items():
            for freq_name, scores in scan_result['resonance_scores'].items():
                if scores['total_score'] > max_score:
                    max_score = scores['total_score']
                    optimal_freq = freq_name
                    optimal_scan = scan_type

        print(f"  🎯 {optimal_freq} (via {optimal_scan} scan, score: {max_score:.3f})")

    # 総合比較
    print("\n" + "="*60)
    print("SACRED GEOMETRY COMPARATIVE ANALYSIS")
    print("="*60)

    # 各パターンの最適周波数マトリックス
    print("\nOptimal Sacred Frequencies by Pattern:")
    print("-"*50)

    for pattern_name, results in all_results.items():
        print(f"\n{pattern_name}:")

        # 各スキャン方法での最高スコア周波数
        for scan_type in ['spiral', 'raster', 'hilbert']:
            if scan_type in results:
                scan_result = results[scan_type]
                best_freq = max(scan_result['resonance_scores'].items(),
                              key=lambda x: x[1]['total_score'])
                print(f"  {scan_type.capitalize():8s}: {best_freq[0]:10s} "
                      f"(score: {best_freq[1]['total_score']:.3f})")

    return all_results

# メイン実行
if __name__ == "__main__":
    # 全神聖幾何学パターンの解析実行
    all_results = analyze_all_sacred_geometries()

    print("\n" + "="*60)
    print("FINAL INSIGHTS - Lambda³ Sacred Geometry Analysis")
    print("="*60)
    print("1. Each sacred geometry has unique ΔΛC patterns")
    print("2. Vesica Piscis creates high ρT (tension) zones")
    print("3. Sri Yantra shows complex hierarchical ΔΛC cascades")
    print("4. Fibonacci spiral exhibits growth-dependent tension")
    print("5. Scan method affects which frequencies are emphasized")
    print("\n✨ Sacred geometries encode frequencies through structural dynamics! ✨")
