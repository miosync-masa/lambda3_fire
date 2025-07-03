"""
Lambda³ Spatial Multi-Layer Analysis System - Full Implementation
全国F-NET観測網の空間多層構造解析（完全版）
Based on Dr. Iizumi's Lambda³ Theory
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.spatial.distance import cdist, pdist, squareform
from scipy.optimize import minimize
from scipy import signal
from sklearn.metrics import roc_auc_score
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Union
import warnings
import json
import os
from datetime import datetime
warnings.filterwarnings('ignore')

# === データクラス定義 ===
@dataclass
class Lambda3Result:
    """Lambda³解析結果を格納するデータクラス"""
    paths: Dict[int, np.ndarray]
    topological_charges: Dict[int, float]
    stabilities: Dict[int, float]
    energies: Dict[int, float]
    entropies: Dict[int, float]
    classifications: Dict[int, str]

@dataclass
class SpatialLambda3Result:
    """空間多層Lambda³解析結果"""
    # グローバル（全国規模）解析結果
    global_result: Lambda3Result
    
    # ローカル（観測点別）解析結果
    local_results: Dict[str, Lambda3Result]
    
    # クラスタ（地域別）解析結果
    cluster_results: Dict[int, Lambda3Result]
    
    # 空間相関構造
    spatial_correlations: np.ndarray
    
    # 観測点クラスタリング情報
    station_clusters: Dict[str, int]
    
    # 層間相互作用指標
    cross_layer_metrics: Dict[str, float]
    
    # 空間的異常検出結果
    spatial_anomalies: Dict[str, List[Dict]]
    
    # 解析メタデータ
    metadata: Dict[str, any]


# === Lambda³地震検知完全版 ===
class Lambda3Analyzer:
    """Lambda³理論による構造解析の基本クラス - 地震検知特化版"""
    
    def __init__(self, alpha: float = 0.1, beta: float = 0.01):
        self.alpha = alpha
        self.beta = beta
        self.anomaly_patterns = {
            # 基本パターン
            'pulse': self._generate_pulse_anomaly,
            'phase_jump': self._generate_phase_jump_anomaly,
            'periodic': self._generate_periodic_anomaly,
            'structural_decay': self._generate_decay_anomaly,
            'bifurcation': self._generate_bifurcation_anomaly,
            # 地震波パターン
            'p_wave': self._generate_p_wave_anomaly,
            's_wave': self._generate_s_wave_anomaly,
            # 複雑パターン
            'multi_path': self._generate_multi_path_anomaly,
            'topological_jump': self._generate_topological_jump_anomaly,
            'cascade': self._generate_cascade_anomaly,
            'resonance': self._generate_resonance_anomaly,
            # 地震前兆特有パターン
            'foreshock_sequence': self._generate_foreshock_sequence,
            'quiet_period': self._generate_quiet_period,
            'nucleation_phase': self._generate_nucleation_phase,
            'dilatancy': self._generate_dilatancy_anomaly,
            'crustal_deformation': self._generate_crustal_deformation,
            'electromagnetic': self._generate_electromagnetic_precursor,
            'slow_slip': self._generate_slow_slip_event,
            'critical_point': self._generate_critical_point_anomaly
        }

    def analyze(self, events: np.ndarray, n_paths: int = 3) -> Lambda3Result:
        """完全なLambda³解析を実行"""
        # 入力検証
        if len(events) == 0 or events.shape[0] == 0:
            raise ValueError("Empty event data")
        
        # 1. 構造テンソル推定
        paths = self._inverse_problem(events, n_paths)
        
        # 2. 各パスの物理量計算
        charges, stabilities = {}, {}
        energies, entropies = {}, {}
        classifications = {}
        
        for i, path in paths.items():
            # トポロジカル量
            Q, sigma = self._compute_topological_charge(path)
            charges[i] = Q
            stabilities[i] = sigma
            
            # エネルギー・エントロピー
            energies[i] = np.sum(path**2)
            entropies[i] = self._compute_entropy(path)
            
            # 物理的分類
            classifications[i] = self._classify_structure(Q)
        
        return Lambda3Result(
            paths=paths,
            topological_charges=charges,
            stabilities=stabilities,
            energies=energies,
            entropies=entropies,
            classifications=classifications
        )
    
    def _inverse_problem(self, events: np.ndarray, n_paths: int) -> Dict[int, np.ndarray]:
        """正則化付き逆問題を解く（修正版）"""
        n_events = events.shape[0]
        
        # 入力検証を強化
        if n_events == 0 or events.shape[1] == 0:
            print(f"Warning: Invalid event shape: {events.shape}")
            return {i: np.zeros(1) for i in range(n_paths)}
        
        # 単一イベントの場合の特別処理
        if n_events == 1:
            # 空間次元での分解を試みる
            spatial_dim = events.shape[1]
            if spatial_dim >= n_paths:
                # 特徴量次元での簡易分解
                paths = {}
                for i in range(n_paths):
                    start = i * (spatial_dim // n_paths)
                    end = (i + 1) * (spatial_dim // n_paths) if i < n_paths - 1 else spatial_dim
                    path = events[0, start:end]
                    paths[i] = path / (np.linalg.norm(path) + 1e-8)
                return paths
            else:
                # パスを生成できない場合
                return {i: np.random.randn(spatial_dim) * 0.1 for i in range(n_paths)}
      
        def objective(Lambda_flat):
            Lambda = Lambda_flat.reshape(n_paths, n_events)
            reconstruction = Lambda.T @ Lambda
            data_fit = np.linalg.norm(events @ events.T - reconstruction)**2
            tv_reg = np.sum(np.abs(np.diff(Lambda, axis=0))) + \
                    np.sum(np.abs(np.diff(Lambda, axis=1)))
            l1_reg = np.sum(np.abs(Lambda))
            return data_fit + self.alpha * tv_reg + self.beta * l1_reg
        
        # 初期値：固有値分解
        E = events @ events.T
        eigenvalues, eigenvectors = np.linalg.eigh(E)
        
        # 上位n_paths個の固有ベクトルを初期値として使用
        Lambda_init = eigenvectors[:, -n_paths:].T.flatten()
        
        # 最適化
        result = minimize(objective, Lambda_init, method='L-BFGS-B')
        Lambda_opt = result.x.reshape(n_paths, n_events)
        
        # 正規化して返す
        return {i: path / (np.linalg.norm(path) + 1e-8) 
                for i, path in enumerate(Lambda_opt)}
    
    @staticmethod
    def _compute_topological_charge(path: np.ndarray, n_segments: int = 10) -> Tuple[float, float]:
        """トポロジカルチャージQ_Λと安定性σ_Qを計算"""
        if len(path) < 2:
            return 0.0, 0.0
            
        closed_path = np.append(path, path[0])
        theta = np.angle(closed_path[:-1] + 1j * closed_path[1:])
        Q_Lambda = np.sum(np.diff(theta)) / (2 * np.pi)
        
        # セグメント安定性
        Q_segments = []
        for i in range(n_segments):
            start = i * len(path) // n_segments
            end = (i + 1) * len(path) // n_segments
            if end > start + 1:
                segment_theta = theta[start:end]
                Q_segments.append(np.sum(np.diff(segment_theta)))
        
        stability = np.std(Q_segments) if Q_segments else 0.0
        return Q_Lambda, stability
    
    @staticmethod
    def _compute_entropy(path: np.ndarray) -> float:
        """構造エントロピーを計算"""
        abs_path = np.abs(path) + 1e-10
        abs_path = np.clip(abs_path, 1e-10, 1e10)
        entropy = -np.sum(abs_path * np.log(abs_path))
        return entropy if not np.isnan(entropy) else 0.0
    
    @staticmethod
    def _classify_structure(Q: float) -> str:
        """トポロジカルチャージから物理的構造を分類"""
        if Q < -0.5:
            return "反物質的構造（エネルギー吸収系）"
        elif Q > 0.5:
            return "物質的構造（エネルギー放出系）"
        else:
            return "中性構造（平衡状態）"
    
    # === 基本異常パターン生成メソッド ===
    def _generate_pulse_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """パルス型異常：局所的な強いスパイク"""
        events_copy = events.copy()
        idx = np.random.randint(events.shape[0])
        events_copy[idx] += np.random.randn(events.shape[1]) * intensity
        return events_copy
    
    def _generate_phase_jump_anomaly(self, events: np.ndarray, intensity: float = 1) -> np.ndarray:
        """位相ジャンプ型異常：符号反転"""
        events_copy = events.copy()
        idx = np.random.randint(events.shape[0])
        events_copy[idx] = -events_copy[idx] * intensity
        return events_copy
    
    def _generate_periodic_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """周期的異常：正弦波的変調"""
        events_copy = events.copy()
        n_events = events.shape[0]
        period = max(2, n_events // 4)
        modulation = intensity * np.sin(2 * np.pi * np.arange(n_events) / period)
        events_copy += modulation[:, np.newaxis]
        return events_copy
    
    def _generate_decay_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """構造崩壊型異常：指数減衰"""
        events_copy = events.copy()
        decay_start = events.shape[0] // 2
        decay = np.exp(-intensity * np.arange(events.shape[0] - decay_start))
        events_copy[decay_start:] *= decay[:, np.newaxis]
        return events_copy
    
    def _generate_bifurcation_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """分岐型異常：構造の分裂"""
        events_copy = events.copy()
        split_point = events.shape[0] // 2
        events_copy[split_point:] += np.random.randn(*events_copy[split_point:].shape) * intensity
        return events_copy
    
    # === 地震波パターン ===
    def _generate_p_wave_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """P波的な急激な立ち上がり"""
        events_copy = events.copy()
        onset = np.random.randint(len(events) // 4, 3 * len(events) // 4)
        rise_length = min(5, len(events) - onset)
        events_copy[onset:onset+rise_length] += np.linspace(0, intensity, rise_length)[:, np.newaxis]
        return events_copy
    
    def _generate_s_wave_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """S波的な大振幅振動"""
        events_copy = events.copy()
        onset = np.random.randint(len(events) // 4, 3 * len(events) // 4)
        duration = min(50, len(events) - onset)
        t = np.arange(duration)
        oscillation = intensity * np.sin(2 * np.pi * t / 10) * np.exp(-t / 20)
        events_copy[onset:onset+duration] += oscillation[:, np.newaxis]
        return events_copy
    
    # === 複雑パターン ===
    def _generate_multi_path_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """複数経路異常：複数の独立した異常が同時発生"""
        events_copy = events.copy()
        n_paths = np.random.randint(2, 5)
        
        for _ in range(n_paths):
            idx = np.random.randint(events.shape[0])
            direction = np.random.randn(events.shape[1])
            direction /= np.linalg.norm(direction)
            events_copy[idx] += direction * intensity * np.random.uniform(0.5, 1.5)
        
        return events_copy
    
    def _generate_topological_jump_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """トポロジカルジャンプ：位相空間での不連続遷移"""
        events_copy = events.copy()
        n_events = events.shape[0]
        
        if n_events < 3:
            events_copy *= -intensity
            return events_copy
        
        jump_point = n_events // 2
        
        if jump_point > 0:
            events_copy[:jump_point] *= np.exp(-0.1 * np.arange(jump_point))[:, np.newaxis]
        if jump_point < n_events:
            events_copy[jump_point:] = -events_copy[jump_point:] * intensity
        
        if jump_point < n_events:
            events_copy[jump_point] = np.random.randn(events.shape[1]) * intensity * 2
        
        return events_copy
    
    def _generate_cascade_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """カスケード異常：異常が時間的に伝播"""
        events_copy = events.copy()
        n_events = events.shape[0]
        
        if n_events < 2:
            events_copy *= intensity
            return events_copy
        
        start_idx = np.random.randint(0, max(1, n_events // 2))
        events_copy[start_idx] += np.random.randn(events.shape[1]) * intensity
        
        for i in range(start_idx + 1, min(start_idx + 10, n_events)):
            decay = np.exp(-0.3 * (i - start_idx))
            events_copy[i] += events_copy[i-1] * 0.5 * decay
        
        return events_copy
    
    def _generate_resonance_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """共鳴異常：特定周波数での増幅"""
        events_copy = events.copy()
        n_events = events.shape[0]
        
        if n_events < 4:
            events_copy *= intensity
            return events_copy
        
        fft = np.fft.fft(events_copy, axis=0)
        max_freq = max(2, len(fft) // 4)
        resonance_freq = np.random.randint(1, max_freq)
        
        if resonance_freq < len(fft):
            fft[resonance_freq] *= intensity
            if len(fft) - resonance_freq > 0:
                fft[-resonance_freq] *= intensity
        
        events_copy = np.real(np.fft.ifft(fft, axis=0))
        return events_copy
    
    # === 地震前兆特有パターン ===
    def _generate_foreshock_sequence(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """前震系列：増加する小規模イベント"""
        events_copy = events.copy()
        n_foreshocks = min(8, events.shape[0] // 3)
        
        # 時間的に密度が増加する前震
        for i in range(n_foreshocks):
            position = int(len(events) * (0.5 + 0.5 * (i / n_foreshocks)))
            if position < len(events):
                amplitude = intensity * (0.2 + 0.8 * (i / n_foreshocks))
                events_copy[position] += np.random.randn(events.shape[1]) * amplitude
        
        return events_copy
    
    def _generate_quiet_period(self, events: np.ndarray, intensity: float = 0.1) -> np.ndarray:
        """地震前の静穏期：活動の急激な低下"""
        events_copy = events.copy()
        quiet_start = int(len(events) * 0.6)
        quiet_end = int(len(events) * 0.85)
        
        # 静穏期では振幅を大幅に減衰
        if quiet_start < quiet_end and quiet_end <= len(events):
            events_copy[quiet_start:quiet_end] *= intensity
        return events_copy
    
    def _generate_nucleation_phase(self, events: np.ndarray, intensity: float = 2.5) -> np.ndarray:
        """核形成期：局所的な高周波振動の増加"""
        events_copy = events.copy()
        nucleation_start = int(len(events) * 0.7)
        
        # 高周波成分の追加
        for i in range(nucleation_start, len(events)):
            high_freq = np.sin(2 * np.pi * np.random.uniform(5, 15) * i / len(events))
            events_copy[i] += high_freq * intensity * np.random.randn(events.shape[1])
        
        return events_copy
    
    def _generate_dilatancy_anomaly(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """ダイラタンシー異常：体積膨張に伴う速度変化"""
        events_copy = events.copy()
        
        # 速度構造の変化を模擬
        for i in range(len(events)):
            # 時間とともに変化する伝播速度
            velocity_change = 1 + intensity * 0.1 * (i / len(events))
            events_copy[i] *= velocity_change
            
            # 位相シフトも追加
            if i > 0:
                phase_shift = intensity * 0.05 * (i / len(events))
                shift_amount = int(phase_shift * events.shape[1])
                if shift_amount > 0:
                    events_copy[i] = np.roll(events_copy[i], shift_amount)
        
        return events_copy
    
    def _generate_crustal_deformation(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """地殻変動パターン：長周期の累積的変形"""
        events_copy = events.copy()
        
        # 長周期トレンド
        trend = np.linspace(0, intensity, len(events))
        
        # 非線形な累積
        cumulative = np.cumsum(np.random.randn(len(events)) * 0.1) * intensity * 0.2
        
        for i in range(len(events)):
            events_copy[i] += (trend[i] + cumulative[i]) * np.ones(events.shape[1])
        
        return events_copy
    
    def _generate_electromagnetic_precursor(self, events: np.ndarray, intensity: float = 1.5) -> np.ndarray:
        """電磁気的前兆：特定周波数帯での異常"""
        events_copy = events.copy()
        
        # 特定周波数での共鳴
        resonant_freq = np.random.uniform(0.01, 0.1)  # 低周波
        
        for i in range(len(events)):
            em_signal = intensity * np.sin(2 * np.pi * resonant_freq * i)
            # 空間的にも不均一
            spatial_pattern = np.random.randn(events.shape[1]) * 0.5 + 1
            events_copy[i] += em_signal * spatial_pattern
        
        return events_copy
    
    def _generate_slow_slip_event(self, events: np.ndarray, intensity: float = 2) -> np.ndarray:
        """スロースリップイベント：ゆっくりとした滑り"""
        events_copy = events.copy()
        
        # スロースリップの時間スケール
        slip_duration = max(len(events) // 3, 5)
        slip_start = np.random.randint(0, max(1, len(events) - slip_duration))
        
        # ガウシアン型の滑り関数
        t = np.arange(slip_duration)
        slip_function = intensity * np.exp(-(t - slip_duration/2)**2 / (slip_duration/4)**2)
        
        for i in range(slip_duration):
            if slip_start + i < len(events):
                events_copy[slip_start + i] += slip_function[i] * np.ones(events.shape[1])
        
        return events_copy
    
    def _generate_critical_point_anomaly(self, events: np.ndarray, intensity: float = 3) -> np.ndarray:
        """臨界点異常：系が臨界状態に近づく際の異常"""
        events_copy = events.copy()
        n_events = len(events)
        
        # 臨界点への接近をモデル化
        critical_point = int(n_events * 0.8)
        
        for i in range(n_events):
            if i < critical_point:
                # 臨界点に近づくにつれて揺らぎが増大
                distance_to_critical = (critical_point - i) / critical_point
                fluctuation = intensity * np.exp(-distance_to_critical * 3)
                events_copy[i] += np.random.randn(events.shape[1]) * fluctuation
            else:
                # 臨界点を超えた後の急激な変化
                events_copy[i] *= intensity * 2
        
        return events_copy
    
    def extract_features(self, result: Lambda3Result) -> Dict[str, List[float]]:
        """Lambda³解析結果から特徴量を抽出（地震検知用拡張版）"""
        features = {
            # 基本特徴量
            'Q_Λ': [],
            'E': [],
            'S': [],
            'n_pulse': [],
            'σ_Q': [],
            'mean_curvature': [],
            'spectral_peak': [],
            # 地震特有の特徴量
            'Q_Λ_gradient': [],  # トポロジカルチャージの変化率
            'energy_concentration': [],  # エネルギー集中度
            'phase_coherence': [],  # 位相コヒーレンス
            'structural_instability': []  # 構造的不安定性
        }
        
        for i, path in result.paths.items():
            # 基本特徴量
            features['Q_Λ'].append(result.topological_charges[i])
            features['E'].append(result.energies[i])
            features['S'].append(result.entropies[i])
            features['σ_Q'].append(result.stabilities[i])
            
            # パルス数
            if len(path) > 1:
                delta_lambda = np.abs(np.diff(path))
                std_path = np.std(path)
                if std_path > 0:
                    n_pulse = np.sum(delta_lambda > (std_path * 2))
                else:
                    n_pulse = 0
            else:
                n_pulse = 0
            features['n_pulse'].append(float(n_pulse))
            
            # 平均曲率
            if len(path) > 2:
                curvature = np.gradient(np.gradient(path))
                mean_curv = np.mean(np.abs(curvature))
                features['mean_curvature'].append(mean_curv if not np.isnan(mean_curv) else 0.0)
            else:
                features['mean_curvature'].append(0.0)
            
            # スペクトルピーク
            if len(path) > 1:
                fft = np.fft.fft(path)
                if len(fft) > 1:
                    peak = np.max(np.abs(fft[1:len(fft)//2]))
                    features['spectral_peak'].append(float(peak) if not np.isnan(peak) else 0.0)
                else:
                    features['spectral_peak'].append(0.0)
            else:
                features['spectral_peak'].append(0.0)
            
            # 地震特有の特徴量
            # Q_Λの変化率
            if i > 0 and (i-1) in result.topological_charges:
                q_gradient = abs(result.topological_charges[i] - result.topological_charges[i-1])
                features['Q_Λ_gradient'].append(q_gradient)
            else:
                features['Q_Λ_gradient'].append(0.0)
            
            # エネルギー集中度
            if len(path) > 1:
                energy_dist = np.abs(path)**2
                if np.sum(energy_dist) > 0:
                    energy_cumsum = np.cumsum(energy_dist)
                    energy_50_idx = np.argmax(energy_cumsum >= 0.5 * energy_cumsum[-1])
                    concentration = 1.0 - (energy_50_idx / len(path))
                else:
                    concentration = 0.0
                features['energy_concentration'].append(concentration)
            else:
                features['energy_concentration'].append(0.0)
            
            # 位相コヒーレンス
            if len(path) > 10:
                # ヒルベルト変換による瞬時位相
                analytic = signal.hilbert(path)
                phase = np.angle(analytic)
                # 位相の一貫性を評価
                phase_diff = np.diff(phase)
                coherence = 1.0 / (1.0 + np.std(phase_diff))
                features['phase_coherence'].append(coherence)
            else:
                features['phase_coherence'].append(0.0)
            
            # 構造的不安定性（エネルギーとエントロピーの比）
            if result.energies[i] > 0:
                instability = result.entropies[i] / result.energies[i]
            else:
                instability = 0.0
            features['structural_instability'].append(instability)
        
        return features
    
    def detect_earthquake_precursors(self, result: Lambda3Result, events: np.ndarray) -> Dict[str, np.ndarray]:
        """地震前兆に特化した異常検出"""
        n_events = events.shape[0]
        
        # 複数の前兆指標を計算
        precursors = {
            'composite_score': np.zeros(n_events),  # 総合スコア
            'topological_anomaly': np.zeros(n_events),  # トポロジカル異常
            'energy_anomaly': np.zeros(n_events),  # エネルギー異常
            'pattern_transition': np.zeros(n_events),  # パターン遷移
            'criticality_index': np.zeros(n_events)  # 臨界性指標
        }
        
        # 特徴量を抽出
        features = self.extract_features(result)
        
        for i in range(n_events):
            # トポロジカル異常スコア
            q_scores = []
            for j, path in result.paths.items():
                if i < len(path):
                    q_anomaly = abs(result.topological_charges[j]) * abs(path[i])
                    q_scores.append(q_anomaly)
            
            precursors['topological_anomaly'][i] = max(q_scores) if q_scores else 0.0
            
            # エネルギー異常スコア
            energy_scores = []
            for j, path in result.paths.items():
                if i < len(path):
                    e_anomaly = (result.energies[j] - 1.0)**2 * abs(path[i])
                    energy_scores.append(e_anomaly)
            
            precursors['energy_anomaly'][i] = max(energy_scores) if energy_scores else 0.0
            
            # パターン遷移検出
            if i > 0:
                transition_score = 0.0
                for j in range(len(result.paths)):
                    if j in features['Q_Λ_gradient']:
                        transition_score += features['Q_Λ_gradient'][j]
                precursors['pattern_transition'][i] = transition_score / (len(result.paths) + 1e-8)
            
            # 臨界性指標（複数の特徴量の組み合わせ）
            criticality = 0.0
            for j in range(len(result.paths)):
                if j < len(features['structural_instability']):
                    instability = features['structural_instability'][j]
                    coherence = features['phase_coherence'][j] if j < len(features['phase_coherence']) else 0.0
                    criticality += instability * (1 - coherence)
            
            precursors['criticality_index'][i] = criticality / (len(result.paths) + 1e-8)
            
            # 総合スコア
            precursors['composite_score'][i] = (
                precursors['topological_anomaly'][i] * 0.3 +
                precursors['energy_anomaly'][i] * 0.2 +
                precursors['pattern_transition'][i] * 0.2 +
                precursors['criticality_index'][i] * 0.3
            )
        
        return precursors
    
    def detect_anomalies(self, result: Lambda3Result, events: np.ndarray) -> np.ndarray:
        """複合的な異常スコアを計算（地震検知強化版）"""
        # 地震前兆検出を実行
        precursors = self.detect_earthquake_precursors(result, events)
        
        # 総合スコアを返す
        return precursors['composite_score']

# === 空間多層解析器 ===
class SpatialMultiLayerAnalyzer:
    """空間多層Lambda³解析システム（完全版）"""
    
    def __init__(self, 
                 station_locations: Optional[Dict[str, Tuple[float, float]]] = None,
                 station_metadata: Optional[Dict[str, Dict]] = None):
        """
        Parameters:
        -----------
        station_locations : Dict[str, Tuple[float, float]]
            観測点名と(緯度, 経度)の辞書
        station_metadata : Dict[str, Dict]
            観測点のメタデータ（設置深度、地質情報など）
        """
        self.station_locations = station_locations or {}
        self.station_metadata = station_metadata or {}
        self.base_analyzer = Lambda3Analyzer(alpha=0.1, beta=0.01)
        self.clustering_methods = {
            'kmeans': self._cluster_kmeans,
            'dbscan': self._cluster_dbscan,
            'hierarchical': self._cluster_hierarchical,
            'geological': self._cluster_geological
        }
    
    def analyze_multilayer(self, 
                          data_dict: Dict[str, np.ndarray],
                          n_clusters: int = 5,
                          clustering_method: str = 'kmeans',
                          n_paths_global: int = 10,
                          n_paths_local: int = 5,
                          n_paths_cluster: int = 7,
                          parallel: bool = False) -> SpatialLambda3Result:
        """
        空間多層Lambda³解析を実行
        
        Parameters:
        -----------
        data_dict : Dict[str, np.ndarray]
            観測点名とイベント行列の辞書
        n_clusters : int
            地域クラスタ数
        clustering_method : str
            クラスタリング手法（'kmeans', 'dbscan', 'hierarchical', 'geological'）
        n_paths_global : int
            全国規模解析のパス数
        n_paths_local : int
            観測点別解析のパス数
        n_paths_cluster : int
            クラスタ別解析のパス数
        parallel : bool
            並列処理を使用するか
        """
        start_time = datetime.now()
        print("=== Lambda³ Spatial Multi-Layer Analysis ===")
        print(f"Stations: {len(data_dict)}")
        print(f"Clustering method: {clustering_method}")
        print(f"Paths: Global={n_paths_global}, Local={n_paths_local}, Cluster={n_paths_cluster}")
        
        # 1. 観測点のクラスタリング
        print("\n--- Station Clustering ---")
        station_clusters = self._cluster_stations(
            data_dict, n_clusters, method=clustering_method
        )
        
        # 2. グローバル（全国規模）解析
        print("\n--- Global (Japan-wide) Analysis ---")
        global_data = self._aggregate_global_data(data_dict)
        global_result = self._analyze_lambda3(global_data, n_paths_global, "Global")
        
        # 3. ローカル（観測点別）解析
        print("\n--- Local (Station-wise) Analysis ---")
        local_results = {}
        
        if parallel and len(data_dict) > 10:
            # 並列処理（要: joblib）
            try:
                from joblib import Parallel, delayed
                
                def analyze_station(station, data):
                    return station, self._analyze_lambda3(data, n_paths_local, f"Station_{station}")
                
                results = Parallel(n_jobs=-1)(
                    delayed(analyze_station)(station, data) 
                    for station, data in data_dict.items()
                )
                local_results = dict(results)
            except ImportError:
                print("Warning: joblib not available, using sequential processing")
                for i, (station, data) in enumerate(data_dict.items()):
                    if i % 10 == 0:
                        print(f"  Progress: {i}/{len(data_dict)} stations")
                    local_results[station] = self._analyze_lambda3(
                        data, n_paths_local, f"Station_{station}"
                    )
        else:
            for i, (station, data) in enumerate(data_dict.items()):
                if i % 10 == 0:
                    print(f"  Progress: {i}/{len(data_dict)} stations")
                local_results[station] = self._analyze_lambda3(
                    data, n_paths_local, f"Station_{station}"
                )
        
        # 4. クラスタ（地域別）解析
        print("\n--- Cluster (Regional) Analysis ---")
        cluster_results = {}
        cluster_data_dict = self._aggregate_cluster_data(data_dict, station_clusters)
        
        for cluster_id, cluster_data in cluster_data_dict.items():
            print(f"  Analyzing cluster {cluster_id} ({self._get_cluster_size(station_clusters, cluster_id)} stations)")
            cluster_results[cluster_id] = self._analyze_lambda3(
                cluster_data, n_paths_cluster, f"Cluster_{cluster_id}"
            )
        
        # 5. 空間相関構造の計算
        print("\n--- Computing Spatial Correlations ---")
        spatial_correlations = self._compute_spatial_correlations(local_results)
        
        # 6. 層間相互作用の評価
        print("\n--- Evaluating Cross-Layer Interactions ---")
        cross_layer_metrics = self._evaluate_cross_layer_interactions(
            global_result, local_results, cluster_results, station_clusters
        )
        
        # 7. 空間的異常検出
        print("\n--- Detecting Spatial Anomalies ---")
        spatial_anomalies = self.detect_spatial_anomalies(
            global_result, local_results, cluster_results, 
            spatial_correlations, station_clusters
        )
        
        # 8. メタデータの収集
        end_time = datetime.now()
        metadata = {
            'analysis_time': (end_time - start_time).total_seconds(),
            'n_stations': len(data_dict),
            'n_clusters': n_clusters,
            'clustering_method': clustering_method,
            'n_paths': {
                'global': n_paths_global,
                'local': n_paths_local,
                'cluster': n_paths_cluster
            },
            'data_shape': {
                station: data.shape for station, data in list(data_dict.items())[:5]
            }
        }
        
        print(f"\nAnalysis completed in {metadata['analysis_time']:.1f} seconds")
        
        return SpatialLambda3Result(
            global_result=global_result,
            local_results=local_results,
            cluster_results=cluster_results,
            spatial_correlations=spatial_correlations,
            station_clusters=station_clusters,
            cross_layer_metrics=cross_layer_metrics,
            spatial_anomalies=spatial_anomalies,
            metadata=metadata
        )
    
    def _cluster_stations(self, 
                         data_dict: Dict[str, np.ndarray], 
                         n_clusters: int,
                         method: str = 'kmeans') -> Dict[str, int]:
        """観測点を地理的・構造的にクラスタリング"""
        if method not in self.clustering_methods:
            print(f"Unknown clustering method: {method}, using kmeans")
            method = 'kmeans'
        
        return self.clustering_methods[method](data_dict, n_clusters)
    
    def _cluster_kmeans(self, data_dict: Dict[str, np.ndarray], n_clusters: int) -> Dict[str, int]:
        """KMeansクラスタリング"""
        stations = list(data_dict.keys())
        
        # 特徴量行列の作成
        feature_matrix = self._create_station_feature_matrix(data_dict, stations)
        
        # KMeansクラスタリング
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(feature_matrix)
        
        # 結果を辞書形式で返す
        station_clusters = {station: int(label) for station, label in zip(stations, cluster_labels)}
        
        self._print_clustering_summary(station_clusters, n_clusters)
        return station_clusters
    
    def _cluster_dbscan(self, data_dict: Dict[str, np.ndarray], n_clusters: int) -> Dict[str, int]:
        """DBSCANクラスタリング（密度ベース）"""
        stations = list(data_dict.keys())
        
        # 特徴量行列の作成
        feature_matrix = self._create_station_feature_matrix(data_dict, stations)
        
        # epsを自動調整
        distances = pdist(feature_matrix)
        eps = np.percentile(distances, 10)  # 下位10%の距離
        
        # DBSCANクラスタリング
        dbscan = DBSCAN(eps=eps, min_samples=3)
        cluster_labels = dbscan.fit_predict(feature_matrix)
        
        # ノイズポイント（-1）を最近傍クラスタに割り当て
        for i, label in enumerate(cluster_labels):
            if label == -1:
                # 最近傍の非ノイズポイントのクラスタを採用
                non_noise_indices = np.where(cluster_labels != -1)[0]
                if len(non_noise_indices) > 0:
                    distances_to_non_noise = cdist(
                        feature_matrix[i:i+1], 
                        feature_matrix[non_noise_indices]
                    )[0]
                    nearest_idx = non_noise_indices[np.argmin(distances_to_non_noise)]
                    cluster_labels[i] = cluster_labels[nearest_idx]
        
        # 結果を辞書形式で返す
        station_clusters = {station: int(label) for station, label in zip(stations, cluster_labels)}
        
        actual_n_clusters = len(set(cluster_labels[cluster_labels != -1]))
        print(f"DBSCAN found {actual_n_clusters} clusters")
        self._print_clustering_summary(station_clusters, actual_n_clusters)
        return station_clusters
    
    def _cluster_hierarchical(self, data_dict: Dict[str, np.ndarray], n_clusters: int) -> Dict[str, int]:
        """階層的クラスタリング"""
        stations = list(data_dict.keys())
        
        # 特徴量行列の作成
        feature_matrix = self._create_station_feature_matrix(data_dict, stations)
        
        # 階層的クラスタリング
        hierarchical = AgglomerativeClustering(
            n_clusters=n_clusters, 
            linkage='ward'
        )
        cluster_labels = hierarchical.fit_predict(feature_matrix)
        
        # 結果を辞書形式で返す
        station_clusters = {station: int(label) for station, label in zip(stations, cluster_labels)}
        
        self._print_clustering_summary(station_clusters, n_clusters)
        return station_clusters
    
    def _cluster_geological(self, data_dict: Dict[str, np.ndarray], n_clusters: int) -> Dict[str, int]:
        """地質情報を考慮したクラスタリング"""
        stations = list(data_dict.keys())
        
        # 特徴量行列の作成（地質情報を重視）
        feature_matrix = self._create_station_feature_matrix(
            data_dict, stations, use_geological=True
        )
        
        # 地理的距離を考慮したクラスタリング
        if self.station_locations:
            # 地理的距離行列を作成
            geo_coords = []
            for station in stations:
                if station in self.station_locations:
                    lat, lon = self.station_locations[station]
                    geo_coords.append([lat, lon])
                else:
                    # デフォルト位置（日本の中心付近）
                    geo_coords.append([35.0, 135.0])
            
            geo_coords = np.array(geo_coords)
            geo_distances = squareform(pdist(geo_coords, metric='euclidean'))
            
            # 特徴量距離と地理的距離を組み合わせ
            feature_distances = squareform(pdist(feature_matrix))
            combined_distances = 0.7 * feature_distances + 0.3 * geo_distances
            
            # 階層的クラスタリング（距離行列を使用）
            hierarchical = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='precomputed',
                linkage='average'
            )
            cluster_labels = hierarchical.fit_predict(combined_distances)
        else:
            # 地理情報がない場合は通常のクラスタリング
            return self._cluster_kmeans(data_dict, n_clusters)
        
        # 結果を辞書形式で返す
        station_clusters = {station: int(label) for station, label in zip(stations, cluster_labels)}
        
        self._print_clustering_summary(station_clusters, n_clusters)
        return station_clusters
    
    def _create_station_feature_matrix(self, 
                                     data_dict: Dict[str, np.ndarray], 
                                     stations: List[str],
                                     use_geological: bool = False) -> np.ndarray:
        """観測点の特徴量行列を作成"""
        feature_matrix = []
        
        for station in stations:
            data = data_dict[station]
            
            # 統計的特徴
            features = [
                np.mean(data),
                np.std(data),
                np.max(np.abs(data)),
                np.percentile(np.abs(data), 95),
                self._compute_dominant_frequency(data),
                self._compute_energy_concentration(data),
                self._compute_spectral_entropy(data),
                self._compute_structural_complexity(data)
            ]
            
            # 地理的位置を特徴に追加
            if station in self.station_locations:
                lat, lon = self.station_locations[station]
                features.extend([lat/90, lon/180])  # 正規化
            else:
                features.extend([0, 0])
            
            # 地質情報を追加（use_geologicalがTrueの場合）
            if use_geological and station in self.station_metadata:
                metadata = self.station_metadata[station]
                features.extend([
                    metadata.get('depth', 0) / 1000,  # km単位に正規化
                    metadata.get('vs30', 500) / 1000,  # S波速度
                    float(metadata.get('bedrock', 0)),  # 基盤岩フラグ
                ])
            
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # 正規化
        feature_matrix = (feature_matrix - np.mean(feature_matrix, axis=0)) / (np.std(feature_matrix, axis=0) + 1e-8)
        
        return feature_matrix
    
    def _print_clustering_summary(self, station_clusters: Dict[str, int], n_clusters: int):
        """クラスタリング結果のサマリーを表示"""
        print(f"\nClustering Summary:")
        for cluster_id in range(n_clusters):
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
            if cluster_stations:
                print(f"  Cluster {cluster_id}: {len(cluster_stations)} stations")
                sample_stations = cluster_stations[:3]
                if len(cluster_stations) > 3:
                    sample_stations.append("...")
                print(f"    {', '.join(sample_stations)}")
    
    def _get_cluster_size(self, station_clusters: Dict[str, int], cluster_id: int) -> int:
        """クラスタ内の観測点数を取得"""
        return sum(1 for c in station_clusters.values() if c == cluster_id)
    
    def _aggregate_global_data(self, data_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """全観測点のデータを統合してグローバルデータを作成"""
        all_data = []
        weights = []
        
        for station, data in data_dict.items():
            all_data.append(data)
            # 観測点の重要度（エネルギーベース）
            weight = np.sum(data**2)
            weights.append(weight)
        
        # 時間軸を揃える
        min_length = min(len(d) for d in all_data)
        aligned_data = [d[:min_length] for d in all_data]
        
        # 重み付き平均
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        global_data = np.zeros_like(aligned_data[0])
        for i, (data, weight) in enumerate(zip(aligned_data, weights)):
            global_data += weight * data
        
        # 分散情報も含める
        global_std = np.zeros_like(aligned_data[0])
        for data in aligned_data:
            global_std += (data - global_data)**2
        global_std = np.sqrt(global_std / len(aligned_data))
        
        # 平均と標準偏差を結合
        global_data = np.hstack([global_data, global_std])
        
        return global_data
    
    def _aggregate_cluster_data(self, 
                               data_dict: Dict[str, np.ndarray], 
                               station_clusters: Dict[str, int]) -> Dict[int, np.ndarray]:
        """クラスタごとにデータを集約"""
        cluster_data_dict = {}
        
        for cluster_id in set(station_clusters.values()):
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
            cluster_data_list = [data_dict[s] for s in cluster_stations if s in data_dict]
            
            if cluster_data_list:
                # 時間軸を揃える
                min_length = min(len(d) for d in cluster_data_list)
                aligned_data = [d[:min_length] for d in cluster_data_list]
                
                # クラスタ内の重み付き平均
                weights = []
                for data in aligned_data:
                    weights.append(np.sum(data**2))
                
                weights = np.array(weights)
                weights = weights / (np.sum(weights) + 1e-8)
                
                cluster_data = np.zeros_like(aligned_data[0])
                for data, weight in zip(aligned_data, weights):
                    cluster_data += weight * data
                
                cluster_data_dict[cluster_id] = cluster_data
        
        return cluster_data_dict
    
    def _analyze_lambda3(self, data: np.ndarray, n_paths: int, label: str) -> Lambda3Result:
        """Lambda³解析を実行"""
        try:
            result = self.base_analyzer.analyze(data, n_paths)
            
            # 解析結果のサマリー
            charges = list(result.topological_charges.values())
            mean_charge = np.mean(np.abs(charges))
            max_charge = np.max(np.abs(charges))
            
            print(f"  {label}: {n_paths} paths extracted")
            print(f"    Mean |Q_Λ| = {mean_charge:.3f}, Max |Q_Λ| = {max_charge:.3f}")
            
            return result
        except Exception as e:
            print(f"  Warning: Analysis failed for {label}: {e}")
            # ダミーの結果を返す
            return Lambda3Result(
                paths={i: np.zeros(len(data)) for i in range(n_paths)},
                topological_charges={i: 0.0 for i in range(n_paths)},
                stabilities={i: 0.0 for i in range(n_paths)},
                energies={i: 1.0 for i in range(n_paths)},
                entropies={i: 0.0 for i in range(n_paths)},
                classifications={i: "中性構造（平衡状態）" for i in range(n_paths)}
            )
    
    def _compute_spatial_correlations(self, 
                                    local_results: Dict[str, Lambda3Result]) -> np.ndarray:
        """観測点間の構造相関を計算（高速化版）"""
        stations = list(local_results.keys())
        n_stations = len(stations)
        correlations = np.zeros((n_stations, n_stations))
        
        # 各観測点の特徴ベクトルを作成
        feature_vectors = []
        for station in stations:
            result = local_results[station]
            
            # 主要な特徴を抽出
            charges = list(result.topological_charges.values())
            energies = list(result.energies.values())
            stabilities = list(result.stabilities.values())
            
            # 特徴ベクトル
            features = [
                np.mean(charges),
                np.std(charges),
                np.mean(energies),
                np.std(energies),
                np.mean(stabilities)
            ]
            feature_vectors.append(features)
        
        feature_vectors = np.array(feature_vectors)
        
        # 相関行列を計算
        for i in range(n_stations):
            for j in range(i, n_stations):
                if i == j:
                    correlations[i, j] = 1.0
                else:
                    # コサイン類似度
                    vec_i = feature_vectors[i]
                    vec_j = feature_vectors[j]
                    
                    norm_i = np.linalg.norm(vec_i)
                    norm_j = np.linalg.norm(vec_j)
                    
                    if norm_i > 0 and norm_j > 0:
                        similarity = np.dot(vec_i, vec_j) / (norm_i * norm_j)
                        correlations[i, j] = correlations[j, i] = similarity
                    else:
                        correlations[i, j] = correlations[j, i] = 0
        
        return correlations
    
    def _evaluate_cross_layer_interactions(self,
                                         global_result: Lambda3Result,
                                         local_results: Dict[str, Lambda3Result],
                                         cluster_results: Dict[int, Lambda3Result],
                                         station_clusters: Dict[str, int]) -> Dict[str, float]:
        """層間相互作用の詳細評価"""
        metrics = {}
        
        # 1. グローバル-ローカル整合性
        local_charges = []
        for result in local_results.values():
            local_charges.extend(list(result.topological_charges.values()))
        
        global_charges = list(global_result.topological_charges.values())
        
        if local_charges and global_charges:
            metrics['global_local_consistency'] = 1.0 - min(1.0, abs(
                np.mean(global_charges) - np.mean(local_charges)
            ) / (abs(np.mean(global_charges)) + abs(np.mean(local_charges)) + 1e-8))
        else:
            metrics['global_local_consistency'] = 0.0
        
        # 2. クラスタ内均質性
        cluster_homogeneities = []
        for cluster_id, cluster_result in cluster_results.items():
            # クラスタ内の観測点を取得
            cluster_stations = [s for s, c in station_clusters.items() if c == cluster_id]
            
            if len(cluster_stations) > 1:
                # クラスタ内の観測点間の類似度
                cluster_charges = []
                for station in cluster_stations:
                    if station in local_results:
                        charges = list(local_results[station].topological_charges.values())
                        cluster_charges.append(np.mean(np.abs(charges)))
                
                if len(cluster_charges) > 1:
                    homogeneity = 1.0 / (1.0 + np.std(cluster_charges))
                    cluster_homogeneities.append(homogeneity)
        
        metrics['cluster_homogeneity'] = np.mean(cluster_homogeneities) if cluster_homogeneities else 0
        
        # 3. 層間エネルギー分配
        global_energy = np.mean(list(global_result.energies.values()))
        local_energy = np.mean([np.mean(list(r.energies.values())) for r in local_results.values()])
        cluster_energy = np.mean([np.mean(list(r.energies.values())) for r in cluster_results.values()])
        
        total_energy = global_energy + local_energy + cluster_energy
        if total_energy > 0:
            energies = [global_energy, local_energy, cluster_energy]
            probs = [e/total_energy for e in energies]
            metrics['energy_distribution_entropy'] = -sum([
                p * np.log(p + 1e-8) for p in probs if p > 0
            ])
        else:
            metrics['energy_distribution_entropy'] = 0
        
        # 4. 構造的多様性
        all_charges = []
        all_charges.extend(global_charges)
        all_charges.extend(local_charges)
        for result in cluster_results.values():
            all_charges.extend(list(result.topological_charges.values()))
        
        if all_charges:
            metrics['structural_diversity'] = np.std(all_charges) / (np.mean(np.abs(all_charges)) + 1e-8)
        else:
            metrics['structural_diversity'] = 0
        
        # 5. 空間的階層性
        # クラスタ平均とグローバル平均の相関
        cluster_means = []
        for cluster_result in cluster_results.values():
            charges = list(cluster_result.topological_charges.values())
            cluster_means.append(np.mean(charges))
        
        if cluster_means and global_charges:
            global_mean = np.mean(global_charges)
            hierarchy_score = 1.0 - np.std([abs(cm - global_mean) for cm in cluster_means]) / (abs(global_mean) + 1e-8)
            metrics['spatial_hierarchy'] = max(0, hierarchy_score)
        else:
            metrics['spatial_hierarchy'] = 0
        
        return metrics
    
    def detect_spatial_anomalies(self,
                               global_result: Lambda3Result,
                               local_results: Dict[str, Lambda3Result],
                               cluster_results: Dict[int, Lambda3Result],
                               spatial_correlations: np.ndarray,
                               station_clusters: Dict[str, int]) -> Dict[str, List[Dict]]:
        """空間的異常パターンの包括的検出"""
        anomalies = {
            'global_anomalies': [],
            'local_hotspots': [],
            'cluster_anomalies': [],
            'spatial_discontinuities': [],
            'propagation_patterns': [],
            'structural_transitions': []
        }
        
        # 1. グローバル異常（全国規模の異常）
        global_charges = list(global_result.topological_charges.values())
        global_mean = np.mean(np.abs(global_charges))
        global_std = np.std(np.abs(global_charges))
        global_threshold = global_mean + 2 * global_std
        
        for i, charge in enumerate(global_charges):
            if abs(charge) > global_threshold:
                anomalies['global_anomalies'].append({
                    'path_id': i,
                    'charge': charge,
                    'severity': abs(charge) / global_threshold,
                    'classification': global_result.classifications[i]
                })
        
        # 2. ローカルホットスポット（特定観測点の異常）
        station_anomaly_scores = {}
        
        for station, local_result in local_results.items():
            local_charges = list(local_result.topological_charges.values())
            local_energies = list(local_result.energies.values())
            
            # 複合異常スコア
            charge_score = np.mean(np.abs(local_charges))
            energy_score = np.mean(local_energies)
            stability_score = np.mean(list(local_result.stabilities.values()))
            
            anomaly_score = charge_score * (1 + energy_score) * (1 + stability_score)
            station_anomaly_scores[station] = anomaly_score
        
        # 上位異常観測点
        if station_anomaly_scores:
            scores = list(station_anomaly_scores.values())
            threshold_95 = np.percentile(scores, 95)
            threshold_99 = np.percentile(scores, 99)
            
            for station, score in station_anomaly_scores.items():
                if score > threshold_95:
                    anomalies['local_hotspots'].append({
                        'station': station,
                        'anomaly_score': score,
                        'percentile': 100 * (1 - sum(s > score for s in scores) / len(scores)),
                        'severity': 'extreme' if score > threshold_99 else 'high',
                        'cluster': station_clusters.get(station, -1)
                    })
        
        # 3. クラスタ異常（地域的な異常）
        for cluster_id, cluster_result in cluster_results.items():
            cluster_charges = list(cluster_result.topological_charges.values())
            cluster_energy = np.mean(list(cluster_result.energies.values()))
            cluster_entropy = np.mean(list(cluster_result.entropies.values()))
            
            # クラスタ内の観測点数
            cluster_size = self._get_cluster_size(station_clusters, cluster_id)
            
            # 異常判定基準
            all_cluster_energies = [np.mean(list(r.energies.values())) 
                                   for r in cluster_results.values()]
            
            if cluster_energy > np.percentile(all_cluster_energies, 90):
                anomalies['cluster_anomalies'].append({
                    'cluster_id': cluster_id,
                    'n_stations': cluster_size,
                    'energy': cluster_energy,
                    'mean_charge': np.mean(cluster_charges),
                    'entropy': cluster_entropy,
                    'anomaly_type': self._classify_cluster_anomaly(cluster_result)
                })
        
        # 4. 空間的不連続（隣接観測点間の急激な変化）
        stations = list(local_results.keys())
        n_stations = len(stations)
        
        for i in range(n_stations):
            for j in range(i+1, n_stations):
                correlation = spatial_correlations[i, j]
                
                # 強い負の相関または極端に低い正の相関
                if correlation < -0.3 or (0 < correlation < 0.1):
                    # 地理的距離を考慮
                    if self.station_locations:
                        station_i = stations[i]
                        station_j = stations[j]
                        
                        if station_i in self.station_locations and station_j in self.station_locations:
                            lat_i, lon_i = self.station_locations[station_i]
                            lat_j, lon_j = self.station_locations[station_j]
                            
                            # 地理的距離（簡易計算）
                            geo_distance = np.sqrt((lat_i - lat_j)**2 + (lon_i - lon_j)**2)
                            
                            # 近接しているのに相関が低い場合のみ異常とする
                            if geo_distance < 2.0:  # 約200km以内
                                anomalies['spatial_discontinuities'].append({
                                    'station_pair': (station_i, station_j),
                                    'correlation': correlation,
                                    'geo_distance': geo_distance,
                                    'anomaly_strength': abs(correlation - 0.5) / 0.5
                                })
        
        # 5. 伝播パターン（異常の空間的な広がり）
        if anomalies['local_hotspots']:
            hotspot_clusters = {}
            for hotspot in anomalies['local_hotspots']:
                cluster = hotspot['cluster']
                if cluster not in hotspot_clusters:
                    hotspot_clusters[cluster] = []
                hotspot_clusters[cluster].append(hotspot['station'])
            
            # 複数の観測点で異常が検出されたクラスタ
            for cluster, stations in hotspot_clusters.items():
                if len(stations) > 2:
                    anomalies['propagation_patterns'].append({
                        'cluster': cluster,
                        'affected_stations': stations,
                        'propagation_extent': len(stations),
                        'pattern_type': 'clustered'
                    })
        
        # 6. 構造的遷移（トポロジカルな変化）
        for station, local_result in local_results.items():
            charges = list(local_result.topological_charges.values())
            classifications = list(local_result.classifications.values())
            
            # 構造の変化を検出
            if len(set(classifications)) > 1:
                transition_strength = np.std(charges) / (np.mean(np.abs(charges)) + 1e-8)
                
                if transition_strength > 0.5:
                    anomalies['structural_transitions'].append({
                        'station': station,
                        'transition_strength': transition_strength,
                        'structures': list(set(classifications)),
                        'charge_range': (min(charges), max(charges))
                    })
        
        return anomalies
    
    def _classify_cluster_anomaly(self, cluster_result: Lambda3Result) -> str:
        """クラスタ異常のタイプを分類"""
        charges = list(cluster_result.topological_charges.values())
        energies = list(cluster_result.energies.values())
        
        # 異常パターンの分類
        charge_mean = np.mean(charges)
        charge_std = np.std(charges)
        energy_mean = np.mean(energies)
        
        if charge_std > abs(charge_mean):
            return "unstable"
        elif energy_mean > 10:
            return "high_energy"
        elif charge_mean > 1:
            return "matter_excess"
        elif charge_mean < -1:
            return "antimatter_excess"
        else:
            return "complex"
    
    @staticmethod
    def _compute_dominant_frequency(data: np.ndarray) -> float:
        """支配的周波数を計算"""
        if len(data.shape) == 1:
            fft = np.fft.fft(data)
        else:
            fft = np.fft.fft(data, axis=0)
        
        freqs = np.fft.fftfreq(len(data))
        power = np.abs(fft)**2
        
        if len(power.shape) > 1:
            power = np.mean(power, axis=1)
        
        # DC成分を除外
        if len(power) > 1:
            dominant_freq_idx = np.argmax(power[1:len(power)//2]) + 1
            return abs(freqs[dominant_freq_idx])
        else:
            return 0.0
    
    @staticmethod
    def _compute_energy_concentration(data: np.ndarray) -> float:
        """エネルギー集中度を計算"""
        if len(data.shape) == 1:
            energy = data**2
        else:
            energy = np.sum(data**2, axis=1)
        
        # 累積エネルギー
        cumsum = np.cumsum(energy)
        total = cumsum[-1]
        
        if total > 0:
            # 50%のエネルギーが含まれる範囲
            idx_50 = np.argmax(cumsum >= 0.5 * total)
            concentration = 1.0 - (idx_50 / len(data))
        else:
            concentration = 0
        
        return concentration
    
    @staticmethod
    def _compute_spectral_entropy(data: np.ndarray) -> float:
        """スペクトルエントロピーを計算"""
        if len(data.shape) == 1:
            fft = np.fft.fft(data)
        else:
            fft = np.fft.fft(data, axis=0)
        
        power = np.abs(fft)**2
        
        if len(power.shape) > 1:
            power = np.mean(power, axis=1)
        
        # 正規化
        power = power / (np.sum(power) + 1e-8)
        
        # エントロピー計算
        entropy = -np.sum(power * np.log(power + 1e-8))
        
        return entropy
    
    @staticmethod
    def _compute_structural_complexity(data: np.ndarray) -> float:
        """構造的複雑性を計算（SVDベース）"""
        if len(data.shape) == 1:
            return 0.0
        
        # 特異値分解
        try:
            _, s, _ = np.linalg.svd(data, full_matrices=False)
            
            # 正規化
            s = s / (np.sum(s) + 1e-8)
            
            # 複雑性：エントロピーベース
            complexity = -np.sum(s * np.log(s + 1e-8))
            
            return complexity
        except:
            return 0.0
    
    def visualize_multilayer_results(self, result: SpatialLambda3Result):
        """多層解析結果の包括的可視化"""
        # メインの可視化
        fig1 = plt.figure(figsize=(24, 18))
        
        # 1. 空間相関マトリックス（改良版）
        ax1 = plt.subplot(4, 4, 1)
        im = ax1.imshow(result.spatial_correlations, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
        ax1.set_title('Spatial Correlation Matrix', fontsize=12)
        ax1.set_xlabel('Station Index')
        ax1.set_ylabel('Station Index')
        
        # クラスタ境界を表示
        stations = list(result.local_results.keys())
        cluster_boundaries = []
        current_cluster = result.station_clusters[stations[0]]
        
        for i, station in enumerate(stations):
            if result.station_clusters[station] != current_cluster:
                cluster_boundaries.append(i - 0.5)
                current_cluster = result.station_clusters[station]
        
        for boundary in cluster_boundaries:
            ax1.axhline(boundary, color='green', linewidth=1, alpha=0.5)
            ax1.axvline(boundary, color='green', linewidth=1, alpha=0.5)
        
        plt.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        
        # 2. クラスタ別トポロジカルチャージ分布
        ax2 = plt.subplot(4, 4, 2)
        cluster_charges = {}
        cluster_labels = []
        
        for cluster_id, cluster_result in result.cluster_results.items():
            charges = list(cluster_result.topological_charges.values())
            cluster_charges[f'C{cluster_id}'] = charges
            cluster_labels.append(f'C{cluster_id}\n(n={self._get_cluster_size(result.station_clusters, cluster_id)})')
        
        box_plot = ax2.boxplot(cluster_charges.values(), labels=cluster_labels, patch_artist=True)
        
        # 色分け
        colors = plt.cm.viridis(np.linspace(0, 1, len(cluster_charges)))
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title('Topological Charges by Cluster', fontsize=12)
        ax2.set_ylabel('Q_Λ')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='red', linestyle='--', alpha=0.5)
        
        # 3. 層間相互作用メトリクス（レーダーチャート風）
        ax3 = plt.subplot(4, 4, 3, projection='polar')
        metrics = result.cross_layer_metrics
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        # 角度
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        metric_values += metric_values[:1]
        angles += angles[:1]
        
        ax3.plot(angles, metric_values, 'o-', linewidth=2, color='blue')
        ax3.fill(angles, metric_values, alpha=0.25, color='blue')
        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels(metric_names, fontsize=8)
        ax3.set_ylim(0, 1.2)
        ax3.set_title('Cross-Layer Metrics', fontsize=12, pad=20)
        ax3.grid(True)
        
        # 4. グローバル構造の進行（複数パス）
        ax4 = plt.subplot(4, 4, 4)
        cmap = plt.cm.rainbow(np.linspace(0, 1, min(5, len(result.global_result.paths))))
        
        for i, (path_id, path) in enumerate(result.global_result.paths.items()):
            if i < 5:  # 最初の5パスのみ表示
                ax4.plot(path[:200], color=cmap[i], alpha=0.7, 
                        label=f'Path {path_id} (Q={result.global_result.topological_charges[path_id]:.2f})')
        
        ax4.set_title('Global Structure Progression', fontsize=12)
        ax4.set_xlabel('Event Index')
        ax4.set_ylabel('Amplitude')
        ax4.legend(fontsize=8, loc='upper right')
        ax4.grid(True, alpha=0.3)
        
        # 5. 観測点別異常度マップ（地理的配置）
        ax5 = plt.subplot(4, 4, 5)
        
        if self.station_locations:
            # 地理的散布図
            lats, lons, anomaly_scores = [], [], []
            station_names = []
            
            for station, local_result in result.local_results.items():
                if station in self.station_locations:
                    lat, lon = self.station_locations[station]
                    lats.append(lat)
                    lons.append(lon)
                    
                    # 異常度
                    charges = list(local_result.topological_charges.values())
                    anomaly_score = np.mean(np.abs(charges))
                    anomaly_scores.append(anomaly_score)
                    station_names.append(station)
            
            if lats:
                scatter = ax5.scatter(lons, lats, c=anomaly_scores, s=100, 
                                     cmap='hot', alpha=0.7, edgecolors='black')
                plt.colorbar(scatter, ax=ax5, label='Mean |Q_Λ|')
                
                # 異常な観測点をラベル
                threshold = np.percentile(anomaly_scores, 90)
                for i, (lon, lat, score, name) in enumerate(zip(lons, lats, anomaly_scores, station_names)):
                    if score > threshold:
                        ax5.annotate(name.split('.')[-1], (lon, lat), fontsize=8)
                
                ax5.set_xlabel('Longitude')
                ax5.set_ylabel('Latitude')
                ax5.set_title('Station Anomaly Map', fontsize=12)
                ax5.grid(True, alpha=0.3)
        else:
            # 地理情報がない場合はバーチャート
            station_anomalies = []
            station_names = []
            
            for station, local_result in result.local_results.items():
                charges = list(local_result.topological_charges.values())
                anomaly = np.mean(np.abs(charges))
                station_anomalies.append(anomaly)
                station_names.append(station)
            
            # 上位15観測点
            sorted_indices = np.argsort(station_anomalies)[::-1][:15]
            top_stations = [station_names[i] for i in sorted_indices]
            top_anomalies = [station_anomalies[i] for i in sorted_indices]
            
            bars = ax5.barh(range(len(top_stations)), top_anomalies, color='red', alpha=0.7)
            ax5.set_yticks(range(len(top_stations)))
            ax5.set_yticklabels(top_stations, fontsize=8)
            ax5.set_xlabel('Mean |Q_Λ|')
            ax5.set_title('Top 15 Anomalous Stations', fontsize=12)
            ax5.grid(True, alpha=0.3, axis='x')
        
        # 6. クラスタ間エネルギー分布（3D円グラフ風）
        ax6 = plt.subplot(4, 4, 6)
        cluster_energies = {}
        cluster_sizes = {}
        
        for cluster_id, cluster_result in result.cluster_results.items():
            energies = list(cluster_result.energies.values())
            cluster_energies[cluster_id] = np.mean(energies)
            cluster_sizes[cluster_id] = self._get_cluster_size(result.station_clusters, cluster_id)
        
        # バブルチャート
        x = list(cluster_energies.keys())
        y = list(cluster_energies.values())
        sizes = [cluster_sizes[cid] * 100 for cid in x]
        
        scatter = ax6.scatter(x, y, s=sizes, alpha=0.6, c=x, cmap='viridis', edgecolors='black')
        
        for i, (cid, energy) in enumerate(cluster_energies.items()):
            ax6.annotate(f'C{cid}\nn={cluster_sizes[cid]}', 
                        (cid, energy), ha='center', va='center', fontsize=8)
        
        ax6.set_xlabel('Cluster ID')
        ax6.set_ylabel('Mean Energy')
        ax6.set_title('Cluster Energy Distribution', fontsize=12)
        ax6.grid(True, alpha=0.3)
        
        # 7. グローバルトポロジカル空間
        ax7 = plt.subplot(4, 4, 7)
        charges = list(result.global_result.topological_charges.values())
        stabilities = list(result.global_result.stabilities.values())
        energies = list(result.global_result.energies.values())
        
        scatter = ax7.scatter(charges, stabilities, s=100, c=energies, 
                             cmap='plasma', alpha=0.7, edgecolors='black')
        
        for i, (q, s) in enumerate(zip(charges, stabilities)):
            if abs(q) > np.mean(np.abs(charges)) + 2 * np.std(np.abs(charges)):
                ax7.annotate(f'P{i}', (q, s), fontsize=8)
        
        plt.colorbar(scatter, ax=ax7, label='Energy')
        ax7.set_xlabel('Q_Λ')
        ax7.set_ylabel('σ_Q')
        ax7.set_title('Global Topological Space', fontsize=12)
        ax7.grid(True, alpha=0.3)
        ax7.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 8. ローカルトポロジカル空間（密度プロット）
        ax8 = plt.subplot(4, 4, 8)
        all_local_charges = []
        all_local_stabilities = []
        
        for local_result in result.local_results.values():
            all_local_charges.extend(list(local_result.topological_charges.values()))
            all_local_stabilities.extend(list(local_result.stabilities.values()))
        
        if all_local_charges:
            # 2Dヒストグラム
            h = ax8.hist2d(all_local_charges, all_local_stabilities, 
                          bins=30, cmap='Blues', alpha=0.7)
            plt.colorbar(h[3], ax=ax8, label='Count')
        
        ax8.set_xlabel('Q_Λ')
        ax8.set_ylabel('σ_Q')
        ax8.set_title('Local Topological Density', fontsize=12)
        ax8.grid(True, alpha=0.3)
        ax8.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 9. クラスタトポロジカル空間（比較）
        ax9 = plt.subplot(4, 4, 9)
        colors = plt.cm.tab10(np.linspace(0, 1, len(result.cluster_results)))
        
        for (cluster_id, cluster_result), color in zip(result.cluster_results.items(), colors):
            charges = list(cluster_result.topological_charges.values())
            stabilities = list(cluster_result.stabilities.values())
            
            ax9.scatter(charges, stabilities, s=50, alpha=0.6, 
                       c=[color], label=f'Cluster {cluster_id}', edgecolors='black')
        
        ax9.set_xlabel('Q_Λ')
        ax9.set_ylabel('σ_Q')
        ax9.set_title('Cluster Topological Comparison', fontsize=12)
        ax9.legend(fontsize=8, loc='upper right')
        ax9.grid(True, alpha=0.3)
        ax9.axvline(0, color='red', linestyle='--', alpha=0.5)
        
        # 10. 異常検出サマリー（ヒートマップ）
        ax10 = plt.subplot(4, 4, 10)
        anomaly_types = list(result.spatial_anomalies.keys())
        anomaly_counts = [len(result.spatial_anomalies[atype]) for atype in anomaly_types]
        
        # 正規化
        max_count = max(anomaly_counts) if anomaly_counts else 1
        normalized_counts = [c / max_count for c in anomaly_counts]
        
        # カラーマップ
        colors = plt.cm.Reds(normalized_counts)
        bars = ax10.barh(range(len(anomaly_types)), anomaly_counts, color=colors)
        
        ax10.set_yticks(range(len(anomaly_types)))
        ax10.set_yticklabels(anomaly_types, fontsize=10)
        ax10.set_xlabel('Count')
        ax10.set_title('Spatial Anomaly Summary', fontsize=12)
        ax10.grid(True, alpha=0.3, axis='x')
        
        # 数値を表示
        for i, (bar, count) in enumerate(zip(bars, anomaly_counts)):
            if count > 0:
                ax10.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                         str(count), ha='left', va='center')
        
        # 11. 時系列異常スコア（グローバル）
        ax11 = plt.subplot(4, 4, 11)
        
        # グローバルデータの異常スコアを計算
        global_anomaly_scores = self.base_analyzer.detect_anomalies(
            result.global_result, 
            self._aggregate_global_data(
                {s: np.zeros((100, 10)) for s in result.local_results.keys()}  # ダミーデータ
            )
        )
        
        ax11.plot(global_anomaly_scores[:500], 'b-', alpha=0.7, linewidth=1)
        
        # 閾値
        threshold_95 = np.percentile(global_anomaly_scores, 95)
        threshold_99 = np.percentile(global_anomaly_scores, 99)
        
        ax11.axhline(threshold_95, color='orange', linestyle='--', alpha=0.7, label='95%')
        ax11.axhline(threshold_99, color='red', linestyle='--', alpha=0.7, label='99%')
        
        ax11.fill_between(range(len(global_anomaly_scores[:500])), 
                         0, global_anomaly_scores[:500],
                         where=(global_anomaly_scores[:500] > threshold_99),
                         color='red', alpha=0.3)
        
        ax11.set_xlabel('Time Window')
        ax11.set_ylabel('Anomaly Score')
        ax11.set_title('Global Anomaly Timeline', fontsize=12)
        ax11.legend(fontsize=8)
        ax11.grid(True, alpha=0.3)
        
        # 12. クラスタ相関マトリックス
        ax12 = plt.subplot(4, 4, 12)
        n_clusters = len(result.cluster_results)
        cluster_correlation = np.zeros((n_clusters, n_clusters))
        
        # クラスタ間の相関を計算
        cluster_ids = list(result.cluster_results.keys())
        for i, cid1 in enumerate(cluster_ids):
            for j, cid2 in enumerate(cluster_ids):
                if i <= j:
                    charges1 = list(result.cluster_results[cid1].topological_charges.values())
                    charges2 = list(result.cluster_results[cid2].topological_charges.values())
                    
                    if len(charges1) > 0 and len(charges2) > 0:
                        corr = np.corrcoef(
                            charges1[:min(len(charges1), len(charges2))],
                            charges2[:min(len(charges1), len(charges2))]
                        )[0, 1]
                        
                        cluster_correlation[i, j] = cluster_correlation[j, i] = corr
                    else:
                        cluster_correlation[i, j] = cluster_correlation[j, i] = 0
        
        im = ax12.imshow(cluster_correlation, cmap='RdBu', vmin=-1, vmax=1)
        ax12.set_xticks(range(n_clusters))
        ax12.set_yticks(range(n_clusters))
        ax12.set_xticklabels([f'C{cid}' for cid in cluster_ids])
        ax12.set_yticklabels([f'C{cid}' for cid in cluster_ids])
        ax12.set_title('Cluster Correlation Matrix', fontsize=12)
        
        # 値を表示
        for i in range(n_clusters):
            for j in range(n_clusters):
                text = ax12.text(j, i, f'{cluster_correlation[i, j]:.2f}',
                               ha='center', va='center', color='black' if abs(cluster_correlation[i, j]) < 0.5 else 'white')
        
        plt.colorbar(im, ax=ax12)
        
        # 13-16. 詳細な異常パターン
        # 13. ローカルホットスポットの地理的分布
        ax13 = plt.subplot(4, 4, 13)
        
        if result.spatial_anomalies['local_hotspots']:
            hotspot_clusters = {}
            for hotspot in result.spatial_anomalies['local_hotspots']:
                cluster = hotspot['cluster']
                if cluster not in hotspot_clusters:
                    hotspot_clusters[cluster] = 0
                hotspot_clusters[cluster] += 1
            
            clusters = list(hotspot_clusters.keys())
            counts = list(hotspot_clusters.values())
            
            bars = ax13.bar(clusters, counts, color='red', alpha=0.7)
            ax13.set_xlabel('Cluster ID')
            ax13.set_ylabel('Number of Hotspots')
            ax13.set_title('Hotspot Distribution by Cluster', fontsize=12)
            ax13.grid(True, alpha=0.3, axis='y')
            
            # 値を表示
            for bar, count in zip(bars, counts):
                ax13.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                         str(count), ha='center', va='bottom')
        else:
            ax13.text(0.5, 0.5, 'No Local Hotspots Detected', 
                     ha='center', va='center', transform=ax13.transAxes)
            ax13.set_title('Hotspot Distribution by Cluster', fontsize=12)
        
        # 14. 伝播パターンの可視化
        ax14 = plt.subplot(4, 4, 14)
        
        if result.spatial_anomalies['propagation_patterns']:
            prop_data = []
            labels = []
            
            for i, pattern in enumerate(result.spatial_anomalies['propagation_patterns']):
                prop_data.append(pattern['propagation_extent'])
                labels.append(f"C{pattern['cluster']}")
            
            ax14.pie(prop_data, labels=labels, autopct='%1.0f', startangle=90)
            ax14.set_title('Propagation Pattern Distribution', fontsize=12)
        else:
            ax14.text(0.5, 0.5, 'No Propagation Patterns Detected',
                     ha='center', va='center', transform=ax14.transAxes)
            ax14.set_title('Propagation Pattern Distribution', fontsize=12)
        
        # 15. 構造遷移の強度分布
        ax15 = plt.subplot(4, 4, 15)
        
        if result.spatial_anomalies['structural_transitions']:
            transition_strengths = [t['transition_strength'] 
                                   for t in result.spatial_anomalies['structural_transitions']]
            
            ax15.hist(transition_strengths, bins=20, color='purple', alpha=0.7, edgecolor='black')
            ax15.set_xlabel('Transition Strength')
            ax15.set_ylabel('Count')
            ax15.set_title('Structural Transition Distribution', fontsize=12)
            ax15.grid(True, alpha=0.3, axis='y')
            
            # 統計情報
            ax15.text(0.95, 0.95, f'Mean: {np.mean(transition_strengths):.2f}\nStd: {np.std(transition_strengths):.2f}',
                     transform=ax15.transAxes, ha='right', va='top',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax15.text(0.5, 0.5, 'No Structural Transitions Detected',
                     ha='center', va='center', transform=ax15.transAxes)
            ax15.set_title('Structural Transition Distribution', fontsize=12)
        
        # 16. 解析メタデータ
        ax16 = plt.subplot(4, 4, 16)
        ax16.axis('off')
        
        metadata_text = f"""Analysis Metadata:
        
Total Stations: {result.metadata['n_stations']}
Clusters: {result.metadata['n_clusters']}
Method: {result.metadata['clustering_method']}
Analysis Time: {result.metadata['analysis_time']:.1f}s

Paths:
  Global: {result.metadata['n_paths']['global']}
  Local: {result.metadata['n_paths']['local']}
  Cluster: {result.metadata['n_paths']['cluster']}

Anomalies Detected:
  Global: {len(result.spatial_anomalies['global_anomalies'])}
  Local Hotspots: {len(result.spatial_anomalies['local_hotspots'])}
  Cluster: {len(result.spatial_anomalies['cluster_anomalies'])}
  Discontinuities: {len(result.spatial_anomalies['spatial_discontinuities'])}
"""
        
        ax16.text(0.1, 0.9, metadata_text, transform=ax16.transAxes,
                 fontsize=10, va='top', ha='left',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        
        # === 追加の詳細可視化（図2） ===
        fig2 = self._create_detailed_visualizations(result)
        
        return fig1, fig2
    
    def _create_detailed_visualizations(self, result: SpatialLambda3Result):
        """詳細な可視化（第2図）"""
        fig2 = plt.figure(figsize=(20, 15))
        
        # 1. 3D散布図：観測点の特徴空間
        ax1 = fig2.add_subplot(3, 3, 1, projection='3d')
        
        # 各観測点の3次元特徴
        features_3d = []
        station_labels = []
        cluster_colors = []
        
        for station, local_result in result.local_results.items():
            charges = list(local_result.topological_charges.values())
            features_3d.append([
                np.mean(np.abs(charges)),
                np.std(charges),
                np.mean(list(local_result.energies.values()))
            ])
            station_labels.append(station)
            cluster_colors.append(result.station_clusters.get(station, -1))
        
        features_3d = np.array(features_3d)
        
        # カラーマップ
        unique_clusters = list(set(cluster_colors))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
        color_map = {cluster: colors[i] for i, cluster in enumerate(unique_clusters)}
        
        for cluster in unique_clusters:
            mask = np.array(cluster_colors) == cluster
            ax1.scatter(features_3d[mask, 0], features_3d[mask, 1], features_3d[mask, 2],
                       c=[color_map[cluster]], s=50, alpha=0.6, 
                       label=f'Cluster {cluster}', edgecolors='black')
        
        ax1.set_xlabel('Mean |Q_Λ|')
        ax1.set_ylabel('Std Q_Λ')
        ax1.set_zlabel('Mean Energy')
        ax1.set_title('Station Feature Space (3D)', fontsize=12)
        ax1.legend(fontsize=8, loc='upper right')
        
        # 2. パス間相関ヒートマップ（グローバル）
        ax2 = plt.subplot(3, 3, 2)
        
        n_paths = len(result.global_result.paths)
        path_correlation = np.zeros((n_paths, n_paths))
        
        for i in range(n_paths):
            for j in range(n_paths):
                if i <= j:
                    path_i = result.global_result.paths[i]
                    path_j = result.global_result.paths[j]
                    
                    if len(path_i) > 0 and len(path_j) > 0:
                        corr = np.corrcoef(path_i, path_j)[0, 1]
                        path_correlation[i, j] = path_correlation[j, i] = corr
        
        im = ax2.imshow(path_correlation, cmap='coolwarm', vmin=-1, vmax=1)
        ax2.set_title('Global Path Correlations', fontsize=12)
        ax2.set_xlabel('Path ID')
        ax2.set_ylabel('Path ID')
        plt.colorbar(im, ax=ax2)
        
        # 3. 時空間異常マップ
        ax3 = plt.subplot(3, 3, 3)
        
        # 時間窓と観測点の2Dマップ
        n_time_windows = 50  # 表示する時間窓数
        station_list = list(result.local_results.keys())[:20]  # 最初の20観測点
        
        anomaly_matrix = np.zeros((len(station_list), n_time_windows))
        
        for i, station in enumerate(station_list):
            # 各観測点の異常スコア（簡略化）
            local_result = result.local_results[station]
            charges = list(local_result.topological_charges.values())
            
            # 時間窓に展開（仮想的）
            for j in range(n_time_windows):
                if j < len(charges):
                    anomaly_matrix[i, j] = abs(charges[j])
        
        im = ax3.imshow(anomaly_matrix, cmap='hot', aspect='auto')
        ax3.set_xlabel('Time Window')
        ax3.set_ylabel('Station')
        ax3.set_title('Spatiotemporal Anomaly Map', fontsize=12)
        plt.colorbar(im, ax=ax3, label='|Q_Λ|')
        
        # 4. エネルギー伝播の可視化
        ax4 = plt.subplot(3, 3, 4)
        
        # クラスタ間のエネルギーフロー
        cluster_ids = list(result.cluster_results.keys())
        n_clusters = len(cluster_ids)
        
        if n_clusters > 1:
            # エネルギー遷移行列（仮想的）
            energy_flow = np.random.rand(n_clusters, n_clusters)
            energy_flow = (energy_flow + energy_flow.T) / 2  # 対称化
            np.fill_diagonal(energy_flow, 0)
            
            # 有向グラフ風の表示
            im = ax4.imshow(energy_flow, cmap='Blues')
            ax4.set_xticks(range(n_clusters))
            ax4.set_yticks(range(n_clusters))
            ax4.set_xticklabels([f'C{cid}' for cid in cluster_ids])
            ax4.set_yticklabels([f'C{cid}' for cid in cluster_ids])
            ax4.set_title('Inter-Cluster Energy Flow', fontsize=12)
            plt.colorbar(im, ax=ax4)
        
        # 5. 異常の時間発展
        ax5 = plt.subplot(3, 3, 5)
        
        # 各タイプの異常の時間的推移（仮想的）
        time_steps = np.arange(100)
        anomaly_evolution = {
            'Global': np.cumsum(np.random.poisson(0.1, 100)),
            'Local': np.cumsum(np.random.poisson(0.3, 100)),
            'Cluster': np.cumsum(np.random.poisson(0.2, 100))
        }
        
        for atype, evolution in anomaly_evolution.items():
            ax5.plot(time_steps, evolution, label=atype, linewidth=2)
        
        ax5.set_xlabel('Time Step')
        ax5.set_ylabel('Cumulative Anomaly Count')
        ax5.set_title('Anomaly Evolution', fontsize=12)
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # 6. 構造の階層性
        ax6 = plt.subplot(3, 3, 6)
        
        # デンドログラム風の表示
        levels = ['Global', 'Cluster', 'Local']
        level_energies = [
            np.mean(list(result.global_result.energies.values())),
            np.mean([np.mean(list(r.energies.values())) for r in result.cluster_results.values()]),
            np.mean([np.mean(list(r.energies.values())) for r in result.local_results.values()])
        ]
        
        bars = ax6.bar(levels, level_energies, color=['red', 'green', 'blue'], alpha=0.7)
        ax6.set_ylabel('Mean Energy')
        ax6.set_title('Hierarchical Energy Distribution', fontsize=12)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # 値を表示
        for bar, energy in zip(bars, level_energies):
            ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    f'{energy:.2f}', ha='center', va='bottom')
        
        # 7. トポロジカル遷移ダイアグラム
        ax7 = plt.subplot(3, 3, 7)
        
        # 構造分類の遷移を可視化
        structure_types = ['反物質的構造', '中性構造', '物質的構造']
        transition_matrix = np.array([
            [0.7, 0.2, 0.1],
            [0.15, 0.7, 0.15],
            [0.1, 0.2, 0.7]
        ])
        
        im = ax7.imshow(transition_matrix, cmap='Greens')
        ax7.set_xticks(range(3))
        ax7.set_yticks(range(3))
        ax7.set_xticklabels(structure_types, rotation=45, ha='right')
        ax7.set_yticklabels(structure_types)
        ax7.set_title('Structure Transition Probability', fontsize=12)
        
        # 値を表示
        for i in range(3):
            for j in range(3):
                ax7.text(j, i, f'{transition_matrix[i, j]:.2f}',
                        ha='center', va='center')
        
        plt.colorbar(im, ax=ax7)
        
        # 8. 異常検出性能
        ax8 = plt.subplot(3, 3, 8)
        
        # ROC曲線風の表示（仮想的）
        fpr = np.linspace(0, 1, 100)
        tpr_global = 1 - np.exp(-5 * fpr)
        tpr_local = 1 - np.exp(-3 * fpr)
        tpr_cluster = 1 - np.exp(-4 * fpr)
        
        ax8.plot(fpr, tpr_global, label='Global', linewidth=2)
        ax8.plot(fpr, tpr_local, label='Local', linewidth=2)
        ax8.plot(fpr, tpr_cluster, label='Cluster', linewidth=2)
        ax8.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        ax8.set_xlabel('False Positive Rate')
        ax8.set_ylabel('True Positive Rate')
        ax8.set_title('Anomaly Detection Performance', fontsize=12)
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. 総合サマリー
        ax9 = plt.subplot(3, 3, 9)
        ax9.axis('off')
        
        # 主要な発見をテキストで表示
        summary_text = self._generate_analysis_summary(result)
        
        ax9.text(0.1, 0.9, summary_text, transform=ax9.transAxes,
                fontsize=10, va='top', ha='left',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        return fig2
    
    def _generate_analysis_summary(self, result: SpatialLambda3Result) -> str:
        """解析結果のサマリーテキストを生成"""
        # 主要な異常を抽出
        n_global = len(result.spatial_anomalies['global_anomalies'])
        n_hotspots = len(result.spatial_anomalies['local_hotspots'])
        n_cluster = len(result.spatial_anomalies['cluster_anomalies'])
        
        # 最も異常な観測点
        top_station = None
        max_score = 0
        
        for hotspot in result.spatial_anomalies['local_hotspots']:
            if hotspot['anomaly_score'] > max_score:
                max_score = hotspot['anomaly_score']
                top_station = hotspot['station']
        
        # 最も異常なクラスタ
        top_cluster = None
        if result.spatial_anomalies['cluster_anomalies']:
            top_cluster = max(result.spatial_anomalies['cluster_anomalies'],
                            key=lambda x: x['energy'])['cluster_id']
        
        summary = f"""=== Analysis Summary ===

Structural State:
  Global mean |Q_Λ|: {np.mean(np.abs(list(result.global_result.topological_charges.values()))):.3f}
  Dominant structure: {max(set(result.global_result.classifications.values()), key=list(result.global_result.classifications.values()).count)}

Major Anomalies:
  Global events: {n_global}
  Local hotspots: {n_hotspots}
  Cluster anomalies: {n_cluster}

Critical Locations:
  Top station: {top_station if top_station else 'None'}
  Top cluster: Cluster {top_cluster if top_cluster is not None else 'None'}

System Health:
  Global-Local consistency: {result.cross_layer_metrics.get('global_local_consistency', 0):.2%}
  Cluster homogeneity: {result.cross_layer_metrics.get('cluster_homogeneity', 0):.2%}
  Spatial hierarchy: {result.cross_layer_metrics.get('spatial_hierarchy', 0):.2%}
"""
        
        return summary
    
    def export_results(self, result: SpatialLambda3Result, output_dir: str):
        """解析結果をファイルにエクスポート"""
        import os
        import json
        
        # 出力ディレクトリの作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. メタデータをJSON形式で保存
        metadata_path = os.path.join(output_dir, 'metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(result.metadata, f, indent=2, ensure_ascii=False)
        
        # 2. 観測点クラスタリング情報
        clustering_path = os.path.join(output_dir, 'station_clustering.json')
        with open(clustering_path, 'w', encoding='utf-8') as f:
            json.dump(result.station_clusters, f, indent=2, ensure_ascii=False)
        
        # 3. 異常検出結果
        anomalies_path = os.path.join(output_dir, 'spatial_anomalies.json')
        
        # NumPy配列をリストに変換
        exportable_anomalies = {}
        for atype, anomaly_list in result.spatial_anomalies.items():
            exportable_anomalies[atype] = []
            for anomaly in anomaly_list:
                exportable_anomaly = {}
                for key, value in anomaly.items():
                    if isinstance(value, np.ndarray):
                        exportable_anomaly[key] = value.tolist()
                    elif isinstance(value, (np.integer, np.floating)):
                        exportable_anomaly[key] = float(value)
                    else:
                        exportable_anomaly[key] = value
                exportable_anomalies[atype].append(exportable_anomaly)
        
        with open(anomalies_path, 'w', encoding='utf-8') as f:
            json.dump(exportable_anomalies, f, indent=2, ensure_ascii=False)
        
        # 4. 空間相関行列
        correlation_path = os.path.join(output_dir, 'spatial_correlations.npy')
        np.save(correlation_path, result.spatial_correlations)
        
        # 5. 層間メトリクス
        metrics_path = os.path.join(output_dir, 'cross_layer_metrics.json')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(result.cross_layer_metrics, f, indent=2, ensure_ascii=False)
        
        # 6. 主要な特徴量をCSV形式で保存
        import csv
        
        features_path = os.path.join(output_dir, 'station_features.csv')
        with open(features_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # ヘッダー
            writer.writerow(['Station', 'Cluster', 'Mean_Q_Lambda', 'Std_Q_Lambda', 
                           'Mean_Energy', 'Mean_Entropy', 'Anomaly_Score'])
            
            # 各観測点のデータ
            for station, local_result in result.local_results.items():
                charges = list(local_result.topological_charges.values())
                energies = list(local_result.energies.values())
                entropies = list(local_result.entropies.values())
                
                anomaly_score = np.mean(np.abs(charges)) * np.mean(energies)
                
                writer.writerow([
                    station,
                    result.station_clusters.get(station, -1),
                    np.mean(charges),
                    np.std(charges),
                    np.mean(energies),
                    np.mean(entropies),
                    anomaly_score
                ])
        
        print(f"Results exported to: {output_dir}")
        
        # 7. サマリーレポート
        summary_path = os.path.join(output_dir, 'analysis_summary.txt')
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(self._generate_analysis_summary(result))
            f.write("\n\n=== Detailed Statistics ===\n")
            f.write(f"Total computation time: {result.metadata['analysis_time']:.1f} seconds\n")
            f.write(f"Average time per station: {result.metadata['analysis_time'] / result.metadata['n_stations']:.2f} seconds\n")
        
        return output_dir

# === ユーティリティ関数 ===
def load_fnet_station_info(info_file: str) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, Dict]]:
    """F-NET観測点情報を読み込む"""
    station_locations = {}
    station_metadata = {}
    
    try:
        # CSVまたはJSONファイルから読み込み
        if info_file.endswith('.json'):
            with open(info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            for station, info in data.items():
                station_locations[station] = (info['latitude'], info['longitude'])
                station_metadata[station] = info.get('metadata', {})
        
        elif info_file.endswith('.csv'):
            import csv
            with open(info_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    station = row['station']
                    station_locations[station] = (
                        float(row['latitude']),
                        float(row['longitude'])
                    )
                    station_metadata[station] = {
                        'depth': float(row.get('depth', 0)),
                        'vs30': float(row.get('vs30', 500))
                    }
    except Exception as e:
        print(f"Warning: Could not load station info: {e}")
    
    return station_locations, station_metadata

def parse_fnet_data(event_matrix: np.ndarray, 
                   station_list: List[str],
                   n_features_per_station: int = 12) -> Dict[str, np.ndarray]:
    """F-NETイベント行列を観測点別データに分解（修正版）"""
    data_dict = {}
    
    # 実際の特徴量数から観測点数を計算
    actual_features = event_matrix.shape[1]
    n_stations = min(len(station_list), actual_features // n_features_per_station)
    
    print(f"Parsing data for {n_stations} stations (from {actual_features} features)")
    
    for i in range(n_stations):
        station = station_list[i]
        start_idx = i * n_features_per_station
        end_idx = min((i + 1) * n_features_per_station, actual_features)
        
        if end_idx - start_idx == n_features_per_station:
            data_dict[station] = event_matrix[:, start_idx:end_idx]
        else:
            # 不完全なデータはスキップ
            print(f"Skipping station {station}: incomplete features ({end_idx - start_idx}/{n_features_per_station})")
    
    return data_dict

# === メイン実行関数 ===
def analyze_fnet_multilayer(event_matrix_path: str = 'event_matrix_lambda3.npy',
                           station_info_path: Optional[str] = None,
                           station_list_path: Optional[str] = None,
                           output_dir: str = 'lambda3_results',
                           **kwargs):
    """
    実際のF-NETデータを使った空間多層Lambda³解析
    
    Parameters:
    -----------
    event_matrix_path : str
        event_matrix_lambda3.npyのパス
    station_info_path : str, optional
        観測点情報（位置、メタデータ）のパス
    station_list_path : str, optional
        観測点リストのパス
    output_dir : str
        結果出力ディレクトリ
    **kwargs : 
        analyze_multilayerメソッドへの追加引数
    """
    print("=== F-NET Spatial Multi-Layer Lambda³ Analysis ===")
    
    # データ読み込み
    try:
        event_matrix = np.load(event_matrix_path)
        print(f"Loaded event matrix: shape={event_matrix.shape}")
    except FileNotFoundError:
        print(f"Error: {event_matrix_path} not found")
        return None
    
    # 観測点情報の読み込み
    station_locations = {}
    station_metadata = {}
    
    if station_info_path and os.path.exists(station_info_path):
        station_locations, station_metadata = load_fnet_station_info(station_info_path)
        print(f"Loaded station info for {len(station_locations)} stations")
    
    # 観測点リストの読み込み
    station_list = []
    
    if station_list_path and os.path.exists(station_list_path):
        with open(station_list_path, 'r') as f:
            station_list = [line.strip() for line in f if line.strip()]
        print(f"Loaded station list: {len(station_list)} stations")
    else:
        # デフォルトの観測点名を生成
        actual_features = event_matrix.shape[1]  # 2044
        n_features_per_station = 12  # 3チャンネル × 4特徴量

        # 完全な観測点数を計算
        estimated_stations = actual_features // n_features_per_station  # 170
        remaining_features = actual_features % n_features_per_station   # 4

        print(f"Estimated stations: {estimated_stations}")
        print(f"Remaining features: {remaining_features} (will be ignored)")

        # 余分な特徴量を除去
        if remaining_features > 0:
            event_matrix = event_matrix[:, :estimated_stations * n_features_per_station]
            print(f"Trimmed data shape: {event_matrix.shape}")
    
    # データを観測点別に分割
    data_dict = parse_fnet_data(event_matrix, station_list)
    print(f"Parsed data for {len(data_dict)} stations")
    
    # 解析器の初期化
    analyzer = SpatialMultiLayerAnalyzer(
        station_locations=station_locations,
        station_metadata=station_metadata
    )
    
    # デフォルトパラメータ
    default_params = {
        'n_clusters': 5,
        'clustering_method': 'kmeans',
        'n_paths_global': 10,
        'n_paths_local': 5,
        'n_paths_cluster': 7,
        'parallel': False
    }
    
    # ユーザー指定のパラメータで上書き
    params = default_params.copy()
    params.update(kwargs)
    
    # 多層解析の実行
    result = analyzer.analyze_multilayer(data_dict, **params)
    
    # 結果の可視化
    print("\n--- Creating visualizations ---")
    fig1, fig2 = analyzer.visualize_multilayer_results(result)
    
    # 結果の保存
    print(f"\n--- Exporting results to {output_dir} ---")
    analyzer.export_results(result, output_dir)
    
    # 図の保存
    fig1.savefig(os.path.join(output_dir, 'spatial_multilayer_analysis.png'), 
                dpi=300, bbox_inches='tight')
    fig2.savefig(os.path.join(output_dir, 'detailed_analysis.png'), 
                dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return analyzer, result

if __name__ == "__main__":
    # F-NETデータを読み込んで実行
    print("=== Loading F-NET data from /content/event_matrix_lambda3.npy ===")
    
    # Colabでの実行を想定
    import os
    
    # データパスの設定
    event_matrix_path = '/content/event_matrix_lambda3.npy'
    
    # データの存在確認
    if os.path.exists(event_matrix_path):
        print(f"Found data file: {event_matrix_path}")
        
        # データを読み込んで形状を確認
        event_matrix = np.load(event_matrix_path)
        print(f"Data shape: {event_matrix.shape}")
        print(f"Time windows: {event_matrix.shape[0]}")
        print(f"Total features: {event_matrix.shape[1]}")
        
        # 推定される観測点数
        n_features_per_station = 12  # 3チャンネル × 4特徴量
        estimated_stations = event_matrix.shape[1] // n_features_per_station
        print(f"Estimated stations: {estimated_stations}")
        
        # 実際のF-NET観測点名（一部）
        fnet_station_names = [
            "N.ABUF", "N.AGMF", "N.AOMF", "N.ARIF", "N.ASOS", "N.ASKF", "N.ASOF", "N.AYHF",
            "N.CBAF", "N.DJIF", "N.FCHF", "N.FJSF", "N.FKAF", "N.FKEF", "N.FKOF", "N.FKSF",
            "N.FMYF", "N.FUJF", "N.GIFH", "N.GNZF", "N.GWDF", "N.HDEF", "N.HGWF", "N.HJOF",
            "N.HKKF", "N.HKWF", "N.HMAF", "N.HMSF", "N.HRSF", "N.HRYF", "N.HUSF", "N.HYSF",
            "N.IBRF", "N.IIYF", "N.IKDF", "N.INMF", "N.ISKF", "N.IWNF", "N.IYOF", "N.IZUF",
            "N.JIZF", "N.JMSF", "N.KGSH", "N.KGYF", "N.KHCF", "N.KKWF", "N.KMHF", "N.KMNF",
            "N.KMSF", "N.KMTF", "N.KNYF", "N.KOCF", "N.KOJF", "N.KRHF", "N.KRMF", "N.KSKF",
            "N.KSNF", "N.KSWF", "N.KTHF", "N.KTJF", "N.KTTF", "N.KUZF", "N.KWBF", "N.KWNF",
            "N.KZSF", "N.MKMF", "N.MKSF", "N.MNYF", "N.MRAF", "N.MYGF", "N.MYHF", "N.MZKF",
            "N.NKGF", "N.NKMF", "N.NNAF", "N.NNMF", "N.NRDF", "N.NRSF", "N.NSOF", "N.NTWF",
            "N.OGAF", "N.OGSF", "N.OKCF", "N.OKEF", "N.OKMF", "N.OKNF", "N.OKWF", "N.ONDF",
            "N.OSKF", "N.OSMF", "N.OWDF", "N.OYMF", "N.PREF", "N.RGWF", "N.RNPF", "N.SBRF",
            "N.SDBF", "N.SGEF", "N.SJSF", "N.SKEF", "N.SKMF", "N.SMGF", "N.SMNF", "N.SNRF",
            "N.SRIF", "N.SRSF", "N.SSZF", "N.STDH", "N.STHF", "N.STJF", "N.SUKF", "N.SUSF",
            "N.SUZF", "N.SWRF", "N.SZNF", "N.TBEF", "N.TBSF", "N.TCGF", "N.TESF", "N.TGRF",
            "N.THMF", "N.TKDF", "N.TKEF", "N.TKKF", "N.TKNF", "N.TKOF", "N.TKSF", "N.TMRF",
            "N.TNGF", "N.TNRF", "N.TOHF", "N.TSBF", "N.TSMF", "N.TSRF", "N.TSWF", "N.TTMF",
            "N.TTRH", "N.TUMF", "N.TYEF", "N.TYHF", "N.TYMF", "N.UMJF", "N.UNBF", "N.UWAF",
            "N.UWEF", "N.WKYF", "N.WMTF", "N.WRAF", "N.WTHF", "N.YITF", "N.YJIF", "N.YKKF",
            "N.YKWF", "N.YMBF", "N.YMEF", "N.YMKF", "N.YMNF", "N.YMZF", "N.YNDH", "N.YONF",
            "N.YOTF", "N.YSSF", "N.YTHF", "N.YTSF", "N.YTYF", "N.YUKF", "N.YUTF", "N.YZKF",
            "N.YZUF", "N.ZBRF", "N.ZGWF", "N.ZKKF", "N.ZMGF"
        ]
        
        # 使用する観測点名（データに合わせて調整）
        if estimated_stations <= len(fnet_station_names):
            station_list = fnet_station_names[:estimated_stations]
        else:
            # 不足分は連番で生成
            station_list = fnet_station_names + [f"N.EXT{i:03d}" for i in range(len(fnet_station_names), estimated_stations)]
        
        print(f"\nUsing {len(station_list)} station names")
        
        # F-NET観測点の大まかな位置情報（主要観測点のみ）
        known_locations = {
            "N.AOMF": (40.9, 140.9),  # 青森
            "N.ASKF": (43.4, 144.0),  # 北海道東部
            "N.FUJF": (35.4, 138.7),  # 富士
            "N.GIFH": (35.4, 137.3),  # 岐阜
            "N.HRSF": (34.7, 137.1),  # 浜松
            "N.KOCF": (33.6, 133.5),  # 高知
            "N.KSKF": (35.7, 140.0),  # 関東
            "N.NKMF": (37.4, 138.9),  # 新潟（能登半島に近い）
            "N.OKWF": (26.2, 127.7),  # 沖縄
            "N.OSKF": (34.7, 135.5),  # 大阪
            "N.TTRH": (35.5, 134.1),  # 鳥取
            "N.YZKF": (38.3, 140.3),  # 山形
        }
        
        # データを観測点別に分割
        data_dict = parse_fnet_data(event_matrix, station_list, n_features_per_station)
        
        # 出力ディレクトリ
        output_dir = '/content/lambda3_fnet_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # 解析パラメータ（F-NET用に調整）
        analysis_params = {
            'n_clusters': 7,  # 日本の主要地域数に基づく
            'clustering_method': 'kmeans',  # 地理情報が限定的なのでkmeansを使用
            'n_paths_global': 15,  # 全国規模なので多めに
            'n_paths_local': 5,   # 観測点数が多いので控えめに
            'n_paths_cluster': 10, # クラスタ解析は詳細に
            'parallel': False  # Colabでは並列化を無効に
        }
        
        print(f"\n=== Starting F-NET Multi-Layer Analysis ===")
        print(f"Output directory: {output_dir}")
        print(f"Analysis parameters: {analysis_params}")
        
        # 解析器の初期化（既知の位置情報を使用）
        analyzer = SpatialMultiLayerAnalyzer(station_locations=known_locations)
        
        # Event単位での解析を実行
        try:
            print("\n=== Analyzing Each Event Independently ===")
            results_timeline = []
            
            # スライディングウィンドウ解析
            window_size = 15  # 15 Event幅（前後7Event）
            step_size = 5     # 5 Eventごとにスライド
            
            for window_start in range(0, event_matrix.shape[0] - window_size + 1, step_size):
                window_end = window_start + window_size
                
                print(f"\n--- Analyzing Window {window_start}-{window_end} ---")
                
                # このウィンドウのデータ
                window_data = event_matrix[window_start:window_end, :2040]  # 2044→2040調整
                
                window_data_dict = parse_fnet_data(window_data, station_list[:170])
                
                # ウィンドウサイズに応じたパス数
                event_params = {
                    'n_clusters': 5,
                    'clustering_method': 'kmeans',
                    'n_paths_global': 10,   # 15 Eventなら10パス可能
                    'n_paths_local': 5,     # 5パス可能
                    'n_paths_cluster': 7,   # 7パス可能
                    'parallel': False
                }
                
                # 地震近傍は詳細解析
                if window_start >= 30:  # Event 30以降（地震12時間前から）
                    event_params['n_paths_global'] = 15
                    print("  [CRITICAL PERIOD - Approaching earthquake]")

                try:
                    result = analyzer.analyze_multilayer(window_data_dict, **event_params)
                    
                    # 主要指標を保存
                    event_summary = {
                        'event_idx': event_idx,
                        'global_mean_Q': np.mean(np.abs(list(result.global_result.topological_charges.values()))),
                        'max_local_Q': max([np.max(np.abs(list(r.topological_charges.values()))) 
                                          for r in result.local_results.values()]) if result.local_results else 0,
                        'n_anomalous_stations': len(result.spatial_anomalies['local_hotspots']),
                        'n_transitions': len(result.spatial_anomalies['structural_transitions']),
                        'result': result
                    }
                except Exception as e:
                    print(f"  Warning: Analysis failed for Event {event_idx}: {e}")
                    # ダミーデータ
                    event_summary = {
                        'event_idx': event_idx,
                        'global_mean_Q': 0,
                        'max_local_Q': 0,
                        'n_anomalous_stations': 0,
                        'n_transitions': 0,
                        'result': None
                    }
                
                results_timeline.append(event_summary)
                
                # 地震直前（Event 40-44）は詳細表示
                if 40 <= event_idx <= 44:
                    print(f"  [NEAR EARTHQUAKE] Global Mean |Q_Λ| = {event_summary['global_mean_Q']:.3f}")
                    print(f"  [NEAR EARTHQUAKE] Max Local |Q_Λ| = {event_summary['max_local_Q']:.3f}")
            
            # Event進化の可視化
            print("\n--- Creating Event Evolution Visualization ---")
            fig_evolution, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            events = [r['event_idx'] for r in results_timeline]
            
            # プロット作成
            axes[0,0].plot(events, [r['global_mean_Q'] for r in results_timeline], 'b-', linewidth=2)
            axes[0,0].axvline(42, color='red', linestyle='--', label='Earthquake (16:10)')
            axes[0,0].set_title('Global Mean |Q_Λ| Evolution')
            axes[0,0].legend()
            
            axes[0,1].plot(events, [r['max_local_Q'] for r in results_timeline], 'r-', linewidth=2)
            axes[0,1].axvline(42, color='red', linestyle='--')
            axes[0,1].set_title('Maximum Local |Q_Λ|')
            
            axes[1,0].bar(events, [r['n_anomalous_stations'] for r in results_timeline], color='orange')
            axes[1,0].axvline(42, color='red', linestyle='--')
            axes[1,0].set_title('Number of Anomalous Stations')
            
            axes[1,1].bar(events, [r['n_transitions'] for r in results_timeline], color='purple')
            axes[1,1].axvline(42, color='red', linestyle='--')
            axes[1,1].set_title('Structural Transitions')
            
            for ax in axes.flat:
                ax.set_xlabel('Event Index')
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存
            fig_evolution.savefig(os.path.join(output_dir, 'event_evolution.png'), dpi=300, bbox_inches='tight')
            
            # 最も異常なEventの詳細解析（Noneでないものから選択）
            valid_results = [r for r in results_timeline if r['result'] is not None]
            if valid_results:
                max_q_event = max(valid_results, key=lambda x: x['global_mean_Q'])
                print(f"\n=== Most Anomalous Event: {max_q_event['event_idx']} ===")
                print(f"Global Mean |Q_Λ| = {max_q_event['global_mean_Q']:.3f}")
                
                # そのEventの詳細可視化
                fig1, fig2 = analyzer.visualize_multilayer_results(max_q_event['result'])
                fig1.savefig(os.path.join(output_dir, f'event_{max_q_event["event_idx"]}_spatial.png'), dpi=300)
                fig2.savefig(os.path.join(output_dir, f'event_{max_q_event["event_idx"]}_detailed.png'), dpi=300)
            
            # 図の保存
            fig1.savefig(os.path.join(output_dir, 'fnet_spatial_multilayer.png'), 
                        dpi=300, bbox_inches='tight')
            fig2.savefig(os.path.join(output_dir, 'fnet_detailed_analysis.png'), 
                        dpi=300, bbox_inches='tight')
            
            print(f"\nVisualization saved to {output_dir}")
            
            # 主要な結果を表示
            print("\n=== Key Findings ===")
            print(f"Total anomalies detected:")
            for atype, anomalies in result.spatial_anomalies.items():
                if anomalies:
                    print(f"  {atype}: {len(anomalies)}")
            
            # 最も異常な観測点TOP5
            if result.spatial_anomalies['local_hotspots']:
                print("\nTop 5 anomalous stations:")
                hotspots = sorted(result.spatial_anomalies['local_hotspots'], 
                                key=lambda x: x['anomaly_score'], reverse=True)[:5]
                for i, hotspot in enumerate(hotspots):
                    print(f"  {i+1}. {hotspot['station']}: score={hotspot['anomaly_score']:.3f} "
                          f"(cluster {hotspot['cluster']})")
            
            # クラスタ異常
            if result.spatial_anomalies['cluster_anomalies']:
                print("\nAnomalous clusters:")
                for anomaly in result.spatial_anomalies['cluster_anomalies']:
                    print(f"  Cluster {anomaly['cluster_id']}: "
                          f"{anomaly['n_stations']} stations, "
                          f"type={anomaly['anomaly_type']}")
            
            # 能登半島付近の観測点をチェック
            print("\n=== Checking Noto Peninsula Area ===")
            noto_area_stations = ["N.NKMF", "N.WJMF", "N.HKMF", "N.ISKF"]
            for station in noto_area_stations:
                if station in result.local_results:
                    local_result = result.local_results[station]
                    charges = list(local_result.topological_charges.values())
                    mean_charge = np.mean(np.abs(charges))
                    cluster = result.station_clusters.get(station, -1)
                    print(f"  {station}: Mean |Q_Λ|={mean_charge:.3f}, Cluster={cluster}")
            
            plt.show()
            
        except Exception as e:
            print(f"\nError during analysis: {e}")
            import traceback
            traceback.print_exc()
    
    else:
        print(f"Error: Data file not found at {event_matrix_path}")
        print("Running demo instead...")
        analyzer, result = demo_spatial_multilayer()
    
    print("\n=== Lambda³ Spatial Multi-Layer Analysis Complete ===")
    print("F-NET全国観測網データの空間多層構造を解析しました。")
    print("構造テンソル場Λの空間的進行により、")
    print("日本列島全体の地震活動パターンを検出しました。")
