#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
异响指纹库与异响识别完整仿真代码
============================================================

本脚本无需外部音频数据，会自动模拟多类正常/异常声音，完成：
1) 异响数据模拟；
2) 直接测量特征提取；
3) 时频变换特征提取，并用 PCA 降维；
4) 融合特征构建；
5) 指纹库构建：类别原型、类内距离阈值、直接特征统计；
6) 已知类分类 + 未知异响开放集拒识；
7) 结果评估、可视化、样例 wav 与指纹库文件导出。

对应“三层级特征框架”：
- Level 1：直接测量特征 direct features
- Level 2：降维变换特征 transform/PCA embedding
- Level 3：融合特征 fusion fingerprint

运行示例：
    python abnormal_sound_fingerprint_simulation.py --out ./runs/asf_demo

加大仿真数据量：
    python abnormal_sound_fingerprint_simulation.py --n-per-class 300 --n-unknown 200 --out ./runs/asf_big

依赖：
    numpy scipy scikit-learn pandas matplotlib joblib
"""

from __future__ import annotations

import argparse
import json
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal, stats
from scipy.io import wavfile
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=RuntimeWarning)


# =============================================================================
# 1. 类别体系
# =============================================================================

KNOWN_CLASSES = [
    "normal",
    "friction",
    "knock",
    "squeal",
    "looseness",
    "leak",
    "bearing",
    "resonance",
    "scrape",
]

ABNORMAL_CLASSES = [c for c in KNOWN_CLASSES if c != "normal"]

UNKNOWN_SUBTYPES = [
    "chirp_sweep",
    "electric_buzz",
    "cavitation_like",
]

CLASS_DESCRIPTIONS = {
    "normal": "正常背景/设备运行声，含低频基频、谐波和少量宽带噪声",
    "friction": "摩擦异响：高频宽带能量增强，伴随不稳定接触噪声",
    "knock": "敲击/碰撞异响：稀疏瞬态冲击，峰值和峭度高",
    "squeal": "啸叫异响：稳定或微调制的高频窄带峰",
    "looseness": "松动异响：与转动相关的周期性或准周期性碰撞",
    "leak": "泄漏异响：连续高频嘶嘶声，谱平坦度较高",
    "bearing": "轴承类异响：高频共振被周期性冲击调制，包络谱有峰",
    "resonance": "共振异响：中低频窄带峰与谐波/调幅结构明显",
    "scrape": "刮擦异响：不规则短时高频摩擦/扫频片段",
    "unknown": "未知/不在库异响，用于开放集拒识验证",
}


@dataclass
class AudioMeta:
    sample_id: str
    label: str
    subtype: str
    device_id: str
    rpm_hz: float
    load: float
    sensor_position: str
    environment: str
    snr_db: float
    duration: float
    fs: int


@dataclass
class RecognitionResult:
    predicted_label: str
    best_known_label: str
    confidence: float
    best_probability: float
    best_distance: float
    distance_threshold: float
    rejected: bool
    reject_reasons: List[str]
    top3: List[Tuple[str, float]]


# =============================================================================
# 2. 声音模拟基础函数
# =============================================================================

def set_random_seed(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def limit_peak(x: np.ndarray, max_abs: float = 0.95) -> np.ndarray:
    """只在可能削波时限制峰值，不做强制归一化，以保留响度/冲击幅度特征。"""
    x = np.asarray(x, dtype=np.float64)
    peak = np.max(np.abs(x)) + 1e-12
    if peak > max_abs:
        x = x / peak * max_abs
    return x


def colored_noise(rng: np.random.Generator, n: int, fs: int, color: str = "white") -> np.ndarray:
    """生成白/粉/棕噪声。"""
    if color == "white":
        y = rng.standard_normal(n)
    else:
        alpha = {"pink": 1.0, "brown": 2.0}.get(color, 0.0)
        freqs = np.fft.rfftfreq(n, d=1.0 / fs)
        spec = rng.standard_normal(len(freqs)) + 1j * rng.standard_normal(len(freqs))
        scale = 1.0 / np.maximum(freqs, 1.0) ** (alpha / 2.0)
        spec *= scale
        spec[0] = 0.0
        y = np.fft.irfft(spec, n=n)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return y


def bandpass_noise(
    rng: np.random.Generator,
    n: int,
    fs: int,
    low_hz: float,
    high_hz: float,
    amp: float = 1.0,
    order: int = 4,
) -> np.ndarray:
    """生成指定频带的带通噪声。"""
    nyq = fs / 2.0
    low_hz = max(1.0, min(low_hz, nyq * 0.90))
    high_hz = max(low_hz + 10.0, min(high_hz, nyq * 0.98))
    y = rng.standard_normal(n)
    sos = signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, y)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return amp * y


def lowpass_noise(
    rng: np.random.Generator,
    n: int,
    fs: int,
    high_hz: float,
    amp: float = 1.0,
    order: int = 4,
) -> np.ndarray:
    """生成低通噪声。"""
    y = rng.standard_normal(n)
    high_hz = min(high_hz, fs * 0.45)
    sos = signal.butter(order, high_hz, btype="lowpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, y)
    y = y - np.mean(y)
    y = y / (np.std(y) + 1e-12)
    return amp * y


def smooth_random_envelope(
    rng: np.random.Generator,
    n: int,
    fs: int,
    rate_hz: float = 5.0,
    floor: float = 0.1,
) -> np.ndarray:
    """生成平滑随机包络，用于模拟接触不稳定、泄漏强度波动等。"""
    duration = n / fs
    num_points = max(4, int(duration * rate_hz))
    xp = np.linspace(0, duration, num_points)
    fp = rng.uniform(floor, 1.0, size=num_points)
    t = np.arange(n) / fs
    env = np.interp(t, xp, fp)
    cutoff = min(rate_hz * 2.0, fs * 0.2)
    sos = signal.butter(2, cutoff, btype="lowpass", fs=fs, output="sos")
    env = signal.sosfiltfilt(sos, env)
    return np.clip(env, floor, 1.0)


def add_damped_sine_pulse(
    x: np.ndarray,
    fs: int,
    t0: float,
    freq_hz: float,
    amp: float,
    decay: float,
    length_s: float = 0.05,
    phase: float = 0.0,
) -> None:
    """原地叠加一个指数衰减正弦冲击。"""
    n = len(x)
    start = int(round(t0 * fs))
    if start < 0 or start >= n:
        return
    length = min(int(round(length_s * fs)), n - start)
    if length <= 8:
        return
    tt = np.arange(length) / fs
    pulse = amp * np.exp(-decay * tt) * np.sin(2.0 * np.pi * freq_hz * tt + phase)
    pulse *= signal.windows.tukey(length, alpha=0.3)
    x[start : start + length] += pulse


def add_room_echo(rng: np.random.Generator, x: np.ndarray, fs: int) -> np.ndarray:
    """加入很弱的房间反射。"""
    y = x.copy()
    if rng.random() < 0.8:
        delay = int(rng.uniform(0.004, 0.025) * fs)
        gain = rng.uniform(0.03, 0.18)
        if 0 < delay < len(y):
            y[delay:] += gain * y[:-delay]
    return y


def base_machine_signal(
    rng: np.random.Generator,
    fs: int,
    duration: float,
    rpm_hz: Optional[float] = None,
    load: Optional[float] = None,
) -> Tuple[np.ndarray, float, float]:
    """正常设备运行声：低频转动基频 + 谐波 + 弱宽带噪声。"""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    rpm_hz = float(rpm_hz if rpm_hz is not None else rng.uniform(18.0, 58.0))
    load = float(load if load is not None else rng.uniform(0.2, 1.0))

    amp = rng.uniform(0.025, 0.065) * (0.7 + 0.6 * load)
    x = amp * np.sin(2.0 * np.pi * rpm_hz * t + rng.uniform(0, 2 * np.pi))
    x += 0.40 * amp * np.sin(2.0 * np.pi * 2.0 * rpm_hz * t + rng.uniform(0, 2 * np.pi))
    x += 0.22 * amp * np.sin(2.0 * np.pi * 3.0 * rpm_hz * t + rng.uniform(0, 2 * np.pi))
    x += 0.010 * colored_noise(rng, n, fs, "pink")
    return x, rpm_hz, load


def finalize_signal(
    rng: np.random.Generator,
    x: np.ndarray,
    fs: int,
    snr_db: float,
    environment: str,
) -> np.ndarray:
    """叠加环境噪声、回声，并限制峰值。"""
    n = len(x)
    signal_rms = np.sqrt(np.mean(x**2) + 1e-12)
    noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    if environment == "quiet_lab":
        noise = colored_noise(rng, n, fs, "white")
    elif environment == "factory":
        noise = 0.6 * colored_noise(rng, n, fs, "pink") + 0.4 * bandpass_noise(
            rng, n, fs, 200, min(4500, fs * 0.45)
        )
    else:  # reverberant_room
        noise = 0.5 * colored_noise(rng, n, fs, "white") + 0.5 * lowpass_noise(rng, n, fs, 1200)
    x = x + noise_rms * noise
    x = add_room_echo(rng, x, fs)
    return limit_peak(x)


# =============================================================================
# 3. 各类异响生成器
# =============================================================================

def simulate_known_sound(
    label: str,
    rng: np.random.Generator,
    fs: int,
    duration: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """模拟一个已知类别声音片段。"""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    x, rpm_hz, load = base_machine_signal(rng, fs, duration)

    if label == "normal":
        pass

    elif label == "friction":
        env = smooth_random_envelope(rng, n, fs, rate_hz=rng.uniform(4, 12), floor=0.2)
        x += bandpass_noise(rng, n, fs, 1200, fs * 0.47, amp=rng.uniform(0.035, 0.085)) * env
        f0 = rng.uniform(2500, 5200)
        dev = rng.uniform(20, 130)
        fm = rng.uniform(2, 9)
        inst = f0 + dev * np.sin(2 * np.pi * fm * t)
        phase = 2 * np.pi * np.cumsum(inst) / fs
        x += rng.uniform(0.008, 0.025) * np.sin(phase) * env

    elif label == "knock":
        n_events = int(rng.integers(4, 16))
        times = np.sort(rng.uniform(0.08, duration - 0.08, size=n_events))
        for ti in times:
            add_damped_sine_pulse(
                x, fs, ti,
                freq_hz=rng.uniform(450, 2100),
                amp=rng.uniform(0.12, 0.36),
                decay=rng.uniform(90, 240),
                length_s=rng.uniform(0.025, 0.075),
                phase=rng.uniform(0, 2 * np.pi),
            )
            add_damped_sine_pulse(
                x, fs, ti,
                freq_hz=rng.uniform(70, 180),
                amp=rng.uniform(0.04, 0.12),
                decay=rng.uniform(25, 80),
                length_s=rng.uniform(0.04, 0.12),
            )

    elif label == "squeal":
        f0 = rng.uniform(2200, 5600)
        dev = rng.uniform(15, 180)
        fm = rng.uniform(1.0, 8.0)
        inst = f0 + dev * np.sin(2 * np.pi * fm * t + rng.uniform(0, 2 * np.pi))
        phase = 2 * np.pi * np.cumsum(inst) / fs
        env = smooth_random_envelope(rng, n, fs, rate_hz=rng.uniform(1.0, 4.0), floor=0.55)
        x += rng.uniform(0.055, 0.145) * np.sin(phase) * env
        x += rng.uniform(0.006, 0.018) * np.sin(2 * phase + rng.uniform(0, 2 * np.pi)) * env

    elif label == "looseness":
        event_freq = max(6.0, min(32.0, rpm_hz * rng.choice([0.4, 0.5, 1.0])))
        period = 1.0 / event_freq
        ti = rng.uniform(0.02, 0.08)
        while ti < duration - 0.04:
            add_damped_sine_pulse(
                x, fs, ti + rng.normal(0.0, 0.006),
                freq_hz=rng.uniform(500, 1500),
                amp=rng.uniform(0.045, 0.14),
                decay=rng.uniform(110, 260),
                length_s=rng.uniform(0.018, 0.055),
            )
            ti += period * rng.uniform(0.85, 1.15)

    elif label == "leak":
        env = smooth_random_envelope(rng, n, fs, rate_hz=rng.uniform(1.0, 6.0), floor=0.65)
        x += bandpass_noise(rng, n, fs, 4200, fs * 0.485, amp=rng.uniform(0.060, 0.130)) * env
        x += bandpass_noise(rng, n, fs, 2500, 6000, amp=rng.uniform(0.012, 0.035))

    elif label == "bearing":
        defect_freq = rng.uniform(65, 190)
        resonance_freq = rng.uniform(2400, 5200)
        ti = rng.uniform(0.0, 1.0 / defect_freq)
        while ti < duration - 0.01:
            amp = rng.uniform(0.035, 0.105) * (0.9 + 0.5 * np.sin(2 * np.pi * rpm_hz * ti))
            add_damped_sine_pulse(
                x, fs, ti + rng.normal(0.0, 0.0007),
                freq_hz=resonance_freq,
                amp=amp,
                decay=rng.uniform(550, 950),
                length_s=rng.uniform(0.006, 0.018),
            )
            ti += (1.0 / defect_freq) * rng.uniform(0.985, 1.015)
        env = 0.5 + 0.5 * np.sin(2 * np.pi * defect_freq * t + rng.uniform(0, 2 * np.pi))
        x += rng.uniform(0.004, 0.012) * np.sin(2 * np.pi * resonance_freq * t) * env

    elif label == "resonance":
        f0 = rng.uniform(180, 900)
        am = 0.65 + 0.35 * np.sin(2 * np.pi * rng.uniform(2.0, 8.0) * t + rng.uniform(0, 2 * np.pi))
        x += rng.uniform(0.065, 0.155) * np.sin(2 * np.pi * f0 * t + rng.uniform(0, 2 * np.pi)) * am
        x += rng.uniform(0.020, 0.050) * np.sin(2 * np.pi * 2 * f0 * t + rng.uniform(0, 2 * np.pi)) * am
        x += rng.uniform(0.006, 0.018) * bandpass_noise(
            rng, n, fs, max(30, f0 - 80), min(fs * 0.45, f0 + 80)
        )

    elif label == "scrape":
        n_bursts = int(rng.integers(5, 18))
        for _ in range(n_bursts):
            start = int(rng.uniform(0.0, duration - 0.08) * fs)
            length = int(rng.uniform(0.025, 0.11) * fs)
            end = min(n, start + length)
            if end <= start + 16:
                continue
            burst_n = end - start
            low = rng.uniform(900, 2500)
            high = rng.uniform(3500, fs * 0.47)
            burst = bandpass_noise(rng, burst_n, fs, low, high, amp=rng.uniform(0.06, 0.18))
            if rng.random() < 0.6:
                tb = np.arange(burst_n) / fs
                chirp = signal.chirp(tb, f0=low, f1=high, t1=tb[-1], method="linear")
                burst += rng.uniform(0.015, 0.045) * chirp
            burst *= signal.windows.hann(burst_n)
            x[start:end] += burst

    else:
        raise ValueError(f"未知已知类别: {label}")

    return x, {"rpm_hz": rpm_hz, "load": load}


def simulate_unknown_sound(
    subtype: str,
    rng: np.random.Generator,
    fs: int,
    duration: float,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """模拟不在库声音。训练时不使用这些类别，只用于开放集测试。"""
    n = int(round(fs * duration))
    t = np.arange(n) / fs
    x, rpm_hz, load = base_machine_signal(rng, fs, duration)

    if subtype == "chirp_sweep":
        for _ in range(int(rng.integers(2, 5))):
            start_s = rng.uniform(0.0, duration - 0.4)
            length_s = rng.uniform(0.22, 0.65)
            start = int(start_s * fs)
            length = min(int(length_s * fs), n - start)
            tb = np.arange(length) / fs
            f0 = rng.uniform(250, 900)
            f1 = rng.uniform(5200, fs * 0.46)
            method = str(rng.choice(["linear", "quadratic"]))
            sweep = signal.chirp(tb, f0=f0, f1=f1, t1=tb[-1], method=method)
            sweep *= signal.windows.tukey(length, alpha=0.5)
            x[start : start + length] += rng.uniform(0.07, 0.15) * sweep

    elif subtype == "electric_buzz":
        base = float(rng.choice([50.0, 60.0, 100.0, 120.0]))
        for h in range(1, int(min(36, fs / (2 * base)))):
            amp = 0.11 / (h ** 0.75) * rng.uniform(0.7, 1.3)
            x += amp * np.sin(2 * np.pi * base * h * t + rng.uniform(0, 2 * np.pi))
        x += 0.015 * bandpass_noise(rng, n, fs, 800, 4500)

    elif subtype == "cavitation_like":
        n_events = int(rng.integers(35, 90))
        times = np.sort(rng.uniform(0.02, duration - 0.02, size=n_events))
        for ti in times:
            add_damped_sine_pulse(
                x, fs, ti,
                freq_hz=rng.uniform(1500, 6500),
                amp=rng.uniform(0.025, 0.09),
                decay=rng.uniform(700, 1600),
                length_s=rng.uniform(0.003, 0.012),
            )
        x += 0.04 * bandpass_noise(rng, n, fs, 1000, fs * 0.48)

    else:
        raise ValueError(f"未知子类不存在: {subtype}")

    return x, {"rpm_hz": rpm_hz, "load": load}


def simulate_dataset(
    fs: int,
    duration: float,
    n_per_class: int,
    n_unknown: int,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, List[AudioMeta], np.ndarray, np.ndarray, List[AudioMeta]]:
    """生成已知类数据和未知类开放集测试数据。"""
    rng = set_random_seed(seed)
    envs = ["quiet_lab", "factory", "reverberant_room"]
    sensors = ["near_left", "near_right", "far_front", "far_side"]

    known_signals: List[np.ndarray] = []
    known_labels: List[str] = []
    known_meta: List[AudioMeta] = []

    for label in KNOWN_CLASSES:
        for i in range(n_per_class):
            x, vals = simulate_known_sound(label, rng, fs, duration)
            environment = str(rng.choice(envs, p=[0.30, 0.45, 0.25]))
            if environment == "quiet_lab":
                snr_db = float(rng.uniform(30, 44))
            elif environment == "factory":
                snr_db = float(rng.uniform(14, 28))
            else:
                snr_db = float(rng.uniform(22, 36))
            x = finalize_signal(rng, x, fs, snr_db, environment)

            sample_id = f"K_{label}_{i:05d}"
            known_signals.append(x)
            known_labels.append(label)
            known_meta.append(
                AudioMeta(
                    sample_id=sample_id,
                    label=label,
                    subtype=label,
                    device_id=f"D{int(rng.integers(1, 11)):02d}",
                    rpm_hz=float(vals["rpm_hz"]),
                    load=float(vals["load"]),
                    sensor_position=str(rng.choice(sensors)),
                    environment=environment,
                    snr_db=snr_db,
                    duration=duration,
                    fs=fs,
                )
            )

    unknown_signals: List[np.ndarray] = []
    unknown_labels: List[str] = []
    unknown_meta: List[AudioMeta] = []
    for i in range(n_unknown):
        subtype = UNKNOWN_SUBTYPES[i % len(UNKNOWN_SUBTYPES)]
        x, vals = simulate_unknown_sound(subtype, rng, fs, duration)
        environment = str(rng.choice(envs, p=[0.25, 0.50, 0.25]))
        if environment == "quiet_lab":
            snr_db = float(rng.uniform(30, 44))
        elif environment == "factory":
            snr_db = float(rng.uniform(14, 28))
        else:
            snr_db = float(rng.uniform(22, 36))
        x = finalize_signal(rng, x, fs, snr_db, environment)
        sample_id = f"U_{subtype}_{i:05d}"
        unknown_signals.append(x)
        unknown_labels.append("unknown")
        unknown_meta.append(
            AudioMeta(
                sample_id=sample_id,
                label="unknown",
                subtype=subtype,
                device_id=f"D{int(rng.integers(1, 11)):02d}",
                rpm_hz=float(vals["rpm_hz"]),
                load=float(vals["load"]),
                sensor_position=str(rng.choice(sensors)),
                environment=environment,
                snr_db=snr_db,
                duration=duration,
                fs=fs,
            )
        )

    return (
        np.stack(known_signals, axis=0),
        np.array(known_labels, dtype=object),
        known_meta,
        np.stack(unknown_signals, axis=0),
        np.array(unknown_labels, dtype=object),
        unknown_meta,
    )


# =============================================================================
# 4. 特征提取：Level 1 直接测量特征 + Level 2 时频变换特征
# =============================================================================

DIRECT_FEATURE_NAMES = [
    "rms",
    "std",
    "mean_abs",
    "peak_abs",
    "peak_to_peak",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "skewness",
    "kurtosis",
    "zcr",
    "env_mean",
    "env_std",
    "env_skewness",
    "env_kurtosis",
    "env_crest",
    "spec_centroid",
    "spec_bandwidth",
    "spec_flatness",
    "spec_entropy",
    "spec_rolloff85",
    "spec_rolloff95",
    "dom_freq",
    "dom_ratio",
    "spec_flux",
    "auto_peak_freq",
    "auto_peak_value",
    "env_peak_freq",
    "env_peak_ratio",
    "band_0_250",
    "band_250_500",
    "band_500_1000",
    "band_1000_2000",
    "band_2000_4000",
    "band_4000_8000",
    "band_ratio_low",
    "band_ratio_mid",
    "band_ratio_high",
]


def _safe_stat(v: float) -> float:
    if not np.isfinite(v):
        return 0.0
    return float(v)


def _band_power_ratio(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> float:
    mask = (freqs >= low) & (freqs < high)
    total = np.trapezoid(psd, freqs) + 1e-18
    if not np.any(mask):
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]) / total)


def _spectral_rolloff(freqs: np.ndarray, psd: np.ndarray, pct: float) -> float:
    c = np.cumsum(psd)
    if c[-1] <= 0:
        return 0.0
    idx = int(np.searchsorted(c, pct * c[-1]))
    idx = min(max(idx, 0), len(freqs) - 1)
    return float(freqs[idx])


def _autocorr_peak(x: np.ndarray, fs: int, min_hz: float = 5.0, max_hz: float = 400.0) -> Tuple[float, float]:
    """返回自相关指定频率范围内的最大峰值对应频率和峰值。"""
    y = x - np.mean(x)
    n = len(y)
    if np.std(y) < 1e-12:
        return 0.0, 0.0
    corr = signal.correlate(y, y, mode="full", method="fft")[n - 1 :]
    corr = corr / (corr[0] + 1e-18)
    lag_min = max(1, int(fs / max_hz))
    lag_max = min(len(corr) - 1, int(fs / min_hz))
    if lag_max <= lag_min:
        return 0.0, 0.0
    seg = corr[lag_min:lag_max]
    idx = int(np.argmax(seg)) + lag_min
    return float(fs / idx), float(corr[idx])


def _envelope_spectrum_peak(x: np.ndarray, fs: int) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """包络谱峰值，用于周期冲击、轴承等。"""
    env = np.abs(signal.hilbert(x))
    env = env - np.mean(env)
    nperseg = min(4096, len(env))
    freqs, psd = signal.welch(env, fs=fs, nperseg=nperseg)
    mask = (freqs >= 5.0) & (freqs <= min(800.0, fs / 2.0))
    if not np.any(mask):
        return 0.0, 0.0, freqs, psd
    f2 = freqs[mask]
    p2 = psd[mask]
    idx = int(np.argmax(p2))
    ratio = float(p2[idx] / (np.sum(p2) + 1e-18))
    return float(f2[idx]), ratio, freqs, psd


def extract_direct_features(x: np.ndarray, fs: int) -> np.ndarray:
    """Level 1：直接测量特征。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)
    n = len(x)

    rms = np.sqrt(np.mean(x**2) + 1e-18)
    std = np.std(x)
    mean_abs = np.mean(np.abs(x)) + 1e-18
    peak_abs = np.max(np.abs(x))
    peak_to_peak = np.ptp(x)
    crest = peak_abs / (rms + 1e-18)
    shape_factor = rms / mean_abs
    impulse_factor = peak_abs / mean_abs
    skewness = stats.skew(x, bias=False)
    kurt = stats.kurtosis(x, fisher=False, bias=False)  # 正态分布约为3
    zcr = np.mean(np.abs(np.diff(np.signbit(x))).astype(float))

    env = np.abs(signal.hilbert(x))
    env_mean = np.mean(env)
    env_std = np.std(env)
    env_skewness = stats.skew(env, bias=False)
    env_kurt = stats.kurtosis(env, fisher=False, bias=False)
    env_crest = np.max(env) / (np.sqrt(np.mean(env**2) + 1e-18))

    nperseg = min(2048, n)
    freqs, psd = signal.welch(x, fs=fs, window="hann", nperseg=nperseg)
    psd = np.maximum(psd, 1e-18)
    total_power = np.sum(psd) + 1e-18
    pnorm = psd / total_power
    spec_centroid = np.sum(freqs * pnorm)
    spec_bandwidth = np.sqrt(np.sum(((freqs - spec_centroid) ** 2) * pnorm))
    spec_flatness = np.exp(np.mean(np.log(psd))) / (np.mean(psd) + 1e-18)
    spec_entropy = -np.sum(pnorm * np.log2(pnorm + 1e-18)) / np.log2(len(pnorm) + 1e-18)
    rolloff85 = _spectral_rolloff(freqs, psd, 0.85)
    rolloff95 = _spectral_rolloff(freqs, psd, 0.95)
    dom_idx = int(np.argmax(psd[1:]) + 1) if len(psd) > 1 else 0
    dom_freq = freqs[dom_idx]
    dom_ratio = psd[dom_idx] / total_power

    # 频谱通量：相邻帧谱变化
    f_stft, t_stft, Z = signal.stft(x, fs=fs, window="hann", nperseg=512, noverlap=384)
    mag = np.abs(Z)
    mag = mag / (np.sum(mag, axis=0, keepdims=True) + 1e-18)
    if mag.shape[1] > 1:
        spec_flux = np.mean(np.sqrt(np.sum(np.diff(mag, axis=1) ** 2, axis=0)))
    else:
        spec_flux = 0.0

    auto_peak_freq, auto_peak_value = _autocorr_peak(x, fs)
    env_peak_freq, env_peak_ratio, _, _ = _envelope_spectrum_peak(x, fs)

    b0 = _band_power_ratio(freqs, psd, 0, 250)
    b1 = _band_power_ratio(freqs, psd, 250, 500)
    b2 = _band_power_ratio(freqs, psd, 500, 1000)
    b3 = _band_power_ratio(freqs, psd, 1000, 2000)
    b4 = _band_power_ratio(freqs, psd, 2000, 4000)
    b5 = _band_power_ratio(freqs, psd, 4000, min(8000, fs / 2.0))
    low = b0 + b1
    mid = b2 + b3
    high = b4 + b5

    values = [
        rms, std, mean_abs, peak_abs, peak_to_peak, crest, shape_factor, impulse_factor,
        skewness, kurt, zcr,
        env_mean, env_std, env_skewness, env_kurt, env_crest,
        spec_centroid, spec_bandwidth, spec_flatness, spec_entropy, rolloff85, rolloff95,
        dom_freq, dom_ratio, spec_flux,
        auto_peak_freq, auto_peak_value, env_peak_freq, env_peak_ratio,
        b0, b1, b2, b3, b4, b5,
        low, mid, high,
    ]
    return np.array([_safe_stat(v) for v in values], dtype=np.float64)


def resample_matrix(mat: np.ndarray, out_rows: int, out_cols: int) -> np.ndarray:
    """把变长谱图重采样成固定尺寸，便于进入 PCA/分类器。"""
    a = signal.resample(mat, out_rows, axis=0)
    a = signal.resample(a, out_cols, axis=1)
    return np.asarray(a, dtype=np.float64)


def extract_transform_features(
    x: np.ndarray,
    fs: int,
    spec_shape: Tuple[int, int] = (32, 32),
    env_bins: int = 96,
) -> np.ndarray:
    """Level 2 的原始高维变换特征：log-STFT 谱图 + 包络谱向量。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - np.mean(x)

    nperseg = min(512, len(x))
    noverlap = int(nperseg * 0.75)
    freqs, times, S = signal.spectrogram(
        x,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        scaling="spectrum",
        mode="magnitude",
    )
    S = np.log1p(S)
    # 限制到可听/可解释频段，不超过 8 kHz
    mask = freqs <= min(8000, fs / 2.0)
    S = S[mask]
    S_small = resample_matrix(S, spec_shape[0], spec_shape[1])

    # 包络谱向量：周期冲击/调制特征
    _, _, f_env, p_env = _envelope_spectrum_peak(x, fs)
    max_env_hz = min(800.0, fs / 2.0)
    grid = np.linspace(0, max_env_hz, env_bins)
    p_interp = np.interp(grid, f_env, p_env, left=0.0, right=0.0)
    p_interp = np.log1p(p_interp)

    return np.concatenate([S_small.ravel(), p_interp], axis=0)


def extract_feature_batch(
    signals: np.ndarray,
    fs: int,
    verbose: bool = True,
    prefix: str = "features",
) -> Tuple[np.ndarray, np.ndarray]:
    """批量提取 Level 1 和 Level 2 原始特征。"""
    direct_list = []
    transform_list = []
    for i, x in enumerate(signals):
        direct_list.append(extract_direct_features(x, fs))
        transform_list.append(extract_transform_features(x, fs))
        if verbose and ((i + 1) % 200 == 0 or i + 1 == len(signals)):
            print(f"[{prefix}] extracted {i + 1}/{len(signals)}")
    return np.vstack(direct_list), np.vstack(transform_list)


# =============================================================================
# 5. 三层级融合特征管线
# =============================================================================

class ThreeLevelFeaturePipeline:
    """
    Level 1: direct_scaled
    Level 2: direct_pca + transform_pca
    Level 3: fusion = scaler([direct_scaled, direct_pca, transform_pca])
    """

    def __init__(self, direct_pca_dim: int = 10, transform_pca_dim: int = 48, seed: int = 42):
        self.direct_pca_dim = direct_pca_dim
        self.transform_pca_dim = transform_pca_dim
        self.seed = seed
        self.direct_scaler = StandardScaler()
        self.transform_scaler = StandardScaler()
        self.direct_pca: Optional[PCA] = None
        self.transform_pca: Optional[PCA] = None
        self.fusion_scaler = StandardScaler()
        self.fusion_feature_names: List[str] = []

    @staticmethod
    def _safe_n_components(requested: int, n_samples: int, n_features: int) -> int:
        if requested <= 0:
            return min(n_samples - 1, n_features)
        return max(1, min(int(requested), n_samples - 1, n_features))

    def fit(self, X_direct: np.ndarray, X_transform: np.ndarray) -> "ThreeLevelFeaturePipeline":
        D = self.direct_scaler.fit_transform(X_direct)
        T = self.transform_scaler.fit_transform(X_transform)

        d_comp = self._safe_n_components(self.direct_pca_dim, D.shape[0], D.shape[1])
        t_comp = self._safe_n_components(self.transform_pca_dim, T.shape[0], T.shape[1])
        self.direct_pca = PCA(n_components=d_comp, random_state=self.seed)
        self.transform_pca = PCA(n_components=t_comp, random_state=self.seed)
        Dp = self.direct_pca.fit_transform(D)
        Tp = self.transform_pca.fit_transform(T)
        fusion_raw = np.hstack([D, Dp, Tp])
        self.fusion_scaler.fit(fusion_raw)

        self.fusion_feature_names = (
            [f"L1_direct::{n}" for n in DIRECT_FEATURE_NAMES]
            + [f"L2_direct_pca::{i:02d}" for i in range(Dp.shape[1])]
            + [f"L2_transform_pca::{i:02d}" for i in range(Tp.shape[1])]
        )
        return self

    def transform(self, X_direct: np.ndarray, X_transform: np.ndarray) -> Dict[str, np.ndarray]:
        if self.direct_pca is None or self.transform_pca is None:
            raise RuntimeError("FeaturePipeline 尚未 fit。")
        D = self.direct_scaler.transform(X_direct)
        T = self.transform_scaler.transform(X_transform)
        Dp = self.direct_pca.transform(D)
        Tp = self.transform_pca.transform(T)
        fusion_raw = np.hstack([D, Dp, Tp])
        Z = self.fusion_scaler.transform(fusion_raw)
        return {
            "level1_direct_scaled": D,
            "level2_direct_pca": Dp,
            "level2_transform_pca": Tp,
            "level3_fusion": Z,
        }

    def fit_transform(self, X_direct: np.ndarray, X_transform: np.ndarray) -> Dict[str, np.ndarray]:
        self.fit(X_direct, X_transform)
        return self.transform(X_direct, X_transform)


# =============================================================================
# 6. 异响指纹库：原型、距离、阈值、开放集拒识
# =============================================================================

class FingerprintLibrary:
    def __init__(self, alpha_prob: float = 0.65):
        self.alpha_prob = float(alpha_prob)
        self.fingerprints: Dict[str, Dict[str, object]] = {}
        self.prob_threshold: float = 0.30
        self.score_threshold: float = 0.30

    @staticmethod
    def _diag_mahalanobis(z: np.ndarray, centroid: np.ndarray, diag_var: np.ndarray) -> float:
        # 除以维度，得到更稳定的“均方马氏距离”。
        return float(np.sqrt(np.mean(((z - centroid) ** 2) / (diag_var + 1e-9))))

    def distance(self, label: str, z: np.ndarray) -> float:
        fp = self.fingerprints[str(label)]
        return self._diag_mahalanobis(z, np.asarray(fp["centroid"]), np.asarray(fp["diag_var"]))

    def build(
        self,
        Z_train: np.ndarray,
        y_train: Sequence[str],
        X_direct_train: np.ndarray,
        sample_ids: Sequence[str],
        distance_quantile: float = 0.98,
        threshold_scale: float = 1.10,
    ) -> "FingerprintLibrary":
        y_arr = np.asarray(y_train, dtype=object)
        for label in sorted(np.unique(y_arr)):
            idx = np.where(y_arr == label)[0]
            Zc = Z_train[idx]
            Dc = X_direct_train[idx]
            centroid = np.mean(Zc, axis=0)
            diag_var = np.var(Zc, axis=0) + 1e-6
            dists = np.array([self._diag_mahalanobis(z, centroid, diag_var) for z in Zc])
            threshold = float(np.quantile(dists, distance_quantile) * threshold_scale)
            threshold = max(threshold, float(np.median(dists) + 3.0 * np.median(np.abs(dists - np.median(dists)))))
            sort_idx = np.argsort(dists)
            prototype_samples = [str(sample_ids[idx[j]]) for j in sort_idx[: min(3, len(sort_idx))]]
            boundary_samples = [str(sample_ids[idx[j]]) for j in sort_idx[-min(3, len(sort_idx)) :]]
            self.fingerprints[str(label)] = {
                "label": str(label),
                "description": CLASS_DESCRIPTIONS.get(str(label), ""),
                "n_train": int(len(idx)),
                "centroid": centroid.astype(float),
                "diag_var": diag_var.astype(float),
                "distance_threshold": float(threshold),
                "distance_quantiles": {
                    "q50": float(np.quantile(dists, 0.50)),
                    "q90": float(np.quantile(dists, 0.90)),
                    "q95": float(np.quantile(dists, 0.95)),
                    "q98": float(np.quantile(dists, 0.98)),
                    "max": float(np.max(dists)),
                },
                "prototype_samples": prototype_samples,
                "boundary_samples": boundary_samples,
                "direct_feature_mean": np.mean(Dc, axis=0).astype(float),
                "direct_feature_std": np.std(Dc, axis=0).astype(float),
            }
        return self

    def calibrate_thresholds_on_validation(
        self,
        Z_val: np.ndarray,
        y_val: Sequence[str],
        clf: RandomForestClassifier,
        q: float = 0.02,
        class_distance_q: float = 0.98,
        known_prob_floor: float = 0.15,
        known_score_floor: float = 0.15,
    ) -> None:
        """用已知类验证集校准拒识阈值，目标是减少已知类误拒。"""
        y_arr = np.asarray(y_val, dtype=object)

        # 更新每个类别的距离阈值。
        for label in sorted(np.unique(y_arr)):
            idx = np.where(y_arr == label)[0]
            if len(idx) == 0 or str(label) not in self.fingerprints:
                continue
            dists = np.array([self.distance(str(label), z) for z in Z_val[idx]])
            new_th = float(np.quantile(dists, class_distance_q) * 1.08)
            old_th = float(self.fingerprints[str(label)]["distance_threshold"])
            self.fingerprints[str(label)]["distance_threshold"] = max(old_th, new_th)
            self.fingerprints[str(label)]["validation_distance_q98"] = float(np.quantile(dists, 0.98))

        # 全局概率阈值和融合分数阈值。
        proba = clf.predict_proba(Z_val)
        class_to_idx = {str(c): i for i, c in enumerate(clf.classes_)}
        true_probs = []
        true_scores = []
        for i, label in enumerate(y_arr):
            label = str(label)
            if label not in class_to_idx or label not in self.fingerprints:
                continue
            p = float(proba[i, class_to_idx[label]])
            d = self.distance(label, Z_val[i])
            th = float(self.fingerprints[label]["distance_threshold"])
            dscore = float(np.exp(-0.5 * (d / (th + 1e-12)) ** 2))
            score = self.alpha_prob * p + (1.0 - self.alpha_prob) * dscore
            true_probs.append(p)
            true_scores.append(score)
        if true_probs:
            self.prob_threshold = float(np.clip(np.quantile(true_probs, q) * 0.80, known_prob_floor, 0.75))
        if true_scores:
            self.score_threshold = float(np.clip(np.quantile(true_scores, q) * 0.80, known_score_floor, 0.75))

    def recognize_one(
        self,
        z: np.ndarray,
        proba: np.ndarray,
        clf_classes: Sequence[str],
    ) -> RecognitionResult:
        clf_classes = [str(c) for c in clf_classes]
        scores = []
        distances = []
        thresholds = []
        for cls, p in zip(clf_classes, proba):
            d = self.distance(cls, z)
            th = float(self.fingerprints[cls]["distance_threshold"])
            dscore = float(np.exp(-0.5 * (d / (th + 1e-12)) ** 2))
            score = self.alpha_prob * float(p) + (1.0 - self.alpha_prob) * dscore
            scores.append(score)
            distances.append(d)
            thresholds.append(th)

        best_idx = int(np.argmax(scores))
        best_cls = clf_classes[best_idx]
        best_prob = float(proba[best_idx])
        best_score = float(scores[best_idx])
        best_distance = float(distances[best_idx])
        best_threshold = float(thresholds[best_idx])

        reasons = []
        if best_prob < self.prob_threshold:
            reasons.append(f"probability<{self.prob_threshold:.3f}")
        if best_score < self.score_threshold:
            reasons.append(f"fusion_score<{self.score_threshold:.3f}")
        if best_distance > best_threshold:
            reasons.append(f"distance>{best_threshold:.3f}")
        rejected = len(reasons) > 0
        predicted = "unknown" if rejected else best_cls

        top3_idx = np.argsort(scores)[::-1][:3]
        top3 = [(clf_classes[j], float(scores[j])) for j in top3_idx]
        return RecognitionResult(
            predicted_label=predicted,
            best_known_label=best_cls,
            confidence=best_score,
            best_probability=best_prob,
            best_distance=best_distance,
            distance_threshold=best_threshold,
            rejected=rejected,
            reject_reasons=reasons,
            top3=top3,
        )

    def recognize_many(self, Z: np.ndarray, clf: RandomForestClassifier) -> List[RecognitionResult]:
        proba = clf.predict_proba(Z)
        return [self.recognize_one(z, p, clf.classes_) for z, p in zip(Z, proba)]

    def to_jsonable(self) -> Dict[str, object]:
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, dict):
                return {str(k): convert(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert(v) for v in obj]
            return obj

        return convert(
            {
                "alpha_prob": self.alpha_prob,
                "prob_threshold": self.prob_threshold,
                "score_threshold": self.score_threshold,
                "direct_feature_names": DIRECT_FEATURE_NAMES,
                "class_descriptions": CLASS_DESCRIPTIONS,
                "fingerprints": self.fingerprints,
            }
        )


# =============================================================================
# 7. 训练、评估、绘图与导出
# =============================================================================

def split_known_indices(
    y: np.ndarray,
    test_size: float,
    val_size: float,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """分层切分为 train/val/test。"""
    if test_size + val_size >= 0.8:
        raise ValueError("test_size + val_size 过大，请保持充足训练样本。")
    idx = np.arange(len(y))
    train_val_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=y
    )
    relative_val = val_size / (1.0 - test_size)
    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=relative_val,
        random_state=seed + 1,
        stratify=y[train_val_idx],
    )
    return train_idx, val_idx, test_idx


def save_metadata(meta: List[AudioMeta], path: Path) -> None:
    df = pd.DataFrame([asdict(m) for m in meta])
    df.to_csv(path, index=False, encoding="utf-8-sig")


def save_example_wavs(
    signals: np.ndarray,
    labels: np.ndarray,
    meta: List[AudioMeta],
    out_dir: Path,
    fs: int,
    max_per_label: int = 1,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    counts: Dict[str, int] = {}
    for x, y, m in zip(signals, labels, meta):
        label = str(y)
        if counts.get(label, 0) >= max_per_label:
            continue
        counts[label] = counts.get(label, 0) + 1
        y16 = x / (np.max(np.abs(x)) + 1e-12)
        y16 = np.asarray(np.clip(y16, -1, 1) * 32767, dtype=np.int16)
        wavfile.write(str(out_dir / f"{m.sample_id}_{m.subtype}.wav"), fs, y16)


def plot_confusion(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str], path: Path, title: str) -> None:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.75), max(6, len(labels) * 0.65)))
    im = ax.imshow(cm)
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_pca_embedding(Z: np.ndarray, labels: Sequence[str], path: Path, title: str, seed: int) -> None:
    pca = PCA(n_components=2, random_state=seed)
    Z2 = pca.fit_transform(Z)
    labels = np.asarray(labels, dtype=object)
    fig, ax = plt.subplots(figsize=(9, 7))
    for lab in sorted(np.unique(labels)):
        idx = labels == lab
        ax.scatter(Z2[idx, 0], Z2[idx, 1], s=16, alpha=0.75, label=str(lab))
    ax.set_title(title)
    ax.set_xlabel("PCA-1")
    ax.set_ylabel("PCA-2")
    ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_feature_importance(df_imp: pd.DataFrame, path: Path, topn: int = 20) -> None:
    df = df_imp.head(topn).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(5, topn * 0.28)))
    ax.barh(df["feature"], df["mutual_info"])
    ax.set_title(f"Top-{topn} direct measured features by mutual information")
    ax.set_xlabel("Mutual information")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def result_rows(results: List[RecognitionResult], y_true: Sequence[str], meta: List[AudioMeta]) -> List[Dict[str, object]]:
    rows = []
    for r, yt, m in zip(results, y_true, meta):
        rows.append(
            {
                "sample_id": m.sample_id,
                "true_label": str(yt),
                "true_subtype": m.subtype,
                "predicted_label": r.predicted_label,
                "best_known_label": r.best_known_label,
                "confidence": r.confidence,
                "best_probability": r.best_probability,
                "best_distance": r.best_distance,
                "distance_threshold": r.distance_threshold,
                "rejected": r.rejected,
                "reject_reasons": ";".join(r.reject_reasons),
                "top3": json.dumps(r.top3, ensure_ascii=False),
                "rpm_hz": m.rpm_hz,
                "load": m.load,
                "environment": m.environment,
                "snr_db": m.snr_db,
            }
        )
    return rows


def train_and_evaluate(args: argparse.Namespace) -> Dict[str, object]:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "plots").mkdir(exist_ok=True)
    (out_dir / "examples_wav").mkdir(exist_ok=True)

    print("\n[1/8] 模拟已知类与未知类音频数据 ...")
    X_known, y_known, meta_known, X_unknown, y_unknown, meta_unknown = simulate_dataset(
        fs=args.fs,
        duration=args.duration,
        n_per_class=args.n_per_class,
        n_unknown=args.n_unknown,
        seed=args.seed,
    )
    print(f"known: {X_known.shape}, unknown: {X_unknown.shape}")

    save_metadata(meta_known, out_dir / "known_metadata.csv")
    save_metadata(meta_unknown, out_dir / "unknown_metadata.csv")
    if args.save_wav:
        save_example_wavs(X_known, y_known, meta_known, out_dir / "examples_wav", args.fs, max_per_label=1)
        save_example_wavs(X_unknown, y_unknown, meta_unknown, out_dir / "examples_wav", args.fs, max_per_label=3)

    print("\n[2/8] 提取三层级中的 Level 1/Level 2 原始特征 ...")
    Xd_known, Xt_known = extract_feature_batch(X_known, args.fs, verbose=True, prefix="known")
    Xd_unknown, Xt_unknown = extract_feature_batch(X_unknown, args.fs, verbose=True, prefix="unknown")

    pd.DataFrame(Xd_known, columns=DIRECT_FEATURE_NAMES).to_csv(
        out_dir / "direct_features_known.csv", index=False, encoding="utf-8-sig"
    )

    print("\n[3/8] 划分 train/val/test ...")
    train_idx, val_idx, test_idx = split_known_indices(
        y_known, test_size=args.test_size, val_size=args.val_size, seed=args.seed
    )
    y_train, y_val, y_test = y_known[train_idx], y_known[val_idx], y_known[test_idx]
    ids_train = [meta_known[i].sample_id for i in train_idx]
    print(f"train={len(train_idx)}, val={len(val_idx)}, test_known={len(test_idx)}, test_unknown={len(y_unknown)}")

    print("\n[4/8] 拟合 Level 2 PCA 与 Level 3 融合特征 ...")
    pipe = ThreeLevelFeaturePipeline(
        direct_pca_dim=args.direct_pca_dim,
        transform_pca_dim=args.transform_pca_dim,
        seed=args.seed,
    )
    feats_train = pipe.fit_transform(Xd_known[train_idx], Xt_known[train_idx])
    feats_val = pipe.transform(Xd_known[val_idx], Xt_known[val_idx])
    feats_test = pipe.transform(Xd_known[test_idx], Xt_known[test_idx])
    feats_unknown = pipe.transform(Xd_unknown, Xt_unknown)
    Z_train = feats_train["level3_fusion"]
    Z_val = feats_val["level3_fusion"]
    Z_test = feats_test["level3_fusion"]
    Z_unknown = feats_unknown["level3_fusion"]

    print(f"fusion dim = {Z_train.shape[1]}")

    print("\n[5/8] 训练已知类分类器 ...")
    clf = RandomForestClassifier(
        n_estimators=args.trees,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=args.seed,
        n_jobs=-1,
    )
    clf.fit(Z_train, y_train)

    y_pred_closed = clf.predict(Z_test)
    closed_acc = accuracy_score(y_test, y_pred_closed)
    closed_macro_f1 = f1_score(y_test, y_pred_closed, average="macro")
    print(f"closed-set known accuracy = {closed_acc:.4f}, macro-F1 = {closed_macro_f1:.4f}")

    print("\n[6/8] 建立异响指纹库并校准开放集拒识阈值 ...")
    fp_lib = FingerprintLibrary(alpha_prob=args.alpha_prob)
    fp_lib.build(
        Z_train,
        y_train,
        Xd_known[train_idx],
        sample_ids=ids_train,
        distance_quantile=args.distance_quantile,
        threshold_scale=args.distance_threshold_scale,
    )
    fp_lib.calibrate_thresholds_on_validation(
        Z_val,
        y_val,
        clf,
        q=args.known_reject_quantile,
        class_distance_q=args.val_distance_quantile,
    )
    print(f"global prob_threshold={fp_lib.prob_threshold:.3f}, score_threshold={fp_lib.score_threshold:.3f}")

    print("\n[7/8] 开放集识别：已知类 + 未知类拒识 ...")
    Z_open = np.vstack([Z_test, Z_unknown])
    y_open = np.concatenate([y_test, y_unknown])
    meta_open = [meta_known[i] for i in test_idx] + meta_unknown
    results_open = fp_lib.recognize_many(Z_open, clf)
    y_pred_open = np.array([r.predicted_label for r in results_open], dtype=object)
    open_labels = sorted(list(np.unique(np.concatenate([y_open, y_pred_open]))))

    open_macro_f1 = f1_score(y_open, y_pred_open, labels=open_labels, average="macro", zero_division=0)
    unknown_mask = y_open == "unknown"
    known_mask = y_open != "unknown"
    unknown_recall = float(np.mean(y_pred_open[unknown_mask] == "unknown")) if np.any(unknown_mask) else 0.0
    known_false_reject = float(np.mean(y_pred_open[known_mask] == "unknown")) if np.any(known_mask) else 0.0
    print(f"open-set macro-F1 = {open_macro_f1:.4f}")
    print(f"unknown recall = {unknown_recall:.4f}, known false reject rate = {known_false_reject:.4f}")

    # 结果文件
    report_closed = classification_report(y_test, y_pred_closed, zero_division=0)
    report_open = classification_report(y_open, y_pred_open, labels=open_labels, zero_division=0)
    (out_dir / "classification_report_closed_set.txt").write_text(report_closed, encoding="utf-8")
    (out_dir / "classification_report_open_set.txt").write_text(report_open, encoding="utf-8")
    pd.DataFrame(result_rows(results_open, y_open, meta_open)).to_csv(
        out_dir / "open_set_predictions.csv", index=False, encoding="utf-8-sig"
    )

    print("\n[8/8] 导出指纹库、模型、特征重要性与图表 ...")
    # 直接测量特征重要性
    mi = mutual_info_classif(Xd_known[train_idx], y_train, random_state=args.seed)
    df_mi = pd.DataFrame({"feature": DIRECT_FEATURE_NAMES, "mutual_info": mi})
    df_mi = df_mi.sort_values("mutual_info", ascending=False)
    df_mi.to_csv(out_dir / "direct_feature_importance.csv", index=False, encoding="utf-8-sig")

    # 融合特征重要性：随机森林内部重要性
    df_fusion_imp = pd.DataFrame(
        {
            "fusion_feature": pipe.fusion_feature_names,
            "importance": clf.feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    df_fusion_imp.to_csv(out_dir / "fusion_feature_importance.csv", index=False, encoding="utf-8-sig")

    fingerprint_json = fp_lib.to_jsonable()
    fingerprint_json["simulation_config"] = vars(args)
    fingerprint_json["feature_pipeline"] = {
        "direct_pca_dim_actual": int(pipe.direct_pca.n_components_) if pipe.direct_pca else None,
        "transform_pca_dim_actual": int(pipe.transform_pca.n_components_) if pipe.transform_pca else None,
        "fusion_dim": int(Z_train.shape[1]),
        "transform_feature_raw_dim": int(Xt_known.shape[1]),
    }
    with open(out_dir / "fingerprint_library.json", "w", encoding="utf-8") as f:
        json.dump(fingerprint_json, f, ensure_ascii=False, indent=2)

    model_bundle = {
        "feature_pipeline": pipe,
        "classifier": clf,
        "fingerprint_library": fp_lib,
        "direct_feature_names": DIRECT_FEATURE_NAMES,
        "known_classes": KNOWN_CLASSES,
        "class_descriptions": CLASS_DESCRIPTIONS,
    }
    joblib.dump(model_bundle, out_dir / "asf_model_bundle.joblib")

    metrics = {
        "closed_set_known_accuracy": float(closed_acc),
        "closed_set_known_macro_f1": float(closed_macro_f1),
        "open_set_macro_f1": float(open_macro_f1),
        "unknown_recall": float(unknown_recall),
        "known_false_reject_rate": float(known_false_reject),
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_known_size": int(len(test_idx)),
        "test_unknown_size": int(len(y_unknown)),
        "fusion_dim": int(Z_train.shape[1]),
        "prob_threshold": float(fp_lib.prob_threshold),
        "score_threshold": float(fp_lib.score_threshold),
    }
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    if not args.no_plots:
        plot_confusion(
            y_test,
            y_pred_closed,
            labels=sorted(np.unique(y_known)),
            path=out_dir / "plots" / "confusion_closed_set.png",
            title="Closed-set known-class classification",
        )
        plot_confusion(
            y_open,
            y_pred_open,
            labels=open_labels,
            path=out_dir / "plots" / "confusion_open_set.png",
            title="Open-set recognition with unknown rejection",
        )
        plot_pca_embedding(
            np.vstack([Z_test, Z_unknown]),
            y_open,
            path=out_dir / "plots" / "fusion_embedding_pca2.png",
            title="Level-3 fusion fingerprint embedding PCA visualization",
            seed=args.seed,
        )
        plot_feature_importance(df_mi, out_dir / "plots" / "direct_feature_importance_top20.png", topn=20)

    print(f"\n完成。输出目录：{out_dir.resolve()}")
    print("主要文件：")
    print("  - fingerprint_library.json：异响指纹库")
    print("  - asf_model_bundle.joblib：特征管线 + 分类器 + 指纹库")
    print("  - open_set_predictions.csv：开放集识别明细")
    print("  - metrics.json：评估指标")
    print("  - plots/*.png：混淆矩阵、融合指纹空间、特征重要性")
    print("  - examples_wav/*.wav：每类仿真音频样例")

    return metrics


# =============================================================================
# 8. 单条新音频识别示例函数
# =============================================================================

def recognize_new_waveform(
    x: np.ndarray,
    fs: int,
    model_bundle_path: str,
) -> RecognitionResult:
    """
    给定一条新波形，载入模型包并输出识别结果。
    实际工程中可把此函数接到在线采集/事件切分模块之后。
    """
    bundle = joblib.load(model_bundle_path)
    pipe: ThreeLevelFeaturePipeline = bundle["feature_pipeline"]
    clf: RandomForestClassifier = bundle["classifier"]
    fp_lib: FingerprintLibrary = bundle["fingerprint_library"]

    Xd = extract_direct_features(x, fs)[None, :]
    Xt = extract_transform_features(x, fs)[None, :]
    Z = pipe.transform(Xd, Xt)["level3_fusion"]
    return fp_lib.recognize_many(Z, clf)[0]


# =============================================================================
# 9. 命令行入口
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="异响指纹库与开放集识别仿真")
    parser.add_argument("--out", type=str, default="./runs/asf_demo", help="输出目录")
    parser.add_argument("--fs", type=int, default=16000, help="采样率")
    parser.add_argument("--duration", type=float, default=2.0, help="每条音频时长，秒")
    parser.add_argument("--n-per-class", type=int, default=80, help="每个已知类别仿真样本数")
    parser.add_argument("--n-unknown", type=int, default=90, help="开放集未知类样本数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--test-size", type=float, default=0.18, help="已知类测试集比例")
    parser.add_argument("--val-size", type=float, default=0.15, help="已知类验证集比例")
    parser.add_argument("--direct-pca-dim", type=int, default=10, help="直接测量特征 PCA 维数")
    parser.add_argument("--transform-pca-dim", type=int, default=48, help="时频变换特征 PCA 维数")
    parser.add_argument("--trees", type=int, default=300, help="随机森林树数量")
    parser.add_argument("--alpha-prob", type=float, default=0.65, help="分类概率在融合判决中的权重")
    parser.add_argument("--distance-quantile", type=float, default=0.98, help="训练集类内距离阈值分位数")
    parser.add_argument("--distance-threshold-scale", type=float, default=1.10, help="距离阈值放大系数")
    parser.add_argument("--val-distance-quantile", type=float, default=0.98, help="验证集类内距离阈值分位数")
    parser.add_argument("--known-reject-quantile", type=float, default=0.02, help="用于全局概率/分数阈值的低分位数")
    parser.add_argument("--no-plots", action="store_true", help="不输出图片")
    parser.add_argument("--save-wav", action="store_true", default=True, help="保存样例 wav")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.n_per_class < 12:
        print("警告：n_per_class 很小，分层切分和 PCA/分类器结果会不稳定。建议 >= 50。")
    metrics = train_and_evaluate(args)
    print("\nMetrics:")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
