
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
from scipy import signal
from scipy.io import wavfile


EPS = 1e-12


@dataclass
class SourceMeta:
    name: str
    band: Tuple[float, float]   # Hz
    rank: int


def smooth_gate(t: np.ndarray, intervals: List[Tuple[float, float]], ramp: float = 0.05) -> np.ndarray:
    """Create a smoothed 0/1 activity envelope."""
    env = np.zeros_like(t, dtype=float)
    for start, end in intervals:
        env += ((t >= start) & (t <= end)).astype(float)
    env = np.clip(env, 0.0, 1.0)

    fs = 1.0 / (t[1] - t[0])
    win_len = max(5, int(ramp * fs))
    if win_len % 2 == 0:
        win_len += 1
    win = np.hanning(win_len)
    win /= win.sum() + EPS
    env = np.convolve(env, win, mode="same")
    return np.clip(env, 0.0, 1.0)


def lowpass_noise(rng: np.random.Generator, n: int, fs: int, cutoff: float, order: int = 4) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = signal.butter(order, cutoff, btype="lowpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, noise)
    return y / (np.std(y) + EPS)


def bandpass_noise(
    rng: np.random.Generator,
    n: int,
    fs: int,
    low: float,
    high: float,
    order: int = 4,
) -> np.ndarray:
    noise = rng.standard_normal(n)
    sos = signal.butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    y = signal.sosfiltfilt(sos, noise)
    return y / (np.std(y) + EPS)


def generate_car_mixture(
    M: int = 3,
    fs: int = 8000,
    duration: float = 8.0,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[SourceMeta]]:
    """
    Generate:
      x = background + M low-rank-ish anomaly sources

    Returns:
      t, mixture, [background, anomaly_1, ..., anomaly_M], metadata for anomalies.
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # Normal driving background: engine harmonics + road/tire colored noise.
    f0 = 70 + 6 * np.sin(2 * np.pi * 0.12 * t) + 2 * np.sin(2 * np.pi * 0.37 * t)
    phase = 2 * np.pi * np.cumsum(f0) / fs
    background = (
        0.12 * np.sin(phase)
        + 0.06 * np.sin(2 * phase + 0.2)
        + 0.035 * np.sin(3 * phase + 0.7)
    )
    background += 0.04 * lowpass_noise(rng, n, fs, cutoff=900)
    background += 0.025 * bandpass_noise(rng, n, fs, low=80, high=700)

    anomalies: List[np.ndarray] = []
    meta: List[SourceMeta] = []

    # 1) Narrowband squeal/whine: approximately rank-1 magnitude spectrogram.
    if M >= 1:
        env = smooth_gate(t, [(0.8, 2.8), (4.2, 6.6)], ramp=0.08)
        freq = 1550 + 40 * np.sin(2 * np.pi * 0.23 * t)
        ph = 2 * np.pi * np.cumsum(freq) / fs
        a = 0.11 * env * (1 + 0.2 * np.sin(2 * np.pi * 3 * t)) * np.sin(ph)
        anomalies.append(a)
        meta.append(SourceMeta(name="squeal", band=(1350, 1750), rank=1))

    # 2) Knock: repeated decaying resonance, close to fixed spectral template × sparse activation.
    if M >= 2:
        a = np.zeros(n)
        times = np.arange(2.2, 7.1, 0.37) + rng.normal(0.0, 0.015, size=len(np.arange(2.2, 7.1, 0.37)))
        decay_len = int(0.16 * fs)
        tau = np.arange(decay_len) / fs
        kernel = np.exp(-tau / 0.045) * np.sin(2 * np.pi * 380 * tau)
        for event_time in times:
            idx = int(event_time * fs)
            if 0 <= idx < n:
                length = min(decay_len, n - idx)
                a[idx:idx + length] += kernel[:length]
        a = 0.13 * a / (np.max(np.abs(a)) + EPS)
        anomalies.append(a)
        meta.append(SourceMeta(name="knock", band=(250, 650), rank=1))

    # 3) Rattle: stable spectral band with bursty activation.
    if M >= 3:
        env = smooth_gate(t, [(1.1, 3.4), (5.3, 7.6)], ramp=0.06)
        carrier = bandpass_noise(rng, n, fs, low=750, high=1150)
        mod = 0.65 + 0.35 * np.sin(2 * np.pi * 18 * t + 0.4)
        a = 0.08 * env * mod * carrier
        anomalies.append(a)
        meta.append(SourceMeta(name="rattle", band=(700, 1200), rank=2))

    # Extra anomalies, for M > 3: additional narrowband low-rank tones.
    base_centers = [2050, 2350, 2750, 3150]
    for q in range(3, M):
        center = base_centers[(q - 3) % len(base_centers)]
        start = min(duration - 1.0, 0.7 + 0.4 * q)
        end = min(duration - 0.4, 2.8 + 0.6 * q)
        env = smooth_gate(t, [(start, end)], ramp=0.07)
        freq = center + 30 * np.sin(2 * np.pi * (0.15 + 0.02 * q) * t)
        ph = 2 * np.pi * np.cumsum(freq) / fs
        a = 0.07 * env * np.sin(ph + rng.uniform(0, 2 * np.pi))
        anomalies.append(a)
        meta.append(SourceMeta(name=f"tone_{q + 1}", band=(center - 180, center + 180), rank=1))

    sources = [background] + anomalies
    mixture = np.sum(sources, axis=0)

    # Global scaling preserves the exact mixture relation and prevents clipping.
    scale = 0.95 / (np.max(np.abs(mixture)) + EPS)
    sources = [s * scale for s in sources]
    mixture = mixture * scale
    return t, mixture, sources, meta


def stft_signal(x: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256):
    freqs, times, X = signal.stft(
        x,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        boundary="zeros",
        padded=True,
    )
    return freqs, times, X


def istft_signal(X: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256, length: int | None = None) -> np.ndarray:
    _, x = signal.istft(
        X,
        fs=fs,
        window="hann",
        nperseg=n_fft,
        noverlap=n_fft - hop,
        input_onesided=True,
        boundary=True,
    )
    if length is not None:
        if len(x) < length:
            x = np.pad(x, (0, length - len(x)))
        x = x[:length]
    return x


def make_group_masks(freqs: np.ndarray, meta: List[SourceMeta], bg_max_freq: float = 1400.0) -> List[np.ndarray]:
    """
    Frequency support priors. These are what make single-channel separation identifiable.
    For real data, replace these with prior bands/templates estimated from labeled normal/anomaly samples.
    """
    masks: List[np.ndarray] = []

    bg = np.zeros_like(freqs, dtype=float)
    bg[freqs <= bg_max_freq] = 1.0
    masks.append(bg)

    for m in meta:
        lo, hi = m.band
        mask = np.zeros_like(freqs, dtype=float)
        mask[(freqs >= lo) & (freqs <= hi)] = 1.0
        masks.append(mask)

    return masks


def constrained_group_nmf(
    V: np.ndarray,
    group_ranks: List[int],
    group_masks: List[np.ndarray],
    n_iter: int = 500,
    lambda_anomaly: float = 0.004,
    seed: int = 0,
    verbose: bool = False,
):
    """
    Solve:
      min_{W,H >= 0} 0.5 ||V - sum_j W_j H_j||_F^2 + sum_{j>=1} lambda_j ||H_j||_1
      s.t. W_j = P_j ⊙ W_j, ||w_k||_1 = 1.

    Multiplicative updates are used, followed by mask projection and column normalization.
    """
    rng = np.random.default_rng(seed)
    F, T = V.shape
    J = len(group_ranks)
    if len(group_masks) != J:
        raise ValueError("len(group_masks) must equal len(group_ranks).")

    atom_to_group = []
    for j, rank in enumerate(group_ranks):
        atom_to_group.extend([j] * rank)
    atom_to_group = np.asarray(atom_to_group, dtype=int)
    K = len(atom_to_group)

    atom_masks = np.column_stack([group_masks[j] for j in atom_to_group]).astype(float)

    W = rng.random((F, K)) + 0.05
    W *= atom_masks
    for k in range(K):
        if W[:, k].sum() <= EPS:
            # Fallback if a mask is empty due to invalid frequency bounds.
            W[:, k] = rng.random(F) + 0.05

    H = rng.random((K, T)) + 0.05

    # Remove scale ambiguity W D, D^{-1} H.
    col_sum = W.sum(axis=0) + EPS
    W /= col_sum[None, :]
    H *= col_sum[:, None]

    # L1 penalty on anomaly activations; no L1 penalty on background.
    lam = np.zeros((K, 1))
    for k, group_idx in enumerate(atom_to_group):
        if group_idx > 0:
            lam[k, 0] = lambda_anomaly

    losses = []
    V = np.maximum(V, EPS)

    for it in range(n_iter):
        WH = W @ H + EPS

        # H update: positive part = W^T W H + lambda; negative part = W^T V.
        H *= (W.T @ V) / (W.T @ WH + lam + EPS)
        H = np.maximum(H, EPS)

        WH = W @ H + EPS

        # W update: positive part = W H H^T; negative part = V H^T.
        W *= (V @ H.T) / (WH @ H.T + EPS)

        # Hard spectral support projection.
        W *= atom_masks
        W = np.maximum(W, EPS * atom_masks)

        # Normalize W columns, absorb scale into H.
        col_sum = W.sum(axis=0) + EPS
        W /= col_sum[None, :]
        H *= col_sum[:, None]

        if verbose and (it % 50 == 0 or it == n_iter - 1):
            WH = W @ H + EPS
            loss = 0.5 * np.mean((V - WH) ** 2) + float(np.sum(lam * H)) / V.size
            losses.append(loss)
            print(f"iter={it:04d}, objective={loss:.6e}")

    group_V = []
    for j in range(J):
        idx = np.where(atom_to_group == j)[0]
        group_V.append(W[:, idx] @ H[idx, :])

    return group_V, W, H, atom_to_group, losses


def reconstruct_sources(
    X: np.ndarray,
    group_magnitudes: List[np.ndarray],
    fs: int,
    n_fft: int,
    hop: int,
    length: int,
    power: float = 2.0,
    mixture_time: np.ndarray | None = None,
    misi_iter: int = 0,
) -> List[np.ndarray]:
    """
    Phase-aware reconstruction. With power=2, this is a Wiener-like complex mask:
      S_j = L_j^2 / sum_k L_k^2 * X
    using the mixture STFT phase.
    """
    numerators = [np.maximum(L, 0.0) ** power for L in group_magnitudes]
    denom = np.sum(numerators, axis=0) + EPS
    S_list = [(num / denom) * X for num in numerators]

    if mixture_time is None or misi_iter <= 0:
        return [istft_signal(S, fs, n_fft, hop, length=length) for S in S_list]

    # Optional MISI-style phase refinement. It may improve perceptual quality, but is not
    # guaranteed to improve SDR for every synthetic example, so it is off by default.
    mags = [np.maximum(L, 0.0) for L in group_magnitudes]
    J = len(S_list)
    for _ in range(misi_iter):
        time_sources = [istft_signal(S, fs, n_fft, hop, length=length) for S in S_list]
        residual = mixture_time[:length] - np.sum(time_sources, axis=0)
        time_sources = [s + residual / J for s in time_sources]

        new_S_list = []
        for s, mag in zip(time_sources, mags):
            _, _, S_new = stft_signal(s, fs, n_fft, hop)
            S_fixed = np.zeros_like(mag, dtype=np.complex128)
            r = min(S_fixed.shape[0], S_new.shape[0])
            c = min(S_fixed.shape[1], S_new.shape[1])
            S_fixed[:r, :c] = S_new[:r, :c]
            new_S_list.append(mag * np.exp(1j * np.angle(S_fixed)))
        S_list = new_S_list

    return [istft_signal(S, fs, n_fft, hop, length=length) for S in S_list]


def separate_lowrank_anomalies(
    x: np.ndarray,
    fs: int,
    meta: List[SourceMeta],
    bg_rank: int = 4,
    n_fft: int = 1024,
    hop: int = 256,
    n_iter: int = 500,
    seed: int = 0,
    use_misi: bool = False,
):
    freqs, times, X = stft_signal(x, fs, n_fft, hop)
    V = np.abs(X)

    # Scale improves numerical conditioning of NMF.
    scale = np.mean(V) + EPS
    Vn = V / scale

    group_ranks = [bg_rank] + [m.rank for m in meta]
    group_masks = make_group_masks(freqs, meta)

    group_Vn, W, H, atom_to_group, losses = constrained_group_nmf(
        Vn,
        group_ranks=group_ranks,
        group_masks=group_masks,
        n_iter=n_iter,
        lambda_anomaly=0.004,
        seed=seed,
        verbose=False,
    )
    group_V = [g * scale for g in group_Vn]

    estimates = reconstruct_sources(
        X,
        group_V,
        fs=fs,
        n_fft=n_fft,
        hop=hop,
        length=len(x),
        power=2.0,
        mixture_time=x,
        misi_iter=8 if use_misi else 0,
    )

    info = {
        "freqs": freqs,
        "times": times,
        "X": X,
        "group_magnitudes": group_V,
        "W": W,
        "H": H,
        "atom_to_group": atom_to_group,
        "losses": losses,
        "ranks": group_ranks,
    }
    return estimates, info


# ---------------------------------------------------------------------------
# RPCA-style low-rank + sparse decomposition (minimal priors)
# ---------------------------------------------------------------------------

def rpca_magnitude_decompose(
    V: np.ndarray,
    lam: float | None = None,
    beta: float | None = None,
    tol: float = 1e-6,
    max_iter: int = 2000,
    verbose: bool = False,
):
    """
    Low-rank (smooth) + sparse decomposition on magnitude spectrogram.

    Uses temporal-sparsity assumption: the background is always (or almost
    always) present, whereas anomalies appear only in some time intervals.
    Identifies time frames dominated by the background by looking at the
    lower envelope of per-frame energy, reconstructs the background
    spectrogram from those frames, and sets the residual as anomalies.

    This is one of the weakest possible priors:
      - No frequency-band assumptions
      - No source-count assumptions
      - No rank assumptions
      - Only assumes anomalies are temporally sparse enough that some
        frames contain mostly background.

    V : (F, T) non-negative magnitude spectrogram.
    Returns L (background), S (anomalies), info dict.
    """
    F, T = V.shape

    # Per-frame total energy
    energy = V.sum(axis=0)  # shape (T,)

    # Use lower percentile of energy to identify background-dominated frames.
    # 15th percentile expects at least 15% of the recording to be anomaly-free.
    lo_pct = 15.0
    energy_thresh = np.percentile(energy, lo_pct)
    bg_mask = energy <= energy_thresh
    n_bg = int(np.sum(bg_mask))

    # Estimate background subspace from the quietest frames.
    # Use SVD to get a low-rank basis that captures the FM-modulated
    # harmonic structure (rank is auto-determined by energy ratio).
    min_rank = 3
    max_rank = 30
    if n_bg >= min_rank:
        V_bg = V[:, bg_mask]
        U_bg, s_bg, _ = np.linalg.svd(V_bg, full_matrices=False)
        # Keep components that capture 98% of background-frame energy.
        energy_cumsum = np.cumsum(s_bg ** 2)
        energy_total = energy_cumsum[-1]
        rank = int(np.searchsorted(energy_cumsum, 0.98 * energy_total))
        rank = max(min_rank, min(rank, max_rank, len(s_bg)))
        basis = U_bg[:, :rank]  # (F, rank)
    else:
        # Fallback: use SVD on all frames with conservative rank.
        U_all, s_all, _ = np.linalg.svd(V, full_matrices=False)
        energy_cumsum = np.cumsum(s_all ** 2)
        rank = int(np.searchsorted(energy_cumsum, 0.98 * energy_cumsum[-1]))
        rank = max(min_rank, min(rank, max_rank, len(s_all)))
        basis = U_all[:, :rank]

    effective_rank = rank
    if verbose:
        print(f"  RPCA temporal-sparsity: {n_bg}/{T} frames below "
              f"{lo_pct}th energy percentile, background rank={effective_rank}")

    # Project each frame onto the background subspace.
    coefs = basis.T @ V  # (rank, T)
    L = np.clip(basis @ coefs, 0.0, None)
    S = V - L
    S = np.maximum(S, 0.0)
    L = V - S

    info = {
        "n_iter": 1,
        "effective_rank": effective_rank,
        "lam": 0.0,
    }
    return L, S, info


def separate_anomalies_rpca(
    x: np.ndarray,
    fs: int,
    n_fft: int = 1024,
    hop: int = 256,
    lam: float | None = None,
    rpca_tol: float = 1e-6,
    rpca_max_iter: int = 2000,
    verbose: bool = False,
):
    """
    Single-channel anomaly separation using RPCA on the magnitude spectrogram.

    Only assumptions:
      1. Background is approximately low-rank in the magnitude spectrogram.
      2. Anomalies are sparse in time-frequency.

    Returns:
      estimates: [background, anomalies_sum]  (2 time-domain signals)
      info:      dict with L, S, effective_rank, etc.
    """
    freqs, times, X = stft_signal(x, fs, n_fft, hop)
    V = np.abs(X)

    L, S, rpca_info = rpca_magnitude_decompose(
        V, lam=lam, tol=rpca_tol, max_iter=rpca_max_iter, verbose=verbose,
    )

    # Phase-aware reconstruction using Wiener-like masking (power=2).
    group_magnitudes = [L, S]
    estimates = reconstruct_sources(
        X, group_magnitudes, fs=fs, n_fft=n_fft, hop=hop, length=len(x), power=2.0,
    )

    info = {
        "freqs": freqs,
        "times": times,
        "X": X,
        "L": L,
        "S": S,
        "rpca": rpca_info,
    }
    return estimates, info


def si_sdr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    alpha = np.dot(estimate, reference) / (np.dot(reference, reference) + EPS)
    target = alpha * reference
    error = estimate - target
    return float(10 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(error ** 2) + EPS)))


def corrcoef(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = reference - np.mean(reference)
    estimate = estimate - np.mean(estimate)
    return float(np.dot(reference, estimate) / ((np.linalg.norm(reference) * np.linalg.norm(estimate)) + EPS))


def bss_projection_metrics(true_sources: List[np.ndarray], estimated_sources: List[np.ndarray]):
    """
    Simple BSS Eval-style projection metrics.
    SDR_proj: target / (interference + artifacts)
    SIR_proj: target / interference
    SAR_proj: (target + interference) / artifacts
    """
    S = np.vstack([s - np.mean(s) for s in true_sources])  # shape: J × N
    metrics = []
    for j, estimate in enumerate(estimated_sources):
        y = estimate - np.mean(estimate)
        coeffs, *_ = np.linalg.lstsq(S.T, y, rcond=None)
        projection = coeffs @ S
        target = coeffs[j] * S[j]
        interference = projection - target
        artifacts = y - projection

        sdr = 10 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(interference ** 2) + np.sum(artifacts ** 2) + EPS))
        sir = 10 * np.log10((np.sum(target ** 2) + EPS) / (np.sum(interference ** 2) + EPS))
        sar = 10 * np.log10((np.sum(target ** 2) + np.sum(interference ** 2) + EPS) / (np.sum(artifacts ** 2) + EPS))
        metrics.append((float(sdr), float(sir), float(sar)))
    return metrics


def evaluate_separation(
    true_sources: List[np.ndarray],
    estimated_sources: List[np.ndarray],
    names: List[str],
) -> List[Dict[str, float | str]]:
    projection = bss_projection_metrics(true_sources, estimated_sources)
    rows = []
    for name, ref, est, (sdr_p, sir_p, sar_p) in zip(names, true_sources, estimated_sources, projection):
        rows.append(
            {
                "source": name,
                "SI_SDR_dB": si_sdr(ref, est),
                "corr": corrcoef(ref, est),
                "SDR_proj_dB": sdr_p,
                "SIR_proj_dB": sir_p,
                "SAR_proj_dB": sar_p,
            }
        )
    return rows


def print_metrics_table(rows: List[Dict[str, float | str]]) -> None:
    headers = ["source", "SI_SDR_dB", "corr", "SDR_proj_dB", "SIR_proj_dB", "SAR_proj_dB"]
    print(" | ".join(f"{h:>12}" for h in headers))
    print("-" * (15 * len(headers)))
    for row in rows:
        print(
            f"{row['source']:>12} | "
            f"{row['SI_SDR_dB']:12.2f} | "
            f"{row['corr']:12.3f} | "
            f"{row['SDR_proj_dB']:12.2f} | "
            f"{row['SIR_proj_dB']:12.2f} | "
            f"{row['SAR_proj_dB']:12.2f}"
        )


def save_wav(path: str, fs: int, x: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    y = x / (np.max(np.abs(x)) + EPS)
    wavfile.write(path, fs, np.int16(np.clip(y, -1.0, 1.0) * 32767))


def main():
    fs = 8000
    M = 3
    duration = 8.0
    seed = 42

    t, mixture, true_sources, meta = generate_car_mixture(M=M, fs=fs, duration=duration, seed=seed)
    names = ["background"] + [m.name for m in meta]
    out_dir = "output_lowrank_demo"

    # ---- Baseline: Constrained Group NMF with frequency masks ----
    print("=" * 70)
    print("Method 1: Group NMF with frequency masks (original)")
    print("=" * 70)
    estimates_nmf, info_nmf = separate_lowrank_anomalies(
        mixture, fs=fs, meta=meta, bg_rank=4, n_fft=1024, hop=256,
        n_iter=500, seed=seed,
    )
    rows_nmf = evaluate_separation(true_sources, estimates_nmf, names)
    print_metrics_table(rows_nmf)

    # ---- RPCA: Low-rank + sparse decomposition (minimal priors) ----
    print("\n" + "=" * 70)
    print("Method 2: RPCA low-rank + sparse (minimal priors)")
    print("=" * 70)
    estimates_rpca, info_rpca = separate_anomalies_rpca(
        mixture, fs=fs, n_fft=1024, hop=256, verbose=True,
    )
    # RPCA outputs [background, anomalies_sum] — combine true anomaly sources
    # for a fair comparison.
    true_anomalies_sum = np.sum(true_sources[1:], axis=0)
    rpca_true = [true_sources[0], true_anomalies_sum]
    rpca_names = ["background", "anomalies_sum"]
    rows_rpca = evaluate_separation(rpca_true, estimates_rpca, rpca_names)
    print_metrics_table(rows_rpca)

    print(f"\n  RPCA effective rank of background: {info_rpca['rpca']['effective_rank']}")
    print(f"  RPCA iterations: {info_rpca['rpca']['n_iter']}")
    print(f"  RPCA lambda (auto): {info_rpca['rpca']['lam']:.4f}")

    # ---- Save all audio files ----
    save_wav(os.path.join(out_dir, "mixture.wav"), fs, mixture)
    for name, s in zip(names, true_sources):
        save_wav(os.path.join(out_dir, f"true_{name}.wav"), fs, s)
    # Also save summed anomalies for RPCA comparison
    save_wav(os.path.join(out_dir, "true_anomalies_sum.wav"), fs, true_anomalies_sum)
    for name, s_hat in zip(names, estimates_nmf):
        save_wav(os.path.join(out_dir, f"estimated_nmf_{name}.wav"), fs, s_hat)
    for name, s_hat in zip(rpca_names, estimates_rpca):
        save_wav(os.path.join(out_dir, f"estimated_rpca_{name}.wav"), fs, s_hat)

    print(f"\nSaved wav files to: {out_dir}/")
    print("NMF source order:", ", ".join(names))
    print("RPCA source order:", ", ".join(rpca_names))


if __name__ == "__main__":
    main()
