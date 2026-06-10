"""
Spectrogram visualization for the lowrank anomaly separation demo.
Shows mixture, true sources, and estimated sources side-by-side.
"""

import os
import numpy as np
from scipy.io import wavfile
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # headless backend


def compute_spectrogram(x: np.ndarray, fs: int, n_fft: int = 1024, hop: int = 256):
    freqs, times, X = signal.stft(
        x, fs=fs, window="hann", nperseg=n_fft,
        noverlap=n_fft - hop, boundary="zeros", padded=True,
    )
    mag = np.abs(X)
    log_mag = 20 * np.log10(mag + 1e-12)
    return freqs, times, log_mag


def plot_spectrogram(ax, freqs, times, log_mag, title, vmin=-80, vmax=10,
                     cmap="inferno", colorbar=True):
    im = ax.pcolormesh(times, freqs, log_mag, shading="gouraud",
                       cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_ylabel("Frequency [Hz]")
    if colorbar:
        plt.colorbar(im, ax=ax, label="dB")
    return im


def main():
    out_dir = "/home/hlb/abnormal_filter_lowrank/output_lowrank_demo"
    output_plot = "/home/hlb/abnormal_filter_lowrank/spectrograms.png"

    # Load all files
    names = ["background", "squeal", "knock", "rattle"]

    fs, mixture = wavfile.read(os.path.join(out_dir, "mixture.wav"))
    mixture = mixture.astype(float) / 32767.0

    true_sources = {}
    est_sources = {}
    for name in names:
        _, s = wavfile.read(os.path.join(out_dir, f"true_{name}.wav"))
        true_sources[name] = s.astype(float) / 32767.0
        _, s = wavfile.read(os.path.join(out_dir, f"estimated_nmf_{name}.wav"))
        est_sources[name] = s.astype(float) / 32767.0

    # STFT params (match the demo)
    n_fft = 1024
    hop = 256

    # Compute all spectrograms
    _, _, mix_spec = compute_spectrogram(mixture, fs, n_fft, hop)

    specs = {"mixture": mix_spec}
    for name in names:
        _, _, spec_t = compute_spectrogram(true_sources[name], fs, n_fft, hop)
        specs[f"true_{name}"] = spec_t
        _, _, spec_e = compute_spectrogram(est_sources[name], fs, n_fft, hop)
        specs[f"est_{name}"] = spec_e

    # Use common frequency/time axes for the true/est pairs
    freqs, times, _ = compute_spectrogram(mixture, fs, n_fft, hop)

    # Dynamic dB range
    all_mags = np.concatenate([s.ravel() for s in specs.values()])
    vmax = np.percentile(all_mags, 99)
    vmin = vmax - 80

    # ---- Plot layout ----
    fig = plt.figure(figsize=(18, 14))

    # Row 1: Mixture (full width)
    ax_mix = fig.add_subplot(5, 2, (1, 2))
    plot_spectrogram(ax_mix, freqs, times, specs["mixture"],
                     "Mixture Spectrogram", vmin=vmin, vmax=vmax)

    # Rows 2-5: True vs Estimated for each source
    freq_bands = {
        "background": (0, 1600),
        "squeal": (1200, 2000),
        "knock": (0, 800),
        "rattle": (500, 1500),
    }

    for i, name in enumerate(names):
        row = i + 2
        ax_true = fig.add_subplot(5, 2, row * 2 - 1)
        ax_est = fig.add_subplot(5, 2, row * 2)

        plot_spectrogram(ax_true, freqs, times, specs[f"true_{name}"],
                         f"True {name}", vmin=vmin, vmax=vmax)
        plot_spectrogram(ax_est, freqs, times, specs[f"est_{name}"],
                         f"Estimated {name}", vmin=vmin, vmax=vmax)

        # Zoom to relevant frequency band
        flo, fhi = freq_bands[name]
        for ax in [ax_true, ax_est]:
            ax.set_ylim(flo, fhi)

    fig.suptitle("Lowrank Anomaly Separation — Spectrogram Comparison",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()
    plt.savefig(output_plot, dpi=150, bbox_inches="tight")
    print(f"Saved spectrogram plot to: {output_plot}")

    # ---- Difference map (error) ----
    fig2, axes2 = plt.subplots(2, 4, figsize=(20, 9))

    for i, name in enumerate(names):
        ax_true = axes2[0, i]
        ax_est = axes2[1, i]

        true_mag = 10 ** (specs[f"true_{name}"] / 20)
        est_mag = 10 ** (specs[f"est_{name}"] / 20)
        error_db = 20 * np.log10(np.abs(est_mag - true_mag) + 1e-12)

        # Choose scale for error
        err_vmax = np.percentile(error_db, 95)
        err_vmin = err_vmax - 60

        plot_spectrogram(ax_est, freqs, times, specs[f"est_{name}"],
                         f"Estimated {name}", vmin=vmin, vmax=vmax)

        im_err = ax_true.pcolormesh(times, freqs, error_db, shading="gouraud",
                                     cmap="coolwarm", vmin=err_vmin,
                                     vmax=err_vmax, rasterized=True)
        ax_true.set_title(f"Residual: True − Est {name}", fontsize=10, fontweight="bold")
        ax_true.set_ylabel("Frequency [Hz]")
        plt.colorbar(im_err, ax=ax_true, label="dB")

        flo, fhi = freq_bands[name]
        for ax in [ax_true, ax_est]:
            ax.set_ylim(flo, fhi)

    fig2.suptitle("Separation Quality — Estimated Spectrograms vs Residuals",
                   fontsize=14, fontweight="bold")
    plt.tight_layout()
    residual_plot = "/home/hlb/abnormal_filter_lowrank/residuals.png"
    plt.savefig(residual_plot, dpi=150, bbox_inches="tight")
    print(f"Saved residual plot to: {residual_plot}")

    # ---- Waveform overview ----
    fig3, axes3 = plt.subplots(5, 1, figsize=(18, 12), sharex=True)
    t = np.arange(len(mixture)) / fs

    axes3[0].plot(t, mixture, color="gray", alpha=0.7, linewidth=0.5)
    axes3[0].set_title("Mixture", fontweight="bold")
    axes3[0].set_ylabel("Amplitude")

    colors = {"background": "#1f77b4", "squeal": "#ff7f0e",
              "knock": "#2ca02c", "rattle": "#d62728"}

    for i, name in enumerate(names):
        ax = axes3[i + 1]
        ax.plot(t, true_sources[name], color=colors[name], alpha=0.6,
                linewidth=0.5, label=f"True {name}")
        ax.plot(t, est_sources[name], color="black", alpha=0.6,
                linewidth=0.5, linestyle="--", label=f"Est {name}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(name.capitalize(), fontweight="bold")

    axes3[-1].set_xlabel("Time [s]")
    fig3.suptitle("Waveform Comparison — True vs Estimated",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    waveform_plot = "/home/hlb/abnormal_filter_lowrank/waveforms.png"
    plt.savefig(waveform_plot, dpi=150, bbox_inches="tight")
    print(f"Saved waveform plot to: {waveform_plot}")


def plot_rpca_results():
    """Visualize RPCA low-rank + sparse decomposition results."""
    out_dir = "/home/hlb/abnormal_filter_lowrank/output_lowrank_demo"

    fs, mixture = wavfile.read(os.path.join(out_dir, "mixture.wav"))
    mixture = mixture.astype(float) / 32767.0

    # Load RPCA outputs and true references
    rpca_pairs = {
        "background": "background",
        "anomalies_sum": "anomalies_sum",
    }

    true_signals = {}
    est_signals = {}
    for key, fname in rpca_pairs.items():
        est_path = os.path.join(out_dir, f"estimated_rpca_{fname}.wav")
        true_path = os.path.join(out_dir, f"true_{fname}.wav")
        if not os.path.exists(est_path):
            print(f"Skipping RPCA visualization: {est_path} not found. Run the demo first.")
            return
        _, s = wavfile.read(true_path)
        true_signals[key] = s.astype(float) / 32767.0
        _, s = wavfile.read(est_path)
        est_signals[key] = s.astype(float) / 32767.0

    n_fft = 1024
    hop = 256
    freqs, times, _ = compute_spectrogram(mixture, fs, n_fft, hop)

    # Compute spectrograms
    _, _, mix_spec = compute_spectrogram(mixture, fs, n_fft, hop)
    specs = {"mixture": mix_spec}
    for key in rpca_pairs:
        _, _, st = compute_spectrogram(true_signals[key], fs, n_fft, hop)
        specs[f"true_{key}"] = st
        _, _, se = compute_spectrogram(est_signals[key], fs, n_fft, hop)
        specs[f"est_{key}"] = se

    # Dynamic dB range
    all_mags = np.concatenate([s.ravel() for s in specs.values()])
    vmax = np.percentile(all_mags, 99)
    vmin = vmax - 80

    # ---- Plot 1: RPCA spectrogram comparison ----
    fig, axes = plt.subplots(2, 2, figsize=(18, 8))

    plot_spectrogram(axes[0, 0], freqs, times, specs["mixture"],
                     "Mixture Spectrogram", vmin=vmin, vmax=vmax)
    plot_spectrogram(axes[0, 1], freqs, times, specs["true_background"],
                     "True Background (clean engine)", vmin=vmin, vmax=vmax)
    plot_spectrogram(axes[1, 0], freqs, times, specs["est_background"],
                     "RPCA Estimated Background (L, low-rank)", vmin=vmin, vmax=vmax)
    plot_spectrogram(axes[1, 1], freqs, times, specs[f"true_anomalies_sum"],
                     "True Anomalies Sum", vmin=vmin, vmax=vmax)

    fig.suptitle("RPCA: Low-Rank + Sparse Decomposition — Spectrograms",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    rpca_spec_plot = "/home/hlb/abnormal_filter_lowrank/spectrograms_rpca.png"
    plt.savefig(rpca_spec_plot, dpi=150, bbox_inches="tight")
    print(f"Saved RPCA spectrogram plot to: {rpca_spec_plot}")

    # ---- Plot 2: Error / residual analysis ----
    fig2, axes2 = plt.subplots(2, 2, figsize=(18, 8))

    for i, key in enumerate(["background", "anomalies_sum"]):
        ax_est = axes2[0, i]
        ax_err = axes2[1, i]

        true_mag = 10 ** (specs[f"true_{key}"] / 20)
        est_mag = 10 ** (specs[f"est_{key}"] / 20)
        error_db = 20 * np.log10(np.abs(est_mag - true_mag) + 1e-12)

        err_vmax = np.percentile(error_db, 95)
        err_vmin = err_vmax - 60

        plot_spectrogram(ax_est, freqs, times, specs[f"est_{key}"],
                         f"RPCA Estimated {key}", vmin=vmin, vmax=vmax)

        im_err = ax_err.pcolormesh(times, freqs, error_db, shading="gouraud",
                                   cmap="coolwarm", vmin=err_vmin,
                                   vmax=err_vmax, rasterized=True)
        ax_err.set_title(f"Residual: True − Est {key}", fontsize=10, fontweight="bold")
        ax_err.set_ylabel("Frequency [Hz]")
        plt.colorbar(im_err, ax=ax_err, label="dB")

    fig2.suptitle("RPCA Separation Quality — Estimated vs Residuals",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    rpca_residual_plot = "/home/hlb/abnormal_filter_lowrank/residuals_rpca.png"
    plt.savefig(rpca_residual_plot, dpi=150, bbox_inches="tight")
    print(f"Saved RPCA residual plot to: {rpca_residual_plot}")

    # ---- Plot 3: Waveform comparison ----
    fig3, axes3 = plt.subplots(3, 1, figsize=(18, 8), sharex=True)
    t_wave = np.arange(len(mixture)) / fs

    axes3[0].plot(t_wave, mixture, color="gray", alpha=0.7, linewidth=0.5)
    axes3[0].set_title("Mixture", fontweight="bold")
    axes3[0].set_ylabel("Amplitude")

    colors = {"background": "#1f77b4", "anomalies_sum": "#d62728"}
    for i, key in enumerate(["background", "anomalies_sum"]):
        ax = axes3[i + 1]
        ax.plot(t_wave, true_signals[key], color=colors[key], alpha=0.6,
                linewidth=0.5, label=f"True {key}")
        ax.plot(t_wave, est_signals[key], color="black", alpha=0.6,
                linewidth=0.5, linestyle="--", label=f"RPCA Est {key}")
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right", fontsize=8)
        ax.set_title(key.replace("_", " ").capitalize(), fontweight="bold")

    axes3[-1].set_xlabel("Time [s]")
    fig3.suptitle("RPCA Waveform Comparison — True vs Estimated",
                  fontsize=14, fontweight="bold")
    plt.tight_layout()
    rpca_waveform_plot = "/home/hlb/abnormal_filter_lowrank/waveforms_rpca.png"
    plt.savefig(rpca_waveform_plot, dpi=150, bbox_inches="tight")
    print(f"Saved RPCA waveform plot to: {rpca_waveform_plot}")


if __name__ == "__main__":
    main()
    plot_rpca_results()
