#!/usr/bin/env python3
"""
NCZ3: float-safe CAN neural compression patch.

Problem fixed:
    NCZ2 encodes arithmetic-coded bytes using integer CDFs derived from an MLP,
    but the decoder recomputes those CDFs from floating point model weights. If a
    copied model, BLAS backend, CPU/GPU, or dtype conversion produces a different
    integer CDF at even one context/symbol boundary, arithmetic decoding can drift.

Solution:
    Use the MLP only on the encoder side. Compute the integer CDF table once in
    the compressor, serialize that exact table into the pack, and make the
    decoder use only the serialized integer CDF table. Decoding is then pure
    integer arithmetic + CRC checks, independent of floating point math.

This script reuses your existing neural_canzip_v2.py and
structured_can_ai_cs_pipeline.py. It also works with uploaded names containing
"(1)" when placed in the same directory.

Typical use:
    python ncz3_cdf_safe_patch.py demo --out-dir /mnt/data/ncz3_demo --duration-sec 20 --epochs 20

Outputs:
    - src.blf
    - fixed.ncz3
    - fixed_model_perturbed.ncz3
    - legacy.ncz2 and legacy_model_perturbed.ncz2 for failure-mode comparison
    - simulation_report.json
"""
from __future__ import annotations

import argparse
import importlib.util
import io
import json
import sys
import time
import zlib
import struct
import lzma
from pathlib import Path
from typing import Any

import numpy as np

MAGIC_NCZ3 = b"NCZ3\x00\x00\x00\x01"


def _import_module_from_candidates(module_name: str, candidates: list[Path]):
    """Import a module, falling back to specific filenames such as name(1).py."""
    try:
        return __import__(module_name)
    except Exception:
        pass
    for path in candidates:
        if path.exists():
            spec = importlib.util.spec_from_file_location(module_name, str(path))
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(f"Cannot import {module_name}; tried: " + ", ".join(str(p) for p in candidates))


def load_user_modules():
    """Load the two original scripts from the same directory as this patch."""
    base = Path(__file__).resolve().parent
    if str(base) not in sys.path:
        sys.path.insert(0, str(base))

    structured = _import_module_from_candidates(
        "structured_can_ai_cs_pipeline",
        [base / "structured_can_ai_cs_pipeline.py", base / "structured_can_ai_cs_pipeline(1).py"],
    )
    # neural_canzip_v2 imports structured_can_ai_cs_pipeline by this exact name.
    sys.modules["structured_can_ai_cs_pipeline"] = structured

    neural = _import_module_from_candidates(
        "neural_canzip_v2",
        [base / "neural_canzip_v2.py", base / "neural_canzip_v2(1).py"],
    )
    return structured, neural


STRUCTURED, N = load_user_modules()


def crc32_bytes(data: bytes) -> int:
    return int(zlib.crc32(data) & 0xFFFFFFFF)


def write_section(f, data: bytes) -> None:
    f.write(struct.pack("<Q", len(data)))
    f.write(data)


def read_section(f) -> bytes:
    raw = f.read(8)
    if len(raw) != 8:
        raise EOFError("truncated section length")
    n = struct.unpack("<Q", raw)[0]
    data = f.read(n)
    if len(data) != n:
        raise EOFError("truncated section body")
    return data


def encode_cdf_blob(cdfs: np.ndarray) -> tuple[bytes, dict[str, Any]]:
    """Serialize exact integer CDFs; decoder will use this, not the MLP.

    We store positive frequencies diff(CDF) rather than cumulative CDF values.
    This is still an exact integer codec model, but compresses much better with zlib.
    """
    if cdfs.ndim != 2 or cdfs.shape[1] != 257:
        raise ValueError(f"expected CDF shape [contexts,257], got {cdfs.shape}")
    if cdfs.dtype not in (np.dtype("uint16"), np.dtype("uint32")):
        raise ValueError(f"expected uint16/uint32 CDFs, got {cdfs.dtype}")
    if not np.all(cdfs[:, 0] == 0):
        raise ValueError("CDF must start at zero")
    if not np.all(cdfs[:, -1] == int(cdfs[0, -1])):
        raise ValueError("all CDF rows must have the same total")
    freq64 = np.diff(cdfs.astype(np.int64), axis=1)
    if not np.all(freq64 > 0):
        raise ValueError("all arithmetic-coder symbols must have positive integer frequency")
    freq = freq64.astype(cdfs.dtype)
    cdf_raw = np.ascontiguousarray(cdfs).tobytes()
    freq_raw = np.ascontiguousarray(freq).tobytes()
    meta = {
        "cdf_encoding": "positive_frequency_table_v1",
        "cdf_shape": [int(cdfs.shape[0]), int(cdfs.shape[1])],
        "cdf_freq_shape": [int(freq.shape[0]), int(freq.shape[1])],
        "cdf_dtype": str(cdfs.dtype),
        "cdf_raw_bytes": len(cdf_raw),
        "cdf_freq_raw_bytes": len(freq_raw),
        "cdf_crc32": crc32_bytes(cdf_raw),
        "cdf_freq_crc32": crc32_bytes(freq_raw),
    }
    return zlib.compress(freq_raw, 9), meta


def decode_cdf_blob(blob: bytes, header: dict[str, Any]) -> np.ndarray:
    raw = zlib.decompress(blob)
    if crc32_bytes(raw) != int(header["cdf_freq_crc32"]):
        raise ValueError("CDF frequency-table CRC mismatch; pack is corrupted or not canonical")
    dtype = np.dtype(header["cdf_dtype"])
    freq_shape = tuple(int(x) for x in header["cdf_freq_shape"])
    freq = np.frombuffer(raw, dtype=dtype).reshape(freq_shape).copy()
    if freq.shape[1] != 256:
        raise ValueError("bad frequency-table width")
    if not np.all(freq.astype(np.int64) > 0):
        raise ValueError("frequency table must be strictly positive")

    cdf64 = np.concatenate(
        [np.zeros((freq.shape[0], 1), dtype=np.int64), np.cumsum(freq.astype(np.int64), axis=1)],
        axis=1,
    )
    if not np.all(cdf64[:, -1] == int(header["cdf_total"])):
        raise ValueError("CDF total mismatch")
    cdfs = cdf64.astype(dtype)

    # Structural and byte-level canonical checks catch accidental bit flips.
    if cdfs.shape != tuple(int(x) for x in header["cdf_shape"]):
        raise ValueError("CDF shape mismatch")
    if crc32_bytes(np.ascontiguousarray(cdfs).tobytes()) != int(header["cdf_crc32"]):
        raise ValueError("reconstructed CDF CRC mismatch")
    return cdfs


def save_pack_ncz3(
    path: Path,
    header: dict[str, Any],
    model_blob: bytes,
    cdf_blob: bytes,
    id_blob: bytes,
    dt_blob: bytes,
    bitstream: bytes,
) -> None:
    header_blob = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    with Path(path).open("wb") as f:
        f.write(MAGIC_NCZ3)
        for blob in (header_blob, model_blob, cdf_blob, id_blob, dt_blob, bitstream):
            write_section(f, blob)


def load_pack_ncz3(path: Path) -> tuple[dict[str, Any], bytes, bytes, bytes, bytes, bytes]:
    with Path(path).open("rb") as f:
        if f.read(len(MAGIC_NCZ3)) != MAGIC_NCZ3:
            raise ValueError("not an NCZ3 float-safe pack")
        header = json.loads(read_section(f).decode("utf-8"))
        model_blob = read_section(f)     # optional/audit only; decoder never uses it
        cdf_blob = read_section(f)       # authoritative probability model for codec
        id_blob = read_section(f)
        dt_blob = read_section(f)
        bitstream = read_section(f)
    return header, model_blob, cdf_blob, id_blob, dt_blob, bitstream


def compress_blf_float_safe(
    blf_path: Path,
    out_path: Path,
    epochs: int = 40,
    hidden: int = 24,
    lr: float = 0.05,
    cdf_total: int = 4096,
    seed: int = 20260516,
    keep_model_blob: bool = True,
) -> dict[str, Any]:
    """Compress BLF to NCZ3. Decoder never evaluates floating point MLP."""
    if cdf_total <= 256:
        raise ValueError("cdf_total must be > 256")
    N.selftest_bayesian_arithmetic()
    t0 = time.time()

    frames = STRUCTURED.read_blf_classic(Path(blf_path))
    ids, t_us, payload = STRUCTURED.frames_to_arrays(frames)
    id_codes, unique_ids = N.build_id_codes(ids)
    num_ids = len(unique_ids)

    manifold_points = N.construct_fisher_rao_context_manifold(id_codes, num_ids)
    innovations = N.compute_gf2_temporal_residuals(id_codes, payload, num_ids)
    posterior_stats = N.compute_empirical_posterior_statistics(
        manifold_points, innovations, num_ids * 8 * 16
    )

    model, train_report = N.optimize_variational_bayes_elbo(
        posterior_stats, hidden=hidden, epochs=epochs, lr=lr, seed=seed
    )
    model_meta = {
        "num_manifold_points": int(model.num_manifold_points),
        "hidden": int(model.hidden),
        "cdf_total": int(cdf_total),
        "note": "audit only; decoder uses serialized integer CDFs",
    }
    model_blob = model.to_blob(model_meta) if keep_model_blob else b""

    # Canonicalize exactly once on the encoder side, then serialize the integer table.
    model_for_codec, _ = N.InformationGeometricNeuralModel.from_blob(model_blob) if model_blob else (model, model_meta)
    cdfs = N.compute_posterior_predictive_cdfs(model_for_codec, cdf_total)
    cdf_blob, cdf_meta = encode_cdf_blob(cdfs)

    t_enc = time.time()
    bitstream = N.bayesian_arithmetic_encode(innovations, manifold_points, cdfs, cdf_total)
    enc_sec = time.time() - t_enc

    id_blob = zlib.compress(id_codes.tobytes(), 9)
    dt = np.diff(t_us, prepend=t_us[0]).astype(np.int32)
    dt_blob = zlib.compress(dt.tobytes(), 9)

    source_bytes = Path(blf_path).stat().st_size
    source_lzma6_bytes = len(lzma.compress(Path(blf_path).read_bytes(), preset=6))

    header: dict[str, Any] = {
        "format": "NCZ3-float-safe-CAN-neural-compression",
        "core_rule": "decoder MUST use serialized integer CDF table; decoder MUST NOT evaluate MLP",
        "decode_uses_float_model": False,
        "fidelity": "CAN-frame-level-lossless; canonical-BLF-output; not raw-BLF-byte-lossless",
        "frame_count": int(len(frames)),
        "payload_len": 8,
        "innovation_count": int(innovations.size),
        "unique_ids": [int(x) for x in unique_ids],
        "id_dtype": "uint8" if id_codes.dtype == np.uint8 else "uint16",
        "first_timestamp_us": int(t_us[0]),
        "cdf_total": int(cdf_total),
        **cdf_meta,
        "model_meta": model_meta,
        "train_report": train_report,
        "payload_crc32": crc32_bytes(payload.tobytes()),
        "innovation_crc32": crc32_bytes(innovations.tobytes()),
        "id_codes_crc32": crc32_bytes(id_codes.tobytes()),
        "timestamp_delta_crc32": crc32_bytes(dt.tobytes()),
        "source_blf_bytes": int(source_bytes),
        "source_blf_lzma6_bytes": int(source_lzma6_bytes),
        "source_payload_bytes": int(payload.nbytes),
        "raw_payload_shannon_entropy_bits_per_byte": STRUCTURED.shannon_entropy_u8(payload),
        "temporal_innovation_shannon_entropy_bits_per_byte": STRUCTURED.shannon_entropy_u8(innovations),
        "temporal_innovation_zero_ratio": float((innovations == 0).mean()),
        "sections": {
            "model_blob_bytes_audit_only": int(len(model_blob)),
            "cdf_blob_bytes_authoritative": int(len(cdf_blob)),
            "id_blob_bytes": int(len(id_blob)),
            "timestamp_delta_blob_bytes": int(len(dt_blob)),
            "arithmetic_bitstream_bytes": int(len(bitstream)),
        },
        "arithmetic_bits_per_innovation_actual": float(len(bitstream) * 8 / max(1, innovations.size)),
        "arithmetic_encode_seconds": float(enc_sec),
    }
    save_pack_ncz3(out_path, header, model_blob, cdf_blob, id_blob, dt_blob, bitstream)
    header["output_pack_bytes"] = int(Path(out_path).stat().st_size)
    header["compression_ratio_vs_source_blf"] = float(header["output_pack_bytes"] / source_bytes)
    header["compression_ratio_vs_lzma6_blf"] = float(header["output_pack_bytes"] / source_lzma6_bytes) if source_lzma6_bytes else None
    header["total_seconds"] = float(time.time() - t0)
    save_pack_ncz3(out_path, header, model_blob, cdf_blob, id_blob, dt_blob, bitstream)
    header["output_pack_bytes"] = int(Path(out_path).stat().st_size)
    return header


def decode_pack_arrays_float_safe(pack_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Decode NCZ3 using only serialized integer CDFs and integer arithmetic."""
    header, _model_blob, cdf_blob, id_blob, dt_blob, bitstream = load_pack_ncz3(pack_path)
    cdfs = decode_cdf_blob(cdf_blob, header)

    id_dtype = np.uint8 if header["id_dtype"] == "uint8" else np.uint16
    id_codes = np.frombuffer(zlib.decompress(id_blob), dtype=id_dtype).copy()
    dt = np.frombuffer(zlib.decompress(dt_blob), dtype=np.int32).copy()
    if crc32_bytes(id_codes.tobytes()) != int(header["id_codes_crc32"]):
        raise ValueError("id code CRC mismatch")
    if crc32_bytes(dt.tobytes()) != int(header["timestamp_delta_crc32"]):
        raise ValueError("timestamp CRC mismatch")

    t_us = np.cumsum(dt.astype(np.int64)) + int(header["first_timestamp_us"])
    manifold_points = N.construct_fisher_rao_context_manifold(id_codes, len(header["unique_ids"]))

    t_dec = time.time()
    innovations = N.bayesian_arithmetic_decode(
        int(header["innovation_count"]), manifold_points, cdfs, int(header["cdf_total"]), bitstream
    )
    header["arithmetic_decode_seconds"] = float(time.time() - t_dec)
    if crc32_bytes(innovations.tobytes()) != int(header["innovation_crc32"]):
        raise ValueError("decoded innovation CRC mismatch")

    payload = N.invert_gf2_temporal_residuals(id_codes, innovations, len(header["unique_ids"]))
    if crc32_bytes(payload.tobytes()) != int(header["payload_crc32"]):
        raise ValueError("decoded payload CRC mismatch")

    unique_ids = np.array(header["unique_ids"], dtype=np.uint16)
    ids = unique_ids[id_codes.astype(np.int64)]
    return ids, t_us, payload, header


def verify_float_safe(blf_path: Path, pack_path: Path) -> dict[str, Any]:
    """Verify NCZ3 against source BLF arrays."""
    frames = STRUCTURED.read_blf_classic(Path(blf_path))
    src_ids, src_t_us, src_payload = STRUCTURED.frames_to_arrays(frames)
    ids, t_us, payload, header = decode_pack_arrays_float_safe(pack_path)
    return {
        "source_blf": str(blf_path),
        "pack": str(pack_path),
        "source_frames": int(len(frames)),
        "decoded_frames": int(len(ids)),
        "ids_match": bool(np.array_equal(src_ids, ids)),
        "timestamps_match": bool(np.array_equal(src_t_us, t_us)),
        "payload_match": bool(np.array_equal(src_payload, payload)),
        "payload_byte_mismatches": int(np.count_nonzero(src_payload != payload)) if src_payload.shape == payload.shape else None,
        "frame_level_lossless": bool(np.array_equal(src_ids, ids) and np.array_equal(src_t_us, t_us) and np.array_equal(src_payload, payload)),
        "decode_uses_float_model": bool(header.get("decode_uses_float_model")),
        "cdf_crc32": int(header["cdf_crc32"]),
        "fidelity": header["fidelity"],
        "decode_seconds_arithmetic_only": header.get("arithmetic_decode_seconds"),
    }


def perturb_model_blob(model_blob: bytes, delta: float = 0.25) -> bytes:
    """Create a deterministic perturbation of the stored audit MLP model."""
    if not model_blob:
        return model_blob
    model, meta = N.InformationGeometricNeuralModel.from_blob(model_blob)
    # Bias perturbation is deterministic and makes CDF changes likely.
    model.b[0] = model.b[0] + float(delta)
    if model.W.size:
        model.W.flat[0] = model.W.flat[0] - float(delta) / 2.0
    return model.to_blob(meta)


def perturb_ncz2_model_only(src: Path, dst: Path, delta: float = 0.25) -> dict[str, Any]:
    """Modify only the MLP section in an NCZ2 pack, leaving bitstream untouched."""
    header, model_blob, id_blob, dt_blob, bitstream = N.load_pack(Path(src))
    original_model, _ = N.InformationGeometricNeuralModel.from_blob(model_blob)
    original_cdfs = N.compute_posterior_predictive_cdfs(original_model, int(header["cdf_total"]))
    new_model_blob = perturb_model_blob(model_blob, delta=delta)
    new_model, _ = N.InformationGeometricNeuralModel.from_blob(new_model_blob)
    new_cdfs = N.compute_posterior_predictive_cdfs(new_model, int(header["cdf_total"]))
    changed_entries = int(np.count_nonzero(original_cdfs != new_cdfs))
    changed_rows = int(np.count_nonzero(np.any(original_cdfs != new_cdfs, axis=1)))
    N.save_pack(Path(dst), header, new_model_blob, id_blob, dt_blob, bitstream)
    return {"changed_cdf_entries": changed_entries, "changed_cdf_rows": changed_rows}


def perturb_ncz3_model_only(src: Path, dst: Path, delta: float = 0.25) -> dict[str, Any]:
    """Modify only the audit MLP model in NCZ3. Decode should still succeed."""
    header, model_blob, cdf_blob, id_blob, dt_blob, bitstream = load_pack_ncz3(Path(src))
    new_model_blob = perturb_model_blob(model_blob, delta=delta)
    save_pack_ncz3(Path(dst), header, new_model_blob, cdf_blob, id_blob, dt_blob, bitstream)
    return {
        "model_blob_changed": bool(model_blob != new_model_blob),
        "cdf_blob_unchanged": True,
        "cdf_crc32": int(header["cdf_crc32"]),
    }


def try_call(fn, *args, **kwargs) -> dict[str, Any]:
    try:
        value = fn(*args, **kwargs)
        return {"ok": True, "result": value}
    except Exception as e:  # keep failure in JSON report for simulation
        return {"ok": False, "error_type": type(e).__name__, "error": str(e)}


def run_simulation(
    out_dir: Path,
    duration_sec: float = 20.0,
    epochs: int = 20,
    hidden: int = 16,
    lr: float = 0.05,
    cdf_total: int = 4096,
    delta: float = 0.25,
) -> dict[str, Any]:
    """Generate data, compare NCZ2 failure mode with NCZ3 robust decoding."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    src_blf = out_dir / "src.blf"
    legacy = out_dir / "legacy.ncz2"
    legacy_perturbed = out_dir / "legacy_model_perturbed.ncz2"
    fixed = out_dir / "fixed.ncz3"
    fixed_perturbed = out_dir / "fixed_model_perturbed.ncz3"

    frames = STRUCTURED.generate_structured_frames(duration_sec)
    STRUCTURED.SimpleBlfWriter(src_blf, compression_level=0).write_frames(frames)

    # Legacy NCZ2: decoder depends on re-evaluating the MLP into CDFs.
    legacy_header = N.compress_blf_via_bayesian_information_geometry(
        src_blf, legacy, epochs=epochs, hidden=hidden, lr=lr, cdf_total=cdf_total, seed=20260516
    )
    legacy_verify = try_call(N.verify_pack, src_blf, legacy)
    legacy_perturb_info = perturb_ncz2_model_only(legacy, legacy_perturbed, delta=delta)
    legacy_perturbed_verify = try_call(N.verify_pack, src_blf, legacy_perturbed)

    # Fixed NCZ3: decoder ignores MLP and uses exact serialized integer CDFs.
    fixed_header = compress_blf_float_safe(
        src_blf, fixed, epochs=epochs, hidden=hidden, lr=lr, cdf_total=cdf_total, seed=20260516, keep_model_blob=True
    )
    fixed_verify = try_call(verify_float_safe, src_blf, fixed)
    fixed_perturb_info = perturb_ncz3_model_only(fixed, fixed_perturbed, delta=delta)
    fixed_perturbed_verify = try_call(verify_float_safe, src_blf, fixed_perturbed)

    report = {
        "simulation": {
            "duration_sec": float(duration_sec),
            "frames": int(len(frames)),
            "epochs": int(epochs),
            "hidden": int(hidden),
            "cdf_total": int(cdf_total),
            "model_perturbation_delta": float(delta),
            "src_blf": str(src_blf),
        },
        "legacy_ncz2": {
            "pack": str(legacy),
            "pack_bytes": int(legacy.stat().st_size),
            "model_blob_bytes": int(legacy_header["sections"]["model_blob_bytes"]),
            "bitstream_bytes": int(legacy_header["sections"]["bayesian_arithmetic_bitstream_bytes"]),
            "normal_verify": legacy_verify,
            "perturbed_pack": str(legacy_perturbed),
            "perturb_info": legacy_perturb_info,
            "verify_after_model_perturbation": legacy_perturbed_verify,
        },
        "fixed_ncz3": {
            "pack": str(fixed),
            "pack_bytes": int(fixed.stat().st_size),
            "cdf_blob_bytes": int(fixed_header["sections"]["cdf_blob_bytes_authoritative"]),
            "model_blob_bytes_audit_only": int(fixed_header["sections"]["model_blob_bytes_audit_only"]),
            "bitstream_bytes": int(fixed_header["sections"]["arithmetic_bitstream_bytes"]),
            "normal_verify": fixed_verify,
            "perturbed_pack": str(fixed_perturbed),
            "perturb_info": fixed_perturb_info,
            "verify_after_model_perturbation": fixed_perturbed_verify,
        },
    }
    report_path = out_dir / "simulation_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="NCZ3 float-safe patch and simulation for CAN neural compression")
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compress", help="compress BLF to float-safe NCZ3")
    c.add_argument("blf")
    c.add_argument("out")
    c.add_argument("--epochs", type=int, default=40)
    c.add_argument("--hidden", type=int, default=24)
    c.add_argument("--lr", type=float, default=0.05)
    c.add_argument("--cdf-total", type=int, default=4096)
    c.add_argument("--seed", type=int, default=20260516)
    c.add_argument("--no-model-blob", action="store_true", help="do not store audit MLP in NCZ3")

    v = sub.add_parser("verify", help="verify NCZ3 against source BLF")
    v.add_argument("blf")
    v.add_argument("pack")

    d = sub.add_parser("decompress", help="decompress NCZ3 to canonical BLF")
    d.add_argument("pack")
    d.add_argument("out_blf")
    d.add_argument("--blf-compression", type=int, default=0)

    demo = sub.add_parser("demo", help="run end-to-end simulation comparing NCZ2 and NCZ3")
    demo.add_argument("--out-dir", default="/mnt/data/ncz3_demo")
    demo.add_argument("--duration-sec", type=float, default=20.0)
    demo.add_argument("--epochs", type=int, default=20)
    demo.add_argument("--hidden", type=int, default=16)
    demo.add_argument("--lr", type=float, default=0.05)
    demo.add_argument("--cdf-total", type=int, default=4096)
    demo.add_argument("--delta", type=float, default=0.25)

    args = ap.parse_args()
    if args.cmd == "compress":
        rep = compress_blf_float_safe(
            Path(args.blf), Path(args.out), epochs=args.epochs, hidden=args.hidden,
            lr=args.lr, cdf_total=args.cdf_total, seed=args.seed,
            keep_model_blob=not args.no_model_blob,
        )
    elif args.cmd == "verify":
        rep = verify_float_safe(Path(args.blf), Path(args.pack))
    elif args.cmd == "decompress":
        ids, t_us, payload, header = decode_pack_arrays_float_safe(Path(args.pack))
        frames = [STRUCTURED.CanFrame(int(t_us[i]), int(ids[i]), bytes(payload[i])) for i in range(len(ids))]
        STRUCTURED.SimpleBlfWriter(Path(args.out_blf), compression_level=args.blf_compression).write_frames(frames)
        rep = {"out_blf": args.out_blf, "frames": len(frames), "out_blf_bytes": Path(args.out_blf).stat().st_size, "fidelity": header["fidelity"]}
    elif args.cmd == "demo":
        rep = run_simulation(
            Path(args.out_dir), duration_sec=args.duration_sec, epochs=args.epochs,
            hidden=args.hidden, lr=args.lr, cdf_total=args.cdf_total, delta=args.delta,
        )
    else:
        raise AssertionError(args.cmd)
    print(json.dumps(rep, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
