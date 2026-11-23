#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""TitaNet speaker-embedding (.nemo) -> GGUF converter.

Converts NeMo `EncDecSpeakerLabelModel` checkpoints with a `ConvASREncoder`
(Jasper-style, separable convolutions with squeeze-excitation) and a
`SpeakerDecoder` with attentive statistics pooling, e.g.
`nvidia/speakerverification_en_titanet_large`, into the GGUF that
`src/asr/spk/titanet_model.cpp` loads.

Output convention:

    * `general.architecture = "speaker"`, hparams under `speaker.*`.
    * All BatchNorms that directly follow a linear operation are folded into
      it (encoder conv-BN pairs, residual conv-BN pairs, the embedding BN in
      front of the final 1x1 conv). The BN in the attention TDNN sits behind a
      ReLU and is stored as a per-channel scale/shift instead.
    * A depthwise conv with kernel 1 (the epilog block) is a per-channel scale
      and is folded into the following pointwise conv.
    * Pointwise (k=1) weights are stored 2D (out, in) so the runtime's Conv1D
      dispatches them through ggml_mul_mat; depthwise weights keep the
      PyTorch-native (C, 1, k) layout.
    * The classification head (`decoder.final`) is training-only and dropped.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import yaml
from gguf import GGMLQuantizationType, GGUFWriter

from .source import extracted_checkpoint, find_checkpoint_files, load_state_dict

ARCH = "speaker"

# NeMo's JasperBlock builds every conv-BN pair with BatchNorm1d(eps=1e-3); the
# decoder's BatchNorms use the PyTorch default.
JASPER_BN_EPS = 1e-3
DECODER_BN_EPS = 1e-5

WEIGHT_TYPES = {
    "f32": (GGMLQuantizationType.F32, 0),
    "f16": (GGMLQuantizationType.F16, 1),
}


def _np(t) -> np.ndarray:
    return t.detach().cpu().float().numpy().astype(np.float64)


def _bn_affine(sd: dict, prefix: str, eps: float) -> tuple[np.ndarray, np.ndarray]:
    """Return (scale, shift) with bn(x) == x * scale + shift."""
    gamma = _np(sd[prefix + ".weight"])
    beta = _np(sd[prefix + ".bias"])
    mean = _np(sd[prefix + ".running_mean"])
    var = _np(sd[prefix + ".running_var"])
    scale = gamma / np.sqrt(var + eps)
    return scale, beta - mean * scale


def _block_layers(sd: dict, block: int) -> dict[int, str]:
    """Map mconv layer index -> kind ("dw", "pw", "bn", "se") for one block."""
    kinds: dict[int, str] = {}
    pattern = re.compile(rf"^encoder\.encoder\.{block}\.mconv\.(\d+)\.(.+)$")
    for key, value in sd.items():
        m = pattern.match(key)
        if not m:
            continue
        index, rest = int(m.group(1)), m.group(2)
        if rest == "conv.weight":
            kinds[index] = "dw" if value.shape[1] == 1 else "pw"
        elif rest == "running_mean":
            kinds[index] = "bn"
        elif rest.startswith("fc."):
            kinds[index] = "se"
    return dict(sorted(kinds.items()))


def convert(nemo_path: Path, out_path: Path, weight_type: str = "f32") -> None:
    if weight_type not in WEIGHT_TYPES:
        raise ValueError(f"speaker models support --outtype {', '.join(WEIGHT_TYPES)}")
    pw_qtype, file_type = WEIGHT_TYPES[weight_type]

    print(f"[convert] reading {nemo_path}")
    with extracted_checkpoint(nemo_path, "speaker-convert-") as tmp:
        files = find_checkpoint_files(tmp)
        if files["config"] is None or files["weights"] is None:
            raise RuntimeError(f"Missing artifacts in {nemo_path}: {files}")
        with open(files["config"], "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        sd = load_state_dict(files["weights"])

    enc_cfg = cfg["encoder"]
    dec_cfg = cfg["decoder"]
    pp_cfg = cfg["preprocessor"]
    if "convasrencoder" not in str(enc_cfg.get("_target_", "")).lower():
        raise RuntimeError("speaker conversion expects a ConvASREncoder (TitaNet)")
    if str(dec_cfg.get("pool_mode", "")) != "attention":
        raise RuntimeError("speaker conversion expects attentive statistics pooling")
    if str(enc_cfg.get("activation", "relu")) != "relu":
        raise RuntimeError("speaker conversion supports ReLU encoders only")
    if str(pp_cfg.get("normalize", "per_feature")) != "per_feature":
        raise RuntimeError("speaker conversion expects per_feature normalization")

    blocks = enc_cfg["jasper"]
    feat_in = int(enc_cfg["feat_in"])
    emb_sizes = dec_cfg["emb_sizes"]
    emb_dim = int(emb_sizes[0] if isinstance(emb_sizes, list) else emb_sizes)

    gw = GGUFWriter(str(out_path), arch=ARCH)
    gw.add_architecture()
    gw.add_string("general.name", cfg.get("name", out_path.stem))
    gw.add_uint32("general.file_type", file_type)

    gw.add_uint32("speaker.preprocessor.sample_rate", int(pp_cfg.get("sample_rate", 16000)))
    gw.add_float32("speaker.preprocessor.window_size", float(pp_cfg.get("window_size", 0.025)))
    gw.add_float32("speaker.preprocessor.window_stride", float(pp_cfg.get("window_stride", 0.01)))
    gw.add_uint32("speaker.preprocessor.n_fft", int(pp_cfg.get("n_fft", 512)))
    gw.add_uint32("speaker.preprocessor.features", int(pp_cfg.get("features", feat_in)))
    gw.add_float32("speaker.preprocessor.preemph", float(pp_cfg.get("preemph", 0.97)))
    gw.add_float32(
        "speaker.preprocessor.log_zero_guard", float(pp_cfg.get("log_zero_guard_value", 2**-24))
    )

    gw.add_uint32("speaker.encoder.feat_in", feat_in)
    gw.add_uint32("speaker.encoder.n_blocks", len(blocks))
    gw.add_uint32("speaker.embedding_dim", emb_dim)

    def emit(name: str, arr: np.ndarray, qtype=GGMLQuantizationType.F32) -> None:
        data = np.ascontiguousarray(arr.astype(np.float32))
        if qtype == GGMLQuantizationType.F16:
            data = data.astype(np.float16)
        gw.add_tensor(name, data, raw_dtype=qtype)
        print(f"  {name:36s} {tuple(arr.shape)} {qtype.name}")

    fb = _np(sd["preprocessor.featurizer.fb"]).reshape(-1, _np(sd["preprocessor.featurizer.fb"]).shape[-1])
    emit("preprocessor.fb", fb)

    filters, repeats, kernels, residuals = [], [], [], []
    se_reduction: int | None = None
    for b, block in enumerate(blocks):
        if not block.get("separable", False) or not block.get("se", False):
            raise RuntimeError(f"block {b}: expected a separable block with squeeze-excitation")
        if int(block.get("se_context_size", -1)) != -1:
            raise RuntimeError(f"block {b}: only global squeeze-excitation is supported")
        if any(int(s) != 1 for s in block.get("stride", [1])) or any(
            int(d) != 1 for d in block.get("dilation", [1])
        ):
            raise RuntimeError(f"block {b}: only stride 1 / dilation 1 is supported")
        out_channels = int(block["filters"])
        repeat = int(block["repeat"])
        kernel = int(block["kernel"][0])
        residual = bool(block.get("residual", False))
        if kernel % 2 != 1:
            raise RuntimeError(f"block {b}: even kernels are not supported")

        layers = _block_layers(sd, b)
        seq = list(layers.items())
        convs = [(i, k) for i, k in seq if k in ("dw", "pw")]
        bns = [i for i, k in seq if k == "bn"]
        se = [i for i, k in seq if k == "se"]
        if len(bns) != repeat or len(se) != 1 or len(convs) != 2 * repeat:
            raise RuntimeError(f"block {b}: unexpected layer layout {layers}")

        for r in range(repeat):
            dw_index, dw_kind = convs[2 * r]
            pw_index, pw_kind = convs[2 * r + 1]
            if dw_kind != "dw" and kernel != 1:
                raise RuntimeError(f"block {b}.{r}: expected depthwise conv at mconv.{dw_index}")
            dw = _np(sd[f"encoder.encoder.{b}.mconv.{dw_index}.conv.weight"])  # (C, 1, k)
            pw = _np(sd[f"encoder.encoder.{b}.mconv.{pw_index}.conv.weight"])[:, :, 0]  # (O, C)
            scale, shift = _bn_affine(sd, f"encoder.encoder.{b}.mconv.{bns[r]}", JASPER_BN_EPS)
            pw_bias = shift.copy()
            if kernel == 1:
                pw = pw * dw[:, 0, 0][None, :]  # fold the per-channel scale
            else:
                emit(f"encoder.{b}.rep.{r}.dw.weight", dw)
            emit(f"encoder.{b}.rep.{r}.pw.weight", pw * scale[:, None], pw_qtype)
            emit(f"encoder.{b}.rep.{r}.pw.bias", pw_bias)

        prefix = f"encoder.encoder.{b}.mconv.{se[0]}"
        reduction = out_channels // int(sd[prefix + ".fc.0.weight"].shape[0])
        if se_reduction not in (None, reduction):
            raise RuntimeError(f"block {b}: squeeze-excitation ratio differs between blocks")
        se_reduction = reduction
        emit(f"encoder.{b}.se.fc1.weight", _np(sd[prefix + ".fc.0.weight"]))
        emit(f"encoder.{b}.se.fc2.weight", _np(sd[prefix + ".fc.2.weight"]))

        if residual:
            res = _np(sd[f"encoder.encoder.{b}.res.0.0.conv.weight"])[:, :, 0]
            scale, shift = _bn_affine(sd, f"encoder.encoder.{b}.res.0.1", JASPER_BN_EPS)
            emit(f"encoder.{b}.res.weight", res * scale[:, None], pw_qtype)
            emit(f"encoder.{b}.res.bias", shift)

        filters.append(out_channels)
        repeats.append(repeat)
        kernels.append(kernel)
        residuals.append(1 if residual else 0)

    gw.add_array("speaker.encoder.filters", filters)
    gw.add_array("speaker.encoder.repeat", repeats)
    gw.add_array("speaker.encoder.kernel", kernels)
    gw.add_array("speaker.encoder.residual", residuals)
    gw.add_uint32("speaker.encoder.se_reduction", se_reduction)

    # Attentive statistics pooling: TDNN(3C -> A) = conv -> ReLU -> BN, tanh,
    # conv(A -> C), softmax over time.
    tdnn = _np(sd["decoder._pooling.attention_layer.0.conv_layer.weight"])[:, :, 0]
    emit("pool.tdnn.weight", tdnn, pw_qtype)
    emit("pool.tdnn.bias", _np(sd["decoder._pooling.attention_layer.0.conv_layer.bias"]))
    scale, shift = _bn_affine(sd, "decoder._pooling.attention_layer.0.bn", DECODER_BN_EPS)
    emit("pool.tdnn.bn_scale", scale)
    emit("pool.tdnn.bn_shift", shift)
    emit(
        "pool.attn.weight",
        _np(sd["decoder._pooling.attention_layer.2.weight"])[:, :, 0],
        pw_qtype,
    )
    emit("pool.attn.bias", _np(sd["decoder._pooling.attention_layer.2.bias"]))
    gw.add_uint32("speaker.pool.attention_channels", int(tdnn.shape[0]))

    # Embedding: BN(2C) followed by a 1x1 conv to emb_dim, folded into one
    # affine map W' = W * scale, b' = b + W @ shift.
    scale, shift = _bn_affine(sd, "decoder.emb_layers.0.0", DECODER_BN_EPS)
    w = _np(sd["decoder.emb_layers.0.1.weight"])[:, :, 0]
    bias = _np(sd["decoder.emb_layers.0.1.bias"])
    emit("emb.weight", w * scale[None, :], pw_qtype)
    emit("emb.bias", bias + w @ shift)

    gw.write_header_to_file()
    gw.write_kv_data_to_file()
    gw.write_tensors_to_file()
    gw.close()
    print(f"\nDone. Wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB)")
