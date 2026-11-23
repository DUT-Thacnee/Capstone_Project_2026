from __future__ import annotations

import math
from typing import Optional, Tuple

from .commands import QuantDesc


def quantize_multiplier(real_multiplier: float) -> Tuple[int, int]: # kiểm tra để khác NaN, vô cùng và <0
    if not math.isfinite(real_multiplier) or real_multiplier < 0:
        raise ValueError(f"Invalid real_multiplier: {real_multiplier}")

    if real_multiplier == 0.0:
        return 0, 0

    significand, exponent = math.frexp(real_multiplier) # đưa số thực x về phần trị (significand) và phần mũ (exponent) sao cho x = significand * 2**exponent, với 0.5 <= |significand| < 1
    q31 = int(round(significand * (1 << 31)))

    if q31 == (1 << 31):
        q31 //= 2
        exponent += 1

    out_shift = 31 - exponent
    if out_shift < -128 or out_shift > 127:
        raise ValueError(f"out_shift out of int8 range: {out_shift}")

    if q31 < -(1 << 31) or q31 > (1 << 31) - 1:
        raise ValueError(f"out_multiplier out of int32 range: {q31}")

    return q31, int(out_shift)



def make_qdesc_from_scales(
    *,
    scale_in: float,
    scale_w: float,
    scale_out: float,
    out_zero_point: int = 0,
    act_min: int = -128,
    act_max: int = 127,
    in_zero_point: int = 0,
    w_zero_point: int = 0,
) -> QuantDesc:
    if scale_in <= 0 or scale_w <= 0 or scale_out <= 0:
        raise ValueError("Scales must be positive")

    real_multiplier = (scale_in * scale_w) / scale_out
    out_multiplier, out_shift = quantize_multiplier(real_multiplier)

    q = QuantDesc(
        out_multiplier=out_multiplier,
        out_shift=out_shift,
        in_zero_point=int(in_zero_point),
        w_zero_point=int(w_zero_point),
        out_zero_point=int(out_zero_point),
        act_min=int(act_min),
        act_max=int(act_max),
    )
    q.validate()
    return q


def extract_qparams_from_torch_quant_conv(module) -> Optional[dict]:
    try:
        import torch  # noqa: F401
    except Exception:
        return None

    scale_out = getattr(module, "scale", None)
    zero_point_out = getattr(module, "zero_point", None)

    try:
        wq = module.weight()
        if hasattr(wq, "q_per_tensor_scale"):
            scale_w = float(wq.q_per_tensor_scale())
        elif hasattr(wq, "q_per_channel_scales"):
            scale_w = float(wq.q_per_channel_scales().max().item())
        else:
            return None
    except Exception:
        return None

    if scale_out is None or zero_point_out is None:
        return None

    return {
        "scale_out": float(scale_out),
        "zero_point_out": int(zero_point_out),
        "scale_w": float(scale_w),
    }
