from __future__ import annotations

import glob
import os
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .commands import LayerCmd, axis_words_exact


@dataclass
class DdrMemory:
    w: object
    b: object
    skip: object


class AxisTensor:
    """DMA-backed NHWC int8 tensor packed into uint32 words."""

    def __init__(self, buf_u32: np.ndarray, *, h: int, w: int, c_padded: int, c_logical: Optional[int] = None):
        self.buf_u32 = buf_u32
        self.h = int(h)
        self.w = int(w)
        self.c_padded = int(c_padded)
        self.c_logical = int(c_logical) if c_logical is not None else int(c_padded)

        exp_words = axis_words_exact(self.h * self.w * self.c_padded, 4)
        if int(self.buf_u32.size) != exp_words:
            raise ValueError(f"Buffer words={self.buf_u32.size} != expected words={exp_words}")
        if self.buf_u32.dtype != np.uint32:
            raise TypeError("AxisTensor requires uint32 DMA buffers")
        if self.c_logical > self.c_padded:
            raise ValueError("c_logical must be <= c_padded")

    def view_nhwc_i8(self) -> np.ndarray:
        nbytes = self.h * self.w * self.c_padded
        return self.buf_u32.view(np.uint8)[:nbytes].view(np.int8).reshape(self.h, self.w, self.c_padded)

    def write_from_nhwc_i8(self, x_nhwc_i8: np.ndarray) -> None:
        if x_nhwc_i8.dtype != np.int8:
            raise TypeError("Expected int8 input")
        if x_nhwc_i8.shape != (self.h, self.w, self.c_logical):
            raise ValueError(f"Expected shape {(self.h, self.w, self.c_logical)}, got {x_nhwc_i8.shape}")
        v = self.view_nhwc_i8()
        v.fill(0)
        v[:, :, : self.c_logical] = x_nhwc_i8

    def read_to_nhwc_i8(self) -> np.ndarray:
        v = self.view_nhwc_i8()
        return v[:, :, : self.c_logical].copy()


@dataclass
class PingPong:
    ping: AxisTensor
    pong: AxisTensor

    def swap(self) -> None:
        self.ping, self.pong = self.pong, self.ping


class CnnAccelerator:
    def __init__(
        self,
        *,
        bitfile: str,
        core_name: str,
        dma_main_name: str,
        dma_residual_name: Optional[str] = None,
        download: bool = True,
    ) -> None:
        from pynq import Overlay

        try:
            self.overlay = Overlay(bitfile, download=download)
        except RuntimeError as exc:
            if "Could not open device with index" in str(exc):
                bitfile_exists = os.path.exists(bitfile)
                dri_devices = glob.glob("/dev/dri/*")
                raise RuntimeError(
                    "PYNQ/XRT cannot open FPGA device index 0 before loading the bitstream.\n"
                    "This is a board/PYNQ/XRT/permission issue, not an IP-name, DMA, or weight-file issue.\n"
                    f"bitfile={bitfile!r}, bitfile_exists={bitfile_exists}, "
                    f"/dev/dri={dri_devices}, XILINX_XRT={os.environ.get('XILINX_XRT')!r}\n"
                    "On the board, check: `ls -l /dev/dri`, `lsmod | grep -E 'zocl|xrt'`, "
                    "`xbutil examine`, and try `sudo -E python3 /home/xilinx/imp_system.py`."
                ) from exc
            raise
        self.core = self._get_ip(self.overlay, core_name)
        self.dma_main = self._get_ip(self.overlay, dma_main_name)
        self.dma_residual = self._get_ip(self.overlay, dma_residual_name) if dma_residual_name else None
        self._ddr: Optional[DdrMemory] = None

    @staticmethod
    def _get_ip(overlay, name: Optional[str]):
        if name is None:
            return None
        if hasattr(overlay, name):
            return getattr(overlay, name)
        if hasattr(overlay, "ip_dict") and name in overlay.ip_dict:
            return overlay.ip_dict[name]
        raise KeyError(f"IP '{name}' not found in overlay")

    def allocate_ddr_tables(self, *, w_bytes: int, b_bytes: int, skip_bytes: int = 0) -> DdrMemory:
        from pynq import allocate

        def alloc_u64(byte_count: int):
            words64 = (int(byte_count) + 7) // 8
            buf = allocate(shape=(words64,), dtype=np.uint64)
            buf.view(np.uint8)[: int(byte_count)] = 0
            buf.flush()
            return buf

        w = alloc_u64(w_bytes)
        b = alloc_u64(b_bytes)
        skip = alloc_u64(skip_bytes if skip_bytes else 8)

        self._ddr = DdrMemory(w=w, b=b, skip=skip)

        self._write_ptr("gmem_w", int(w.physical_address))
        self._write_ptr("gmem_b", int(b.physical_address))
        self._write_ptr("gmem_skip", int(skip.physical_address))

        return self._ddr

    def load_ddr_bytes(self, *, w_data: bytes, b_data: bytes, skip_data: Optional[bytes] = None) -> None:
        if self._ddr is None:
            raise RuntimeError("Call allocate_ddr_tables() first")
        self._ddr.w.view(np.uint8)[: len(w_data)] = np.frombuffer(w_data, dtype=np.uint8)
        self._ddr.b.view(np.uint8)[: len(b_data)] = np.frombuffer(b_data, dtype=np.uint8)
        if skip_data is not None:
            self._ddr.skip.view(np.uint8)[: len(skip_data)] = np.frombuffer(skip_data, dtype=np.uint8)
        self._ddr.w.flush()
        self._ddr.b.flush()
        self._ddr.skip.flush()

    @staticmethod
    def allocate_axis_words(words: int, *, dtype=np.uint32):
        from pynq import allocate

        return allocate(shape=(int(words),), dtype=dtype)

    def make_axis_tensor(self, *, h: int, w: int, c_padded: int, c_logical: Optional[int] = None) -> AxisTensor:
        words = axis_words_exact(int(h) * int(w) * int(c_padded), 4)
        buf = self.allocate_axis_words(words)
        return AxisTensor(buf, h=h, w=w, c_padded=c_padded, c_logical=c_logical)

    def _write_ptr(self, reg: str, addr: int) -> None:
        rm = getattr(self.core, "register_map", None)
        if rm is not None and hasattr(rm, reg):
            setattr(rm, reg, int(addr))
            return

        lo_name = f"{reg}_1"
        hi_name = f"{reg}_2"
        if rm is not None and hasattr(rm, lo_name) and hasattr(rm, hi_name):
            addr = int(addr) & 0xFFFFFFFFFFFFFFFF
            setattr(rm, lo_name, addr & 0xFFFFFFFF)
            setattr(rm, hi_name, (addr >> 32) & 0xFFFFFFFF)
            return

        available = [n for n in dir(rm) if not n.startswith("_")] if rm is not None else []
        raise AttributeError(
            f"Pointer register '{reg}' not found in register_map. "
            f"Expected '{reg}' or split registers '{lo_name}'/'{hi_name}'. "
            f"Available (sample): {available[:50]}"
        )

    def write_cmd(self, cmd: LayerCmd, *, prefix: str = "cmd") -> None:
        rm = getattr(self.core, "register_map", None)
        if rm is None:
            raise RuntimeError("Core does not expose register_map")

        if hasattr(cmd, "as_register_words") and hasattr(rm, f"{prefix}_1"):
            words = cmd.as_register_words()
            missing_seq = [f"{prefix}_{i}" for i in range(1, len(words) + 1) if not hasattr(rm, f"{prefix}_{i}")]
            if missing_seq:
                raise KeyError(
                    "Command struct register sequence is incomplete.\n"
                    f"Missing (first 10): {missing_seq[:10]}"
                )
            for i, value in enumerate(words, start=1):
                setattr(rm, f"{prefix}_{i}", int(value))
            return

        fields = cmd.as_flat_register_dict(prefix=prefix)
        missing = []
        for k, v in fields.items():
            if hasattr(rm, k):
                setattr(rm, k, int(v))
            else:
                missing.append(k)

        if missing:
            available = [n for n in dir(rm) if not n.startswith("_")]
            raise KeyError(
                "Some cmd fields were not found in register_map.\n"
                f"Missing (first 10): {missing[:10]}\n"
                f"Available (sample): {available[:50]}"
            )

    def start(self) -> None:
        self.core.write(0x00, 0x01)

    def wait_done(self, timeout_s: float = 5.0) -> None:
        t0 = time.time()
        while True:
            ctrl = int(self.core.read(0x00))
            if (ctrl >> 1) & 0x1:
                return
            if (time.time() - t0) > timeout_s:
                raise TimeoutError("HLS core timeout")
            time.sleep(0.0001)

    def run_layer(
        self,
        *,
        cmd: LayerCmd,
        in_buf_u32: object,
        out_buf_u32: object,
        residual_buf_u32: Optional[object] = None,
        axis_bytes: int = 4,
        cmd_prefix: str = "cmd",
        timeout_s: float = 5.0,
    ) -> None:
        cmd.validate_axis_counts(axis_bytes=axis_bytes)

        if int(in_buf_u32.size) != int(cmd.in_words):
            raise ValueError(f"in_buf_u32 words={in_buf_u32.size} != cmd.in_words={cmd.in_words}")
        if int(out_buf_u32.size) != int(cmd.out_words):
            raise ValueError(f"out_buf_u32 words={out_buf_u32.size} != cmd.out_words={cmd.out_words}")

        if cmd.requires_residual_stream():
            if self.dma_residual is None:
                raise RuntimeError("Residual stream requested but no residual DMA configured")
            if residual_buf_u32 is None:
                raise ValueError("Residual stream requested but residual_buf_u32 is None")
            if int(residual_buf_u32.size) != int(cmd.out_words):
                raise ValueError("residual_buf_u32 words != cmd.out_words")

        if hasattr(in_buf_u32, "flush"):
            in_buf_u32.flush()
        if cmd.requires_residual_stream() and hasattr(residual_buf_u32, "flush"):
            residual_buf_u32.flush()

        self.write_cmd(cmd, prefix=cmd_prefix)

        self.dma_main.recvchannel.transfer(out_buf_u32)
        if cmd.requires_residual_stream():
            self.dma_residual.sendchannel.transfer(residual_buf_u32)
        self.dma_main.sendchannel.transfer(in_buf_u32)

        self.start()

        t0 = time.time()
        self.dma_main.sendchannel.wait()
        if cmd.requires_residual_stream():
            self.dma_residual.sendchannel.wait()
        self.dma_main.recvchannel.wait()

        if hasattr(out_buf_u32, "invalidate"):
            out_buf_u32.invalidate()

        if (time.time() - t0) > timeout_s:
            raise TimeoutError("DMA wait exceeded timeout")

    def run_layer_tensor(
        self,
        *,
        cmd: LayerCmd,
        in_tensor: AxisTensor,
        out_tensor: AxisTensor,
        residual_tensor: Optional[AxisTensor] = None,
        **kwargs,
    ) -> None:
        self.run_layer(
            cmd=cmd,
            in_buf_u32=in_tensor.buf_u32,
            out_buf_u32=out_tensor.buf_u32,
            residual_buf_u32=(residual_tensor.buf_u32 if residual_tensor is not None else None),
            **kwargs,
        )
