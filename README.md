# Energy-Aware Multi-Mode Reconfigurable Accelerator for CNN (MobileNetV2) on ZCU106 (xczu7ev-ffvc1156-2-e)

## Project Abstract
This thesis-grade project presents an energy-aware, command-driven CNN accelerator for MobileNetV2 on the Xilinx ZCU106 (xczu7ev-ffvc1156-2-e). The core innovation is a **single reconfigurable engine** that executes multiple CNN primitives (1x1 pointwise, 3x3 depthwise, global average pooling) under a unified runtime command interface. By moving layer configuration into an AXI-Lite `cnn_layer_cmd_t` structure and decoupling data movement from computation with **DATAFLOW**, the design achieves deterministic streaming, eliminates deadlock conditions, and fits within stringent DSP/BRAM constraints. On the software side, a PYNQ-based runtime orchestrates DMA transfers with **zero-copy ping-pong buffers**, performs channel alignment for SIMD efficiency, and bridges modern INT8 quantization into fixed-point hardware parameters.

## Hardware Architecture (Vitis HLS)
Key sources:
- `Hardware/HLS/src_hw/parameters.h`
- `Hardware/HLS/src_hw/hardware_functions.cpp`

### Command-Driven Single Engine
Traditional CNN accelerators often hardcode each layer in a separate kernel, leading to excessive resource duplication and rigid dataflow. We replace that approach with a **single MultiMode_Compute engine** driven by a compact command structure (`cnn_layer_cmd_t`). The command specifies:
- **Opcode** (CONV2D, AVGPOOL, NOP)
- **Tensor descriptors** (NHWC dimensions, element size)
- **Convolution parameters** (kernel, stride, groups)
- **Quantization parameters** (multiplier, shift, zero point, activation clamp)
- **Sparse metadata** (RLE offsets and nonzero count)

This architecture turns CNN layers into **software-defined transactions**, enabling the same silicon to execute the full MobileNetV2 topology without hardware duplication or re-synthesis.

### DATAFLOW Paradigm
The kernel is organized into pipeline stages: **Read Input -> Read Weights/Bias -> Compute -> Write Output**. Each stage consumes and produces a fixed number of AXI-Stream words derived from `in_words` and `out_words` in the command. This yields two critical properties:
1. **Deadlock elimination**: every producer has a bounded consumer, and all streams are deterministically drained.
2. **Composability**: runtime can dispatch arbitrary layer sequences without modifying the kernel.

The design also enforces a strict `TLAST` policy (asserted on the final output word), which prevents DMA hangs and makes software/hardware synchronization provably deterministic.

### SIMD Unrolling and NHWC Data Layout
Compute throughput is controlled by two tunable unroll factors: `CNN_ENGINE_SIMD` (channels per vector lane) and `CNN_ENGINE_PE` (parallel output accumulators). This **SIMD x PE** structure replaces monolithic unrolling and allows a precise trade-off between performance and resource use. In practice:
- The unroll factors are chosen to **fit ZCU106 DSP/BRAM budgets** while keeping the inner MAC loops fully pipelined.
- **NHWC** layout is mandatory because it guarantees contiguous channel vectors, which maximizes AXI burst efficiency and aligns naturally with SIMD lane grouping.

### Energy-Aware Sparsity (RLE)
Sparse weights are stored using **Run-Length Encoding** with `(distance, value)` pairs. Each distance encodes how many zeros to skip before the next nonzero value. The hardware reconstructs the dense stream on the fly, which produces two advantages:
- **Bandwidth reduction**: fewer bytes are read from DDR when weights are sparse.
- **Dynamic power reduction**: skipping zero-weight regions reduces unnecessary MAC switching (effectively behaving like clock gating for inactive DSPs).

This sparsity path is orthogonal to the core compute pipeline, allowing dense and sparse execution under a unified interface.

## Software Orchestration (Python/PYNQ)
Key runtime files:
- `Python/pynq_cnn/accelerator.py`
- `Python/pynq_cnn/commands.py`
- `Python/pynq_cnn/packing.py`
- `Python/pynq_cnn/quant.py`
- `Python/pynq_cnn/mobilenetv2.py`

### Zero-Copy Ping-Pong Buffering
The PYNQ runtime allocates physically contiguous DMA buffers via `pynq.allocate`. Two buffers are reused in a **ping-pong** pattern so the output of layer *N* becomes the input of layer *N+1* without CPU copying. This yields:
- Lower CPU overhead and less cache pollution.
- Predictable DMA throughput, which is essential for deterministic streaming.

### Software-Hardware Alignment
The hardware requires channels to be a multiple of both the AXI packing width (4 bytes per word) and SIMD width. The software dynamically pads channel dimensions and packs NHWC tensors into 32-bit words. This alignment step ensures:
- No cross-word channel fragmentation.
- SIMD lanes remain fully utilized.

### Quantization Bridge
Modern PyTorch INT8 quantization produces floating-point scales and zero points. The runtime converts those to fixed-point parameters for `cnn_quant_desc_t`:
- `out_multiplier` and `out_shift` implement a fixed-point approximation of the real multiplier.
- `out_zero_point` and activation limits enforce the output range.

Residual fusion in MobileNetV2 is only valid when the residual tensor and projection output share the **same quantization domain**, a condition explicitly enforced by the software scheduler.

## End-to-End Workflow
```
Offline Packer (PC)
  -> INT8 quantization + RLE compression
  -> weights_and_biases.bin + metadata.json

Hardware Synthesis (Vitis HLS + Vivado)
  -> MultiMode kernel IP
  -> accelerator.bit + accelerator.hwh

Online Orchestration (PYNQ)
  -> DMA streams + AXI-Lite commands
  -> ping-pong activations + residual stream
```

This workflow is deliberately modular: **offline compression**, **hardware generation**, and **runtime orchestration** are decoupled but precisely aligned through the shared command format.
