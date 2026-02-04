# TAO-LoRA (Time-Aware Outlier Low-Rank Adaptation)

TAO-LoRA is a novel approach to quantized neural networks for time-series sensor data that preserves critical physical events during quantization. The method distinguishes between steady-state and transient signals, applying different quantization strategies to each while maintaining computational efficiency on edge devices.

## Core Principles

1. **Time-Aware Quantization**: Quantization strategies account for temporal characteristics of sensor data, distinguishing between steady-state and transient signals
2. **Dynamic Path Routing**: Model inference dynamically routes through different computational paths based on signal characteristics (steady vs. transient)
3. **Hardware-Aware Optimization**: Implementation considers edge computing constraints, especially DRAM bandwidth limitations
4. **Event Preservation**: Quantization methods preserve critical physical events/transients that are essential for downstream tasks
5. **System-Level Integration**: Solutions address the full stack from mathematical modeling to hardware implementation

## Architecture

The architecture implements dual computational paths:
- **Steady Path (INT4)**: Optimized for steady-state signals with low-rank characteristics
- **Transient Path (FP16/BF16)**: Preserves high-amplitude, wide-frequency transient signals using LoRA as a "high-precision buffer"

Mathematical formulation:
```
Y = Q4(X)·Q4(W_base) + (X·A)·B·I(M_transient)
```

Where `Q4` represents INT4 quantization, `A,B` are LoRA matrices, and `I(M_transient)` is a dynamic gate based on transient detection. 
