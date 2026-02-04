# Data Model: Time-Aware Outlier Low-Rank Adaptation (TAO-LoRA)

**Feature**: 001-time-aware-lora
**Date**: 2026-02-04
**Status**: Completed

## Overview

This document describes the data structures and entities used in the TAO-LoRA system for processing time-series sensor data with specialized quantization that preserves critical physical events.

## Core Entities

### Sensor Data Stream
- **Description**: Time-series data from sensors (accelerometers, ECG, etc.) with temporal characteristics and potential physical transients
- **Attributes**:
  - `timestamp`: float - Time of measurement
  - `values`: array[float] - Sensor readings (can be multi-dimensional for multi-axis sensors)
  - `sensor_type`: str - Type of sensor (accelerometer, gyroscope, ECG, etc.)
  - `metadata`: dict - Additional sensor-specific metadata
- **Relationships**: Used by QuantizationPathSelector, EventPreservationModule

### Quantization Path Selector
- **Description**: Component that analyzes input windows and determines whether to use INT4-only or dual INT4/FP16 processing
- **Attributes**:
  - `window_size`: int - Size of the sliding window for analysis
  - `variance_threshold`: float - Threshold for determining path selection
  - `signal_characteristics`: dict - Current analysis of signal properties
- **Relationships**: Processes SensorDataStream, controls DualPathArchitecture

### Dual-Path Architecture
- **Description**: Computational framework with steady path (INT4) and transient path (FP16 LoRA) with dynamic gating
- **Attributes**:
  - `steady_path_model`: object - INT4 quantized model for steady-state processing
  - `transient_path_model`: object - FP16 LoRA model for transient processing
  - `gate_controller`: object - Logic for controlling path selection
  - `output_combiner`: object - Combines outputs from both paths
- **Relationships**: Uses QuantizationPathSelector for routing decisions, interacts with EventPreservationModule

### Event Preservation Module
- **Description**: Mechanism that ensures critical physical events are maintained during quantization
- **Attributes**:
  - `lora_matrices`: tuple(A, B) - LoRA matrices for outlier buffering
  - `residual_calculator`: object - Computes quantization residuals
  - `loss_coefficient`: float - Lambda coefficient for residual fitting
- **Relationships**: Works with DualPathArchitecture to preserve events, uses SensorDataStream for analysis

## Supporting Entities

### QuantizationParameters
- **Description**: Configuration for quantization processes
- **Attributes**:
  - `bit_width`: int - Bit width for quantization (typically 4 for INT4)
  - `clipping_threshold`: float - Threshold for quantization clipping
  - `scale_factor`: float - Scaling factor for quantized values
  - `zero_point`: int - Zero point for asymmetric quantization

### TransientDetector
- **Description**: Component that identifies transient events in sensor data
- **Attributes**:
  - `energy_threshold`: float - Threshold for high-frequency energy detection
  - `difference_order`: int - Order of differences for detection (typically 2nd order)
  - `window_analysis`: object - Sliding window analysis logic
- **Relationships**: Used by QuantizationPathSelector to detect transients

### HardwareOptimizationConfig
- **Description**: Configuration for hardware-specific optimizations
- **Attributes**:
  - `device_type`: str - Target hardware (e.g., "jetson_orin_nano")
  - `memory_bandwidth_limit`: float - Memory bandwidth constraint in GB/s
  - `compute_capability`: str - CUDA compute capability for GPU optimizations
  - `kernel_config`: dict - Triton kernel configuration parameters

## Data Flow

1. **Input**: SensorDataStream enters the system
2. **Analysis**: TransientDetector analyzes the stream to identify signal characteristics
3. **Routing**: QuantizationPathSelector determines which path(s) to use based on variance and other metrics
4. **Processing**: DualPathArchitecture processes the data through appropriate path(s)
5. **Preservation**: EventPreservationModule ensures critical events are maintained
6. **Output**: Combined result from both paths (if both used) is returned

## Relationships

```
SensorDataStream → QuantizationPathSelector → DualPathArchitecture ← EventPreservationModule
                    ↑                              ↓
         TransientDetector ←→ HardwareOptimizationConfig
```

## Serialization Formats

### Model Storage
- **Format**: PyTorch state dictionaries with custom serialization for quantized weights
- **Structure**: 
  - `base_weights`: Quantized base model weights (INT4)
  - `lora_weights`: LoRA adapter weights (FP16)
  - `config`: Model configuration including quantization parameters
  - `routing_params`: Parameters for path selection logic

### Configuration Files
- **Format**: YAML
- **Structure**:
  - `quantization`: Quantization parameters
  - `routing`: Path selection thresholds and parameters
  - `hardware`: Hardware-specific optimization settings
  - `training`: Training hyperparameters for LoRA adaptation