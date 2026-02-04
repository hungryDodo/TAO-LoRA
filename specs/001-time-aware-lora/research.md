# Research: Time-Aware Outlier Low-Rank Adaptation (TAO-LoRA)

**Feature**: 001-time-aware-lora
**Date**: 2026-02-04
**Status**: Completed

## Overview

This document outlines the research and technical decisions made during the planning phase of the TAO-LoRA project. The goal is to implement a novel approach to quantized neural networks for time-series sensor data that preserves critical physical events during quantization.

## Technical Approach

### Core Innovation

TAO-LoRA introduces a dual-path architecture that separates steady-state and transient signal processing:
- **Steady Path (INT4)**: Optimized for steady-state signals with low-rank characteristics
- **Transient Path (FP16/BF16)**: Preserves high-amplitude, wide-frequency transient signals using LoRA as a "high-precision buffer"

Mathematical formulation:
```
Y = Q4(X)·Q4(W_base) + (X·A)·B·I(M_transient)
```

Where `Q4` represents INT4 quantization, `A,B` are LoRA matrices, and `I(M_transient)` is a dynamic gate based on transient detection.

### Key Technical Decisions

#### 1. Time-Aware Quantization Strategy
- Traditional W4A4 quantization fails for time-series sensor data because it clips important physical transients (events) like acceleration peaks during falls or R-waves in ECGs
- Steady-state activations show low-rank properties while transients exhibit high-amplitude, wide-frequency characteristics that must be preserved
- Solution: Use lower clipping thresholds for INT4 quantization to deliberately "cut off" outliers in steady-state, achieving higher resolution for normal signals

#### 2. Dynamic Path Routing Mechanism
- Sliding window mechanism detects high-frequency energy or second-order differences to trigger the high-precision path
- Variance-based detection: If variance < threshold, use INT4-only; if variance > threshold, activate both INT4 and FP16 paths
- CUDA stream parallelism enables simultaneous computation when both paths are active

#### 3. Event Preservation Method
- Loss function includes a term that forces LoRA to fit the "quantization residual" - the high-amplitude signal components that would otherwise be lost during quantization
- Formula: `L = L_task + λ||(W - Q4(W)) - AB^T||_F`
- This ensures that anomaly detection accuracy remains high even under aggressive quantization

#### 4. Hardware-Aware Optimizations
- Memory footprint optimization is critical as loading full FP16 LoRA parameters during frequent switching creates bottlenecks
- Zipper-style execution with bit-serial or integer-shift operators
- Triton kernels for fusing dequantization and addition steps in SRAM to minimize HBM access

## Architecture Components

### Quantization Module
- Implements time-aware quantization with adaptive thresholds
- Handles the separation of steady-state and transient signals
- Manages the INT4 quantization process

### Router Module
- Analyzes incoming signal characteristics
- Determines which computational path to use
- Implements variance-based detection algorithm

### Dual-Path Architecture
- Coordinates the two computational paths
- Manages switching between INT4-only and dual-path modes
- Ensures seamless integration of results

### Event Preservation Module
- Implements the LoRA-based outlier buffer
- Maintains critical physical events during quantization
- Applies the specialized loss function

## Competitive Analysis

### Comparison with Existing Approaches
- **QLoRA**: Focuses on general quantization but doesn't distinguish between steady and transient signals
- **LoftQ**: Emphasizes quantization-aware fine-tuning but lacks dynamic path routing
- **R-Sparse**: Addresses sparsity but doesn't consider temporal characteristics of sensor data

### Advantages of TAO-LoRA
- Preserves critical physical events that are essential for downstream tasks
- Optimizes for edge computing constraints, especially DRAM bandwidth
- Addresses the full stack from mathematical modeling to hardware implementation
- Dynamically adapts to signal characteristics rather than using static quantization

## Implementation Challenges

### Memory Management
- Loading full FP16 LoRA parameters during frequent switching creates DRAM bandwidth bottlenecks
- Solution: Implement Zipper-style execution and kernel fusion to minimize memory access

### Real-Time Processing
- Need to maintain >100Hz processing for typical sensor feeds
- Solution: Optimize critical path with Triton kernels and parallel computation

### Accuracy Preservation
- Ensuring >95% recall for anomaly detection compared to full-precision models
- Solution: Carefully tune the balance between quantization and LoRA compensation

## Validation Strategy

### Metrics
- **Accuracy (Anomaly Detection)**: Focus on recall for outlier detection
- **Edge Latency**: Measure jitter between steady and transient periods on Jetson Orin Nano
- **Memory Footprint**: Verify reduction compared to traditional mixed-precision approaches

### Testing Approach
- Synthetic sensor data with known transients for controlled testing
- Real-world sensor datasets for practical validation
- Comparative studies against QLoRA, LoftQ, and R-Sparse