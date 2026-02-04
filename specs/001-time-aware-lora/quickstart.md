# Quickstart Guide: Time-Aware Outlier Low-Rank Adaptation (TAO-LoRA)

**Feature**: 001-time-aware-lora
**Date**: 2026-02-04

## Overview

This guide provides a quick introduction to setting up and using the TAO-LoRA system for processing time-series sensor data with specialized quantization that preserves critical physical events.

## Prerequisites

- Python 3.11+
- PyTorch 2.0+
- CUDA-compatible GPU (for optimal performance)
- At least 8GB RAM (16GB recommended for larger models)
- Linux environment (tested on x86_64 and ARM64)

## Setup

### 1. Clone and Navigate to Project

```bash
git clone <repository-url>
cd TAO-LoRA
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy triton
# Install other project-specific dependencies
```

### 4. Install Project Package

```bash
pip install -e .
```

## Basic Usage

### 1. Import TAO-LoRA Components

```python
from src.model.talora_model import TALORAModel
from src.quantization.time_aware_quant import TimeAwareQuantizer
from src.router.dynamic_router import DynamicRouter
```

### 2. Initialize the Model

```python
# Load a pre-trained base model
base_model = load_pretrained_model("path/to/model")

# Initialize TAO-LoRA with the base model
talora_model = TALORAModel(
    base_model=base_model,
    quantization_bit_width=4,
    lora_rank=16,
    variance_threshold=0.5  # Adjust based on your data characteristics
)
```

### 3. Process Sensor Data

```python
import torch
import numpy as np

# Example sensor data (batch_size, sequence_length, features)
sensor_data = torch.randn(1, 100, 3)  # 1 sample, 100 timesteps, 3 features (e.g., x,y,z accelerometer)

# Process through TAO-LoRA
with torch.no_grad():
    output = talora_model(sensor_data)
    
print(f"Output shape: {output.shape}")
```

### 4. Dynamic Path Selection

```python
# The router automatically selects the appropriate path based on signal characteristics
router = DynamicRouter(variance_threshold=0.5)

# Route selection happens internally during forward pass
for batch in sensor_dataloader:
    # Model automatically chooses between INT4-only and dual-path based on input characteristics
    output = talora_model(batch)
```

## Advanced Configuration

### Custom Quantization Parameters

```python
from src.quantization.time_aware_quant import QuantizationParameters

quant_params = QuantizationParameters(
    bit_width=4,
    clipping_threshold=2.0,
    scale_factor=1.0,
    zero_point=0
)

talora_model = TALORAModel(
    base_model=base_model,
    quantization_params=quant_params,
    lora_rank=16
)
```

### Hardware-Specific Optimizations

```python
from src.hardware.optimization import HardwareOptimizer

# Apply hardware-specific optimizations for Jetson Orin Nano
optimizer = HardwareOptimizer(device_type="jetson_orin_nano")
optimized_model = optimizer.apply(talora_model)
```

## Testing the Implementation

### Run Unit Tests

```bash
pytest tests/unit/
```

### Run Integration Tests

```bash
pytest tests/integration/
```

### Benchmark Performance

```bash
# Memory footprint test
python -m tests.benchmark.test_memory_footprint

# Latency test
python -m tests.benchmark.test_latency
```

## Example: Processing Accelerometer Data

```python
import torch
import numpy as np
from src.model.talora_model import TALORAModel
from src.utils.sliding_window import SlidingWindowProcessor

# Simulate accelerometer data with a transient event (e.g., sudden movement)
def generate_test_data():
    # Steady state: low variance signal
    steady_signal = torch.randn(50, 3) * 0.1
    
    # Transient event: high variance spike
    transient_event = torch.randn(10, 3) * 2.0
    
    # Return to steady state
    return torch.cat([steady_signal, transient_event, steady_signal], dim=0)

# Load and configure the model
base_model = load_pretrained_model("path/to/model")  # Replace with actual model loading
talora_model = TALORAModel(base_model=base_model, lora_rank=16)

# Process the data
sensor_sequence = generate_test_data().unsqueeze(0)  # Add batch dimension

with torch.no_grad():
    # The model will automatically use the appropriate path based on signal characteristics
    output = talora_model(sensor_sequence)
    
    # The output should preserve the transient event while maintaining efficiency
    print(f"Processed {sensor_sequence.shape[1]} timesteps")
    print(f"Output shape: {output.shape}")
```

## Troubleshooting

### Common Issues

1. **CUDA Memory Errors**: Reduce batch size or use gradient checkpointing
2. **Path Selection Not Working**: Adjust variance_threshold parameter based on your data
3. **Accuracy Degradation**: Fine-tune the lambda coefficient in the loss function

### Performance Tips

- Use the sliding window mechanism for real-time processing
- Adjust the variance threshold based on your specific sensor data characteristics
- Monitor memory usage during dual-path activation
- Consider using Triton kernels for hardware-specific optimizations

## Next Steps

1. Explore the full API documentation
2. Review the benchmarking results for your specific use case
3. Fine-tune hyperparameters for your specific sensor modality
4. Test with your actual sensor data to validate performance