<!-- SYNC IMPACT REPORT
Version Change: N/A -> 1.0.0
Modified Principles: N/A (New constitution)
Added Sections: All sections (New constitution)
Removed Sections: N/A
Templates Requiring Updates: ✅ Updated - plan-template.md, spec-template.md, tasks-template.md
Templates Requiring Updates: ✅ Updated - README.md
Follow-up TODOs: None
-->

# TAO-LoRA Constitution
<!-- Example: Spec Constitution, TaskFlow Constitution, etc. -->

## Core Principles

### Time-Aware Quantization
<!-- Example: I. Library-First -->
Quantization strategies must account for temporal characteristics of sensor data, distinguishing between steady-state and transient signals. Traditional W4A4 quantization fails for time-series sensor data because it clips important physical transients (events) like acceleration peaks during falls or R-waves in ECGs. Steady-state activations show low-rank properties while transients exhibit high-amplitude, wide-frequency characteristics that must be preserved.
<!-- Example: Every feature starts as a standalone library; Libraries must be self-contained, independently testable, documented; Clear purpose required - no organizational-only libraries -->

### Dynamic Path Routing
<!-- Example: II. CLI Interface -->
Model inference should dynamically route through different computational paths based on signal characteristics (steady vs. transient). The architecture implements dual paths: INT4 quantized computation for steady-state signals and high-precision (FP16/BF16) LoRA computation for transient detection. A sliding window mechanism detects high-frequency energy or second-order differences to trigger the high-precision path.
<!-- Example: Every library exposes functionality via CLI; Text in/out protocol: stdin/args → stdout, errors → stderr; Support JSON + human-readable formats -->

### Hardware-Aware Optimization
<!-- Example: III. Test-First (NON-NEGOTIABLE) -->
Implementation must consider edge computing constraints, especially DRAM bandwidth limitations. Memory footprint optimization is critical as loading full FP16 LoRA parameters during frequent switching creates bottlenecks. Solutions must implement Zipper-style execution with bit-serial or integer-shift operators and kernel fusion to minimize HBM access.
<!-- Example: TDD mandatory: Tests written → User approved → Tests fail → Then implement; Red-Green-Refactor cycle strictly enforced -->

### Event Preservation
<!-- Example: IV. Integration Testing -->
Quantization methods must preserve critical physical events/transients that are essential for downstream tasks. The loss function includes a term that forces LoRA to fit the "quantization residual" - the high-amplitude signal components that would otherwise be lost during quantization. This ensures that anomaly detection accuracy remains high even under aggressive quantization.
<!-- Example: Focus areas requiring integration tests: New library contract tests, Contract changes, Inter-service communication, Shared schemas -->

### System-Level Integration
<!-- Example: V. Observability, VI. Versioning & Breaking Changes, VII. Simplicity -->
Solutions must address the full stack from mathematical modeling to hardware implementation. The approach combines mathematical modeling (Y = Q4(X)·Q4(W_base) + (X·A)·B·I(M_transient)) with system optimization (CUDA stream parallelism, Triton kernels for SRAM fusion). Implementation spans Python/PyTorch framework layer and low-level kernel optimizations.
<!-- Example: Text I/O ensures debuggability; Structured logging required; Or: MAJOR.MINOR.BUILD format; Or: Start simple, YAGNI principles -->

## Technical Implementation Requirements
<!-- Example: Additional Constraints, Security Requirements, Performance Standards, etc. -->

The implementation must include: Time-Aware Router in forward pass, lightweight statistical computations (variance) for windowed inputs, CUDA stream parallelism for simultaneous INT4/F16 computation, Zipper-style bit-serial operators, and Triton kernels for fusing dequantization and addition steps in SRAM.
<!-- Example: Technology stack requirements, compliance standards, deployment policies, etc. -->

## Evaluation and Validation Framework
<!-- Example: Development Workflow, Review Process, Quality Gates, etc. -->

Validation must include: Accuracy evaluation for anomaly detection (focusing on outlier recall), Edge latency measurements on platforms like Jetson Orin Nano comparing steady vs. transient period jitter, and memory footprint verification to confirm elimination of traditional mixed-precision casting overhead. Competitor comparisons should include QLoRA, LoftQ, and R-Sparse.
<!-- Example: Code review requirements, testing gates, deployment approval process, etc. -->

## Governance
<!-- Example: Constitution supersedes all other practices; Amendments require documentation, approval, migration plan -->

All implementations must comply with hardware constraints and preserve critical signal features. Mathematical models must be validated against physical event detection requirements. Performance benchmarks must be conducted on target edge hardware. Any deviation from dual-path architecture or event preservation requirements must be documented and approved.
<!-- Example: All PRs/reviews must verify compliance; Complexity must be justified; Use [GUIDANCE_FILE] for runtime development guidance -->

**Version**: 1.0.0 | **Ratified**: 2026-02-04 | **Last Amended**: 2026-02-04
<!-- Example: Version: 2.1.1 | Ratified: 2025-06-13 | Last Amended: 2025-07-16 -->