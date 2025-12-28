# ⚡ Optimized Whisper ASR (CUDA Graphs + Flash Attention)

This project explores low-level optimizations for the OpenAI Whisper model (Small) on NVIDIA GPUs. Moving beyond standard Hugging Face implementations, this project applies **Flash Attention**, **CUDA Graphs**, and **INT8 Quantization** to minimize latency and memory bandwidth bottlenecks.

## 🎯 The Optimization Journey (The "4 Scopes")

The project followed a strict performance engineering roadmap based on profiling data from **NVIDIA Nsight Systems**.

### 🔍 Phase 1: Profiling & Bottleneck Analysis
Initial profiling revealed two distinct gaps:
1.  **Launch Gaps:** 10-20µs gaps between kernels where the GPU was idle (CPU-bound).
2.  **Memory Gaps:** Extended stalls during Attention and Linear layers (Memory-bound).

### 🛠 Phase 2: Implementation & Pivots

| Scope | Optimization | Status | Outcome |
| :--- | :--- | :--- | :--- |
| **1** | **Flash Attention (v2)** | ✅ Done | Replaced vanilla attention with `SDPA`, significantly reducing memory I/O. |
| **2** | **CUDA Graphs** | ✅ Done | Used `torch.compile(mode="reduce-overhead")` to capture the graph, eliminating CPU launch overhead. |
| **3** | **Custom C++ Kernel** | ❌ Skipped(Failed due to Cuda Graphs implementation) | Wrote a raw CUDA `fused_topk` kernel to replace PyTorch's decoder search. **Outcome:** While 10x faster in isolation, integration with CUDA Graphs caused memory pointer conflicts (Static vs. Dynamic memory), leading to a strategic pivot. |
| **4** | **Speculative Decoding** | ❌ Skippe(Failed due to Cuda Graphs implementation) | Determined architectural incompatibility with CUDA Graphs due to dynamic control flow requirements. |

## 📂 Project Structure

This repository documents the evolution of the optimization:

* `asr16.py`: The baseline FP16 implementation.
* `asr16toflash.py`: Scope 1 implementation (Flash Attention).
* `asr16toflashcudagraphs.py`: Scope 2 implementation (CUDA Graphs).
* **`asr_optimized.py`**: **The Final Production Pipeline** (combines Flash + Graphs + INT8).
* `fused_topk.cu`: The experimental C++ kernel (Scope 3 research artifact).
* `whisper_profile.nsys-rep`: Nsight Systems profiling data.

## 🚀 How to Run the Optimized Model

### 1. Requirements
```bash
pip install torch transformers librosa torchao
