# MobileFineTuner (GenAI)

**A High-Performance, Mobile-Native C++17 Framework for On-Device LLM Fine-Tuning, Real-Time Monitoring, and Embedded AI Deployment**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-C%2B%2B17%20%7C%20Android%20Native%20%7C%20Web%20Dashboard-green.svg)]()
[![CMake](https://img.shields.io/badge/Build-CMake%203.10%2B-orange.svg)]()
[![Android NDK](https://img.shields.io/badge/Android%20NDK-26.1%2B%20(API%2035)-darkgreen.svg)]()
[![Paper](https://img.shields.io/badge/Sensys%202027-Accepted-brightgreen.svg)]()
[![arXiv](https://img.shields.io/badge/arXiv-2512.08211-b31b1b.svg)](https://arxiv.org/abs/2512.08211)

---

## News & Highlights

- **[2026-09-01]** Our research paper **"MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications"** has been accepted by **Sensys 2027**!
- **[2026-07-10]** **MobileRLHF** introduced, bringing reinforcement learning from human feedback and preference optimization to edge devices.
- **[2025-12-01]** Open-sourced **MobileFineTuner**, establishing an end-to-end native C++ training core, Android SDK (`mft-sdk`), Android Visualizer App, and Web Dashboard.

<div align="center">
  <img src="logo.jpg" alt="Duke University & Duke Kunshan University" width="280"/>
</div>

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Key Innovations & Technical Capabilities](#key-innovations--technical-capabilities)
3. [End-to-End System Architecture](#end-to-end-system-architecture)
4. [Repository Structure](#repository-structure)
5. [Supported Model Architectures](#supported-model-architectures)
6. [Core C++ Engine (`operator/`)](#core-c-engine-operator)
7. [Android Ecosystem (`android-visualizer/`)](#android-ecosystem-android-visualizer)
   - [MFT Android SDK (`mft-sdk`)](#mft-android-sdk-mft-sdk)
   - [Android Visualizer Dashboard App](#android-visualizer-dashboard-app)
   - [Standalone Consumer Sample (`sdk-sample`)](#standalone-consumer-sample-sdk-sample)
8. [Interactive Web UI Dashboard (`web_ui/`)](#interactive-web-ui-dashboard-web_ui)
9. [Runnable Model Targets (`examples/`)](#runnable-model-targets-examples)
10. [Asset Contract & Data Management (`docs/` & `scripts/`)](#asset-contract--data-management)
11. [Build & Installation Guide](#build--installation-guide)
12. [Quick Start & Training Execution](#quick-start--training-execution)
    - [Synthetic Smoke Testing](#1-synthetic-smoke-testing-5-models)
    - [Real Asset Fine-Tuning (WikiText-2 & MMLU)](#2-real-asset-fine-tuning)
    - [Unified C++ Auto API](#3-unified-c-auto-api)
13. [Evaluation & Device Telemetry](#evaluation--device-telemetry)
14. [Maintained Scripts Catalog](#maintained-scripts-catalog)
15. [Citation & Academic Credit](#citation--academic-credit)
16. [License & Acknowledgments](#license--acknowledgments)

---

## Executive Summary

**MobileFineTuner** is an industrial-grade, privacy-preserving C++17 framework designed specifically for on-device parameter-efficient fine-tuning (PEFT/LoRA) and full fine-tuning (Full-FT) of Large Language Models (LLMs) on resource-constrained edge hardware.

Traditional LLM fine-tuning relies on heavy Python environments (PyTorch, Hugging Face Transformers, DeepSpeed) that require gigabytes of RAM, huge virtual environments, and substantial runtime overhead. MobileFineTuner replaces this with:

1. **A Lean, Pure C++17 Core**: Zero Python dependencies in the training execution path. Custom autograd, tensor ops, memory arenas, and native tokenizers.
2. **First-Class Mobile & Embedded Integration**: Android SDK (`mft-sdk`) packaged as an AAR with a native JNI bridge for direct in-app training.
3. **Dual Telemetry & Visualization Platforms**:
   - **Android Visualizer App**: Research-grade companion app rendering Catmull-Rom bezier curves, memory dynamics phase charts, step-by-step progress rings, and live log categorizers.
   - **Interactive Web Dashboard**: Web-based dashboard for desktop and server environments with live loss monitoring, parameter deltas, configuration generators, and test suites.
4. **Hugging Face Compatibility**: Out-of-the-box loading of Hugging Face SafeTensors checkpoints (both single-file and sharded layouts), standard tokenizer vocabularies, and PEFT-compliant adapter persistence.
5. **Strict Privacy by Design**: All dataset handling, tokenization, gradient computation, and weight adaptation happen entirely on-device with zero telemetry egress.

---

## Key Innovations & Technical Capabilities

| Capability | MobileFineTuner Native Approach | Conventional Desktop / Python Stacks |
| :--- | :--- | :--- |
| **Training Runtime** | Pure C++17 native binary / Android JNI | Python interpreter + PyTorch C++ bindings + CUDA/ROCm |
| **Memory Footprint** | Step Arena allocation + pooled memory + peak RSS monitor | Python GC overhead, dynamic tensor fragmentation |
| **Model Ingestion** | Direct SafeTensors parsing (single & sharded index) | Heavy safetensors / torch weight deserialization |
| **Tokenization** | Native C++ BPE and Gemma SentencePiece tokenizers | Hugging Face `tokenizers` Python / Rust bindings |
| **Adaptation** | Native LoRA injection on attention (Q, K, V, O) & MLP (Gate, Up, Down) | Hugging Face PEFT library |
| **Mobile Deployment** | Plug-and-play Android AAR (`mft-sdk`), C++ umbrella header | Requires custom ONNX / TFLite workarounds (mostly inference only) |
| **Live Telemetry** | Dual-track: Native Android Visualizer + Lightweight Web Dashboard | TensorBoard / Weights & Biases (requires cloud / Python) |

---

## End-to-End System Architecture

```text
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                                 MODEL & DATA ASSETS                                      │
│   • Hugging Face SafeTensors (Single / Sharded)       • config.json / generation_config │
│   • BPE Vocab / Gemma Tokenizer JSON                 • WikiText-2 / MMLU / QNLI Data   │
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           PUBLIC DISCOVERY & UNIFIED AUTO API                           │
│   ops::ModelRegistry           ops::TokenizerFactory         ops::AutoModelForCausalLM  │
│   (inspects config.json)       (instantiates tokenizer)      (loads graph + SafeTensors)│
└────────────────────────────────────────┬────────────────────────────────────────────────┘
                                         │
                                         ▼
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                            MOBILEFINETUNER C++ CORE ENGINE                              │
│  ┌───────────────────────┐   ┌────────────────────────┐   ┌──────────────────────────┐  │
│  │     Graph Models      │   │    Autograd Engine     │   │      Memory & Ops        │  │
│  │ • GPT-2 (124M/355M)   │   │ • Tape-based backward  │   │ • Step Arena Allocator   │  │
│  │ • Gemma 3 (270M/1B-PT)│   │ • Topological sorting  │   │ • Fused Softmax / Attention││
│  │ • Qwen 2.5 (0.5B)     │   │ • Gradient accum       │   │ • RMSNorm / RoPE / GeGLU │  │
│  └───────────────────────┘   └────────────────────────┘   └──────────────────────────┘  │
│  ┌───────────────────────────────────────────────────────────────────────────────────┐  │
│  │                AutoTrainer / Adam Optimizer / LoRA Linear Layer                   │  │
│  │ • Injects trainable low-rank adapters (A & B) into Attention & Feed-Forward blocks│  │
│  │ • Adam updates with weight decay, gradient clipping, and PEFT adapter saving      │  │
│  └───────────────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────┬───────────────────────────────────────────┬──────────────────┘
                           │                                           │
                           ▼                                           ▼
┌──────────────────────────────────────────────┐ ┌────────────────────────────────────────┐
│             ANDROID ECOSYSTEM                │ │             DESKTOP & SERVER           │
│  ┌────────────────────────────────────────┐  │ │  ┌──────────────────────────────────┐  │
│  │ mft-sdk (AAR + JNI Bridge)             │  │ │  │ Standalone Training Binaries     │  │
│  │ • Java/Kotlin API (MobileFineTuner)    │  │ │  │ • train_full / train (examples/) │  │
│  │ • Native self-test & synthetic smoke   │  │ │  │ • eval_ppl / eval_mmlu           │  │
│  └───────────────────┬────────────────────┘  │ │  └─────────────────┬────────────────┘  │
│                      │                       │ │                    │                   │
│                      ▼                       │ │                    ▼                   │
│  ┌────────────────────────────────────────┐  │ │  ┌──────────────────────────────────┐  │
│  │ Android Visualizer Companion App       │  │ │  │ Interactive Web Dashboard       │  │
│  │ • Live Catmull-Rom Bezier Curves       │  │ │  │ (web_ui/server.py :8080)         │  │
│  │ • Real-time Memory Dynamics & RSS      │  │ │  │ • Browser-based loss graphs      │  │
│  │ • Run Comparison ("Versus") & Terminal │  │ │  │ • Live log streamer & smoke UI   │  │
│  └────────────────────────────────────────┘  │ │  └──────────────────────────────────┘  │
└──────────────────────────────────────────────┘ └────────────────────────────────────────┘
```

---

## Repository Structure

```text
GenAI/
├── operator/                       # Core C++17 framework
│   ├── include/mobile_finetuner/   # Stable umbrella header (mobile_finetuner.h)
│   ├── finetune_ops/
│   │   ├── core/                   # Tensor, Autograd engine, memory allocators, native tokenizers
│   │   ├── graph/                  # GPT-2, Gemma, Qwen model graphs & SafeTensors loader
│   │   ├── memory/                 # Arena allocator for micro-step zero overhead memory
│   │   ├── nn/                     # LoRA linear layers and adapter modules
│   │   └── optim/                  # Adam optimizer, AutoTrainer, step utilities
│   ├── cmake/                      # CMake package export configurations
│   └── CMakeLists.txt              # Primary CMake build specification
│
├── android-visualizer/             # Complete Android development suite
│   ├── mft-sdk/                    # Android library module (builds release AAR + JNI bridge)
│   │   ├── src/main/java/          # Java API: com.mobilefinetuner.sdk.MobileFineTuner
│   │   └── src/main/cpp/           # JNI bindings into C++ core
│   ├── app/                        # Android Visualizer research-grade monitoring application
│   ├── sdk-sample/                 # Standalone sample app demonstrating AAR consumption
│   ├── scripts/qa/                 # Mobile device QA test scripts
│   └── build.gradle.kts            # Multi-module Gradle configuration
│
├── web_ui/                         # Interactive browser-based training dashboard
│   ├── server.py                   # Lightweight multithreaded Python server & REST API
│   └── public/                     # Modern UI assets (index.html, style.css, app.js)
│
├── examples/                       # Executable fine-tuning and evaluation targets
│   ├── common/                     # Shared MMLU & Gemma evaluation runners
│   ├── gpt2_small_lora_finetune/   # GPT-2 Small (124M) LoRA and Full-FT entrypoints
│   ├── gpt2_medium_lora_finetune/  # GPT-2 Medium (355M) LoRA fine-tuning
│   ├── gemma_3_270m_lora_finetune/ # Gemma-3 270M GQA LoRA fine-tuning
│   ├── gemma_3_1b_pt_lora_finetune/# Gemma-3 1B-PT LoRA fine-tuning
│   └── qwen_lora_finetune/         # Qwen2.5-0.5B RoPE LoRA fine-tuning
│
├── scripts/                        # Maintenance, benchmarking, and asset scripts
│   ├── android/                    # Android build, ADB monitoring, and staging scripts
│   ├── lib/                        # Shared bash path resolution helpers
│   ├── run_training_smoke.sh       # 5-model synthetic training smoke suite
│   ├── run_training_real_assets.sh # 5-model real-asset sanity check suite
│   ├── check_local_assets.sh       # Verifies local weights & dataset resolution
│   ├── check_release_tree.sh       # Validates release integrity & environment hygiene
│   └── generate_tokenizer_hf_golden_fixtures.py # Generates HF alignment test fixtures
│
├── docs/                           # Technical documentation & API references
│   ├── ARCHITECTURE.md             # System architecture & extension guide
│   ├── PUBLIC_API.md               # C++ and Android public API contracts
│   ├── ANDROID_SDK.md              # Android SDK build, ABI, and integration manual
│   └── MODEL_ASSETS.md             # Hugging Face asset formatting requirements
│
├── logo.jpg                        # Academic institutional branding
├── LICENSE                         # Apache License 2.0
└── README.md                       # Master project documentation (this file)
```

---

## Supported Model Architectures

MobileFineTuner provides optimized graph implementations for leading open-weight LLM families:

| Architecture | Model Variant | Parameters | Context | Key Architectural Attributes | LoRA Injection Targets |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **GPT-2** | GPT-2 Small | 124M | 1024 | Absolute Positional Embeddings, MHA, LayerNorm | Attention QKV, Attn Proj |
| **GPT-2** | GPT-2 Medium | 355M | 1024 | Absolute Positional Embeddings, MHA, LayerNorm | Attention QKV, Attn Proj |
| **Gemma 3** | Gemma-3 270M | 270M | 2048 | Grouped-Query Attention (GQA), RoPE, RMSNorm, GeGLU | Q, K, V, O projections + MLP Gate, Up, Down |
| **Gemma 3** | Gemma-3 1B-PT | 1.0B | 2048 | Scaled GQA, 18 layers, RoPE, RMSNorm, GeGLU | Q, K, V, O projections + MLP Gate, Up, Down |
| **Qwen 2.5** | Qwen2.5-0.5B | 490M | 2048 | SwiGLU, Rotary Embeddings, Tied Word Embeddings | Q, K, V, O projections + MLP Gate, Up, Down |

---

## Core C++ Engine (`operator/`)

The core engine is implemented in standard C++17 with modular subsystems:

### 1. Autograd & Tensor Engine
- **`Tensor`**: Custom multi-dimensional tensor supporting continuous memory buffers, copy-on-write semantics, slicing, and broadcasting.
- **`AutogradEngine`**: Tape-based automatic differentiation engine that builds a dynamic execution graph and executes backward passes via topological sort.
- **`step_arena.h`**: Ultra-low-overhead arena allocator dedicated to micro-step forward/backward passes, recycling memory across steps without OS `malloc`/`free` thrashing.

### 2. Operators & Kernels
- **Attention**: Standard causal multi-head attention and fused memory-efficient attention (`memory_efficient_attention.cpp`).
- **Normalizations**: Pre-LayerNorm and high-performance RMSNorm with backward gradient kernels.
- **Positional Encodings**: Native Rotary Position Embedding (RoPE) kernel supporting half-dimension complex rotations.
- **Loss**: Language modeling cross-entropy with selected-token masking (`lm_loss.cpp`), preventing activation explosion during loss evaluation.

### 3. Native Tokenizers
Zero external Python dependencies are needed to tokenize text:
- **`TokenizerBPE`**: Byte-level Byte-Pair Encoding matching Hugging Face GPT-2 and Qwen pre-tokenization.
- **`TokenizerGemma`**: SentencePiece-compatible tokenization for Gemma-3 models.
- **`TokenizerFactory`**: Unified auto-discovery layer that inspects `config.json` and instantiates the correct tokenizer automatically.

### 4. Adapter Management & SafeTensors
- **`SafeTensorsLoader`**: Fast zero-copy memory-mapped SafeTensors parser that handles single `model.safetensors` files as well as sharded index manifests (`model.safetensors.index.json`).
- **`LoRALinear`**: Dynamic forward layer executing:
  $$\mathbf{y} = \mathbf{x}\mathbf{W}_0^T + \frac{\alpha}{r} (\mathbf{x}\mathbf{A}^T)\mathbf{B}^T$$
- **`LoRASaver`**: Writes fine-tuned adapters out as standard Hugging Face PEFT-compatible SafeTensors checkpoints.

---

## Android Ecosystem (`android-visualizer/`)

MobileFineTuner features a comprehensive Android solution spanning SDK embedding and real-time visualization:

```text
android-visualizer/
├── mft-sdk/        ──> Android Library (AAR + libmobilefinetuner_jni.so)
├── app/            ──> Research-Grade Real-Time Visualizer App
└── sdk-sample/     ──> Standalone Consumer Integration Example
```

### MFT Android SDK (`mft-sdk`)
The SDK packages the native C++ operator core behind an intuitive Java interface.

**Building the Release AAR:**
```bash
bash scripts/android/build_mft_sdk_aar.sh
```
*Output: `android-visualizer/mft-sdk/build/outputs/aar/mft-sdk-release.aar`*

**Publishing to Local Maven:**
```bash
bash scripts/android/publish_mft_sdk_local.sh
```

**Gradle Integration in Android Apps:**
```kotlin
repositories {
    maven { url = uri("/path/to/MobileFineTuner/android-visualizer/mft-sdk/build/repo") }
}

dependencies {
    implementation("com.mobilefinetuner:mobilefinetuner-android:0.1.0")
}
```

**Java API Usage Example:**
```java
import com.mobilefinetuner.sdk.MobileFineTuner;

// 1. Open model from device storage
File modelDir = new File(context.getFilesDir(), "qwen-0.5b");
try (MobileFineTuner mft = MobileFineTuner.open(modelDir)) {
    // 2. Initialize LoRA configuration
    mft.initLora(8 /* rank */, 16.0f /* alpha */);
    
    // 3. Execute training step directly on-device
    float loss = mft.trainStep(inputIds, attentionMask, labels, 2e-4f /* lr */);
    Log.i("MFT", "Step Loss: " + loss);
    
    // 4. Save trained adapter weights
    File adapterOut = new File(context.getFilesDir(), "adapter.safetensors");
    mft.saveLoraAdapter(adapterOut);
}
```

### Android Visualizer Dashboard App
A full-featured Jetpack/Kotlin Android application engineered for on-device live training observation:

- **Home Tab**: Live circular progress ring, real-time memory dynamics phase chart, sparkline trend cards for Loss, PPL, LR, and Peak RSS, and an event timeline.
- **Train Tab**: Interactive time-series charts rendering Catmull-Rom bezier curves with horizontal pan, double-tap zoom, and individual step inspection.
- **Versus Tab**: Side-by-side run comparative benchmark with step alignment, delta cards, and winner indicators.
- **Logs Tab**: Terminal-style live log viewer with instant filter chips (`[Train]`, `[Eval]`, `[Ckpt]`, `[RSS]`, `[Warn]`, `[Error]`) and free-text search.
- **Runs Tab**: Automatic discovery and status tracking of active and historical runs.

---

## Interactive Web UI Dashboard (`web_ui/`)

For desktop workstations, edge servers, and headless test runners, MobileFineTuner provides an interactive web dashboard.

### Features
- **Live Training Visualizer**: Real-time multi-metric graphs (Loss, Learning Rate, Perplexity) refreshed via asynchronous polling.
- **Live Console Stream**: Websocket/HTTP terminal streaming stdout/stderr with syntax highlighting.
- **One-Click Synthetic Smoke**: Trigger 5-step synthetic smoke training on any supported model family directly from the browser.
- **Integrated Test Runner**: Run the full CTest operator suite and inspect granular test pass/fail results interactively.
- **Model Configurator**: Interactive parameter tuning (Batch Size, Sequence Length, Rank, Alpha, Learning Rate).

### Launching the Dashboard
```bash
python web_ui/server.py 8080
```
Open your browser and navigate to:
```text
http://localhost:8080
```

---

## Runnable Model Targets (`examples/`)

MobileFineTuner provides five pre-configured training applications under `examples/`:

### 1. `examples/gpt2_small_lora_finetune/`
- Full fine-tuning (`train_full`) and LoRA fine-tuning (`train`).
- WikiText-2 language modeling and MMLU multi-task classification.
- Evaluation tools: `eval_ppl` and `eval_mmlu`.

### 2. `examples/gpt2_medium_lora_finetune/`
- Scaled 355M parameter configuration.
- Supports gradient accumulation for low-memory execution on 4GB-8GB devices.

### 3. `examples/gemma_3_270m_lora_finetune/`
- Ultra-compact Google Gemma 3 architecture with GQA and RMSNorm.
- Ideal baseline for ultra-low-power edge nodes.

### 4. `examples/gemma_3_1b_pt_lora_finetune/`
- 1.0B parameter model with full LoRA adaptation across attention and GeGLU feed-forward blocks.

### 5. `examples/qwen_lora_finetune/`
- Alibaba Qwen 2.5 0.5B model.
- Dedicated pipelines for WikiText-2 (`run_wikitext.sh`), MMLU (`run_mmlu.sh`), and QNLI classification (`run_qnli.sh`).

---

## Asset Contract & Data Management

In accordance with modern ML framework practices, MobileFineTuner does not bundle proprietary model checkpoints or multi-gigabyte datasets inside git. Assets are dynamically mapped via environment variables or shared directory roots:

### Recommended Environment Setup
```bash
export MFT_MODEL_ROOT=/path/to/mft-models
export MFT_DATA_ROOT=/path/to/mft-data
```

### Directory Layout
```text
$MFT_MODEL_ROOT/
├── gpt2/                  # config.json, vocab.json, merges.txt, model.safetensors
├── gpt2-medium/           # config.json, vocab.json, merges.txt, model.safetensors
├── gemma-3-270m/          # config.json, tokenizer.json, model.safetensors
├── gemma-3-1b-pt/         # config.json, tokenizer.json, model.safetensors (or sharded)
└── Qwen2.5-0.5B/          # config.json, tokenizer.json, model.safetensors

$MFT_DATA_ROOT/
├── wikitext2/wikitext-2-raw/   # wiki.train.raw, wiki.valid.raw, wiki.test.raw
└── mmlu/data/                  # dev/, test/, val/ CSV subdirectories
```

### Validating Asset Discovery
Run the asset verification script to check your local configuration:
```bash
bash scripts/check_local_assets.sh
```

---

## Build & Installation Guide

### Prerequisites
- **C++ Compiler**: GCC 9+, Clang 10+, or MSVC 2019+ (C++17 standard required)
- **Build System**: CMake ≥ 3.10
- **Threading**: Native pthreads / C++11 std::thread
- **Optional Acceleration**: BLAS (Apple Accelerate, OpenBLAS, or Intel MKL)
- **Android Builds**: Android NDK `26.1.10909125`, Android SDK API 35, JDK 17+

### 1. Building the Core Library & Unit Tests
```bash
# Configure with BLAS acceleration enabled
cmake -S operator -B operator/build -DUSE_BLAS=ON -DCMAKE_BUILD_TYPE=Release

# Compile using all CPU cores
cmake --build operator/build -j

# Execute unit and operator test suite
ctest --test-dir operator/build --output-on-failure
```

### 2. Installing the CMake Package
```bash
cmake --install operator/build --prefix /usr/local
```

Downstream C++ projects can link MobileFineTuner cleanly via CMake:
```cmake
find_package(MobileFineTuner REQUIRED)
target_link_libraries(my_app PRIVATE MobileFineTuner::operators)
```

And include the single umbrella header:
```cpp
#include "mobile_finetuner/mobile_finetuner.h"
```

---

## Quick Start & Training Execution

### 1. Synthetic Smoke Testing (5 Models)
Validate all 5 training pipelines in seconds without needing any pretrained weights:
```bash
bash scripts/run_training_smoke.sh
```
This tests forward, backward, gradient accumulation, and parameter updates for:
1. GPT-2 Small
2. GPT-2 Medium
3. Gemma 270M
4. Gemma 1B-PT
5. Qwen 2.5-0.5B

### 2. Real Asset Fine-Tuning

#### WikiText-2 LoRA Fine-Tuning
```bash
# GPT-2 Small
cd examples/gpt2_small_lora_finetune
TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=4 GRAD_ACCUM_STEPS=2 ./run_train.sh

# Gemma 270M
cd examples/gemma_3_270m_lora_finetune
TRAIN_MODE=wt2 STEPS=200 BATCH_SIZE=4 GRAD_ACCUM_STEPS=1 ./run_train.sh

# Qwen2.5-0.5B
cd examples/qwen_lora_finetune
MAX_STEPS=200 BATCH_SIZE=1 GRAD_ACCUM_STEPS=1 ./run_wikitext.sh
```

#### MMLU Multi-Task Fine-Tuning
```bash
cd examples/gpt2_small_lora_finetune
./run_prepare_data.sh
TRAIN_MODE=mmlu STEPS=200 ./run_train.sh
```

#### GPT-2 Full Fine-Tuning (Full-FT)
```bash
cmake -S examples/gpt2_small_lora_finetune -B examples/gpt2_small_lora_finetune/build
cmake --build examples/gpt2_small_lora_finetune/build --target train_full -j

examples/gpt2_small_lora_finetune/build/train_full \
  --data_dir "$MFT_DATA_ROOT/wikitext2/wikitext-2-raw" \
  --pretrained_dir "$MFT_MODEL_ROOT/gpt2" \
  --output_path runs/gpt2_small_full_ft.safetensors \
  --epochs 1 --batch_size 4 --grad_accum_steps 2 --seq_len 128 --lr 1e-4
```

### 3. Unified C++ Auto API
For custom applications, use the high-level `AutoModel` and `AutoTrainer` abstractions:

```cpp
#include "mobile_finetuner/mobile_finetuner.h"

int main() {
    std::string model_dir = "/path/to/model";

    // 1. Inspect model metadata & instantiate tokenizer
    auto spec = ops::ModelRegistry::inspect_pretrained(model_dir);
    auto tokenizer = ops::TokenizerFactory::from_pretrained(model_dir);

    // 2. Load model graph with family-correct SafeTensors mapping
    auto model = ops::AutoModelForCausalLM::from_pretrained(model_dir);

    // 3. Inject LoRA adapters (Rank=8, Alpha=16.0)
    model->init_lora(ops::AutoLoraConfig::attention_qkvo(8, 16.0f));

    // 4. Initialize Trainer
    ops::AutoTrainerConfig cfg;
    cfg.learning_rate = 2e-4f;
    cfg.weight_decay = 0.01f;
    ops::AutoTrainer trainer(*model, cfg);

    // 5. Run single-step forward + LM loss + backward + optimizer step
    auto step_result = trainer.train_step(input_ids, attention_mask, labels);
    std::cout << "Loss: " << step_result.loss << std::endl;

    return 0;
}
```

---

## Evaluation & Device Telemetry

### WikiText-2 Perplexity (PPL)
Evaluate model loss and perplexity with merged or isolated LoRA adapters:
```bash
examples/gpt2_small_lora_finetune/build/eval_ppl \
  --data_root "$MFT_DATA_ROOT/wikitext2/wikitext-2-raw" \
  --pretrained_dir "$MFT_MODEL_ROOT/gpt2" \
  --lora_path examples/gpt2_small_lora_finetune/outputs/lora_final.safetensors \
  --lora_merge 1
```

### MMLU Few-Shot Benchmark
```bash
examples/gpt2_small_lora_finetune/build/eval_mmlu \
  --mmlu_root "$MFT_DATA_ROOT/mmlu/data" \
  --split dev \
  --pretrained_dir "$MFT_MODEL_ROOT/gpt2" \
  --lora_path examples/gpt2_small_lora_finetune/outputs/lora_final.safetensors \
  --lora_merge 1 \
  --fewshot 0
```

### Mobile Hardware Resource Profiling
When running native binaries on real Android smartphones via ADB, capture continuous memory and thermal telemetry:
```bash
# Terminal 1: Run native training on phone
bash scripts/android/run_qwen_qnli_native_phone.sh

# Terminal 2: Monitor RSS, PSS, CPU, and thermals at 100ms intervals
bash scripts/android/adb_resource_monitor.sh
```

---

## Maintained Scripts Catalog

All core utility scripts located in `scripts/` are maintained and verified:

| Script | Purpose |
| :--- | :--- |
| `scripts/run_training_smoke.sh` | Orchestrates 5-model synthetic training smoke test suite. |
| `scripts/run_training_real_assets.sh` | Executes 1-step real-asset verification for all supported models. |
| `scripts/check_local_assets.sh` | Validates asset resolution (`MFT_MODEL_ROOT` & `MFT_DATA_ROOT`). |
| `scripts/check_release_tree.sh` | Enforces repository cleanliness and absence of local machine paths. |
| `scripts/generate_tokenizer_hf_golden_fixtures.py` | Generates golden tokenization JSONL files from Hugging Face for C++ tests. |
| `scripts/android/build_mft_sdk_aar.sh` | Compiles the Android SDK into a distributable release AAR. |
| `scripts/android/publish_mft_sdk_local.sh` | Publishes `mft-sdk` into a local Maven repository for app consumers. |
| `scripts/android/adb_resource_monitor.sh` | ADB-based sampling monitor capturing memory RSS and CPU utilization. |
| `scripts/android/run_qwen_qnli_native_phone.sh` | Pushes binary and assets to Android device and runs on-phone Qwen training. |
| `scripts/plot_loss_curve.py` | Generates PNG loss curve plots from structured training logs. |

---

## Citation & Academic Credit

If you use MobileFineTuner or components of this framework in your research, please cite our paper:

```bibtex
@inproceedings{mobilefinetuner2027,
  title={MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications},
  author={Geng, Jiaxiang and Zhao, Lunyu and Lu, Yiyi and Luo, Bing},
  booktitle={ACM Conference on Embedded Networked Sensor Systems (Sensys)},
  year={2027}
}

@article{geng2025mobilefinetuner,
  title={MobileFineTuner: A Mobile-Native Framework for On-Device LLM Fine-Tuning in Real-World Embedded AI Applications},
  author={Geng, Jiaxiang and Zhao, Lunyu and Lu, Yiyi and Luo, Bing},
  journal={arXiv preprint arXiv:2512.08211},
  year={2025}
}
```

### Research Team
- **Jiaxiang Geng** — Duke Kunshan University & The University of Hong Kong (`jg645@duke.edu`)
- **Lunyu Zhao** — Duke Kunshan University (`lz269@duke.edu`)
- **Yiyi Lu** — Duke Kunshan University (`yl996@duke.edu`)
- **Bing Luo** — Duke Kunshan University (`bl291@duke.edu`)

---

## License & Acknowledgments

This project is licensed under the **Apache License 2.0** — see the [LICENSE](LICENSE) file for complete details.

```text
Copyright 2024-2026 Mobile LLM Fine-Tuning Project Contributors

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```

### Acknowledgments
We extend our gratitude to the open-source community:
- **Hugging Face** for the Transformers architecture definitions, tokenizers, and SafeTensors format.
- **Google Research** & **Alibaba Cloud** for open-weight Gemma and Qwen models.
- **DeepSpeed** and **PEFT** authors for pioneering parameter-efficient training formulations.
- **Duke University** & **Duke Kunshan University** for research computing support.

---

<div align="center">
  <b>MobileFineTuner: Bringing Private, Native On-Device LLM Adaptation to Everyone.</b>
</div>
