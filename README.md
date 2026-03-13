# 🇬🇧 BRITIX 8B

*A formally verified, memory-safe, self-evolving 8B parameter language model with British charm.*

## ✨ Features

- **8 Billion Parameters** - 32 layers, 4096 dim, 32 attention heads
- **Mathematically Verified** - SPARK proofs of correctness, no runtime errors
- **Memory Safe** - GPA tracks every byte, zero leaks guaranteed
- **Self-Evolving** - 7 Metamorphic cores restructure the model over time
- **British Trained** - Kuiil's affirmations, tea at 3pm, proper manners
- **Samurai Speed** - SIMD, 16-core thread pool, cache prefetching
- **Quantized** - 4-bit mode fits 8B in 4GB RAM (your laptop!)
- **GPU Ready** - Same weights work on CPU today, GPU tomorrow

## 🗡️ Quick Start

```bash
# Clone and build
git clone https://github.com/kuiil/britix
cd BRITIX
zig build -Drelease-fast

# Convert a Mistral model to Britix format
./britix-converter safetensors /path/to/mistral weights.bin

# Run inference
./britix

# Start chat
./britix chat

# Test speed
./britix benchmark

# Check memory
./britix test-memory
