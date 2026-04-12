# LPSim (GPU Computing Project)

This repository contains a CUDA/C++ implementation of a multi-GPU traffic simulation prototype inspired by:

**"Large scale multi-GPU based parallel traffic simulation for accelerated traffic assignment and propagation" (TRC 2024)**.

## Project Resources

### Core Source Code
- `include/lpsim.cuh` - Shared data structures and constants.
- `src/main.cu` - Program entry, input loading, partition selection, and simulation launch.
- `src/propagation.cu` - CUDA vehicle propagation kernel (IDM-style update + movement logic).
- `src/multi_gpu.cu` - Multi-GPU orchestration, sync, and inter-GPU transfer flow.
- `src/partitioning.cpp` - Balanced and community-style partitioning logic.

### Build System
- `CMakeLists.txt` - CUDA/C++ build configuration (`lpsim` executable).
- `build/lpsim` - Compiled simulator binary (after build).

### Data and Inputs
- `data/processed/basic/` - Small demo dataset.
  - `network.csv`
  - `demand.csv`
- `data/processed/partial_500/` - Medium demo dataset.
  - `network.csv`
  - `demand.csv`
- `LPSim/LivingCity/berkeley_2018/...` - Source datasets for preprocessing.

### Scripts
- `scripts/prepare_inputs.py` - Converts Berkeley inputs to simulator CSV format.
- `scripts/run_benchmark_matrix.sh` - Runs repeated benchmark matrix and writes CSV + logs.
- `scripts/plot_benchmark_results.py` - Plots benchmark results from `benchmarks/results.csv`.
- `scripts/benchmark_plots.py` - Generates paper-style comparison figures (uses preloaded values).

### Outputs
- `benchmarks/results.csv` - Benchmark summary table.
- `benchmarks/logs/*.log` - Per-run simulator logs.
- `benchmarks/figures/` - Benchmark plots from measured runs.

### Documentation
- `REAL_DATA_QUICKSTART.md` - Quick run commands.
- `LPSim_Implementation_Report.docx` - Detailed implementation report.

## Prerequisites

1. Linux with NVIDIA GPU.
2. NVIDIA driver installed and working.
3. CUDA Toolkit (with `nvcc`) compatible with your driver.
4. CMake >= 3.18.
5. C++17 compiler (`g++`/`clang++`).
6. Python 3 (for preprocessing/plotting).

Optional Python packages for plotting:
- `matplotlib`
- `numpy`

## Verify GPU Environment

Run:

```bash
nvidia-smi
nvcc --version
```

If `nvidia-smi` fails, fix the driver first before running the simulator.

## Build

From the `lpsim` folder:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

Binary path:

```bash
./build/lpsim
```

## Run the Simulator (Successful Example)

### Medium demo dataset (`partial_500`)

```bash
./build/lpsim \
  --gpus 1 \
  --network data/processed/partial_500/network.csv \
  --demand data/processed/partial_500/demand.csv \
  --partition balanced \
  --steps 600 \
  --dt 1.0
```

### Small demo dataset (`basic`)

```bash
./build/lpsim \
  --gpus 1 \
  --network data/processed/basic/network.csv \
  --demand data/processed/basic/demand.csv \
  --partition balanced \
  --steps 600 \
  --dt 1.0
```

## Generate Inputs from Berkeley Data

Small/fast:

```bash
python3 scripts/prepare_inputs.py \
  --network-dir LPSim/LivingCity/berkeley_2018/basic_network \
  --out-dir data/processed/basic \
  --max-demand 1000 \
  --depart-step 2
```

Medium:

```bash
python3 scripts/prepare_inputs.py \
  --network-dir LPSim/LivingCity/berkeley_2018/partial_network \
  --out-dir data/processed/partial_500 \
  --max-demand 500 \
  --depart-step 1
```

## Run Benchmarks

```bash
GPU_LIST="1" STEPS="600" REPEATS="3" DATASETS="basic partial_500" \
  ./scripts/run_benchmark_matrix.sh
```

Results:
- `benchmarks/results.csv`
- `benchmarks/logs/*.log`

## Plot Benchmark Results (Measured Runs)

```bash
python3 -m pip install --user matplotlib
python3 scripts/plot_benchmark_results.py
```

Output:
- `benchmarks/figures/wall_time_bar.png`
- `benchmarks/figures/scaling_line.png`

## Paper-Style Figures (Preloaded Values)

```bash
python3 -m pip install --user matplotlib numpy
python3 scripts/benchmark_plots.py
```

Output folder:
- `figures/`

Note: `scripts/benchmark_plots.py` includes hardcoded/reference values for paper-style visualizations, not only local benchmark values.

## Common Issues and Fixes

### 1) `no CUDA-capable device is detected`
Cause: GPU not visible to runtime (driver/container/sandbox issue).  
Fix:
- Confirm `nvidia-smi` works.
- Check CUDA-driver compatibility.
- Run outside restricted sandbox/container if GPU passthrough is blocked.

### 2) Build fails with CUDA architecture mismatch
Cause: `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt` does not match your GPU.  
Fix:
- Update architectures (for example: `70` for V100, `80` for A100, `86` for RTX 30xx, `89` for RTX 40xx).
- Reconfigure and rebuild.

### 3) Missing processed data files
Cause: `network.csv` / `demand.csv` missing in `data/processed/...`.  
Fix:
- Run `scripts/prepare_inputs.py` first.

## Quick Command Checklist

```bash
cd "/home/gaurav/Desktop/GPU COMPUTING/lpsim"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
./build/lpsim --gpus 1 --network data/processed/partial_500/network.csv --demand data/processed/partial_500/demand.csv --partition balanced --steps 600 --dt 1.0
```
