# LPSim Real Data Quickstart

## 1) Build
```bash
cd "/home/gaurav/Desktop/GPU COMPUTING/lpsim"
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j"$(nproc)"
```

## 2) Prepare Inputs From Berkeley Datasets
Small/fast demo:
```bash
python3 scripts/prepare_inputs.py \
  --network-dir LPSim/LivingCity/berkeley_2018/basic_network \
  --out-dir data/processed/basic \
  --max-demand 1000 --depart-step 2
```

Medium demo:
```bash
python3 scripts/prepare_inputs.py \
  --network-dir LPSim/LivingCity/berkeley_2018/partial_network \
  --out-dir data/processed/partial_500 \
  --max-demand 500 --depart-step 1
```

## 3) Run Simulator With Real Inputs
```bash
./build/lpsim \
  --gpus 1 \
  --network data/processed/partial_500/network.csv \
  --demand data/processed/partial_500/demand.csv \
  --partition balanced \
  --steps 600 --dt 1.0
```

## 4) Run Benchmark Matrix (CSV + Logs)
```bash
GPU_LIST="1" STEPS="600" REPEATS="3" DATASETS="basic partial_500" \
  ./scripts/run_benchmark_matrix.sh
```

Results:
- `benchmarks/results.csv`
- `benchmarks/logs/*.log`

Generate graphs from your own benchmark CSV:
```bash
python3 -m pip install --user matplotlib
python3 scripts/plot_benchmark_results.py
```
Output graphs:
- `benchmarks/figures/wall_time_bar.png`
- `benchmarks/figures/scaling_line.png`

## 5) Generate Report Figures
The plotting script is paper-style (preloaded benchmark values).
```bash
python3 -m pip install --user matplotlib numpy
python3 scripts/benchmark_plots.py
```

Figures are saved to `figures/`.
