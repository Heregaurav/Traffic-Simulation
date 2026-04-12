"""
LPSim Benchmarking & Visualization Script
==========================================
Reproduces key figures from the paper:
  - Fig 1  : Simulation time comparison across simulators
  - Fig 13 : Strong scaling (simulation time vs. demand size vs. GPU count)
  - Table 7: Simulation times for different partitions
  - Amdahl's Law speedup curves
  - Roofline model visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import os

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------
plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size':   11,
    'axes.spines.top':    False,
    'axes.spines.right':  False,
    'figure.dpi':         120,
})
COLORS = ['#1f77b4','#ff7f0e','#2ca02c','#d62728',
          '#9467bd','#8c564b','#e377c2','#7f7f7f']

os.makedirs('figures', exist_ok=True)

# ===========================================================================
# 1. Simulation Time Comparison (Fig. 1)
# ===========================================================================
def plot_sim_comparison():
    simulators = ['LPSim', 'MANTA',
                  'SUMO meso simplified', 'SUMO micro simplified',
                  'SUMO meso advanced', 'SUMO micro advanced']
    times_min  = [2.4, 4.6, 1620.0, 114858.0, 1740.0, 123500.0]
    colors     = ['#1f77b4','#ff7f0e'] + ['#d62728']*4

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(simulators, times_min, color=colors, edgecolor='white', height=0.6)
    ax.set_xscale('log')
    ax.set_xlabel('Simulation Time (minutes, log scale)')
    ax.set_title('Regional Traffic Simulation Time Comparison\n(3.2M OD trips, SF Bay Area)', pad=12)

    for bar, t in zip(bars, times_min):
        ax.text(t * 1.1, bar.get_y() + bar.get_height()/2,
                f'{t:,.1f}', va='center', fontsize=9)

    ax.axvline(10, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
    ax.set_xlim(1, 5e5)
    fig.tight_layout()
    fig.savefig('figures/fig1_sim_comparison.png')
    print("  Saved figures/fig1_sim_comparison.png")
    plt.close()

# ===========================================================================
# 2. Strong Scaling (Fig. 13 style)
# ===========================================================================
def plot_strong_scaling():
    # Data from Table 7 of the paper (times in ms)
    demands = [3_000_000, 6_000_000, 12_000_000, 20_000_000, 24_000_000]
    gpu_counts = [1, 2, 4, 8]

    balanced = {
        3_000_000:  [381997,  353399,  439829,  689898],
        6_000_000:  [752695,  610608,  653128,  928397],
        12_000_000: [1535082, 1182821, 1087738, 1306006],
        20_000_000: [2554437, 1746744, 1459479, 1639141],
        24_000_000: [None,    None,    1744777, 1938588],
    }
    unbalanced = {
        3_000_000:  [381997,  343435,  446019,  678993],
        6_000_000:  [752695,  597259,  632208,  868127],
        12_000_000: [1535082, 1134121, 1111049, 1304821],
        20_000_000: [2554437, 1768196, 1717089, 1635972],
        24_000_000: [None,    None,    2011620, 1928708],
    }

    fig, ax = plt.subplots(figsize=(9, 5))
    cmap = plt.colormaps["tab10"]

    for idx, dem in enumerate(demands):
        col = cmap(idx)
        label_m = f'{dem//1_000_000}M'

        gpus_b = [g for g, t in zip(gpu_counts, balanced[dem]) if t is not None]
        vals_b = [t/1e3 for t in balanced[dem] if t is not None]
        gpus_u = [g for g, t in zip(gpu_counts, unbalanced[dem]) if t is not None]
        vals_u = [t/1e3 for t in unbalanced[dem] if t is not None]

        ax.plot(gpus_b, vals_b, '-o',  color=col, linewidth=2,
                label=f'Balanced-{label_m}')
        ax.plot(gpus_u, vals_u, '--s', color=col, linewidth=2,
                alpha=0.8, label=f'Unbalanced-{label_m}')

    ax.set_xlabel('Number of GPUs')
    ax.set_ylabel('Response Time (seconds)')
    ax.set_title('Strong Scaling: Simulation Time vs. Number of GPUs')
    ax.set_xticks(gpu_counts)
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=8)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    fig.tight_layout()
    fig.savefig('figures/fig13_strong_scaling.png')
    print("  Saved figures/fig13_strong_scaling.png")
    plt.close()

# ===========================================================================
# 3. Amdahl's Law Speedup Prediction
# ===========================================================================
def plot_amdahls_law():
    n_range = np.arange(1, 17)
    serial_fractions = [0.01, 0.05, 0.10, 0.20]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for f, col in zip(serial_fractions, COLORS):
        speedup = 1.0 / (f + (1.0 - f) / n_range)
        ax.plot(n_range, speedup, '-o', color=col,
                label=f'f_serial = {f*100:.0f}%', linewidth=2)

    # Ideal (f=0)
    ax.plot(n_range, n_range, 'k--', label='Ideal (f=0)', linewidth=1)

    # Mark actual observed speedups from Table 7 (balanced, 20M demand)
    actual_gpus    = [1, 2, 4, 8]
    actual_speedup = [1.0,
                      2554437/1746744,
                      2554437/1459479,
                      2554437/1639141]
    ax.scatter(actual_gpus, actual_speedup, marker='*', s=150,
               color='black', zorder=5, label='LPSim Observed (20M)')

    ax.set_xlabel('Number of GPUs (n)')
    ax.set_ylabel('Speedup S(n)')
    ax.set_title("Amdahl's Law vs. LPSim Observed Speedup")
    ax.legend(fontsize=9)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/amdahls_law.png')
    print("  Saved figures/amdahls_law.png")
    plt.close()

# ===========================================================================
# 4. Roofline Model Visualization
# ===========================================================================
def plot_roofline():
    # A100 hardware specs
    peak_gips      = 609.12   # Giga instructions/sec (warp-level)
    bandwidth_gb_s = [2000, 600, 80]   # HBM, L2, L1 (approximate)
    bw_labels      = ['HBM', 'L2', 'L1']

    intensity_range = np.logspace(-3, 3, 200)

    fig, ax = plt.subplots(figsize=(8, 5))

    # Memory-bound rooflines
    for bw, label, col in zip(bandwidth_gb_s, bw_labels,
                               ['#1f77b4','#ff7f0e','#2ca02c']):
        roof = np.minimum(peak_gips, intensity_range * bw)
        ax.loglog(intensity_range, roof, '-', color=col,
                  linewidth=2, label=f'{label} ({bw} GB/s)')

    # Compute-bound horizontal line
    ax.axhline(peak_gips, color='black', linestyle='--',
               linewidth=1.5, label=f'Peak Compute ({peak_gips:.0f} GIPS)')

    # LPSim operating points (unsorted and sorted - from Table 6)
    # Approximate intensity = instructions / bytes
    unsorted_intensity = 5533942 / 14665154
    sorted_intensity   = 830574  / 14547567
    unsorted_perf = 5533942 / 195.28 / 1e9
    sorted_perf   = 830574  / 170.75 / 1e9

    ax.scatter([unsorted_intensity], [unsorted_perf],
               marker='o', s=120, color='red', zorder=6,
               label=f'LPSim Unsorted ({unsorted_perf:.4f} GIPS)')
    ax.scatter([sorted_intensity], [sorted_perf],
               marker='D', s=120, color='purple', zorder=6,
               label=f'LPSim Sorted ({sorted_perf:.4f} GIPS)')

    ax.set_xlabel('Arithmetic Intensity (Instructions / Byte)')
    ax.set_ylabel('Performance (GIPS, warp-level)')
    ax.set_title('Instruction Roofline Model — NVIDIA A100')
    ax.legend(fontsize=9)
    ax.grid(True, which='both', linestyle='--', alpha=0.3)
    fig.tight_layout()
    fig.savefig('figures/roofline.png')
    print("  Saved figures/roofline.png")
    plt.close()

# ===========================================================================
# 5. Partition Performance Table (Table 4 visualization)
# ===========================================================================
def plot_partition_comparison():
    labels = ['Balanced\n2 GPUs', 'Balanced\n4 GPUs', 'Balanced\n8 GPUs',
              'Unbalanced\n2 GPUs', 'Unbalanced\n4 GPUs', 'Unbalanced\n8 GPUs']
    times_ms = [2_498_466, 1_483_472, 1_269_893,
                2_014_554, 1_679_861, 1_783_668]
    colors   = ['#1f77b4']*3 + ['#ff7f0e']*3

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(labels, [t/1e3 for t in times_ms],
                  color=colors, edgecolor='white', width=0.6)
    ax.set_ylabel('Simulation Time (seconds)')
    ax.set_title('Multi-GPU Simulation Time by Partition Strategy\n(Full SF Bay Area Demand)')
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    for bar, t in zip(bars, times_ms):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                f'{t/1e3:.0f}s', ha='center', fontsize=9)
    from matplotlib.patches import Patch
    legend_items = [Patch(color='#1f77b4', label='Balanced Partition'),
                    Patch(color='#ff7f0e', label='Unbalanced Partition')]
    ax.legend(handles=legend_items)
    fig.tight_layout()
    fig.savefig('figures/partition_comparison.png')
    print("  Saved figures/partition_comparison.png")
    plt.close()

# ===========================================================================
# 6. Device_Vector vs Array Performance (Table 5)
# ===========================================================================
def plot_device_vector_vs_array():
    methods = ['Device_Vector\n(Balanced)', 'Device_Vector\n(Unbalanced)',
               'Array\n(Balanced)', 'Array\n(Unbalanced)']
    times_ms = [1_342_463, 1_287_098, 70_725_162, 71_131_945]
    colors   = ['#2ca02c','#2ca02c','#d62728','#d62728']

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(methods, [t/1e3 for t in times_ms],
           color=colors, edgecolor='white', width=0.55)
    ax.set_yscale('log')
    ax.set_ylabel('Simulation Time (seconds, log scale)')
    ax.set_title('Device_Vector vs Array: Vehicle Storage Performance\n(Dual A100, 9M Trips)')
    ax.yaxis.grid(True, which='both', linestyle='--', alpha=0.3)
    from matplotlib.patches import Patch
    legend_items = [Patch(color='#2ca02c', label='Device_Vector (~52x faster)'),
                    Patch(color='#d62728', label='Array')]
    ax.legend(handles=legend_items)
    fig.tight_layout()
    fig.savefig('figures/device_vector_vs_array.png')
    print("  Saved figures/device_vector_vs_array.png")
    plt.close()

# ===========================================================================
# Run all
# ===========================================================================
if __name__ == '__main__':
    print("Generating LPSim benchmark figures...\n")
    plot_sim_comparison()
    plot_strong_scaling()
    plot_amdahls_law()
    plot_roofline()
    plot_partition_comparison()
    plot_device_vector_vs_array()
    print("\nAll figures saved to ./figures/")
