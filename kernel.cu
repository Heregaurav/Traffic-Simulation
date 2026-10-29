// kernel.cu
// GPU utility kernels for LPSim
// Complements simulation.cu with helper operations:
//   - counter resets between ticks
//   - lane-map clearing
//   - per-tick statistics gathering
//
// Each kernel here follows the same pattern as simulation.cu:
// one thread = one element (vehicle or lane cell).

#include <cuda_runtime.h>
#include <math.h>
#include "simulation.h"

// ----------------------------------------------------------------
// reset_tick_counters
// Zero out active_count and finished_count at the start of each
// simulation tick so the per-tick progress report is accurate.
// Launch with a single thread: <<<1, 1>>>
// ----------------------------------------------------------------
__global__ void reset_tick_counters(int* active_count,
                                    int* finished_count)
{
    *active_count   = 0;
    *finished_count = 0;
}

// ----------------------------------------------------------------
// clear_lane_map
// Set every cell of the lane map back to 255 (empty).
// Useful for re-initialisation or warm-restart runs.
// Launch with enough threads to cover lane_map_size.
// ----------------------------------------------------------------
__global__ void clear_lane_map(unsigned char* lane_map,
                               int            lane_map_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < lane_map_size)
        lane_map[idx] = 255u;
}

// ----------------------------------------------------------------
// gather_speed_stats
// Computes the average speed (m/s) of all currently active vehicles
// and writes it into avg_speed_out.
// Uses a simple parallel reduction via atomicAdd on a shared
// accumulator — appropriate for diagnostic/reporting use.
// Launch with enough threads to cover num_vehicles.
// ----------------------------------------------------------------
__global__ void gather_speed_stats(const Vehicle* vehicles,
                                   int            num_vehicles,
                                   float*         speed_sum_out,
                                   int*           active_n_out)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vehicles) return;

    const Vehicle& v = vehicles[vid];
    if (v.active && !v.finished) {
        atomicAdd(speed_sum_out, v.speed);
        atomicAdd(active_n_out,  1);
    }
}

// ----------------------------------------------------------------
// mark_finished_vehicles
// Any vehicle whose current_edge_id has gone past total_edges
// (edge case from rapid multi-step advancement) is clamped and
// marked finished here as a safety pass.
// ----------------------------------------------------------------
__global__ void mark_finished_vehicles(Vehicle* vehicles,
                                       int      num_vehicles,
                                       int*     finished_count)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vehicles) return;

    Vehicle& v = vehicles[vid];
    if (!v.finished && v.active &&
        v.current_edge_id >= v.total_edges)
    {
        v.finished = 1;
        atomicAdd(finished_count, 1);
    }
}

// ----------------------------------------------------------------
// Host-side launcher wrappers
// Called from main.cu or simulation.cu as needed.
// ----------------------------------------------------------------

void launch_reset_counters(int* d_active_count,
                           int* d_finished_count)
{
    reset_tick_counters<<<1, 1>>>(d_active_count, d_finished_count);
}

void launch_clear_lane_map(unsigned char* d_lane_map,
                           int            lane_map_size)
{
    int threads = 256;
    int blocks  = (lane_map_size + threads - 1) / threads;
    clear_lane_map<<<blocks, threads>>>(d_lane_map, lane_map_size);
}

void launch_gather_speed_stats(const Vehicle* d_vehicles,
                               int            num_vehicles,
                               float*         d_speed_sum,
                               int*           d_active_n)
{
    int threads = 256;
    int blocks  = (num_vehicles + threads - 1) / threads;
    gather_speed_stats<<<blocks, threads>>>(
        d_vehicles, num_vehicles, d_speed_sum, d_active_n);
}

void launch_mark_finished(Vehicle* d_vehicles,
                          int      num_vehicles,
                          int*     d_finished_count)
{
    int threads = 256;
    int blocks  = (num_vehicles + threads - 1) / threads;
    mark_finished_vehicles<<<blocks, threads>>>(
        d_vehicles, num_vehicles, d_finished_count);
}
