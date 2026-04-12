/**
 * LPSim - Multi-GPU Simulation Orchestrator
 *
 * Manages:
 *  - GPU context initialization per device
 *  - Main simulation loop (time-driven, BSP model)
 *  - Ghost-zone creation between adjacent GPUs
 *  - Inter-GPU vehicle transfer (P2P via cudaMemcpyPeer or D-H-D fallback)
 *  - Device_Vector dynamic resizing for transferred vehicles
 *
 * Architecture reference: Fig. 4 and Section 3.3 of the paper.
 */

#include "../include/lpsim.cuh"
#include <chrono>
#include <fstream>
#include <sstream>
#include <algorithm>

// Forward declaration of CUDA kernel (defined in propagation.cu)
__global__ void kernel_propagate_vehicles(
    VehicleState* vehicles,
    int num_vehicles,
    int* lane_map,
    const EdgeData* edges,
    const NodeData* nodes,
    const int* route_paths,
    int current_tick,
    float dt,
    IDMParams idm,
    VehicleState* to_copy,
    VehicleState* to_remove,
    int* copy_counter,
    int* remove_counter);

// ===========================================================================
// GPU Context Initialization
// ===========================================================================
void init_gpu_context(GPUContext& ctx,
                      const SimConfig& cfg,
                      const std::vector<EdgeData>& h_edges,
                      const std::vector<NodeData>& h_nodes,
                      const std::vector<VehicleState>& h_vehicles,
                      const std::vector<int>& h_routes,
                      int lane_map_size)
{
    cudaSetDevice(ctx.gpu_id);
    cudaStreamCreate(&ctx.stream);

    // Initialize lane map (all cells empty = 255)
    ctx.d_lane_map.assign(lane_map_size, EMPTY_CELL);

    // Copy network data
    ctx.d_edges = h_edges;
    ctx.d_nodes = h_nodes;

    // Copy vehicle data (only vehicles belonging to this GPU's starting edges)
    ctx.d_vehicles = h_vehicles;

    // Route paths (shared across all GPUs via replicated global data)
    ctx.d_route_paths = h_routes;

    // Initialize transfer buffers (pre-allocate reasonable max size)
    int max_transfer = (int)(h_vehicles.size() * 0.05f) + 1024;
    ctx.d_to_copy.resize(max_transfer);
    ctx.d_to_remove.resize(max_transfer);

    ctx.d_copy_counter.resize(1, 0);
    ctx.d_remove_counter.resize(1, 0);

    std::cout << "[GPU " << ctx.gpu_id << "] Initialized: "
              << h_vehicles.size() << " vehicles, "
              << h_edges.size() << " edges, "
              << h_nodes.size() << " nodes, "
              << lane_map_size  << " lane cells\n";
}

// ===========================================================================
// Check P2P Support Between GPU Pair
// ===========================================================================
bool check_p2p(int src, int dst)
{
    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, src, dst);
    return can_access == 1;
}

void enable_p2p(int num_gpus)
{
    for (int i = 0; i < num_gpus; ++i)
        for (int j = 0; j < num_gpus; ++j)
            if (i != j && check_p2p(i, j)) {
                cudaSetDevice(i);
                cudaDeviceEnablePeerAccess(j, 0);
            }
}

// ===========================================================================
// Inter-GPU Vehicle Transfer
//   Uses P2P (NVLink/PCIe) when available; falls back to Device-Host-Device
// ===========================================================================
void transfer_vehicles(GPUContext& src_ctx,
                       GPUContext& dst_ctx,
                       bool use_p2p = true)
{
    // Retrieve copy count from device
    int h_copy_count = 0;
    cudaSetDevice(src_ctx.gpu_id);
    cudaMemcpy(&h_copy_count,
               thrust::raw_pointer_cast(src_ctx.d_copy_counter.data()),
               sizeof(int), cudaMemcpyDeviceToHost);

    if (h_copy_count == 0) return;

    if (use_p2p && check_p2p(src_ctx.gpu_id, dst_ctx.gpu_id)) {
        // ----- P2P path: direct GPU-to-GPU -----
        // Grow dst vehicle vector to accommodate incoming vehicles
        int old_size = dst_ctx.d_vehicles.size();
        dst_ctx.d_vehicles.resize(old_size + h_copy_count);

        cudaMemcpyPeer(
            thrust::raw_pointer_cast(dst_ctx.d_vehicles.data()) + old_size,
            dst_ctx.gpu_id,
            thrust::raw_pointer_cast(src_ctx.d_to_copy.data()),
            src_ctx.gpu_id,
            h_copy_count * sizeof(VehicleState));
    } else {
        // ----- D-H-D fallback -----
        std::vector<VehicleState> h_buffer(h_copy_count);

        cudaSetDevice(src_ctx.gpu_id);
        cudaMemcpy(h_buffer.data(),
                   thrust::raw_pointer_cast(src_ctx.d_to_copy.data()),
                   h_copy_count * sizeof(VehicleState),
                   cudaMemcpyDeviceToHost);

        cudaSetDevice(dst_ctx.gpu_id);
        int old_size = dst_ctx.d_vehicles.size();
        dst_ctx.d_vehicles.resize(old_size + h_copy_count);
        cudaMemcpy(thrust::raw_pointer_cast(dst_ctx.d_vehicles.data()) + old_size,
                   h_buffer.data(),
                   h_copy_count * sizeof(VehicleState),
                   cudaMemcpyHostToDevice);
    }

    // Reset transfer counter
    cudaSetDevice(src_ctx.gpu_id);
    thrust::fill(src_ctx.d_copy_counter.begin(), src_ctx.d_copy_counter.end(), 0);
    thrust::fill(src_ctx.d_remove_counter.begin(), src_ctx.d_remove_counter.end(), 0);
}

// ===========================================================================
// CUDA Kernel Launch Wrappers
// ===========================================================================
void launch_propagate(GPUContext& ctx, int tick, const SimConfig& cfg)
{
    cudaSetDevice(ctx.gpu_id);

    int num_vehicles = (int)ctx.d_vehicles.size();
    if (num_vehicles == 0) return;

    int grid = (num_vehicles + BLOCK_SIZE - 1) / BLOCK_SIZE;

    kernel_propagate_vehicles<<<grid, BLOCK_SIZE, 0, ctx.stream>>>(
        thrust::raw_pointer_cast(ctx.d_vehicles.data()),
        num_vehicles,
        thrust::raw_pointer_cast(ctx.d_lane_map.data()),
        thrust::raw_pointer_cast(ctx.d_edges.data()),
        thrust::raw_pointer_cast(ctx.d_nodes.data()),
        thrust::raw_pointer_cast(ctx.d_route_paths.data()),
        tick,
        cfg.dt,
        cfg.idm,
        thrust::raw_pointer_cast(ctx.d_to_copy.data()),
        thrust::raw_pointer_cast(ctx.d_to_remove.data()),
        thrust::raw_pointer_cast(ctx.d_copy_counter.data()),
        thrust::raw_pointer_cast(ctx.d_remove_counter.data())
    );
}

// ===========================================================================
// Pre-process: Sort OD pairs by departure time to reduce warp divergence
//   (Section 4.2, Table 6 of paper: sorting reduces instructions by 6.7x)
// ===========================================================================
void sort_vehicles_by_departure(std::vector<VehicleState>& vehicles)
{
    std::sort(vehicles.begin(), vehicles.end(),
              [](const VehicleState& a, const VehicleState& b){
                  return a.depart_time < b.depart_time;
              });
    std::cout << "[Pre-process] Vehicles sorted by departure time ("
              << vehicles.size() << " vehicles)\n";
}

// ===========================================================================
// Main Simulation Loop
//   Implements BSP (Bulk Synchronous Parallel) model:
//     For each time step:
//       1. All GPUs compute vehicle propagation in parallel (superstep)
//       2. Synchronize (cudaStreamSynchronize)
//       3. Communicate ghost-zone vehicles between GPUs (h-relation)
// ===========================================================================
void run_simulation(std::vector<GPUContext>& gpu_contexts,
                    const SimConfig& cfg,
                    bool use_p2p = true)
{
    int num_gpus = (int)gpu_contexts.size();

    std::cout << "\n[Simulation] Starting LPSim with " << num_gpus << " GPU(s)\n";
    std::cout << "[Simulation] Time steps: " << cfg.total_time_steps
              << "  dt=" << cfg.dt << "s\n\n";

    auto wall_start = std::chrono::high_resolution_clock::now();

    for (int tick = 0; tick < cfg.total_time_steps; ++tick)
    {
        // ------------------------------------------------------------------
        // Phase 1: Superstep - launch kernels on all GPUs asynchronously
        // ------------------------------------------------------------------
        for (auto& ctx : gpu_contexts) {
            launch_propagate(ctx, tick, cfg);
        }

        // ------------------------------------------------------------------
        // Phase 2: Barrier - wait for all GPUs to finish this time step
        // ------------------------------------------------------------------
        for (auto& ctx : gpu_contexts) {
            cudaSetDevice(ctx.gpu_id);
            cudaStreamSynchronize(ctx.stream);
        }

        // ------------------------------------------------------------------
        // Phase 3: h-relation - inter-GPU vehicle transfers
        //   Each GPU pair exchanges vehicles that crossed the boundary
        // ------------------------------------------------------------------
        for (int i = 0; i < num_gpus; ++i) {
            for (int j = 0; j < num_gpus; ++j) {
                if (i != j) {
                    transfer_vehicles(gpu_contexts[i], gpu_contexts[j], use_p2p);
                }
            }
        }

        // Progress reporting every 60 seconds of simulation time
        if (tick % (int)(60.0f / cfg.dt) == 0) {
            auto now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            std::cout << "[t=" << (tick * cfg.dt / 60.0f) << " min] "
                      << "Wall time: " << elapsed << "s\n";
        }
    }

    auto wall_end = std::chrono::high_resolution_clock::now();
    double total_s = std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << "\n[Simulation] DONE. Total wall time: "
              << total_s << " s  ("
              << total_s / 60.0 << " min)\n";
}

// ===========================================================================
// Roofline Analysis Helper
//   Estimates arithmetic intensity for comparison with hardware rooflines
// ===========================================================================
struct RooflineStats {
    long long total_instructions;
    long long total_memory_bytes;
    double    arithmetic_intensity;  // Instructions / Byte
    double    achieved_gips;         // Giga-instructions per second
};

RooflineStats compute_roofline_stats(long long instructions,
                                     long long bytes,
                                     double elapsed_s)
{
    RooflineStats s;
    s.total_instructions    = instructions;
    s.total_memory_bytes    = bytes;
    s.arithmetic_intensity  = (bytes > 0) ? (double)instructions / bytes : 0.0;
    s.achieved_gips         = (elapsed_s > 0)
                              ? (double)instructions / elapsed_s / 1e9
                              : 0.0;
    return s;
}

// ===========================================================================
// Amdahl's Law Speedup Prediction
//   S(n) = 1 / (f_serial + (1-f_serial)/n)
// ===========================================================================
double amdahl_speedup(double f_serial, int n_processors)
{
    return 1.0 / (f_serial + (1.0 - f_serial) / n_processors);
}

// ===========================================================================
// Strong Scaling Efficiency
//   E(n) = S(n) / n  where S(n) = T(1) / T(n)
// ===========================================================================
double scaling_efficiency(double t1, double tn, int n)
{
    double speedup = t1 / tn;
    return speedup / n;
}
