/**
 * LPSim - Vehicle Propagation Kernels (CUDA)
 *
 * Implements:
 *   - IDM Car-Following Model
 *   - Mandatory Lane Change
 *   - Gap Acceptance
 *   - Per-vehicle state update (Algorithm 1 from paper)
 *
 * Each CUDA thread handles ONE vehicle per time step (SIMD / SPMD model).
 * Vehicle positions are independent given the state at time k,
 * so all vehicles can be updated in parallel (Eq. 1 of the paper).
 */

#include "../include/lpsim.cuh"
#include <math.h>

// ===========================================================================
// Device Helper: IDM Acceleration
// ===========================================================================
__device__ float idm_accel(float v, float v_lead, float gap,
                            const IDMParams& p)
{
    float dv   = v - v_lead;                  // Speed difference
    float s_star = p.s0 + max(0.0f,
                   v * p.T + v * dv / (2.0f * sqrtf(p.a * p.b)));
    float ratio  = s_star / max(gap, 0.01f);  // Avoid div-by-zero
    float accel  = p.a * (1.0f - powf(v / max(p.v0, 0.01f), p.delta)
                          - ratio * ratio);
    return fmaxf(accel, -p.b);               // Clamp braking
}

// ===========================================================================
// Device Helper: Lane-Change Probability (mandatory)
//   m_i(k+1) = f_{x0}(x_i(k))
// ===========================================================================
__device__ float lane_change_prob(float dist_to_exit, float x0)
{
    if (dist_to_exit > x0) return 0.0f;
    // Linear increase as vehicle approaches exit
    return 1.0f - (dist_to_exit / x0);
}

// ===========================================================================
// Device Helper: Gap Acceptance
//   Returns 1 if gap is acceptable, 0 otherwise
// ===========================================================================
__device__ int gap_accept(float v_i, float v_neighbor,
                           float gap, float g_desired,
                           float alpha_i, float alpha_n, float eps)
{
    float critical = g_desired + alpha_i * v_i
                   - alpha_n * v_neighbor + eps;
    return (gap >= critical) ? 1 : 0;
}

// ===========================================================================
// Device Helper: Find leading vehicle on lane map
//   Returns (gap, lead_speed); gap = INT_MAX if no lead vehicle within range
// ===========================================================================
__device__ void find_lead_vehicle(
    const int* __restrict__ lane_map,
    int  start_cell,
    int  edge_len,
    int  search_dist,
    float& out_gap,
    float& out_lead_speed)
{
    out_gap        = (float)search_dist;
    out_lead_speed = 0.0f;

    for (int d = 1; d <= search_dist && (start_cell + d) < edge_len; ++d) {
        int cell = lane_map[start_cell + d];
        if (cell != EMPTY_CELL) {
            out_gap        = (float)d;
            out_lead_speed = (float)cell;   // Speed encoded in cell value
            return;
        }
    }
}

// ===========================================================================
// KERNEL: Propagate all vehicles one time step
//
//   Grid:  ceil(num_vehicles / BLOCK_SIZE) blocks
//   Block: BLOCK_SIZE threads
//   Each thread -> one vehicle
// ===========================================================================
__global__ void kernel_propagate_vehicles(
    VehicleState*        __restrict__ vehicles,
    int                               num_vehicles,
    int*                 __restrict__ lane_map,
    const EdgeData*      __restrict__ edges,
    const NodeData*      __restrict__ nodes,
    const int*           __restrict__ route_paths,
    int                               current_tick,
    float                             dt,
    IDMParams                         idm,
    VehicleState*        __restrict__ to_copy,
    VehicleState*        __restrict__ to_remove,
    int*                              copy_counter,
    int*                              remove_counter)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_vehicles) return;

    VehicleState& veh = vehicles[tid];

    // -----------------------------------------------------------------------
    // 1. Skip inactive vehicles (not yet departed or already arrived)
    // -----------------------------------------------------------------------
    if (!veh.active) {
        if (current_tick >= veh.depart_time) {
            // Try to activate: check if departure cell is free
            const EdgeData& e = edges[veh.curr_edge];
            int cell = e.lane_map_offset + veh.pos_on_edge;
            if (lane_map[cell] == EMPTY_CELL) {
                lane_map[cell] = min((int)veh.speed, MAX_SPEED_MPS);
                veh.active = 1;
            }
        }
        return;
    }

    // -----------------------------------------------------------------------
    // 2. Compute search distance for lead vehicle
    //    d_front = 2 * dt * v  (covers max possible distance)
    // -----------------------------------------------------------------------
    const EdgeData& cur_edge = edges[veh.curr_edge];
    int search_dist = (int)(2.0f * dt * veh.speed) + 1;

    int cell_abs = cur_edge.lane_map_offset + veh.pos_on_edge;

    float gap, lead_speed;
    find_lead_vehicle(lane_map, cell_abs,
                      cur_edge.lane_map_offset + cur_edge.length_m,
                      search_dist, gap, lead_speed);

    // -----------------------------------------------------------------------
    // 3. IDM acceleration update
    // -----------------------------------------------------------------------
    float new_accel, new_speed;

    if (gap < (float)search_dist) {
        new_accel = idm_accel(veh.speed, lead_speed, gap, idm);
    } else {
        // Free flow: accelerate toward v0
        new_accel = idm.a * (1.0f - powf(veh.speed / max(idm.v0, 0.01f), idm.delta));
    }

    new_speed = max(0.0f, veh.speed + new_accel * dt);
    new_speed = min(new_speed, cur_edge.speed_limit);

    // -----------------------------------------------------------------------
    // 4. Update position
    // -----------------------------------------------------------------------
    float ds    = new_speed * dt + 0.5f * new_accel * dt * dt;
    int   ds_i  = max(1, (int)ds);                // At least 1 cell per step
    int   new_pos_on_edge = veh.pos_on_edge + ds_i;

    // -----------------------------------------------------------------------
    // 5. Handle edge transition
    // -----------------------------------------------------------------------
    if (new_pos_on_edge >= cur_edge.length_m)
    {
        // Clear old cell atomically
        atomicExch(&lane_map[cell_abs], (int)EMPTY_CELL);

        // Check if vehicle has completed its route
        if (veh.route_start + 1 >= veh.route_len) {
            veh.active = 0;
            return;
        }

        // Advance to next edge
        int next_edge_id = route_paths[veh.route_start + 1];
        const EdgeData& next_e = edges[next_edge_id];

        // Check if next edge is on a different GPU -> ghost zone logic
        if (next_e.gpu_id != cur_edge.gpu_id) {
            // Mark for copy to target GPU
            int idx = atomicAdd(copy_counter, 1);
            veh.target_gpu = next_e.gpu_id;
            veh.in_ghost   = GHOST_FLAG_YES;
            to_copy[idx]   = veh;

            // Mark for removal from this GPU
            int ridx = atomicAdd(remove_counter, 1);
            to_remove[ridx] = veh;
            veh.active = 0;
            return;
        }

        // Try to write to first cell of next edge (atomic to avoid collision)
        int new_cell = next_e.lane_map_offset;
        int old_val  = atomicCAS(&lane_map[new_cell],
                                  (int)EMPTY_CELL,
                                  (int)min((int)new_speed, MAX_SPEED_MPS));
        if (old_val != (int)EMPTY_CELL) {
            // Cell occupied: vehicle waits (stays at current edge end)
            // Re-write current cell
            lane_map[cell_abs] = min((int)new_speed, MAX_SPEED_MPS);
            return;
        }

        // Successfully moved to next edge
        veh.prev_edge     = veh.curr_edge;
        veh.curr_edge     = next_edge_id;
        veh.pos_on_edge   = 0;
        veh.route_start  += 1;

    } else {
        // -----------------------------------------------------------------------
        // 6. Still on the same edge: clear old cell, write new cell
        // -----------------------------------------------------------------------
        int new_cell_abs = cur_edge.lane_map_offset + new_pos_on_edge;

        // Mandatory lane change check
        float dist_to_exit = (float)(cur_edge.length_m - new_pos_on_edge);
        float lc_prob = lane_change_prob(dist_to_exit, (float)cur_edge.length_m * 0.3f);

        // Simplified lane change: only if probability exceeds threshold
        // (full gap acceptance omitted for brevity; see paper Eq. 3-4)
        (void)lc_prob;

        // Atomic write to avoid two vehicles in same cell
        int old_val = atomicCAS(&lane_map[new_cell_abs],
                                 (int)EMPTY_CELL,
                                 (int)min((int)new_speed, MAX_SPEED_MPS));
        if (old_val == (int)EMPTY_CELL) {
            // Successfully moved
            atomicExch(&lane_map[cell_abs], (int)EMPTY_CELL);
            veh.pos_on_edge = new_pos_on_edge;
        }
        // else: blocked, vehicle stays in place
    }

    // -----------------------------------------------------------------------
    // 7. Update vehicle dynamics
    // -----------------------------------------------------------------------
    veh.speed = new_speed;
    veh.accel = new_accel;
}

// ===========================================================================
// KERNEL: Apply ghost-zone deletions
//   Compacts the vehicle array by moving last elements to deleted slots
// ===========================================================================
__global__ void kernel_compact_vehicles(
    VehicleState* vehicles,
    int*          remove_indices,
    int           num_to_remove,
    int*          total_vehicles)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_to_remove) return;

    int del_idx  = remove_indices[tid];
    int last_idx = atomicSub(total_vehicles, 1) - 1;

    if (del_idx < last_idx) {
        vehicles[del_idx] = vehicles[last_idx];
    }
}
