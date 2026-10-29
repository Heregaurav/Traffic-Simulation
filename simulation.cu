#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>
#include "simulation.h"

// ----------------------------------------------------------------
// IDM (Intelligent Driver Model)
// ----------------------------------------------------------------
__device__ float idm_acceleration(
    float my_speed,
    float lead_speed,
    float gap_metres,
    float speed_limit)
{
    float max_accel     = 1.5f;
    float comfort_brake = 3.0f;
    float min_gap       = 2.0f;
    float time_headway  = 1.5f;
    float delta         = 4.0f;

    float dv     = my_speed - lead_speed;
    float s_star = min_gap
                 + my_speed * time_headway
                 + (my_speed * dv) / (2.0f * sqrtf(max_accel * comfort_brake));
    s_star = fmaxf(s_star, min_gap);

    float safe_gap  = fmaxf(gap_metres, 0.5f);
    float free_road = 1.0f - powf(my_speed / fmaxf(speed_limit, 0.1f), delta);
    float braking   = powf(s_star / safe_gap, 2.0f);

    return fmaxf(max_accel * (free_road - braking), -comfort_brake);
}

// ----------------------------------------------------------------
// Main kernel: one thread = one vehicle
// ----------------------------------------------------------------
__global__ void simulate_vehicles_kernel(
    Vehicle*       vehicles,
    int            num_vehicles,
    EdgeData*      edges,
    int            num_edges,
    unsigned char* lane_map,
    int            lane_map_size,
    int            current_tick,
    float          dt,
    int*           active_count,
    int*           finished_count)
{
    int vid = blockIdx.x * blockDim.x + threadIdx.x;
    if (vid >= num_vehicles) return;

    Vehicle& v = vehicles[vid];

    if (current_tick < v.depart_tick) return;
    if (v.finished) return;

    // First activation
    if (!v.active) {
        v.active = 1;
        v.speed  = 0.0f;
        atomicAdd(active_count, 1);
    }

    // Safety: clamp edge id
    if (v.current_edge_id < 0 || v.current_edge_id >= num_edges) {
        v.finished = 1;
        atomicAdd(finished_count, 1);
        return;
    }

    EdgeData& edge = edges[v.current_edge_id];

    // Clamp pos_on_edge to valid range
    if (v.pos_on_edge < 0) v.pos_on_edge = 0;
    if (v.pos_on_edge >= edge.length_metres) v.pos_on_edge = edge.length_metres - 1;

    int my_cell = edge.lane_map_offset + v.pos_on_edge;

    // ---- Find nearest car ahead --------------------------------
    float gap        = 50.0f;
    float lead_speed = edge.speed_limit_ms;
    int   scan_dist  = (int)(2.0f * fmaxf(v.speed, 1.0f) * dt) + 3;

    for (int d = 1; d <= scan_dist; d++) {
        int check = my_cell + d;
        if (check < 0 || check >= lane_map_size) break;
        unsigned char val = lane_map[check];
        if (val != 255) {
            gap        = (float)d;
            lead_speed = (float)val;
            break;
        }
    }

    // ---- IDM: compute new speed --------------------------------
    float new_accel = idm_acceleration(v.speed, lead_speed, gap,
                                       edge.speed_limit_ms);
    float new_speed = fmaxf(0.0f, v.speed + new_accel * dt);
    new_speed = fminf(new_speed, edge.speed_limit_ms);

    // ---- Move forward ------------------------------------------
    // FIX: replaced illegal atomicCAS/atomicExch casts from unsigned char*
    // to unsigned int* (caused silent kernel abort on Ampere/sm_86).
    // Plain byte reads/writes are correct here — each vehicle owns its cell.
    int move    = (int)fmaxf(1.0f, new_speed * dt);
    int new_pos = v.pos_on_edge + move;

    if (new_pos >= edge.length_metres) {
        // Vacate old cell
        if (my_cell >= 0 && my_cell < lane_map_size)
            lane_map[my_cell] = 255;

        // Advance to next road segment
        v.current_edge_id++;
        v.pos_on_edge = 0;

        if (v.current_edge_id >= v.total_edges) {
            v.finished = 1;
            atomicAdd(finished_count, 1);
            return;
        }
    } else {
        int new_cell = edge.lane_map_offset + new_pos;
        if (new_cell >= 0 && new_cell < lane_map_size) {
            if (lane_map[new_cell] == 255) {
                // Cell is free — move in
                if (my_cell >= 0 && my_cell < lane_map_size)
                    lane_map[my_cell] = 255;
                lane_map[new_cell] = (unsigned char)fminf(new_speed, 254.0f);
                v.pos_on_edge = new_pos;
            }
            // else: cell taken, stay put this tick
        }
    }

    v.speed = new_speed;
    v.accel = new_accel;
}

// ----------------------------------------------------------------
// Host launcher — called from main.cu once per tick
// ----------------------------------------------------------------
void launch_simulation_step(
    Vehicle*       d_vehicles,
    int            num_vehicles,
    EdgeData*      d_edges,
    int            num_edges,
    unsigned char* d_lane_map,
    int            lane_map_size,
    int            current_tick,
    float          dt,
    int*           d_active_count,
    int*           d_finished_count)
{
    int threads_per_block = 256;
    int blocks = (num_vehicles + threads_per_block - 1) / threads_per_block;

    simulate_vehicles_kernel<<<blocks, threads_per_block>>>(
        d_vehicles, num_vehicles,
        d_edges,    num_edges,
        d_lane_map, lane_map_size,
        current_tick, dt,
        d_active_count, d_finished_count);

    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("CUDA kernel error at tick %d: %s\n",
               current_tick, cudaGetErrorString(err));
    }
}
