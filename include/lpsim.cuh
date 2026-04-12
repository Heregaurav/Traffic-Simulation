/**
 * LPSim - Large Scale Multi-GPU Parallel Traffic Simulation
 * Main Header File
 *
 * Based on: "Large scale multi-GPU based parallel traffic simulation for
 * accelerated traffic assignment and propagation"
 * Jiang et al., Transportation Research Part C (2024)
 */

#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sort.h>
#include <vector>
#include <string>
#include <unordered_map>
#include <iostream>
#include <cassert>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
#define EMPTY_CELL       255   // Sentinel indicating no vehicle
#define MAX_SPEED_MPS    254   // Max representable speed (m/s)
#define CELL_SIZE_M      1     // 1 cell = 1 meter
#define BLOCK_SIZE       256   // CUDA threads per block
#define MAX_LANES        8     // Maximum lanes per road segment
#define GHOST_FLAG_NO    0
#define GHOST_FLAG_YES   1

// ---------------------------------------------------------------------------
// IDM Parameters  (Intelligent Driver Model)
// ---------------------------------------------------------------------------
struct IDMParams {
    float a;      // Max acceleration (m/s^2)
    float b;      // Comfortable braking deceleration (m/s^2)
    float delta;  // Acceleration exponent (usually 4)
    float s0;     // Minimum gap at standstill (m)
    float T;      // Desired time headway (s)
    float v0;     // Desired speed / speed limit (m/s)
};

// ---------------------------------------------------------------------------
// Vehicle State  (stored in GPU global memory per vehicle)
// ---------------------------------------------------------------------------
struct VehicleState {
    int   vehicle_id;
    int   prev_edge;
    int   curr_edge;
    int   next_edge;
    int   pos_on_edge;   // Position (cell index) on current edge
    float speed;         // Current speed (m/s)
    float accel;         // Current acceleration (m/s^2)
    int   route_start;   // Index into route_path array
    int   route_len;     // Number of edges in route
    int   depart_time;   // Departure time (simulation ticks)
    int   active;        // 1 = active, 0 = not yet / finished
    int   in_ghost;      // GHOST_FLAG_YES / GHOST_FLAG_NO
    int   target_gpu;    // For inter-GPU transfer (-1 if local)
};

// ---------------------------------------------------------------------------
// Edge (Road Segment) Descriptor
// ---------------------------------------------------------------------------
struct EdgeData {
    int   edge_id;
    int   num_lanes;
    int   length_m;           // Length in meters
    int   lane_map_offset;    // Starting index in the flat LaneMap array
    int   upstream_node;
    int   downstream_node;
    float speed_limit;        // m/s
    int   gpu_id;             // Which GPU owns this edge
    int   is_ghost;           // 1 = ghost zone edge
};

// ---------------------------------------------------------------------------
// Intersection / Node Descriptor
// ---------------------------------------------------------------------------
struct NodeData {
    int   node_id;
    int   gpu_id;
    int   in_ghost;
    float signal_phase;       // Current green phase (seconds elapsed)
    int   signal_cycle;       // Total cycle length (seconds)
    int   num_in_edges;
    int   num_out_edges;
    int   in_edges[8];        // Up to 8 in-edges
    int   out_edges[8];       // Up to 8 out-edges
};

// ---------------------------------------------------------------------------
// Simulation Config
// ---------------------------------------------------------------------------
struct SimConfig {
    int   num_gpus;
    int   total_time_steps;
    float dt;                 // Time step size (seconds)
    int   num_vehicles;
    int   num_edges;
    int   num_nodes;
    int   lane_map_size;      // Total cells in flat LaneMap
    IDMParams idm;
};

// ---------------------------------------------------------------------------
// Per-GPU Simulation Context
// ---------------------------------------------------------------------------
struct GPUContext {
    int gpu_id;

    // Lane map: 1 int per meter; value = 255 (empty) or speed in m/s
    // Using int keeps CUDA atomic operations naturally aligned.
    thrust::device_vector<int> d_lane_map;

    // Edge and node descriptors
    thrust::device_vector<EdgeData>  d_edges;
    thrust::device_vector<NodeData>  d_nodes;

    // Vehicle states (dynamic size)
    thrust::device_vector<VehicleState> d_vehicles;

    // Ghost-zone transfer buffers
    thrust::device_vector<VehicleState> d_to_copy;   // Vehicles entering ghost zone
    thrust::device_vector<VehicleState> d_to_remove; // Vehicles leaving this GPU

    // Route paths (flat array, indexed via VehicleState.route_start / route_len)
    thrust::device_vector<int> d_route_paths;

    // Atomic counter for ghost-zone transfer lists
    thrust::device_vector<int> d_copy_counter;
    thrust::device_vector<int> d_remove_counter;

    cudaStream_t stream;
};
