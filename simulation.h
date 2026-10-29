#pragma once
#include <cuda_runtime.h>

// One vehicle stored in GPU memory
struct Vehicle {
    int   vehicle_id;
    int   depart_tick;       // which time step this car starts moving
    int   current_edge_id;   // which road segment it is on right now
    int   pos_on_edge;       // how many metres along that segment
    float speed;             // current speed in m/s
    float accel;             // current acceleration in m/s^2
    int   total_edges;       // how many road segments are in its route
    int   active;            // 0 = not yet started, 1 = on the road
    int   finished;          // 1 = reached destination
};

// One road segment stored in GPU memory
struct EdgeData {
    int   edge_id;
    int   from_node;
    int   to_node;
    int   num_lanes;
    int   length_metres;
    float speed_limit_ms;    // speed limit in metres per second
    int   lane_map_offset;   // start index of this edge in the flat lane_map array
};

// Function defined in simulation.cu, called from main.cu
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
    int*           d_finished_count
);
