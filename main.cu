#include <iostream>
#include <iomanip>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cuda_runtime.h>
#include "simulation.h"

// ----------------------------------------------------------------
// Load network.csv
// ----------------------------------------------------------------
std::vector<EdgeData> load_network(const std::string& filepath,
                                   int& total_lane_cells)
{
    std::vector<EdgeData> edges;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << filepath << std::endl;
        return edges;
    }

    std::string line;
    std::getline(file, line); // skip header
    total_lane_cells = 0;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        EdgeData e{};

        try {
            std::getline(ss, tok, ','); e.edge_id        = std::stoi(tok);
            std::getline(ss, tok, ','); e.from_node      = std::stoi(tok);
            std::getline(ss, tok, ','); e.to_node        = std::stoi(tok);
            std::getline(ss, tok, ','); e.num_lanes      = std::stoi(tok);
            std::getline(ss, tok, ','); e.length_metres  = (int)std::stof(tok);
            std::getline(ss, tok, ','); e.speed_limit_ms = std::stof(tok);
        } catch (...) { continue; }

        if (e.length_metres  <= 0) e.length_metres  = 10;
        if (e.num_lanes      <= 0) e.num_lanes       = 1;
        if (e.speed_limit_ms <= 0) e.speed_limit_ms  = 8.3f;

        e.lane_map_offset = total_lane_cells;
        total_lane_cells += e.num_lanes * e.length_metres;
        edges.push_back(e);
    }
    return edges;
}

// ----------------------------------------------------------------
// Simple struct to hold one raw demand row (avoids tuple/C++17)
// ----------------------------------------------------------------
struct RawTrip {
    int   vehicle_id;
    int   origin;
    float depart_s;
};

// ----------------------------------------------------------------
// Load demand.csv
// Scales departure times so all vehicles depart within the run.
// ----------------------------------------------------------------
std::vector<Vehicle> load_demand(const std::string& filepath,
                                 int num_edges, float dt, int max_steps)
{
    std::vector<Vehicle> vehicles;
    std::vector<RawTrip> raw;

    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "ERROR: cannot open " << filepath << std::endl;
        return vehicles;
    }

    std::string line;
    std::getline(file, line); // skip header

    float max_depart_s = 1.0f;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        RawTrip t;
        try {
            std::getline(ss, tok, ','); t.vehicle_id = std::stoi(tok);
            std::getline(ss, tok, ','); t.origin     = std::stoi(tok);
            std::getline(ss, tok, ',');               // skip destination
            std::getline(ss, tok, ','); t.depart_s   = std::stof(tok);
        } catch (...) { continue; }

        if (t.depart_s > max_depart_s) max_depart_s = t.depart_s;
        raw.push_back(t);
    }

    // Scale so the largest departure time maps to 90% of total steps
    float scale = ((float)max_steps * 0.9f) / max_depart_s;

    std::cerr << "  demand: " << raw.size() << " trips, "
              << "max_depart=" << (int)max_depart_s << "s, "
              << "tick_scale=" << scale << "\n";

    for (int i = 0; i < (int)raw.size(); i++) {
        RawTrip& t = raw[i];
        Vehicle v;
        v.vehicle_id      = i;
        v.depart_tick     = (int)(t.depart_s * scale / dt);
        if (v.depart_tick >= max_steps) v.depart_tick = max_steps - 1;
        v.current_edge_id = (int)((unsigned int)t.origin % (unsigned int)num_edges);
        v.total_edges     = 5 + (i % 8);
        v.pos_on_edge     = 0;
        v.speed           = 0.0f;
        v.accel           = 0.0f;
        v.active          = 0;
        v.finished        = 0;
        vehicles.push_back(v);
    }

    std::sort(vehicles.begin(), vehicles.end(),
        [](const Vehicle& a, const Vehicle& b){
            return a.depart_tick < b.depart_tick;
        });

    return vehicles;
}

// ----------------------------------------------------------------
// main
// ----------------------------------------------------------------
int main(int argc, char* argv[])
{
    int         gpus      = 1;
    std::string network   = "";
    std::string demand    = "";
    std::string partition = "balanced";
    int         steps     = 3600;
    float       dt        = 1.0f;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--gpus"      && i+1 < argc) gpus      = std::stoi(argv[++i]);
        else if (arg == "--network"   && i+1 < argc) network   = argv[++i];
        else if (arg == "--demand"    && i+1 < argc) demand    = argv[++i];
        else if (arg == "--partition" && i+1 < argc) partition = argv[++i];
        else if (arg == "--steps"     && i+1 < argc) steps     = std::stoi(argv[++i]);
        else if (arg == "--dt"        && i+1 < argc) dt        = std::stof(argv[++i]);
    }

    std::cout << "\n========================================" << std::endl;
    std::cout << "  LPSim — GPU Traffic Simulator"           << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  GPUs      : " << gpus      << std::endl;
    std::cout << "  Partition : " << partition << std::endl;
    std::cout << "  Steps     : " << steps
              << "  (" << (steps*dt/60.0f) << " min simulated)" << std::endl;
    std::cout << "  dt        : " << dt << " s" << std::endl;
    std::cout << "========================================\n" << std::endl;

    std::cout << "[1/4] Loading data..." << std::endl;
    int total_lane_cells = 0;
    std::vector<EdgeData> h_edges    = load_network(network, total_lane_cells);
    std::vector<Vehicle>  h_vehicles = load_demand(demand, (int)h_edges.size(), dt, steps);

    if (h_edges.empty() || h_vehicles.empty()) {
        std::cerr << "Failed to load data. Exiting." << std::endl;
        return 1;
    }

    int num_vehicles = (int)h_vehicles.size();
    int num_edges    = (int)h_edges.size();

    int departs_in_run = 0;
    for (int i = 0; i < num_vehicles; i++)
        if (h_vehicles[i].depart_tick < steps) departs_in_run++;

    std::cout << "  Road edges      : " << num_edges      << std::endl;
    std::cout << "  Vehicles total  : " << num_vehicles   << std::endl;
    std::cout << "  Depart in run   : " << departs_in_run << "  <- should be > 0" << std::endl;
    std::cout << "  Lane map        : " << total_lane_cells/1000 << " KB\n" << std::endl;

    std::cout << "[2/4] Allocating GPU memory..." << std::endl;

    Vehicle*       d_vehicles       = nullptr;
    EdgeData*      d_edges          = nullptr;
    unsigned char* d_lane_map       = nullptr;
    int*           d_active_count   = nullptr;
    int*           d_finished_count = nullptr;

    cudaError_t err;
    err = cudaMalloc(&d_vehicles,  num_vehicles     * sizeof(Vehicle));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc vehicles failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_edges,     num_edges        * sizeof(EdgeData));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc edges failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    err = cudaMalloc(&d_lane_map,  total_lane_cells * sizeof(unsigned char));
    if (err != cudaSuccess) {
        std::cerr << "cudaMalloc lane_map failed: " << cudaGetErrorString(err) << std::endl;
        return 1;
    }
    cudaMalloc(&d_active_count,   sizeof(int));
    cudaMalloc(&d_finished_count, sizeof(int));

    cudaMemcpy(d_vehicles, h_vehicles.data(),
               num_vehicles * sizeof(Vehicle),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_edges,    h_edges.data(),
               num_edges    * sizeof(EdgeData), cudaMemcpyHostToDevice);

    cudaMemset(d_lane_map,       255, total_lane_cells * sizeof(unsigned char));
    cudaMemset(d_active_count,   0,   sizeof(int));   // init ONCE before loop
    cudaMemset(d_finished_count, 0,   sizeof(int));   // init ONCE before loop

    int threads_per_block = 256;
    int blocks_needed     = (num_vehicles + threads_per_block - 1) / threads_per_block;
    int total_threads     = blocks_needed * threads_per_block;

    std::cout << "  Blocks       : " << blocks_needed     << std::endl;
    std::cout << "  Threads/blk  : " << threads_per_block << std::endl;
    std::cout << "  TOTAL THREADS: " << total_threads
              << "  (one per vehicle)\n" << std::endl;

    std::cout << "[3/4] Running " << steps << " time steps..." << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    auto wall_start   = std::chrono::high_resolution_clock::now();
    int  report_every = steps / 10;
    if (report_every < 1) report_every = 1;

    for (int tick = 0; tick < steps; tick++) {

        // Counters are NEVER reset inside the loop — they accumulate totals
        launch_simulation_step(
            d_vehicles,    num_vehicles,
            d_edges,       num_edges,
            d_lane_map,    total_lane_cells,
            tick, dt,
            d_active_count, d_finished_count);

        if (tick % report_every == 0 || tick == steps - 1) {
            int h_active = 0, h_finished = 0;
            cudaMemcpy(&h_active,   d_active_count,   sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&h_finished, d_finished_count, sizeof(int), cudaMemcpyDeviceToHost);

            auto   now     = std::chrono::high_resolution_clock::now();
            double elapsed = std::chrono::duration<double>(now - wall_start).count();
            float  sim_min = (tick * dt) / 60.0f;
            int    pct     = (tick * 100) / steps;

            std::cout << "  " << std::setw(3) << pct << "% | "
                      << "sim=" << std::fixed << std::setprecision(1) << sim_min << "min | "
                      << "wall=" << std::setprecision(2) << elapsed << "s | "
                      << "active=" << h_active << " | "
                      << "done=" << h_finished
                      << std::endl;
        }
    }

    int h_active = 0, h_finished = 0;
    cudaMemcpy(&h_active,   d_active_count,   sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_finished, d_finished_count, sizeof(int), cudaMemcpyDeviceToHost);

    auto   wall_end = std::chrono::high_resolution_clock::now();
    double total_s  = std::chrono::duration<double>(wall_end - wall_start).count();

    std::cout << std::string(60, '-') << std::endl;
    std::cout << "\n[4/4] Results:" << std::endl;
    std::cout << "\n========================================" << std::endl;
    std::cout << "  LPSim Simulation Results"                << std::endl;
    std::cout << "========================================" << std::endl;
    std::cout << "  Vehicles loaded     : " << num_vehicles  << std::endl;
    std::cout << "  Vehicles activated  : " << h_active      << std::endl;
    std::cout << "  Vehicles finished   : " << h_finished    << std::endl;
    std::cout << "  Time steps run      : " << steps         << std::endl;
    std::cout << "  Simulated duration  : "
              << (steps * dt / 60.0f) << " minutes"          << std::endl;
    std::cout << "  Wall clock time     : "
              << std::fixed << std::setprecision(2)
              << total_s << " seconds"                        << std::endl;
    std::cout << "  Speed vs real time  : "
              << std::setprecision(1)
              << (steps * dt) / total_s << "x faster"        << std::endl;
    std::cout << "  GPU threads used    : " << total_threads
              << " (" << num_vehicles << " vehicles)"         << std::endl;
    std::cout << "========================================" << std::endl;

    cudaFree(d_vehicles);
    cudaFree(d_edges);
    cudaFree(d_lane_map);
    cudaFree(d_active_count);
    cudaFree(d_finished_count);

    return 0;
}
