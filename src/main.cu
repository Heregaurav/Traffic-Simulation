/**
 * LPSim - Main Driver
 *
 * Entry point for the Large Scale Multi-GPU Parallel Traffic Simulation.
 * Handles:
 *   - Command-line configuration
 *   - Network / demand data loading (OSM-style CSV)
 *   - Pre-processing (sorting, routing)
 *   - Partitioning strategy selection
 *   - Multi-GPU context setup
 *   - Simulation execution
 *   - Performance reporting
 *
 * Usage:
 *   ./lpsim --gpus 2 --network network.csv --demand demand.csv \
 *           --partition balanced --steps 43200 --dt 1.0
 */

#include "../include/lpsim.cuh"
#include <cstring>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <chrono>
#include <unordered_map>

// Forward declarations
void sort_vehicles_by_departure(std::vector<VehicleState>& vehicles);
void init_gpu_context(GPUContext&, const SimConfig&,
                      const std::vector<EdgeData>&,
                      const std::vector<NodeData>&,
                      const std::vector<VehicleState>&,
                      const std::vector<int>&, int);
void run_simulation(std::vector<GPUContext>&, const SimConfig&, bool);
void enable_p2p(int);

// Partitioning functions (from partitioning.cpp)
struct GraphEdge { int src, dst; float weight; };
struct Partition {
    std::vector<int> node_to_gpu;
    std::vector<std::vector<int>> gpu_to_nodes;
    float imbalance_ratio;
    int cut_edges;
};
Partition run_partitioning(int, int, const std::vector<GraphEdge>&,
                            const std::vector<float>&,
                            const std::vector<float>&,
                            bool = true, float = 0.1f);

// ===========================================================================
// CSV Data Loader - Network Edges
//   Format: edge_id, src_node, dst_node, num_lanes, length_m, speed_limit_mps
// ===========================================================================
std::vector<EdgeData> load_network(const std::string& path,
                                   int& total_lane_cells)
{
    std::vector<EdgeData> edges;
    std::ifstream f(path);
    if (!f.is_open()) {
        // Generate synthetic small network for testing
        std::cerr << "[Warning] Network file not found, using synthetic 5-node ring\n";
        total_lane_cells = 0;
        int offset = 0;
        for (int i = 0; i < 5; ++i) {
            EdgeData e{};
            e.edge_id          = i;
            e.num_lanes        = 2;
            e.length_m         = 500;
            e.lane_map_offset  = offset;
            e.upstream_node    = i;
            e.downstream_node  = (i + 1) % 5;
            e.speed_limit      = 13.9f;  // ~50 km/h
            e.gpu_id           = 0;
            e.is_ghost         = 0;
            edges.push_back(e);
            offset += e.num_lanes * e.length_m;
        }
        total_lane_cells = offset;
        return edges;
    }

    std::string line;
    std::getline(f, line); // skip header
    total_lane_cells = 0;
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        EdgeData e{};
        char comma;
        ss >> e.edge_id >> comma >> e.upstream_node >> comma
           >> e.downstream_node >> comma >> e.num_lanes >> comma
           >> e.length_m >> comma >> e.speed_limit;
        e.lane_map_offset = total_lane_cells;
        e.gpu_id  = 0;
        e.is_ghost = 0;
        total_lane_cells += e.num_lanes * e.length_m;
        edges.push_back(e);
    }
    std::cout << "[Loader] " << edges.size() << " edges loaded\n";
    return edges;
}

// ===========================================================================
// CSV Data Loader - Demand (OD Pairs)
//   Format: vehicle_id, origin_edge, dest_edge, depart_time_s, route_edge_0, ...
// ===========================================================================
std::vector<VehicleState> load_demand(const std::string& path,
                                      std::vector<int>& route_paths,
                                      int num_edges)
{
    std::vector<VehicleState> vehicles;
    std::ifstream f(path);

    if (!f.is_open()) {
        std::cerr << "[Warning] Demand file not found, generating 1000 synthetic trips\n";
        int N = 1000;
        for (int i = 0; i < N; ++i) {
            VehicleState v{};
            v.vehicle_id    = i;
            v.curr_edge     = i % 5;
            v.pos_on_edge   = 0;
            v.speed         = 10.0f;
            v.accel         = 0.0f;
            v.depart_time   = i * 2;   // Stagger departures
            v.active        = 0;
            v.in_ghost      = 0;
            v.target_gpu    = -1;
            v.route_start   = (int)route_paths.size();
            v.route_len     = 2;
            // Simple route: curr_edge -> next_edge
            route_paths.push_back(i % 5);
            route_paths.push_back((i + 1) % 5);
            vehicles.push_back(v);
        }
        return vehicles;
    }

    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        VehicleState v{};
        char comma;
        ss >> v.vehicle_id >> comma >> v.curr_edge >> comma;
        int dest; ss >> dest >> comma >> v.depart_time;
        v.route_start = (int)route_paths.size();
        v.route_len   = 0;
        v.speed       = 5.0f;
        v.accel       = 0.0f;
        v.active      = 0;
        v.in_ghost    = 0;
        v.target_gpu  = -1;
        v.pos_on_edge = 0;
        // Read route edges
        int edge_id;
        while (ss >> comma >> edge_id) {
            route_paths.push_back(edge_id);
            v.route_len++;
        }
        vehicles.push_back(v);
    }
    std::cout << "[Loader] " << vehicles.size() << " vehicles loaded\n";
    return vehicles;
}

// ===========================================================================
// Performance Report
// ===========================================================================
void print_performance_report(int num_vehicles, int num_gpus,
                               double elapsed_s, double elapsed_cpu_s = 0.0)
{
    std::cout << "\n";
    std::cout << "==================================================\n";
    std::cout << "         LPSim Performance Report\n";
    std::cout << "==================================================\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "  Vehicles simulated : " << num_vehicles << "\n";
    std::cout << "  GPUs used          : " << num_gpus << "\n";
    std::cout << "  Wall time (GPU)    : " << elapsed_s   << " s"
              << "  (" << elapsed_s/60.0 << " min)\n";
    if (elapsed_cpu_s > 0.0) {
        double speedup = elapsed_cpu_s / elapsed_s;
        std::cout << "  CPU baseline time  : " << elapsed_cpu_s << " s\n";
        std::cout << "  Speedup over CPU   : " << speedup << "x\n";
    }
    // Amdahl prediction for various GPU counts
    std::cout << "\n  Amdahl Speedup Predictions (f_serial=0.05):\n";
    double f = 0.05;
    for (int n : {1, 2, 4, 8}) {
        double s = 1.0 / (f + (1.0 - f) / n);
        std::cout << "    " << n << " GPUs: " << std::setprecision(2)
                  << s << "x\n";
    }
    std::cout << "==================================================\n\n";
}

// ===========================================================================
// Main
// ===========================================================================
int main(int argc, char** argv)
{
    // ---- Parse arguments ----
    int         num_gpus         = 1;
    std::string network_file     = "";
    std::string demand_file      = "";
    std::string partition_type   = "balanced";  // "balanced" or "community"
    int         total_time_steps = 43200;        // 12 hours at dt=1s
    float       dt               = 1.0f;

    for (int i = 1; i < argc; i++) {
        if      (!strcmp(argv[i], "--gpus"))      num_gpus         = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--network"))   network_file     = argv[++i];
        else if (!strcmp(argv[i], "--demand"))    demand_file      = argv[++i];
        else if (!strcmp(argv[i], "--partition")) partition_type   = argv[++i];
        else if (!strcmp(argv[i], "--steps"))     total_time_steps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--dt"))        dt               = atof(argv[++i]);
    }

    // Clamp to available CUDA devices
    int avail_gpus = 0;
    cudaError_t dev_count_status = cudaGetDeviceCount(&avail_gpus);
    if (dev_count_status != cudaSuccess) {
        std::cerr << "[Error] cudaGetDeviceCount failed: "
                  << cudaGetErrorString(dev_count_status) << "\n";
    }
    num_gpus = std::min(num_gpus, avail_gpus);
    if (num_gpus == 0) {
        std::cerr << "[Error] No CUDA-capable GPU found.\n";
        return 1;
    }
    std::cout << "[LPSim] Starting with " << num_gpus
              << "/" << avail_gpus << " GPU(s)\n";

    // ---- Load Data ----
    int total_lane_cells = 0;
    auto edges   = load_network(network_file, total_lane_cells);
    std::vector<int> route_paths;
    auto vehicles = load_demand(demand_file, route_paths, (int)edges.size());

    // Remap arbitrary node IDs to dense [0..N-1] indices.
    // Real OSM-style IDs can be very large and sparse, which would make
    // nodes.resize(max_id+1) impossible.
    std::unordered_map<int, int> node_to_dense;
    node_to_dense.reserve(edges.size() * 2);
    int next_dense_id = 0;
    for (auto& e : edges) {
        auto it_u = node_to_dense.find(e.upstream_node);
        if (it_u == node_to_dense.end()) {
            it_u = node_to_dense.emplace(e.upstream_node, next_dense_id++).first;
        }
        auto it_v = node_to_dense.find(e.downstream_node);
        if (it_v == node_to_dense.end()) {
            it_v = node_to_dense.emplace(e.downstream_node, next_dense_id++).first;
        }
        e.upstream_node = it_u->second;
        e.downstream_node = it_v->second;
    }

    // ---- Pre-process: Sort by departure time (reduces warp divergence) ----
    sort_vehicles_by_departure(vehicles);

    // ---- Build synthetic nodes ----
    std::vector<NodeData> nodes;
    // (In practice, loaded from network file; here we create minimal stubs)
    int max_node = 0;
    for (const auto& e : edges)
        max_node = std::max(max_node, std::max(e.upstream_node, e.downstream_node));
    nodes.resize(max_node + 1);
    for (int i = 0; i <= max_node; ++i) {
        nodes[i].node_id     = i;
        nodes[i].gpu_id      = 0;
        nodes[i].in_ghost    = 0;
        nodes[i].signal_cycle = 90;
        nodes[i].signal_phase = 0.0f;
    }

    // ---- Graph Partitioning ----
    if (num_gpus > 1) {
        // Build graph edges weighted by traffic flow
        std::vector<GraphEdge> graph_edges;
        graph_edges.reserve(edges.size());
        for (const auto& e : edges)
            graph_edges.push_back({e.upstream_node, e.downstream_node, 1.0f});

        // Dummy spatial coords (use node_id as proxy)
        std::vector<float> nx(max_node + 1), ny(max_node + 1);
        for (int i = 0; i <= max_node; ++i) { nx[i] = (float)(i % 100); ny[i] = (float)(i / 100); }

        bool use_balanced = (partition_type == "balanced");
        auto partition = run_partitioning(
            max_node + 1, num_gpus, graph_edges, nx, ny, use_balanced);

        std::cout << "[Partition] Imbalance ratio: " << partition.imbalance_ratio << "\n";

        // Apply partition: set gpu_id on edges and nodes
        for (auto& e : edges)
            e.gpu_id = partition.node_to_gpu[e.upstream_node];
        for (auto& n : nodes)
            n.gpu_id = partition.node_to_gpu[n.node_id];
    }

    // ---- Simulation Configuration ----
    SimConfig cfg{};
    cfg.num_gpus         = num_gpus;
    cfg.total_time_steps = total_time_steps;
    cfg.dt               = dt;
    cfg.num_vehicles     = (int)vehicles.size();
    cfg.num_edges        = (int)edges.size();
    cfg.num_nodes        = (int)nodes.size();
    cfg.lane_map_size    = total_lane_cells;
    // IDM defaults (calibrated for urban driving)
    cfg.idm.a     = 1.5f;
    cfg.idm.b     = 3.0f;
    cfg.idm.delta = 4.0f;
    cfg.idm.s0    = 2.0f;
    cfg.idm.T     = 1.5f;
    cfg.idm.v0    = 13.9f;  // 50 km/h

    // ---- Enable P2P if multi-GPU ----
    if (num_gpus > 1) enable_p2p(num_gpus);

    // ---- Initialize GPU Contexts ----
    std::vector<GPUContext> gpu_contexts(num_gpus);
    for (int g = 0; g < num_gpus; ++g) {
        gpu_contexts[g].gpu_id = g;

        // Filter edges/nodes/vehicles for this GPU
        std::vector<EdgeData>     g_edges;
        std::vector<NodeData>     g_nodes;
        std::vector<VehicleState> g_vehicles;

        for (const auto& e : edges)
            if (e.gpu_id == g) g_edges.push_back(e);
        for (const auto& n : nodes)
            if (n.gpu_id == g) g_nodes.push_back(n);
        for (const auto& v : vehicles) {
            int edge_gpu = edges[v.curr_edge].gpu_id;
            if (edge_gpu == g) g_vehicles.push_back(v);
        }

        // Lane map size for this GPU (sum of lane_cells for its edges)
        int g_lane_map_size = 0;
        for (const auto& e : g_edges)
            g_lane_map_size += e.num_lanes * e.length_m;

        init_gpu_context(gpu_contexts[g], cfg,
                         g_edges, g_nodes, g_vehicles,
                         route_paths, g_lane_map_size);
    }

    // ---- Run Simulation ----
    bool use_p2p = (avail_gpus > 1);
    auto t_start = std::chrono::high_resolution_clock::now();
    run_simulation(gpu_contexts, cfg, use_p2p);
    auto t_end = std::chrono::high_resolution_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // ---- Report ----
    print_performance_report((int)vehicles.size(), num_gpus, elapsed);

    return 0;
}
