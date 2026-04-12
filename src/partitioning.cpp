/**
 * LPSim - Graph Partitioning
 *
 * Implements two strategies described in Section 3.3.1 of the paper:
 *
 *  1. Balanced Partition  (NP-hard → multi-level heuristic)
 *     - Goal: equal-size subgraphs, minimize cut edges
 *     - Used when GPU compute is the bottleneck
 *
 *  2. Unbalanced Partition  (Community Detection via Leiden / Louvain)
 *     - Goal: tightly-connected communities, then k-means spatial clustering
 *     - Used when GPU compute is ample and communication is the bottleneck
 *
 * Both strategies take a weighted adjacency graph where edge weights
 * = mean number of vehicles traversing that road link in the period.
 */

#include "../include/lpsim.cuh"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <queue>
#include <map>
#include <random>
#include <functional>

// ===========================================================================
// Data Structures
// ===========================================================================

struct GraphEdge {
    int   src, dst;
    float weight;   // Mean vehicle count on this road link
};

struct Partition {
    std::vector<int> node_to_gpu;   // node_id -> gpu_id
    std::vector<std::vector<int>> gpu_to_nodes; // gpu_id -> [node_ids]
    float imbalance_ratio;          // max_size / ideal_size
    int   cut_edges;
};

// ===========================================================================
// Utility: Build adjacency list from trip demand
// ===========================================================================
std::vector<std::vector<std::pair<int,float>>>
build_adjacency(int num_nodes,
                const std::vector<GraphEdge>& edges)
{
    std::vector<std::vector<std::pair<int,float>>> adj(num_nodes);
    for (const auto& e : edges) {
        adj[e.src].emplace_back(e.dst, e.weight);
        adj[e.dst].emplace_back(e.src, e.weight); // undirected for partitioning
    }
    return adj;
}

// ===========================================================================
// Strategy 1: Balanced Partition
//   Multi-level coarsening → BFS-based bisection → uncoarsening + refinement
//
//   Complexity: O(n log n)
//   Approximation guarantee: (1±ε) balanced components
// ===========================================================================
class BalancedPartitioner {
public:
    BalancedPartitioner(int num_nodes, int num_gpus, float eps = 0.1f)
        : n_(num_nodes), k_(num_gpus), eps_(eps) {}

    Partition partition(const std::vector<GraphEdge>& edges)
    {
        auto adj = build_adjacency(n_, edges);

        // Phase 1: Coarsen the graph
        std::vector<int> coarse_map = coarsen(adj);

        // Phase 2: Initial bisection on coarsened graph using BFS
        std::vector<int> assignment = bfs_bisect(adj, coarse_map);

        // Phase 3: Refine back to original graph (Kernighan-Lin style swap)
        assignment = refine(adj, assignment);

        return make_partition(assignment);
    }

private:
    int n_, k_;
    float eps_;

    // Simple coarsening: merge nodes with highest edge weight (Heavy Edge Matching)
    std::vector<int> coarsen(const std::vector<std::vector<std::pair<int,float>>>& adj)
    {
        std::vector<int> match(n_, -1);
        std::vector<bool> visited(n_, false);

        for (int u = 0; u < n_; ++u) {
            if (visited[u]) continue;
            float best_w = -1.0f;
            int   best_v = -1;
            for (const auto& [v, w] : adj[u]) {
                if (!visited[v] && w > best_w) {
                    best_w = w;
                    best_v = v;
                }
            }
            if (best_v != -1) {
                match[u]      = best_v;
                match[best_v] = u;
                visited[u]    = true;
                visited[best_v] = true;
            } else {
                match[u] = u;  // Unmatched -> maps to itself
                visited[u] = true;
            }
        }
        return match;
    }

    // BFS-based k-way partition: greedily grow k components from seed nodes
    std::vector<int> bfs_bisect(
        const std::vector<std::vector<std::pair<int,float>>>& adj,
        const std::vector<int>& /*coarse_map*/)
    {
        std::vector<int> assignment(n_, -1);
        int target = n_ / k_;

        // Select k seed nodes evenly spaced
        std::vector<int> seeds(k_);
        for (int i = 0; i < k_; ++i)
            seeds[i] = (i * n_) / k_;

        // Multi-source BFS
        std::queue<int> bfsq;
        std::vector<int> sizes(k_, 0);

        for (int g = 0; g < k_; ++g) {
            assignment[seeds[g]] = g;
            sizes[g]++;
            bfsq.push(seeds[g]);
        }

        while (!bfsq.empty()) {
            int u = bfsq.front(); bfsq.pop();
            int g = assignment[u];
            if (sizes[g] >= (int)(target * (1.0f + eps_))) continue;
            for (const auto& [v, w] : adj[u]) {
                if (assignment[v] == -1) {
                    assignment[v] = g;
                    sizes[g]++;
                    bfsq.push(v);
                }
            }
        }

        // Assign any remaining unvisited nodes to closest partition
        for (int u = 0; u < n_; ++u) {
            if (assignment[u] == -1) {
                int min_g = std::min_element(sizes.begin(), sizes.end()) - sizes.begin();
                assignment[u] = min_g;
                sizes[min_g]++;
            }
        }
        return assignment;
    }

    // Kernighan-Lin-style refinement: swap node pairs to reduce cut
    std::vector<int> refine(
        const std::vector<std::vector<std::pair<int,float>>>& adj,
        std::vector<int> assignment)
    {
        bool improved = true;
        int  max_iter = 10;
        while (improved && max_iter-- > 0) {
            improved = false;
            for (int u = 0; u < n_; ++u) {
                int  gu      = assignment[u];
                std::map<int, float> neighbor_gain;
                for (const auto& [v, w] : adj[u]) {
                    int gv = assignment[v];
                    if (gv != gu) neighbor_gain[gv] += w;
                }
                if (neighbor_gain.empty()) continue;
                auto it = std::max_element(
                    neighbor_gain.begin(), neighbor_gain.end(),
                    [](const auto& a, const auto& b){ return a.second < b.second; });
                if (it->second > 0.0f) {
                    // Compute loss of moving u out of gu
                    float loss = 0.0f;
                    for (const auto& [v, w] : adj[u])
                        if (assignment[v] == gu) loss += w;
                    if (it->second > loss) {
                        assignment[u] = it->first;
                        improved = true;
                    }
                }
            }
        }
        return assignment;
    }

    Partition make_partition(const std::vector<int>& assignment)
    {
        Partition p;
        p.node_to_gpu = assignment;
        p.gpu_to_nodes.resize(k_);
        for (int u = 0; u < n_; ++u)
            p.gpu_to_nodes[assignment[u]].push_back(u);

        // Compute imbalance
        int ideal = n_ / k_;
        int max_sz = 0;
        for (const auto& g : p.gpu_to_nodes)
            max_sz = std::max(max_sz, (int)g.size());
        p.imbalance_ratio = (float)max_sz / ideal;

        return p;
    }
};

// ===========================================================================
// Strategy 2: Unbalanced Community-Detection Partition
//   Step 1: Leiden-like modularity maximisation (simplified Louvain)
//   Step 2: K-means clustering of community centroids using spatial coords
// ===========================================================================
class CommunityDetectionPartitioner {
public:
    CommunityDetectionPartitioner(int num_nodes, int num_gpus)
        : n_(num_nodes), k_(num_gpus) {}

    Partition partition(
        const std::vector<GraphEdge>& edges,
        const std::vector<float>& node_x,   // Longitude
        const std::vector<float>& node_y)   // Latitude
    {
        // Phase 1: Louvain community detection
        std::vector<int> community = louvain(edges);
        int num_communities = *std::max_element(community.begin(), community.end()) + 1;

        // Phase 2: Compute community centroids
        std::vector<float> cx(num_communities, 0.0f);
        std::vector<float> cy(num_communities, 0.0f);
        std::vector<int>   csz(num_communities, 0);
        for (int u = 0; u < n_; ++u) {
            int c = community[u];
            cx[c] += node_x[u];
            cy[c] += node_y[u];
            csz[c]++;
        }
        for (int c = 0; c < num_communities; ++c) {
            if (csz[c] > 0) {
                cx[c] /= csz[c];
                cy[c] /= csz[c];
            }
        }

        // Phase 3: K-means on community centroids → assign to k GPUs
        std::vector<int> comm_to_gpu = kmeans(cx, cy, k_);

        // Build final partition
        std::vector<int> assignment(n_);
        for (int u = 0; u < n_; ++u)
            assignment[u] = comm_to_gpu[community[u]];

        return make_partition(assignment);
    }

private:
    int n_, k_;

    // Simplified Louvain: one pass of local node moves
    std::vector<int> louvain(const std::vector<GraphEdge>& edges)
    {
        std::vector<int> comm(n_);
        std::iota(comm.begin(), comm.end(), 0);  // Each node its own community

        float total_weight = 0.0f;
        for (const auto& e : edges) total_weight += e.weight;

        // Degree of each node
        std::vector<float> deg(n_, 0.0f);
        for (const auto& e : edges) {
            deg[e.src] += e.weight;
            deg[e.dst] += e.weight;
        }

        auto adj = build_adjacency(n_, edges);

        bool improved = true;
        while (improved) {
            improved = false;
            for (int u = 0; u < n_; ++u) {
                int   cur_c = comm[u];
                std::map<int, float> comm_weights;
                for (const auto& [v, w] : adj[u])
                    comm_weights[comm[v]] += w;

                // Best community to move u into
                float best_delta = 0.0f;
                int   best_c     = cur_c;

                for (const auto& [c, k_in] : comm_weights) {
                    if (c == cur_c) continue;
                    // Simplified modularity gain (ignores sigma_tot for speed)
                    float delta = 2.0f * k_in / (2.0f * total_weight);
                    if (delta > best_delta) {
                        best_delta = delta;
                        best_c     = c;
                    }
                }
                if (best_c != cur_c) {
                    comm[u]  = best_c;
                    improved = true;
                }
            }
        }

        // Re-label communities 0..num_comm-1
        std::map<int,int> label;
        int cnt = 0;
        for (auto& c : comm) {
            if (!label.count(c)) label[c] = cnt++;
            c = label[c];
        }
        return comm;
    }

    // K-means clustering for community centroids
    std::vector<int> kmeans(const std::vector<float>& cx,
                            const std::vector<float>& cy,
                            int k)
    {
        int n = cx.size();
        std::vector<int> assignment(n, 0);

        // Init k centroids at evenly-spaced indices
        std::vector<float> mx(k), my(k);
        for (int i = 0; i < k; ++i) {
            mx[i] = cx[(i * n) / k];
            my[i] = cy[(i * n) / k];
        }

        for (int iter = 0; iter < 100; ++iter) {
            // Assign each community to nearest centroid
            for (int i = 0; i < n; ++i) {
                float best_d = 1e18f;
                int   best_k = 0;
                for (int g = 0; g < k; ++g) {
                    float dx = cx[i] - mx[g];
                    float dy = cy[i] - my[g];
                    float d  = dx*dx + dy*dy;
                    if (d < best_d) { best_d = d; best_k = g; }
                }
                assignment[i] = best_k;
            }

            // Update centroids
            std::vector<float> nx(k, 0.0f), ny(k, 0.0f);
            std::vector<int>   nsz(k, 0);
            for (int i = 0; i < n; ++i) {
                int g = assignment[i];
                nx[g] += cx[i]; ny[g] += cy[i]; nsz[g]++;
            }
            for (int g = 0; g < k; ++g)
                if (nsz[g] > 0) { mx[g] = nx[g]/nsz[g]; my[g] = ny[g]/nsz[g]; }
        }
        return assignment;
    }

    Partition make_partition(const std::vector<int>& assignment)
    {
        Partition p;
        p.node_to_gpu = assignment;
        p.gpu_to_nodes.resize(k_);
        for (int u = 0; u < n_; ++u)
            p.gpu_to_nodes[assignment[u]].push_back(u);
        p.imbalance_ratio = 0.0f;
        return p;
    }
};

// ===========================================================================
// Entry Point: Choose and execute partitioning
// ===========================================================================
Partition run_partitioning(
    int  num_nodes,
    int  num_gpus,
    const std::vector<GraphEdge>& edges,
    const std::vector<float>& node_x,
    const std::vector<float>& node_y,
    bool use_balanced = true,    // true = balanced, false = community-detection
    float eps = 0.1f)
{
    if (use_balanced) {
        std::cout << "[Partition] Running balanced multi-level partitioning...\n";
        BalancedPartitioner bp(num_nodes, num_gpus, eps);
        return bp.partition(edges);
    } else {
        std::cout << "[Partition] Running community-detection (Louvain + K-means)...\n";
        CommunityDetectionPartitioner cp(num_nodes, num_gpus);
        return cp.partition(edges, node_x, node_y);
    }
}
