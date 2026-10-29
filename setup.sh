#!/bin/bash
# LPSim complete setup script
# Run this once from your home directory: bash setup.sh

set -e  # stop on any error

echo ""
echo "========================================"
echo "  LPSim Complete Setup"
echo "========================================"

# ── Step 1: Install system dependencies ──────────────────────────
echo ""
echo "[1/7] Installing dependencies..."
sudo apt update -qq
sudo apt install -y cmake build-essential git python3-pip python3-venv

# ── Step 2: Clone the repo ────────────────────────────────────────
echo ""
echo "[2/7] Cloning LPSim repo..."
cd ~
rm -rf gpu              # clean start
mkdir  gpu
cd     gpu
git clone https://github.com/Xuan-1998/LPSim.git
cd LPSim
echo "  Cloned into ~/gpu/LPSim"
echo "  Files in repo:"
ls -la

# ── Step 3: Replace source files with working versions ───────────
echo ""
echo "[3/7] Installing working source files..."

# Backup originals
cp main.cu main_original.cu 2>/dev/null || true

# Copy our files (they must be in the same folder as this script)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

cp "$SCRIPT_DIR/main.cu"       ~/gpu/LPSim/main.cu
cp "$SCRIPT_DIR/simulation.cu" ~/gpu/LPSim/simulation.cu
cp "$SCRIPT_DIR/simulation.h"  ~/gpu/LPSim/simulation.h
cp "$SCRIPT_DIR/CMakeLists.txt" ~/gpu/LPSim/CMakeLists.txt

echo "  Files replaced."

# ── Step 4: Build ─────────────────────────────────────────────────
echo ""
echo "[4/7] Building (compiling CUDA code)..."
cd ~/gpu/LPSim
rm -rf build
mkdir  build
cd     build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
echo "  Build complete."

# ── Step 5: Set up Python environment ────────────────────────────
echo ""
echo "[5/7] Setting up Python environment..."
cd ~/gpu/LPSim
python3 -m venv .venv
source .venv/bin/activate
pip install --quiet osmnx networkx pandas matplotlib
echo "  Python packages installed."

# ── Step 6: Download Chennai road network ────────────────────────
echo ""
echo "[6/7] Downloading Chennai road network from OpenStreetMap..."
echo "  (this may take 2-5 minutes depending on internet speed)"

python3 << 'PYEOF'
import osmnx as ox
import pandas as pd
import sys

print("  Connecting to OpenStreetMap...")
try:
    G = ox.graph_from_place('Chennai, Tamil Nadu, India', network_type='drive')
    print(f"  Downloaded: {len(G.nodes):,} intersections, {len(G.edges):,} road segments")
except Exception as e:
    print(f"  Chennai download failed ({e}), trying smaller area...")
    G = ox.graph_from_place('Anna Nagar, Chennai, India', network_type='drive')
    print(f"  Downloaded smaller area: {len(G.nodes):,} nodes, {len(G.edges):,} edges")

nodes_df, edges_df = ox.graph_to_gdfs(G)
edges_df = edges_df.reset_index()
edges_df['edge_id']        = range(len(edges_df))
edges_df['from_node']      = edges_df['u']
edges_df['to_node']        = edges_df['v']
edges_df['num_lanes']      = pd.to_numeric(
                                 edges_df.get('lanes', 2),
                                 errors='coerce').fillna(2).astype(int)
edges_df['length_metres']  = edges_df['length'].round(1)
edges_df['speed_limit_ms'] = 13.9
edges_df['road_type']      = edges_df.get('highway', 'residential').astype(str)

cols = ['edge_id','from_node','to_node','num_lanes','length_metres','speed_limit_ms','road_type']
edges_df[cols].to_csv('network.csv', index=False)
print(f"  Saved network.csv ({len(edges_df):,} rows)")
PYEOF

# ── Step 7: Generate synthetic demand ────────────────────────────
echo ""
echo "[7/7] Generating 50,000 synthetic vehicle trips..."

python3 << 'PYEOF'
import pandas as pd
import random
random.seed(42)

NUM_TRIPS   = 50000
MAX_NODE    = 5000
MAX_DEPART  = 7200   # 2 hours in seconds

trips = []
for i in range(NUM_TRIPS):
    origin = random.randint(0, MAX_NODE - 1)
    dest   = random.randint(0, MAX_NODE - 1)
    while dest == origin:
        dest = random.randint(0, MAX_NODE - 1)
    depart = random.randint(0, MAX_DEPART)
    trips.append([i, origin, dest, depart])

df = pd.DataFrame(trips, columns=['vehicle_id','origin_node','destination_node','departure_time_s'])
df.sort_values('departure_time_s', inplace=True)
df.reset_index(drop=True, inplace=True)
df.to_csv('demand.csv', index=False)
print(f"  Saved demand.csv ({len(df):,} trips, sorted by departure time)")
PYEOF

# ── Done ──────────────────────────────────────────────────────────
echo ""
echo "========================================"
echo "  Setup complete!"
echo "========================================"
echo ""
echo "  To run the simulation:"
echo ""
echo "  cd ~/gpu/LPSim/build"
echo "  ./lpsim --gpus 1 \\"
echo "          --network ../network.csv \\"
echo "          --demand  ../demand.csv \\"
echo "          --partition balanced \\"
echo "          --steps 3600 \\"
echo "          --dt 1.0"
echo ""
