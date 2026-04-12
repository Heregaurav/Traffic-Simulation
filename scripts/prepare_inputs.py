#!/usr/bin/env python3
"""
Prepare LPSim inputs from LivingCity Berkeley network folders.

Input folder is expected to contain:
  - edges.csv
  - od_demand.csv

Outputs:
  - network.csv (edge_id,src_node,dst_node,num_lanes,length_m,speed_limit_mps)
  - demand.csv  (vehicle_id,origin_edge,dest_edge,depart_time_s,route_edge_0,...)
"""

from __future__ import annotations

import argparse
import csv
import heapq
from pathlib import Path
from typing import Dict, List, Tuple


def pick_col(candidates: List[str], options: List[str]) -> str:
    lowered = {c.lower(): c for c in candidates}
    for opt in options:
        if opt.lower() in lowered:
            return lowered[opt.lower()]
    raise KeyError(f"Could not find any of {options} in columns {candidates}")


def to_int(v: str) -> int:
    return int(float(v))


def to_float(v: str) -> float:
    return float(v)


def dijkstra_path(
    adj: Dict[int, List[Tuple[int, int, float]]], src: int, dst: int
) -> List[int]:
    if src == dst:
        return []

    dist: Dict[int, float] = {src: 0.0}
    prev_node: Dict[int, int] = {}
    prev_edge: Dict[int, int] = {}
    pq: List[Tuple[float, int]] = [(0.0, src)]

    while pq:
        d, u = heapq.heappop(pq)
        if d != dist.get(u, float("inf")):
            continue
        if u == dst:
            break
        for edge_id, v, w in adj.get(u, []):
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev_node[v] = u
                prev_edge[v] = edge_id
                heapq.heappush(pq, (nd, v))

    if dst not in dist:
        return []

    route: List[int] = []
    cur = dst
    while cur != src:
        route.append(prev_edge[cur])
        cur = prev_node[cur]
    route.reverse()
    return route


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--network-dir", required=True, help="Folder with edges.csv and od_demand.csv")
    p.add_argument("--out-dir", default="data/processed", help="Output folder")
    p.add_argument("--max-demand", type=int, default=20000, help="Max OD rows to process")
    p.add_argument("--depart-step", type=int, default=2, help="Departure spacing in seconds")
    args = p.parse_args()

    network_dir = Path(args.network_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    edges_path = network_dir / "edges.csv"
    od_path = network_dir / "od_demand.csv"
    if not edges_path.exists() or not od_path.exists():
        raise FileNotFoundError(f"Missing edges.csv or od_demand.csv in {network_dir}")

    # Load and normalize edges.
    edge_rows = []
    adj: Dict[int, List[Tuple[int, int, float]]] = {}
    with edges_path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        u_col = pick_col(cols, ["u", "src_node", "src", "from"])
        v_col = pick_col(cols, ["v", "dst_node", "dst", "to"])
        len_col = pick_col(cols, ["length", "length_m"])
        lanes_col = pick_col(cols, ["lanes", "num_lanes"])
        speed_col = pick_col(cols, ["speed_mph", "speed_limit_mps", "speed"])

        for i, r in enumerate(reader):
            src = to_int(r[u_col])
            dst = to_int(r[v_col])
            length_m = to_float(r[len_col])
            lanes = max(1, to_int(r[lanes_col]))

            if speed_col.lower() == "speed_mph":
                speed_mps = max(1.0, to_float(r[speed_col]) * 0.44704)
            else:
                speed_mps = max(1.0, to_float(r[speed_col]))

            edge_rows.append((i, src, dst, lanes, length_m, speed_mps))
            # weight = travel time
            adj.setdefault(src, []).append((i, dst, length_m / max(speed_mps, 0.1)))

    network_out = out_dir / "network.csv"
    with network_out.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["edge_id", "src_node", "dst_node", "num_lanes", "length_m", "speed_limit_mps"]
        )
        for row in edge_rows:
            w.writerow(row)

    # Load OD and build routed demand.
    demand_out = out_dir / "demand.csv"
    total = 0
    routed = 0
    skipped = 0
    with od_path.open(newline="") as f_in, demand_out.open("w", newline="") as f_out:
        r = csv.DictReader(f_in)
        cols = r.fieldnames or []
        o_col = pick_col(cols, ["origin", "orig"])
        d_col = pick_col(cols, ["destination", "dest"])

        w = csv.writer(f_out)
        w.writerow(
            ["vehicle_id", "origin_edge", "dest_edge", "depart_time_s", "route_edge_0", "route_edge_1"]
        )

        for i, row in enumerate(r):
            if i >= args.max_demand:
                break
            total += 1
            origin = to_int(row[o_col])
            dest = to_int(row[d_col])
            path = dijkstra_path(adj, origin, dest)
            if len(path) == 0:
                skipped += 1
                continue
            routed += 1
            depart = i * args.depart_step
            out = [i, path[0], path[-1], depart] + path
            w.writerow(out)

    print(f"Prepared network: {network_out}")
    print(f"Prepared demand : {demand_out}")
    print(f"Edges: {len(edge_rows)}")
    print(f"OD processed: {total}, routed: {routed}, skipped: {skipped}")


if __name__ == "__main__":
    main()

