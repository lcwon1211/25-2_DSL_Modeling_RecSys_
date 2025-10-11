"""
Running Route Generator with Dijkstra Algorithm
================================================
Generates optimal running routes considering various factors:
- Grade/elevation
- Night safety (streetlights, CCTV)
- Scenic views (waterside, parks)
- Crossings
- Amenities (toilets, convenience stores, subway)
"""

import math
import itertools
from typing import List, Tuple, Optional, Set, Dict
import pandas as pd
import numpy as np
import networkx as nx
import osmnx as ox
import folium
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)


# ============================================================================
# Utility Functions
# ============================================================================

def _nz(x, v=0.0):
    """Return default value if x is None or NaN"""
    try:
        return v if (x is None or (isinstance(x, float) and math.isnan(x))) else x
    except Exception:
        return v if x is None else x


def clip01(x: float) -> float:
    """Clip value to [0, 1] range"""
    return max(0.0, min(1.0, float(x)))


def robust_minmax_series(s: pd.Series, q_low=0.01, q_high=0.99) -> pd.Series:
    """Robust min-max normalization using quantiles"""
    s = s.astype(float).fillna(0.0)
    lo, hi = s.quantile(q_low), s.quantile(q_high)
    if hi == lo:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return ((s - lo) / (hi - lo)).clip(0, 1)


def combine_night_norm(light_norm: float, cctv_norm: float, mode: str="min", alpha: float=0.5) -> float:
    """Combine streetlight and CCTV normalization for night safety"""
    light_norm = clip01(light_norm)
    cctv_norm = clip01(cctv_norm)
    
    if mode == "min":
        return min(light_norm, cctv_norm)
    if mode == "mean":
        return clip01(alpha * light_norm + (1.0 - alpha) * cctv_norm)
    if mode == "geom":
        return clip01(math.sqrt(light_norm * cctv_norm))
    if mode == "harmonic":
        if light_norm <= 0 or cctv_norm <= 0:
            return 0.0
        return clip01(2.0 / (1.0/light_norm + 1.0/cctv_norm))
    return min(light_norm, cctv_norm)


# ============================================================================
# Grade Scaling
# ============================================================================

def add_grade_scaled(
    df: pd.DataFrame,
    difficulty: str = '중',
    alpha: float = 1.0,
    beta: float = 3.0,
    gamma: float = 2.0,
) -> pd.DataFrame:
    """
    Add grade_scaled column to MegaDB DataFrame
    
    Parameters:
    - difficulty: '최하'|'하'|'중'|'상' - target difficulty level
    - alpha: uphill cost multiplier
    - beta: downhill cost multiplier
    - gamma: downhill exponential factor
    """
    df = df.copy()

    # Type safety
    for col in ['length', 'grade_net']:
        if col not in df.columns:
            raise KeyError(f"Required column missing: {col}")
    df['length'] = pd.to_numeric(df['length'], errors='coerce')
    df['grade_net'] = pd.to_numeric(df['grade_net'], errors='coerce')
    df = df.dropna(subset=['length','grade_net']).reset_index(drop=True)

    # Scale grade
    df['grade_net_x100'] = df['grade_net'] * 100.0

    # Label grade direction
    df['w'] = 0
    df.loc[df['grade_net_x100'] > 0, 'w'] = 1    # uphill
    df.loc[df['grade_net_x100'] < 0, 'w'] = -1   # downhill

    # Calculate raw grade cost
    def _raw_cost(row):
        g = row['grade_net_x100']
        L = row['length']
        if g > 0:       # uphill
            return alpha * g * L
        elif g < 0:     # downhill
            return beta * (abs(g) ** gamma) * L
        else:           # flat
            return L
    df['grade_cost_raw'] = df.apply(_raw_cost, axis=1)

    # Set difficulty target (uphill only)
    uphill = df[df['grade_net_x100'] > 0].copy()
    if uphill.empty:
        q0 = q1 = q2 = q3 = 0.0
    else:
        q0 = 0.0
        q1 = uphill['grade_cost_raw'].quantile(0.25)
        q2 = uphill['grade_cost_raw'].quantile(0.50)
        q3 = uphill['grade_cost_raw'].quantile(0.75)

    if difficulty == '최하':
        target_grade = q0
    elif difficulty == '하':
        target_grade = q1
    elif difficulty == '중':
        target_grade = q2
    else:
        target_grade = q3

    # Tune cost to target difficulty
    df['grade_cost_tuned'] = df['grade_cost_raw']
    mask_flat_up = df['w'].isin([0, 1])
    df.loc[mask_flat_up, 'grade_cost_tuned'] = (df.loc[mask_flat_up, 'grade_cost_raw'] - target_grade).abs()

    # MinMax scaling
    vals = df['grade_cost_tuned'].to_numpy().reshape(-1, 1)
    if np.nanmax(vals) == np.nanmin(vals):
        df['grade_scaled'] = 0.0
    else:
        scaler = MinMaxScaler()
        df['grade_scaled'] = scaler.fit_transform(vals)

    return df


# ============================================================================
# Graph Attachment
# ============================================================================

def attach_by_uvk(G: nx.MultiDiGraph, df_edges: pd.DataFrame):
    """
    Attach MegaDB attributes to graph edges by (u, v, key) matching
    Requires df_edges to already have 'grade_scaled' column
    """
    df_e = df_edges.copy()

    # Check required keys
    for col in ["u","v","key"]:
        if col not in df_e.columns:
            raise ValueError(f"Mega DB missing '{col}'")

    # Verify grade_scaled exists
    if "grade_scaled" not in df_e.columns:
        raise ValueError(
            "[attach_by_uvk] 'grade_scaled' is missing. "
            "Call add_grade_scaled(df_edges, difficulty=...) first."
        )
    df_e["grade_scaled"] = pd.to_numeric(df_e["grade_scaled"], errors="coerce").clip(0.0, 1.0).fillna(0.0)

    # Safety features
    def _safe_div(a, b):
        a = _nz(a, 0.0)
        b = _nz(b, 0.0)
        return float(a)/float(b) if float(b) > 0 else 0.0

    # Streetlight density
    df_e["sl_per_m"] = df_e.apply(
        lambda r: _safe_div(r.get("streetlight_count", 0.0), r.get("length", 0.0)), axis=1
    )
    
    # CCTV density
    if "cctv_density" in df_e.columns:
        base_cctv = pd.to_numeric(df_e["cctv_density"], errors="coerce").fillna(0.0)
    else:
        base_cctv = df_e.apply(
            lambda r: _safe_div(r.get("cctv_count", 0.0), r.get("length", 0.0)), axis=1
        )
    df_e["cctv_base"] = base_cctv

    # Robust normalization
    df_e["streetlight_norm"] = robust_minmax_series(df_e["sl_per_m"])
    df_e["cctv_norm"] = robust_minmax_series(df_e["cctv_base"])

    # String casting for matching
    df_e["u_str"] = df_e["u"].astype(str)
    df_e["v_str"] = df_e["v"].astype(str)
    df_e["k_str"] = df_e["key"].astype(str)
    df_e["_key_tuple"] = list(zip(df_e["u_str"], df_e["v_str"], df_e["k_str"]))

    recs = df_e.set_index("_key_tuple").to_dict(orient="index")

    hit = miss = 0
    for u, v, k, d in G.edges(keys=True, data=True):
        tup = (str(u), str(v), str(k))
        row = recs.get(tup)
        if row is None:
            # Try reverse direction
            tup_rev = (str(v), str(u), str(k))
            row = recs.get(tup_rev)
        if row is None:
            miss += 1
            continue
        # Copy all columns
        for col, val in row.items():
            if col in {"_key_tuple","u","v","key","u_str","v_str","k_str"}:
                continue
            d[col] = val
        hit += 1
    print(f"attach_by_uvk: hit={hit}, miss={miss}")


# ============================================================================
# Crossing Policy
# ============================================================================

def apply_crossing_policy(G: nx.MultiDiGraph, policy: str="penalize", penalty_meters: float=30.0):
    """
    Apply crossing policy
    - forbid: remove edges with crossing_count > 0
    - penalize: add soft penalty to edges with crossings
    """
    if policy == "forbid":
        to_remove = []
        for u, v, k, d in G.edges(keys=True, data=True):
            if int(_nz(d.get("crossing_count"), 0)) > 0:
                to_remove.append((u, v, k))
        G.remove_edges_from(to_remove)
        print(f"Removed edges with crossing_count>0: {len(to_remove)}")
    elif policy == "penalize":
        for u, v, k, d in G.edges(keys=True, data=True):
            cnt = int(_nz(d.get("crossing_count"), 0))
            d["crossing_penalty"] = float(cnt) * float(penalty_meters)
        print("Applied soft crossing penalties to edges")
    else:
        print("Crossing policy: allow (no changes)")


# ============================================================================
# Edge Cost Calculation
# ============================================================================

def build_edge_costs(
    G: nx.MultiDiGraph,
    include_length: bool=True,
    w_grade: float=0.6,
    night_mode: bool=False,
    W_NIGHT: float=10.0,
    night_exp: float=3.0,
    dark_threshold: float=0.5,
    night_combine: str="min",
    alpha_streetlight: float=0.5,
    w_cross: float=1.0,
    scenic_bias: float=0.0
):
    """Build edge costs considering all factors"""
    scenic_bias = max(0.0, min(0.9, float(scenic_bias)))
    
    for u, v, k, d in G.edges(keys=True, data=True):
        L = float(_nz(d.get("length"), 0.0))
        base = L if include_length else 1.0
        if base <= 0:
            base = 1e-6

        # Grade
        g = clip01(float(_nz(d.get("grade_scaled"), 0.0)))

        # Night mode
        if night_mode:
            sln = float(_nz(d.get("streetlight_norm"), 0.0))
            ccn = float(_nz(d.get("cctv_norm"), 0.0))
            safe_norm = combine_night_norm(sln, ccn, mode=night_combine, alpha=alpha_streetlight)
            deficit = max(0.0, dark_threshold - safe_norm)
            night_factor = 1.0 + W_NIGHT * (deficit ** night_exp)
        else:
            night_factor = 1.0

        # Scenic bonus
        scenic_flag = (int(_nz(d.get("waterside_count"), 0)) > 0) or (int(_nz(d.get("park_garden_count"), 0)) > 0)
        scenic_factor = (1.0 - scenic_bias) if scenic_flag else 1.0
        scenic_factor = max(0.1, scenic_factor)

        # Crossing penalty
        cross_pen = float(_nz(d.get("crossing_penalty"), 0.0))

        # Final cost
        cost = base * (1.0 + w_grade * g) * night_factor
        cost = cost * scenic_factor + w_cross * cross_pen

        d["cost"] = float(max(1e-6, cost))
        d["_scenic_len"] = float(L if scenic_flag else 0.0)


# ============================================================================
# Must-Pass Sets
# ============================================================================

def nodes_from_edge_count(G: nx.MultiDiGraph, edge_key: str, min_val: int=1) -> Set[int]:
    """Get nodes from edges with specific attribute count"""
    S = set()
    for u, v, k, d in G.edges(keys=True, data=True):
        if int(_nz(d.get(edge_key), 0)) >= min_val:
            S.add(u)
            S.add(v)
    return S


def build_must_sets(
    G: nx.MultiDiGraph,
    require_convenience: bool=False,
    require_toilets: bool=False,
    require_scenic_once: bool=False,
    require_subway: bool=False,
) -> List[Set[int]]:
    """Build list of must-pass node sets"""
    sets = []
    if require_convenience:
        sets.append(nodes_from_edge_count(G, "convenience_supermarket_count", 1))
    if require_toilets:
        sets.append(nodes_from_edge_count(G, "toilets_count", 1))
    if require_scenic_once:
        scenic_nodes = set()
        for key in ["waterside_count", "park_garden_count"]:
            scenic_nodes |= nodes_from_edge_count(G, key, 1)
        sets.append(scenic_nodes)
    if require_subway:
        sets.append(nodes_from_edge_count(G, "subway_locker_count", 1))
    return [s for s in sets if len(s) > 0]


# ============================================================================
# Routing Utilities
# ============================================================================

def stitch_path_cost(G, path, w="cost"):
    """Calculate total cost along a path"""
    tot = 0.0
    for a, b in zip(path[:-1], path[1:]):
        best = min(G[a][b].values(), key=lambda d: d.get(w, 1e9))
        tot += float(best.get(w, 0.0))
    return tot


def path_cost_and_scenic_ratio(G: nx.MultiDiGraph, path: List[int], weight_attr="cost") -> Tuple[float, float, float]:
    """Calculate path cost, length, and scenic ratio"""
    tot_cost, tot_len, scenic_len = 0.0, 0.0, 0.0
    for a, b in zip(path[:-1], path[1:]):
        best = min(G[a][b].values(), key=lambda d: d.get(weight_attr, 1e9))
        tot_cost += float(best.get(weight_attr, 0.0))
        tot_len += float(_nz(best.get("length"), 0.0))
        scenic_len += float(_nz(best.get("_scenic_len"), 0.0))
    ratio = (scenic_len / tot_len) if tot_len > 0 else 0.0
    return tot_cost, tot_len, ratio


def path_via_must_sets_greedy(G: nx.MultiDiGraph, source, target, must_sets: List[Set[int]], weight_attr="cost") -> Optional[List[int]]:
    """Find path that passes through all must-pass sets using greedy approach"""
    if not must_sets:
        try:
            return nx.shortest_path(G, source, target, weight=weight_attr)
        except nx.NetworkXNoPath:
            return None
    
    best_path, best_cost = None, float("inf")
    for order in itertools.permutations(must_sets):
        cur = source
        ok = True
        acc_path = [source]
        total_cost = 0.0
        
        for S in order:
            try:
                dist, prev = nx.single_source_dijkstra(G, cur, weight=weight_attr)
            except Exception:
                ok = False
                break
            cand = [(n, dist[n]) for n in S if n in dist]
            if not cand:
                ok = False
                break
            nxt, _ = min(cand, key=lambda x: x[1])
            seg = nx.shortest_path(G, cur, nxt, weight=weight_attr)
            total_cost += stitch_path_cost(G, seg, weight_attr)
            acc_path += seg[1:]
            cur = nxt
        
        if not ok:
            continue
        
        try:
            seg = nx.shortest_path(G, cur, target, weight=weight_attr)
        except Exception:
            continue
        total_cost += stitch_path_cost(G, seg, weight_attr)
        acc_path += seg[1:]
        
        if total_cost < best_cost:
            best_cost, best_path = total_cost, acc_path
    
    return best_path


# ============================================================================
# Route Diversification
# ============================================================================

def edges_from_path(path):
    """Get edge list from node path"""
    return [(path[i], path[i+1]) for i in range(len(path)-1)]


def poison_along(G, path_nodes, gain=140.0, neighbor_decay=0.5):
    """
    Increase cost along path and neighboring edges for route diversification
    - Main edges: cost += gain
    - 1-hop neighbors: cost += gain * neighbor_decay
    """
    H = G.copy()
    main_edges = edges_from_path(path_nodes)
    touched = set()

    # Poison main path
    for (u, v) in main_edges:
        if H.has_edge(u, v):
            for k in H[u][v]:
                d = H[u][v][k]
                d["cost"] = float(d.get("cost", 0.0)) + float(gain)
                touched.add((u, v))

    # Poison neighbors
    if neighbor_decay > 0:
        nbr_gain = float(gain) * float(neighbor_decay)
        nbrs = set()
        for (u, v) in main_edges:
            if H.has_node(u):
                for x in H.successors(u):
                    nbrs.add((u, x))
            if H.has_node(v):
                for x in H.predecessors(v):
                    nbrs.add((x, v))
        # Exclude main edges
        nbrs = {e for e in nbrs if e not in touched}
        for (a, b) in nbrs:
            if H.has_edge(a, b):
                for k in H[a][b]:
                    d = H[a][b][k]
                    d["cost"] = float(d.get("cost", 0.0)) + nbr_gain
    return H


def edge_overlap_ratio(G, path_a, path_b):
    """Calculate edge overlap ratio (Jaccard similarity)"""
    def edge_keys(G, u, v):
        return [(u, v, k) for k in G[u][v].keys()] if G.has_edge(u, v) else []
    
    Ea = set(sum([edge_keys(G, u, v) for (u, v) in edges_from_path(path_a)], []))
    Eb = set(sum([edge_keys(G, u, v) for (u, v) in edges_from_path(path_b)], []))
    if not Ea or not Eb:
        return 0.0
    inter = len(Ea & Eb)
    union = len(Ea | Eb)
    return inter / union if union else 0.0


# ============================================================================
# Main Route Generation
# ============================================================================

def generate_running_route(
    place_name: str,
    start_lat: float,
    start_lon: float,
    desired_distance_m: float = 5000,
    tolerance_ratio: float = 0.15,
    difficulty: str = '최하',
    include_length: bool = True,
    night_mode: bool = True,
    W_NIGHT: float = 10.0,
    night_exp: float = 3.0,
    dark_threshold: float = 0.5,
    w_grade: float = 0.6,
    w_cross: float = 1.0,
    crossing_policy: str = "penalize",
    crossing_penalty_meters: float = 30.0,
    target_scenic_ratio: float = 0.3,
    require_convenience: bool = False,
    require_toilets: bool = True,
    require_scenic_once: bool = False,
    require_subway: bool = False,
    night_combine: str = "min",
    alpha_streetlight: float = 0.5,
    scenic_search_iters: int = 8,
    csv_path: str = "dataset/Mega DB.csv",
    output_html: str = "running_loop_route.html"
):
    """
    Generate optimal running loop route
    
    Parameters:
    - place_name: Location name (e.g., "Seodaemun-gu, Seoul, South Korea")
                  ⚠️ IMPORTANT: MegaDB must match this region!
    - start_lat, start_lon: Starting coordinates
    - desired_distance_m: Target distance in meters
    - tolerance_ratio: Distance tolerance (±ratio)
    - difficulty: '최하'|'하'|'중'|'상'
    - night_mode: Enable night safety mode
    - crossing_policy: 'forbid'|'penalize'|'allow'
    - target_scenic_ratio: Target scenic route ratio (0-1)
    - require_*: Must-pass requirements
    - csv_path: Path to Mega DB CSV
    - output_html: Output HTML file path
    """
    
    print(f" Starting route generation for {place_name}")
    print(f" Start: ({start_lat}, {start_lon})")
    print(f" Target distance: {desired_distance_m}m")
    print(f"\n  NOTE: Current MegaDB is built for Seodaemun-gu, Seoul")
    print(f"   If using different region, you must build custom MegaDB first!")
    
    # Load graph
    print("\n Loading graph...")
    G = ox.graph_from_place(place_name, network_type="walk")
    start_node = ox.distance.nearest_nodes(G, X=start_lon, Y=start_lat)
    print(f"✅ Graph loaded. Start node: {start_node}")
    
    # Load and process MegaDB
    print("\n Loading MegaDB...")
    df_edges = pd.read_csv(csv_path)
    print(f"✅ MegaDB loaded: {df_edges.shape[0]} edges")
    
    print(f"\n  Processing grade scaling (difficulty: {difficulty})...")
    df_edges = add_grade_scaled(df_edges, difficulty=difficulty, alpha=1.0, beta=3.0, gamma=2.0)
    
    print("\n Attaching attributes to graph...")
    attach_by_uvk(G, df_edges)
    
    # Build must-pass sets
    print("\n Building must-pass sets...")
    base_sets = build_must_sets(
        G,
        require_convenience=require_convenience,
        require_toilets=require_toilets,
        require_scenic_once=require_scenic_once,
        require_subway=require_subway
    )
    print(f"✅ {len(base_sets)} must-pass set(s) created")
    
    # Create working graph
    print(f"\n  Applying crossing policy: {crossing_policy}")
    Gp = G.copy()
    apply_crossing_policy(Gp, policy=crossing_policy, penalty_meters=crossing_penalty_meters)
    
    # Find anchor candidates
    print(f"\n Finding anchor candidates (tolerance: ±{tolerance_ratio*100}%)...")
    dist_by_len = nx.single_source_dijkstra_path_length(Gp, source=start_node, weight="length")
    target_half = desired_distance_m / 2
    lo = target_half * (1 - tolerance_ratio)
    hi = target_half * (1 + tolerance_ratio)
    
    cands = [n for n, d in dist_by_len.items() if lo <= d <= hi]
    K = 30
    cands = sorted(cands, key=lambda n: abs(dist_by_len[n]-target_half))[:K]
    print(f"✅ {len(cands)} anchor candidates found")
    
    if not cands:
        raise RuntimeError("❌ No anchor candidates found. Try increasing tolerance_ratio or K.")
    
    # Helper functions
    def _build_costs_on(Gx, scenic_bias):
        build_edge_costs(
            Gx,
            include_length=include_length,
            w_grade=w_grade,
            night_mode=night_mode,
            W_NIGHT=W_NIGHT,
            night_exp=night_exp,
            dark_threshold=dark_threshold,
            night_combine=night_combine,
            alpha_streetlight=alpha_streetlight,
            w_cross=w_cross,
            scenic_bias=scenic_bias
        )
    
    def _remaining_sets_after(path_nodes, sets_list):
        visited = set(path_nodes)
        rem = []
        for S in sets_list:
            if len(visited & set(S)) == 0:
                rem.append(S)
        return rem
    
    poison_gain = 140.0
    neighbor_decay = 0.5
    overlap_threshold = 0.25
    
    def _route_with_bias(bias):
        Gx = Gp.copy()
        _build_costs_on(Gx, scenic_bias=bias)
        
        best = {"path": None, "len": float("inf"), "cost": float("inf"), "anchor": None, "ratio": 0.0}
        
        for anchor in cands:
            # Path 1: start -> anchor
            must1 = base_sets + [{anchor}]
            path1 = path_via_must_sets_greedy(Gx, start_node, anchor, must1, weight_attr="cost")
            if not path1:
                continue
            
            # Path 2: anchor -> start (with poisoning)
            Gp2 = poison_along(Gx, path1, gain=poison_gain, neighbor_decay=neighbor_decay)
            remain = _remaining_sets_after(path1, base_sets)
            path2 = path_via_must_sets_greedy(Gp2, anchor, start_node, remain, weight_attr="cost")
            if not path2:
                continue
            
            # Combine loop
            path_loop = path1 + path2[1:]
            if not (path_loop and path_loop[0] == start_node and path_loop[-1] == start_node):
                continue
            
            # Check overlap
            ov = edge_overlap_ratio(Gx, path1, path2)
            if ov > overlap_threshold:
                continue
            
            # Evaluate
            c, L, r = path_cost_and_scenic_ratio(Gx, path_loop, "cost")
            better = (
                best["path"] is None
                or abs(L - desired_distance_m) < abs(best["len"] - desired_distance_m)
                or (
                    abs(L - desired_distance_m) == abs(best["len"] - desired_distance_m)
                    and c < best["cost"]
                )
            )
            if better:
                best.update({"path": path_loop, "len": L, "cost": c, "anchor": anchor, "ratio": r})
        
        return best if best["path"] is not None else None
    
    # Generate route with scenic ratio search
    print(f"\n Searching for optimal scenic ratio (target: {target_scenic_ratio})...")
    if target_scenic_ratio > 0:
        lo_b, hi_b = 0.0, 0.9
        best_overall = None
        for i in range(scenic_search_iters):
            bias = (lo_b + hi_b) / 2
            print(f"  Iteration {i+1}/{scenic_search_iters}: bias={bias:.3f}")
            res = _route_with_bias(bias)
            if not res:
                hi_b = bias
                continue
            if best_overall is None or abs(res["ratio"] - target_scenic_ratio) < abs(best_overall["ratio"] - target_scenic_ratio):
                res["bias"] = bias
                best_overall = res
            if res["ratio"] < target_scenic_ratio:
                lo_b = bias
            else:
                hi_b = bias
        best = best_overall
    else:
        print("  Using scenic_bias=0.0")
        best = _route_with_bias(0.0)
    
    if not best:
        raise RuntimeError("❌ Failed to generate loop route. Try adjusting parameters.")
    
    # Results
    print(f"\n✅ Route generated successfully!")
    print(f"   - Length: {best['len']:.1f}m (target: {desired_distance_m}m)")
    print(f"   - Cost: {best['cost']:.1f}")
    print(f"   - Scenic ratio: {best['ratio']:.3f} (target: {target_scenic_ratio})")
    print(f"   - Anchor: {best['anchor']}")
    print(f"   - Scenic bias: {best.get('bias', 0):.3f}")
    
    # Generate map
    print(f"\n  Generating map...")
    loop_path = best["path"]
    coords = [(G.nodes[n]["y"], G.nodes[n]["x"]) for n in loop_path]
    center = coords[len(coords)//2]
    
    # Create folium map
    m = folium.Map(location=center, tiles="CartoDB positron", zoom_start=15)
    folium.PolyLine(coords, weight=6, color='blue', opacity=0.8).add_to(m)
    
    # Add markers
    start_latlon = (G.nodes[start_node]["y"], G.nodes[start_node]["x"])
    folium.Marker(
        start_latlon, 
        tooltip="START/END", 
        icon=folium.Icon(color="green", icon="play")
    ).add_to(m)
    
    if best.get("anchor"):
        a = best["anchor"]
        folium.Marker(
            (G.nodes[a]["y"], G.nodes[a]["x"]), 
            tooltip="ANCHOR", 
            icon=folium.Icon(color="blue", icon="flag")
        ).add_to(m)
    
    # Add info popup
    summary = f"""
    <div style="font-family: Arial; font-size: 14px;">
    <b>🏃 Running Route Summary</b><br><br>
    <b>Length:</b> {best['len']:.1f} m<br>
    <b>Cost:</b> {best['cost']:.1f}<br>
    <b>Scenic ratio:</b> {best['ratio']:.3%}<br>
    <b>Scenic bias:</b> {best.get('bias', 0):.3f}<br>
    <b>Difficulty:</b> {difficulty}<br>
    <b>Night mode:</b> {'✅' if night_mode else '❌'}<br>
    <b>Anchor node:</b> {best['anchor']}
    </div>
    """
    folium.Marker(
        start_latlon,
        popup=folium.Popup(summary, max_width=300),
        icon=folium.Icon(color="green", icon="info-sign")
    ).add_to(m)
    
    # Save map
    m.save(output_html)
    print(f"✅ Map saved: {output_html}")
    
    return {
        "path": loop_path,
        "length": best["len"],
        "cost": best["cost"],
        "scenic_ratio": best["ratio"],
        "scenic_bias": best.get("bias", 0),
        "anchor": best["anchor"],
        "graph": G,
        "map": m
    }


# ============================================================================
# CLI Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate optimal running routes using Dijkstra algorithm",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default parameters
  python main.py --lat 37.5603066 --lon 126.9368383
  
  # Custom distance and difficulty
  python main.py --lat 37.5603066 --lon 126.9368383 --distance 8000 --difficulty 중
  
  # Night mode with strict crossing avoidance
  python main.py --lat 37.5603066 --lon 126.9368383 --night-mode --crossing-policy forbid
  
  # Scenic route with 40% scenic ratio target
  python main.py --lat 37.5603066 --lon 126.9368383 --scenic-ratio 0.4
        """
    )
    
    # Location parameters
    parser.add_argument("--place", type=str, default="Seodaemun-gu, Seoul, South Korea",
                        help="Place name for graph (default: Seodaemun-gu, Seoul)")
    parser.add_argument("--lat", type=float, required=True,
                        help="Starting latitude")
    parser.add_argument("--lon", type=float, required=True,
                        help="Starting longitude")
    
    # Route parameters
    parser.add_argument("--distance", type=float, default=5000,
                        help="Target distance in meters (default: 5000)")
    parser.add_argument("--tolerance", type=float, default=0.15,
                        help="Distance tolerance ratio (default: 0.15)")
    parser.add_argument("--difficulty", type=str, choices=['최하', '하', '중', '상'], default='최하',
                        help="Difficulty level (default: 최하)")
    
    # Safety parameters
    parser.add_argument("--night-mode", action="store_true",
                        help="Enable night safety mode")
    parser.add_argument("--night-weight", type=float, default=10.0,
                        help="Night penalty weight (default: 10.0)")
    parser.add_argument("--dark-threshold", type=float, default=0.5,
                        help="Dark threshold for safety (default: 0.5)")
    
    # Crossing parameters
    parser.add_argument("--crossing-policy", type=str, choices=['forbid', 'penalize', 'allow'], 
                        default='penalize',
                        help="Crossing policy (default: penalize)")
    parser.add_argument("--crossing-penalty", type=float, default=30.0,
                        help="Crossing penalty in meters (default: 30.0)")
    
    # Scenic parameters
    parser.add_argument("--scenic-ratio", type=float, default=0.3,
                        help="Target scenic ratio 0-1 (default: 0.3)")
    
    # Requirements
    parser.add_argument("--require-convenience", action="store_true",
                        help="Require convenience store")
    parser.add_argument("--require-toilets", action="store_true",
                        help="Require toilets")
    parser.add_argument("--require-scenic", action="store_true",
                        help="Require scenic location")
    parser.add_argument("--require-subway", action="store_true",
                        help="Require subway station")
    
    # File paths
    parser.add_argument("--csv", type=str, default="dataset/Mega DB.csv",
                        help="Path to Mega DB CSV (default: dataset/Mega DB.csv)")
    parser.add_argument("--output", type=str, default="running_loop_route.html",
                        help="Output HTML file (default: running_loop_route.html)")
    
    # Advanced parameters
    parser.add_argument("--w-grade", type=float, default=0.6,
                        help="Grade weight (default: 0.6)")
    parser.add_argument("--w-cross", type=float, default=1.0,
                        help="Crossing weight (default: 1.0)")
    
    args = parser.parse_args()
    
    try:
        result = generate_running_route(
            place_name=args.place,
            start_lat=args.lat,
            start_lon=args.lon,
            desired_distance_m=args.distance,
            tolerance_ratio=args.tolerance,
            difficulty=args.difficulty,
            include_length=True,
            night_mode=args.night_mode,
            W_NIGHT=args.night_weight,
            night_exp=3.0,
            dark_threshold=args.dark_threshold,
            w_grade=args.w_grade,
            w_cross=args.w_cross,
            crossing_policy=args.crossing_policy,
            crossing_penalty_meters=args.crossing_penalty,
            target_scenic_ratio=args.scenic_ratio,
            require_convenience=args.require_convenience,
            require_toilets=args.require_toilets,
            require_scenic_once=args.require_scenic,
            require_subway=args.require_subway,
            night_combine="min",
            alpha_streetlight=0.5,
            scenic_search_iters=8,
            csv_path=args.csv,
            output_html=args.output
        )
        
        print("\n" + "="*60)
        print("🎉 SUCCESS! Route generation completed.")
        print("="*60)
        print(f"\n Final Results:")
        print(f"   Distance: {result['length']:.1f}m")
        print(f"   Scenic ratio: {result['scenic_ratio']:.1%}")
        print(f"   Output: {args.output}")
        print("\n Open the HTML file in your browser to view the route!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)