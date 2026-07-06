from typing import Dict, List

import numpy as np

from tactile_ssl.graph.types import WeightedSensorGraph


def _connected_components(num_nodes: int, edge_index: np.ndarray) -> List[List[int]]:
    adjacency = [[] for _ in range(num_nodes)]
    for left, right in edge_index.T.tolist():
        adjacency[left].append(right)
        adjacency[right].append(left)

    seen = np.zeros(num_nodes, dtype=bool)
    components = []
    for node in range(num_nodes):
        if seen[node]:
            continue
        stack = [node]
        seen[node] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor in adjacency[current]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def sensor_graph_stats(graph: WeightedSensorGraph) -> Dict[str, float]:
    degrees = np.zeros(graph.num_nodes, dtype=np.int64)
    if graph.edge_index.shape[1]:
        np.add.at(degrees, graph.edge_index[0], 1)
        np.add.at(degrees, graph.edge_index[1], 1)
    components = _connected_components(graph.num_nodes, graph.edge_index)
    component_sizes = [len(component) for component in components]
    edge_weight = graph.edge_weight
    return {
        "num_nodes": int(graph.num_nodes),
        "num_edges": int(graph.edge_index.shape[1]),
        "avg_degree": float(degrees.mean()),
        "min_degree": int(degrees.min()),
        "max_degree": int(degrees.max()),
        "num_components": int(len(components)),
        "largest_component": int(max(component_sizes) if component_sizes else 0),
        "min_edge_weight": float(edge_weight.min()) if edge_weight.size else 0.0,
        "mean_edge_weight": float(edge_weight.mean()) if edge_weight.size else 0.0,
        "max_edge_weight": float(edge_weight.max()) if edge_weight.size else 0.0,
    }
