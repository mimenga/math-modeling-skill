# -*- coding: utf-8 -*-
"""
08 图论与网络
用途：最短路、最小生成树、最大流（TSP 等组合路径题用 07_智能优化）
依赖：networkx, numpy
"""
import numpy as np
import networkx as nx

# ---------- 1. 最短路（Dijkstra） ----------
def shortest_path(edges, source, target):
    """edges: [(u, v, weight), ...]"""
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return nx.dijkstra_path(G, source, target), nx.dijkstra_path_length(G, source, target)

# ---------- 2. 最小生成树 ----------
def mst(edges):
    G = nx.Graph()
    G.add_weighted_edges_from(edges)
    return nx.minimum_spanning_tree(G)

# ---------- 3. 最大流 ----------
def max_flow(edges, source, sink):
    """edges: [(u, v, capacity), ...]。同一对节点的重复边容量自动累加
    （add_edge 会直接覆盖，导致总容量偏小）。"""
    G = nx.DiGraph()
    for u, v, c in edges:
        if G.has_edge(u, v):
            G[u][v]["capacity"] += c
        else:
            G.add_edge(u, v, capacity=c)
    return nx.maximum_flow_value(G, source, sink)

if __name__ == "__main__":
    edges = [(0, 1, 4), (0, 2, 2), (1, 2, 1), (1, 3, 5), (2, 3, 8)]
    path, dist = shortest_path(edges, 0, 3)
    print("最短路径:", path, "距离:", dist)
    print("最小生成树边:", list(mst(edges).edges(data=True)))
